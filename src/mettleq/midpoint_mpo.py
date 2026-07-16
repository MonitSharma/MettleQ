"""Midpoint-MPO/TNO simulation with greedy unswapping.

This is a distinct, circuit-level tensor-network method. It is intentionally
not selected by MettleQ's ordinary statevector/MPS dispatcher: the algorithm
splits a circuit at its midpoint, cancels the two halves into an MPO, greedily
removes routing swaps, materializes an MPS, and samples that MPS.

The low-level compression core is derived from the Apache-2.0 licensed
``alexgalda-m/peaked-mpo-solver``. MettleQ owns the stable API, trust report,
convergence report, CLI, and benchmark contract around that core.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np


PUBLISHED_P9_EXPECTED_BITSTRING = (
    "01101110111001100000100000001010011100101101010111110111"
)
VENDORED_SOLVER_COMMIT = "3bcdc1e5bfd6abb9425f71bd43e560d2b27f45c1"
_QUIMB_SVD_TELEMETRY = {
    "calls": 0,
    "matrix_size_buckets": {
        "le_256": 0,
        "257_to_1024": 0,
        "1025_to_4096": 0,
        "4097_to_16384": 0,
        "16385_to_65536": 0,
        "65537_to_262144": 0,
        "gt_262144": 0,
    },
    "unscaled_numpy_failures": 0,
    "unscaled_failure_details": [],
    "isolated_scipy_gesvd_calls": 0,
    "isolated_scipy_gesvd_successes": 0,
    "isolated_scipy_gesvd_failures": 0,
    "rescaled_calls": 0,
    "numpy_complex128_failures": 0,
    "scipy_gesdd_failures": 0,
    "eigh_fallbacks": 0,
}


class MidpointMPOError(RuntimeError):
    """Raised when midpoint-MPO compression cannot produce a sampled state."""

    def __init__(self, message: str, *, diagnostics: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


class MidpointMPODependencyError(ImportError):
    """Raised when the optional tensor-network dependencies are unavailable."""


class MidpointMPOWorkerError(MidpointMPOError):
    """Raised when the isolated pinned midpoint-MPO worker fails."""


class _IsolatedSVDProcessError(RuntimeError):
    """Raised when the killable SciPy SVD child cannot return factors."""


def _reset_quimb_safe_svd_telemetry() -> None:
    for key, value in _QUIMB_SVD_TELEMETRY.items():
        if isinstance(value, list):
            value.clear()
        elif isinstance(value, dict):
            for item in value:
                value[item] = 0
        else:
            _QUIMB_SVD_TELEMETRY[key] = 0


def _record_svd_matrix_size(size: int) -> None:
    buckets = _QUIMB_SVD_TELEMETRY["matrix_size_buckets"]
    if size <= 256:
        buckets["le_256"] += 1
    elif size <= 1024:
        buckets["257_to_1024"] += 1
    elif size <= 4096:
        buckets["1025_to_4096"] += 1
    elif size <= 16_384:
        buckets["4097_to_16384"] += 1
    elif size <= 65_536:
        buckets["16385_to_65536"] += 1
    elif size <= 262_144:
        buckets["65537_to_262144"] += 1
    else:
        buckets["gt_262144"] += 1


def _eigh_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct an SVD from a Hermitian eigensolve as a final safe fallback."""

    rows, columns = matrix.shape
    if rows >= columns:
        values, vectors = np.linalg.eigh(matrix.conj().T @ matrix)
        order = np.argsort(values)[::-1]
        singular = np.sqrt(np.maximum(values[order], 0.0))
        right_h = vectors[:, order].conj().T
        left = matrix @ right_h.conj().T
        nonzero = singular > np.finfo(singular.dtype).eps * max(
            float(singular[0]) if singular.size else 0.0, 1.0
        )
        left[:, nonzero] /= singular[nonzero]
        if np.any(~nonzero):
            left[:, ~nonzero] = 0.0
        return left, singular, right_h
    values, vectors = np.linalg.eigh(matrix @ matrix.conj().T)
    order = np.argsort(values)[::-1]
    singular = np.sqrt(np.maximum(values[order], 0.0))
    left = vectors[:, order]
    right_h = left.conj().T @ matrix
    nonzero = singular > np.finfo(singular.dtype).eps * max(
        float(singular[0]) if singular.size else 0.0, 1.0
    )
    right_h[nonzero, :] /= singular[nonzero, None]
    if np.any(~nonzero):
        right_h[~nonzero, :] = 0.0
    return left, singular, right_h


def _scipy_gesvd_child(connection, matrix: np.ndarray) -> None:
    """Run the Quimb-compatible fallback outside the simulation process."""

    try:
        from scipy.linalg import svd

        factors = svd(
            matrix,
            full_matrices=False,
            lapack_driver="gesvd",
            check_finite=False,
        )
        connection.send(("ok", factors))
    except BaseException as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _isolated_scipy_gesvd(
    matrix: np.ndarray,
    *,
    timeout_seconds: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contain SciPy ``gesvd`` so a native failure cannot kill the worker."""

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_scipy_gesvd_child,
        args=(sender, np.asarray(matrix)),
        daemon=True,
    )
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout_seconds):
            raise _IsolatedSVDProcessError(
                f"isolated scipy gesvd exceeded {timeout_seconds:g} seconds"
            )
        try:
            status, payload = receiver.recv()
        except EOFError as error:
            process.join(timeout=5.0)
            raise _IsolatedSVDProcessError(
                f"isolated scipy gesvd exited {process.exitcode} without a result"
            ) from error
    finally:
        receiver.close()
        if process.is_alive():
            process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
        process.join(timeout=5.0)
    if process.exitcode != 0:
        raise _IsolatedSVDProcessError(
            f"isolated scipy gesvd exited {process.exitcode}"
        )
    if status != "ok":
        raise _IsolatedSVDProcessError(f"isolated scipy gesvd failed: {payload}")
    return payload


def _install_quimb_safe_svd() -> None:
    """Replace Quimb's native SVD with a recoverable userspace ladder."""

    from quimb.tensor import decomp
    import inspect

    if getattr(decomp, "_mettleq_safe_svd_installed", False):
        return
    trim_supports_error = "calc_error" in inspect.signature(
        decomp._trim_and_renorm_svd_result_numba
    ).parameters

    def safe_svd_truncated(
        matrix,
        cutoff=-1.0,
        cutoff_mode=4,
        max_bond=-1,
        absorb=0,
        renorm=0,
        calc_error=False,
        **_ignored,
    ):
        array = np.asarray(matrix)
        _QUIMB_SVD_TELEMETRY["calls"] += 1
        _record_svd_matrix_size(int(array.size))
        attempts = []

        def require_finite(left, singular, right_h, driver):
            if not (
                np.all(np.isfinite(left))
                and np.all(np.isfinite(singular))
                and np.all(np.isfinite(right_h))
            ):
                raise np.linalg.LinAlgError(f"{driver} returned non-finite factors")
            return left, singular, right_h

        def trim(left, singular, right_h):
            arguments = (
                left,
                singular,
                right_h,
                cutoff,
                cutoff_mode,
                max_bond,
                absorb,
                renorm,
            )
            try:
                if trim_supports_error:
                    return decomp._trim_and_renorm_svd_result_numba(
                        *arguments, calc_error=calc_error
                    )
                return decomp._trim_and_renorm_svd_result_numba(*arguments)
            except ValueError as error:
                # Quimb catches ValueError outside this function and enters
                # its unsafe in-process ``gesvd`` fallback.
                raise MidpointMPOError(
                    f"midpoint-MPO singular-value trimming failed: {error}"
                ) from error

        # Quimb's Numba wrapper can terminate inside native LAPACK before a
        # Python exception exists. Calling NumPy directly preserves the
        # unscaled numerical trajectory while turning non-convergence into a
        # catchable LinAlgError.
        try:
            unscaled_factors = require_finite(
                *np.linalg.svd(array, full_matrices=False),
                "numpy_unscaled",
            )
        except Exception as error:
            attempts.append(f"numpy_unscaled: {error}")
            _QUIMB_SVD_TELEMETRY["unscaled_numpy_failures"] += 1
            failure = {
                "call": _QUIMB_SVD_TELEMETRY["calls"],
                "shape": [int(size) for size in array.shape],
                "dtype": str(array.dtype),
                "max_abs": float(np.max(np.abs(array))) if array.size else 0.0,
                "error": f"{type(error).__name__}: {error}",
            }
            _QUIMB_SVD_TELEMETRY["unscaled_failure_details"].append(failure)
            print(f"[safe-svd] unscaled failure {failure}", flush=True)
        else:
            return trim(*unscaled_factors)

        # Quimb normally enters scipy ``gesvd`` here. Preserve that numerical
        # path, but run it in a fresh child because the native driver has been
        # observed terminating the long-lived simulation process.
        _QUIMB_SVD_TELEMETRY["isolated_scipy_gesvd_calls"] += 1
        try:
            isolated_factors = require_finite(
                *_isolated_scipy_gesvd(array),
                "isolated_scipy_gesvd",
            )
        except Exception as error:
            attempts.append(f"isolated_scipy_gesvd: {error}")
            _QUIMB_SVD_TELEMETRY["isolated_scipy_gesvd_failures"] += 1
        else:
            _QUIMB_SVD_TELEMETRY["isolated_scipy_gesvd_successes"] += 1
            return trim(*isolated_factors)

        scale = float(np.max(np.abs(array))) if array.size else 0.0
        if not np.isfinite(scale):
            raise MidpointMPOError("midpoint-MPO SVD input contains NaN or infinity")
        if scale == 0.0:
            scale = 1.0
        _QUIMB_SVD_TELEMETRY["rescaled_calls"] += 1
        scaled = np.asarray(array / scale, dtype=np.complex128)

        try:
            left, singular, right_h = require_finite(
                *np.linalg.svd(scaled, full_matrices=False),
                "numpy_complex128",
            )
        except Exception as error:
            attempts.append(f"numpy_complex128: {error}")
            _QUIMB_SVD_TELEMETRY["numpy_complex128_failures"] += 1
            try:
                from scipy.linalg import svd

                left, singular, right_h = require_finite(
                    *svd(
                        scaled,
                        full_matrices=False,
                        lapack_driver="gesdd",
                        check_finite=False,
                    ),
                    "scipy_gesdd",
                )
            except Exception as error:
                attempts.append(f"scipy_gesdd: {error}")
                _QUIMB_SVD_TELEMETRY["scipy_gesdd_failures"] += 1
                try:
                    left, singular, right_h = require_finite(
                        *_eigh_svd(scaled), "eigh"
                    )
                    _QUIMB_SVD_TELEMETRY["eigh_fallbacks"] += 1
                except Exception as error:
                    attempts.append(f"eigh: {error}")
                    raise MidpointMPOError(
                        "all recoverable midpoint-MPO SVD drivers failed: "
                        + "; ".join(attempts)
                    ) from error
        return trim(left, singular * scale, right_h)

    # ``svd_truncated_numpy`` resolves this global on every call. Replacing it
    # bypasses Quimb's ``gesvd`` fallback, which can terminate the worker before
    # Python can report the numerical failure.
    decomp.svd_truncated_numba = safe_svd_truncated
    decomp._mettleq_safe_svd_installed = True


@dataclass(frozen=True)
class MidpointMPOOptions:
    """Controls the midpoint-MPO compression and unswapping contract."""

    max_bond: int = 512
    cutoff: float = 6e-4
    unswap_threshold: float = 500_000.0
    center_ratio: float = 0.5
    max_unswap_iterations: int = 20
    sabre_trials: int = 90
    post_sabre_trials: int = 50
    seed: int = 123
    abort_after_no_progress_unswap_cycles: Optional[int] = 20
    max_unswap_cycles: Optional[int] = None
    max_work_gates: Optional[int] = None
    swap_gate_representation: str = "current"
    reuse_full_swap_probe: bool = True
    parallel_rewire: bool = False
    parallel_absorb_probes: bool = False

    def __post_init__(self) -> None:
        if self.max_bond < 1:
            raise ValueError("max_bond must be positive")
        if not 0.0 <= self.cutoff < 1.0:
            raise ValueError("cutoff must be in [0, 1)")
        if self.unswap_threshold <= 0:
            raise ValueError("unswap_threshold must be positive")
        if not 0.0 < float(self.center_ratio) < 1.0:
            raise ValueError("center_ratio must be between zero and one")
        if self.max_unswap_iterations < 1:
            raise ValueError("max_unswap_iterations must be positive")
        if self.sabre_trials < 1 or self.post_sabre_trials < 1:
            raise ValueError("Sabre trial counts must be positive")
        if self.swap_gate_representation not in {"current", "cx", "block"}:
            raise ValueError(
                "swap_gate_representation must be 'current', 'cx', or 'block'"
            )


@dataclass
class MidpointMPOResult:
    """Inspectable result returned by :class:`MidpointMPOSimulator`."""

    counts: dict[str, int]
    shots: int
    predicted_bitstring: Optional[str]
    expected_bitstring: Optional[str]
    expected_peak_count: Optional[int]
    expected_peak_fraction: Optional[float]
    matches_expected_bitstring: Optional[bool]
    compression_time_s: float
    materialize_time_s: float
    sampling_time_s: float
    measurement_permutation: list[int]
    samples: list[str]
    raw_samples: list[str]
    diagnostics: dict[str, Any]
    stats: list[dict[str, Any]] = field(repr=False)

    @property
    def total_time_s(self) -> float:
        return self.compression_time_s + self.materialize_time_s + self.sampling_time_s

    def to_summary(self, *, include_counts: bool = True) -> dict[str, Any]:
        summary = {
            "method": "midpoint_mpo_unswapping",
            "shots": self.shots,
            "predicted_bitstring": self.predicted_bitstring,
            "expected_bitstring": self.expected_bitstring,
            "expected_peak_count": self.expected_peak_count,
            "expected_peak_fraction": self.expected_peak_fraction,
            "matches_expected_bitstring": self.matches_expected_bitstring,
            "compression_time_s": self.compression_time_s,
            "materialize_time_s": self.materialize_time_s,
            "sampling_time_s": self.sampling_time_s,
            "total_time_s": self.total_time_s,
            "measurement_permutation": self.measurement_permutation,
            "diagnostics": self.diagnostics,
        }
        if include_counts:
            summary["counts"] = self.counts
        return summary


def _require_tensor_network() -> tuple[Any, Any, Any, Any]:
    try:
        import qiskit
        import quimb
        import qiskit_quimb  # noqa: F401 - validates the bridge at the boundary
        from ._vendor.peaked_mpo.pipeline import (
            mpo_compress_unswap,
            mpo_to_mps,
        )
        _install_quimb_safe_svd()
    except ImportError as error:
        raise MidpointMPODependencyError(
            "The midpoint-MPO method needs MettleQ's optional tensor-network "
            "dependencies. Install with `pip install 'mettleq[tensor-network]'`."
        ) from error
    return qiskit, quimb, mpo_compress_unswap, mpo_to_mps


def _numeric_values(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _last_stage(rows: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any]:
    return dict(next((row for row in reversed(rows) if row.get("stage") == stage), {}))


def _summarize_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    termination = _last_stage(rows, "termination")
    timing = _last_stage(rows, "timing_summary")
    stage_counts = Counter(str(row.get("stage", "unknown")) for row in rows)
    consumed = _numeric_values(rows, "u_consumed_total")
    result = {
        "termination_reason": termination.get("termination_reason"),
        "termination_detail": termination.get("termination_detail"),
        "stall_mode": termination.get("stall_mode"),
        "work_gates_consumed": int(consumed[-1]) if consumed else 0,
        "total_work_gates": termination.get("total_work_gates"),
        "peak_max_bond": max(_numeric_values(rows, "max_bond"), default=None),
        "peak_total_elements": max(_numeric_values(rows, "total_elems"), default=None),
        "stage_rows": dict(stage_counts),
    }
    result.update(
        {
            key: value
            for key, value in timing.items()
            if key.endswith("_time_s")
        }
    )
    return result


def _sample_mps(mps: Any, shots: int, seed: int) -> list[tuple[Any, Any]]:
    """Use an explicit seed where supported, retaining old-quimb compatibility."""

    try:
        return list(mps.sample(shots, seed=seed))
    except TypeError:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            return list(mps.sample(shots))
        finally:
            np.random.set_state(state)


class MidpointMPOSimulator:
    """Qiskit-circuit simulator using midpoint MPO cancellation and unswapping.

    The class deliberately does not inherit from the general MettleQ backend.
    Its approximation and resource controls are materially different from a
    forward MPS simulation, so callers must opt into it explicitly.
    """

    method = "midpoint_mpo_unswapping"

    def __init__(self, options: Optional[MidpointMPOOptions] = None, **overrides: Any):
        if options is not None and overrides:
            raise TypeError("pass either options or keyword overrides, not both")
        self.options = options or MidpointMPOOptions(**overrides)
        self.last_result: Optional[MidpointMPOResult] = None

    @staticmethod
    def consolidate_circuit(circuit: Any) -> Any:
        try:
            from qiskit import QuantumCircuit
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
        except ImportError as error:
            raise MidpointMPODependencyError(
                "Qiskit is required for midpoint-MPO circuit input"
            ) from error
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("midpoint-MPO input must be a qiskit.QuantumCircuit")
        return PassManager(
            [Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]
        ).run(circuit.copy())

    def run(
        self,
        circuit: Any,
        *,
        shots: int = 1000,
        expected_bitstring: Optional[str] = None,
        on_progress: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> MidpointMPOResult:
        if shots < 1:
            raise ValueError("shots must be positive")
        qiskit, quimb, compress, materialize = _require_tensor_network()
        source_operations = len(circuit.data)
        compiled = self.consolidate_circuit(circuit)
        compiled_operations = len(compiled.data)
        rows: list[dict[str, Any]] = []

        def collect(row: Mapping[str, Any]) -> None:
            item = dict(row)
            rows.append(item)
            if on_progress is not None:
                on_progress(item)

        options = self.options
        _reset_quimb_safe_svd_telemetry()
        compression_started = time.perf_counter()
        mpo, layers_left, layers_right, returned_rows = compress(
            compiled,
            max_bond=options.max_bond,
            cutoff=options.cutoff,
            unswap_threshold=options.unswap_threshold,
            early_stopping_gates=0,
            center_ratio=options.center_ratio,
            equal=False,
            flip_freq=None,
            max_its=options.max_unswap_iterations,
            to_backend=None,
            seed=options.seed,
            hows=("both", "left", "right"),
            sabre_trials=options.sabre_trials,
            post_sabre_trials=options.post_sabre_trials,
            post_sabre_seed=None,
            sabre_heuristic="decay",
            on_stats=collect,
            max_unswap_cycles=options.max_unswap_cycles,
            max_work_gates=options.max_work_gates,
            abort_after_no_progress_unswap_cycles=(
                options.abort_after_no_progress_unswap_cycles
            ),
            absorb_score="total_elems",
            parallel_absorb_probes=options.parallel_absorb_probes,
            parallel_rewire=options.parallel_rewire,
            adaptive_parallel_rewire=False,
            route_candidates=1,
            route_score="none",
            swap_apply_method="mpo",
            swap_gate_representation=options.swap_gate_representation,
            unswap_select_mode="bond",
            reuse_full_swap_probe=options.reuse_full_swap_probe,
        )
        compression_time_s = time.perf_counter() - compression_started
        if returned_rows is not rows:
            rows = [dict(row) for row in returned_rows]
        stats_summary = _summarize_stats(rows)
        termination = stats_summary.get("termination_reason")
        if termination not in (None, "completed", "early_stopping_gates"):
            raise MidpointMPOError(
                f"midpoint-MPO compression terminated with {termination!r}",
                diagnostics=stats_summary,
            )
        if options.max_unswap_cycles is not None or options.max_work_gates is not None:
            raise MidpointMPOError(
                "partial midpoint-MPO runs cannot be sampled",
                diagnostics=stats_summary,
            )

        materialize_started = time.perf_counter()
        mps, measurement_permutation = materialize(
            mpo,
            layers_left[:-2],
            layers_right,
            max_bond=options.max_bond,
            cutoff=options.cutoff,
            to_backend=None,
        )
        materialize_time_s = time.perf_counter() - materialize_started

        sampling_started = time.perf_counter()
        raw_pairs = _sample_mps(mps, shots, options.seed)
        sampling_time_s = time.perf_counter() - sampling_started
        raw_samples = ["".join(str(bit) for bit in bits) for bits, _ in raw_pairs]
        permutation = [int(index) for index in measurement_permutation]
        samples = ["".join(raw[index] for index in permutation) for raw in raw_samples]
        counts = dict(Counter(samples))
        predicted = max(counts, key=counts.get) if counts else None
        expected_count = counts.get(expected_bitstring, 0) if expected_bitstring else None
        expected_fraction = expected_count / shots if expected_count is not None else None
        matches = predicted == expected_bitstring if expected_bitstring else None
        diagnostics = {
            **stats_summary,
            "source_operation_count": source_operations,
            "consolidated_operation_count": compiled_operations,
            "final_max_bond": int(mpo.max_bond()),
            "final_tensor_elements": int(sum(np.prod(tensor.shape) for tensor in mpo)),
            "options": asdict(options),
            "implementation": "mettleq.midpoint_mpo",
            "vendor_reference_commit": VENDORED_SOLVER_COMMIT,
            "qiskit_version": qiskit.__version__,
            "quimb_version": quimb.__version__,
            "quimb_safe_svd": dict(_QUIMB_SVD_TELEMETRY),
        }
        result = MidpointMPOResult(
            counts=counts,
            shots=shots,
            predicted_bitstring=predicted,
            expected_bitstring=expected_bitstring,
            expected_peak_count=expected_count,
            expected_peak_fraction=expected_fraction,
            matches_expected_bitstring=matches,
            compression_time_s=compression_time_s,
            materialize_time_s=materialize_time_s,
            sampling_time_s=sampling_time_s,
            measurement_permutation=permutation,
            samples=samples,
            raw_samples=raw_samples,
            diagnostics=diagnostics,
            stats=rows,
        )
        self.last_result = result
        return result


class IsolatedMidpointMPOSimulator:
    """Run midpoint MPO in a pinned interpreter from a normal Qiskit process.

    Qiskit 2.x remains loaded only in the caller. The circuit crosses the
    process boundary as OpenQASM 2, while the worker imports the independently
    pinned Qiskit/Quimb stack and returns MettleQ's ordinary result schema.
    """

    method = "isolated_midpoint_mpo_unswapping"

    def __init__(
        self,
        options: Optional[MidpointMPOOptions] = None,
        *,
        worker_python: Optional[os.PathLike[str] | str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.options = options or MidpointMPOOptions()
        configured_python = worker_python or os.environ.get("METTLEQ_MPO_PYTHON")
        if configured_python is None:
            candidate = Path(__file__).resolve().parents[2] / ".venv-mpo/bin/python"
            configured_python = candidate if candidate.exists() else None
        if configured_python is None:
            raise MidpointMPODependencyError(
                "No isolated midpoint-MPO interpreter was configured. Create "
                "`.venv-mpo` from tools/requirements-midpoint-mpo-p9.txt or set "
                "METTLEQ_MPO_PYTHON."
            )
        # Preserve a virtual environment's interpreter symlink: resolving it
        # would bypass pyvenv.cfg and silently lose the pinned site-packages.
        self.worker_python = Path(
            os.path.abspath(os.fspath(Path(configured_python).expanduser()))
        )
        if not self.worker_python.exists():
            raise MidpointMPODependencyError(
                f"isolated midpoint-MPO interpreter does not exist: {self.worker_python}"
            )
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive or None")
        self.timeout_seconds = timeout_seconds
        self.last_result: Optional[MidpointMPOResult] = None

    @staticmethod
    def _validate_circuit(circuit: Any) -> tuple[Any, Any]:
        try:
            import qiskit
            from qiskit import QuantumCircuit, qasm2
        except ImportError as error:
            raise MidpointMPODependencyError(
                "Qiskit is required in the caller to export the worker circuit"
            ) from error
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("isolated midpoint-MPO input must be a qiskit.QuantumCircuit")
        if circuit.parameters:
            raise MidpointMPOError(
                "isolated midpoint-MPO input must bind all circuit parameters"
            )
        unsupported = []
        for instruction in circuit.data:
            operation = getattr(instruction, "operation", None)
            if operation is None:
                operation = instruction[0]
            if operation.name in {
                "measure",
                "reset",
                "delay",
                "initialize",
                "store",
            } or getattr(operation, "condition", None) is not None:
                unsupported.append(operation.name)
        if unsupported:
            names = ", ".join(sorted(set(unsupported)))
            raise MidpointMPOError(
                "isolated midpoint-MPO accepts unitary circuits without dynamic "
                f"semantics; unsupported operations: {names}"
            )
        return qiskit, qasm2

    def _command(
        self,
        *,
        qasm_path: Path,
        output_dir: Path,
        shots: int,
        expected_bitstring: Optional[str],
    ) -> list[str]:
        options = self.options
        command = [
            str(self.worker_python),
            "-m",
            "mettleq.midpoint_mpo",
            "--qasm",
            str(qasm_path),
            "--output-dir",
            str(output_dir),
            "--shots",
            str(shots),
            "--max-bond",
            str(options.max_bond),
            "--cutoff",
            str(options.cutoff),
            "--unswap-threshold",
            str(options.unswap_threshold),
            "--center-ratio",
            str(options.center_ratio),
            "--max-unswap-iterations",
            str(options.max_unswap_iterations),
            "--seed",
            str(options.seed),
            "--sabre-trials",
            str(options.sabre_trials),
            "--post-sabre-trials",
            str(options.post_sabre_trials),
            "--swap-gate-representation",
            options.swap_gate_representation,
        ]
        if expected_bitstring is not None:
            command.extend(["--expected-bitstring", expected_bitstring])
        if options.abort_after_no_progress_unswap_cycles is not None:
            command.extend(
                [
                    "--no-progress-limit",
                    str(options.abort_after_no_progress_unswap_cycles),
                ]
            )
        else:
            command.append("--allow-unbounded-no-progress")
        if options.max_unswap_cycles is not None:
            command.extend(["--max-unswap-cycles", str(options.max_unswap_cycles)])
        if options.max_work_gates is not None:
            command.extend(["--max-work-gates", str(options.max_work_gates)])
        if not options.reuse_full_swap_probe:
            command.append("--no-reuse-full-swap-probe")
        if options.parallel_rewire:
            command.append("--parallel-rewire")
        if options.parallel_absorb_probes:
            command.append("--parallel-absorb-probes")
        return command

    @staticmethod
    def _read_result(output_dir: Path) -> MidpointMPOResult:
        summary = json.loads((output_dir / "summary.json").read_text())
        stats = json.loads((output_dir / "stats.json").read_text())
        raw_samples = []
        samples = []
        with (output_dir / "samples.tsv").open() as handle:
            next(handle, None)
            for line in handle:
                raw, permuted = line.rstrip("\n").split("\t", 1)
                raw_samples.append(raw)
                samples.append(permuted)
        return MidpointMPOResult(
            counts={str(key): int(value) for key, value in summary["counts"].items()},
            shots=int(summary["shots"]),
            predicted_bitstring=summary.get("predicted_bitstring"),
            expected_bitstring=summary.get("expected_bitstring"),
            expected_peak_count=summary.get("expected_peak_count"),
            expected_peak_fraction=summary.get("expected_peak_fraction"),
            matches_expected_bitstring=summary.get("matches_expected_bitstring"),
            compression_time_s=float(summary["compression_time_s"]),
            materialize_time_s=float(summary["materialize_time_s"]),
            sampling_time_s=float(summary["sampling_time_s"]),
            measurement_permutation=[
                int(value) for value in summary["measurement_permutation"]
            ],
            samples=samples,
            raw_samples=raw_samples,
            diagnostics=dict(summary["diagnostics"]),
            stats=[dict(row) for row in stats],
        )

    def run(
        self,
        circuit: Any,
        *,
        shots: int = 1000,
        expected_bitstring: Optional[str] = None,
        output_dir: Optional[os.PathLike[str] | str] = None,
    ) -> MidpointMPOResult:
        if shots < 1:
            raise ValueError("shots must be positive")
        caller_qiskit, qasm2 = self._validate_circuit(circuit)
        temporary = None
        if output_dir is None:
            temporary = tempfile.TemporaryDirectory(prefix="mettleq-mpo-worker-")
            worker_dir = Path(temporary.name)
        else:
            worker_dir = Path(output_dir).expanduser().resolve()
            worker_dir.mkdir(parents=True, exist_ok=True)
        qasm_path = worker_dir / "input.qasm"
        with qasm_path.open("w") as handle:
            qasm2.dump(circuit, handle)
        command = self._command(
            qasm_path=qasm_path,
            output_dir=worker_dir,
            shots=shots,
            expected_bitstring=expected_bitstring,
        )
        log_path = worker_dir / "worker.log"
        source_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (str(source_root), existing_pythonpath)
            if value
        )
        started = time.perf_counter()
        try:
            with log_path.open("w") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    timeout=self.timeout_seconds,
                    check=False,
                )
        except subprocess.TimeoutExpired as error:
            raise MidpointMPOWorkerError(
                f"isolated midpoint-MPO worker exceeded {self.timeout_seconds}s",
                diagnostics={"command": command, "worker_log": str(log_path)},
            ) from error
        worker_wall_time_s = time.perf_counter() - started
        if completed.returncode != 0:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
            raise MidpointMPOWorkerError(
                f"isolated midpoint-MPO worker exited {completed.returncode}:\n{tail}",
                diagnostics={"command": command, "worker_log": str(log_path)},
            )
        try:
            result = self._read_result(worker_dir)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise MidpointMPOWorkerError(
                "isolated midpoint-MPO worker returned an invalid result bundle",
                diagnostics={"command": command, "worker_log": str(log_path)},
            ) from error
        result.diagnostics.update(
            {
                "execution_mode": "isolated_worker",
                "caller_qiskit_version": caller_qiskit.__version__,
                "worker_python": str(self.worker_python),
                "worker_wall_time_s": worker_wall_time_s,
                "worker_output_dir": str(worker_dir) if output_dir is not None else None,
            }
        )
        self.last_result = result
        if temporary is not None:
            temporary.cleanup()
        return result


def build_convergence_report(
    results: Sequence[MidpointMPOResult],
    *,
    peak_fraction_atol: float = 0.03,
) -> dict[str, Any]:
    """Classify independent cutoff/bond convergence using peak evidence.

    A qualified bond comparison holds cutoff fixed; a qualified cutoff
    comparison holds maximum bond fixed. This prevents two diagonal points
    that change both approximation controls from being called converged.
    """

    if peak_fraction_atol <= 0:
        raise ValueError("peak_fraction_atol must be positive")
    points = []
    for result in results:
        options = result.diagnostics.get("options", {})
        points.append(
            {
                "max_bond": options.get("max_bond"),
                "cutoff": options.get("cutoff"),
                "shots": result.shots,
                "predicted_bitstring": result.predicted_bitstring,
                "expected_peak_count": result.expected_peak_count,
                "expected_peak_fraction": result.expected_peak_fraction,
                "matches_expected_bitstring": result.matches_expected_bitstring,
                "total_time_s": result.total_time_s,
            }
        )
    expected_fractions = [
        result.expected_peak_fraction
        for result in results
        if result.expected_peak_fraction is not None
    ]
    predictions = [result.predicted_bitstring for result in results]
    same_prediction = len(set(predictions)) == 1 if predictions else False
    expected_recovered = bool(results) and all(
        result.matches_expected_bitstring is True for result in results
    )
    spread = (
        max(expected_fractions) - min(expected_fractions)
        if len(expected_fractions) >= 2
        else None
    )
    axis_reports = []

    def add_axis_reports(
        *,
        axis: str,
        varied_key: str,
        fixed_key: str,
    ) -> None:
        groups: dict[Any, list[tuple[MidpointMPOResult, dict[str, Any]]]] = {}
        for result, point in zip(results, points):
            fixed_value = point[fixed_key]
            varied_value = point[varied_key]
            if fixed_value is None or varied_value is None:
                continue
            groups.setdefault(fixed_value, []).append((result, point))
        for fixed_value, group in groups.items():
            varied_values = sorted({point[varied_key] for _, point in group})
            if len(varied_values) < 2:
                continue
            group_predictions = [result.predicted_bitstring for result, _ in group]
            group_fractions = [
                result.expected_peak_fraction
                for result, _ in group
                if result.expected_peak_fraction is not None
            ]
            group_spread = (
                max(group_fractions) - min(group_fractions)
                if len(group_fractions) >= 2
                else None
            )
            group_same_prediction = len(set(group_predictions)) == 1
            group_expected_recovered = all(
                result.matches_expected_bitstring is True for result, _ in group
            )
            group_converged = bool(
                group_same_prediction
                and group_expected_recovered
                and group_spread is not None
                and group_spread <= peak_fraction_atol
            )
            axis_reports.append(
                {
                    "axis": axis,
                    "fixed_parameter": fixed_key,
                    "fixed_value": fixed_value,
                    "varied_parameter": varied_key,
                    "varied_values": varied_values,
                    "point_count": len(group),
                    "same_predicted_bitstring": group_same_prediction,
                    "expected_peak_recovered_all_points": group_expected_recovered,
                    "expected_peak_fraction_spread": group_spread,
                    "converged": group_converged,
                }
            )

    add_axis_reports(axis="bond", varied_key="max_bond", fixed_key="cutoff")
    add_axis_reports(axis="cutoff", varied_key="cutoff", fixed_key="max_bond")
    converged = bool(axis_reports) and all(
        report["converged"] for report in axis_reports
    )
    return {
        "classification": "converged" if converged else "not_converged",
        "converged": converged,
        "required_points": 2,
        "peak_fraction_atol": peak_fraction_atol,
        "expected_peak_fraction_spread": spread,
        "same_predicted_bitstring": same_prediction,
        "expected_peak_recovered_all_points": expected_recovered,
        "qualified_comparison_count": len(axis_reports),
        "convergence_axes": sorted({report["axis"] for report in axis_reports}),
        "axis_reports": axis_reports,
        "points": points,
    }


def _write_result(output_dir: Path, result: MidpointMPOResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result.to_summary(), indent=2, default=str) + "\n"
    )
    (output_dir / "stats.json").write_text(
        json.dumps(result.stats, indent=2, default=str) + "\n"
    )
    with (output_dir / "samples.tsv").open("w") as handle:
        handle.write("raw\tpermuted\n")
        for raw, sample in zip(result.raw_samples, result.samples):
            handle.write(f"{raw}\t{sample}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run MettleQ's midpoint-MPO/TNO plus unswapping simulator."
    )
    parser.add_argument("--qasm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--expected-bitstring", default=None)
    parser.add_argument("--max-bond", type=int, default=512)
    parser.add_argument("--cutoff", type=float, default=6e-4)
    parser.add_argument("--unswap-threshold", type=float, default=500_000.0)
    parser.add_argument("--center-ratio", type=float, default=0.5)
    parser.add_argument("--max-unswap-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sabre-trials", type=int, default=90)
    parser.add_argument("--post-sabre-trials", type=int, default=50)
    parser.add_argument("--no-progress-limit", type=int, default=20)
    parser.add_argument("--allow-unbounded-no-progress", action="store_true")
    parser.add_argument("--max-unswap-cycles", type=int, default=None)
    parser.add_argument("--max-work-gates", type=int, default=None)
    parser.add_argument(
        "--swap-gate-representation",
        choices=("current", "cx", "block"),
        default="current",
    )
    parser.add_argument("--no-reuse-full-swap-probe", action="store_true")
    parser.add_argument("--parallel-rewire", action="store_true")
    parser.add_argument("--parallel-absorb-probes", action="store_true")
    args = parser.parse_args(argv)
    try:
        from qiskit import qasm2
    except ImportError as error:
        raise MidpointMPODependencyError("Qiskit is required to load QASM input") from error
    circuit = qasm2.load(
        str(args.qasm), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
    )
    options = MidpointMPOOptions(
        max_bond=args.max_bond,
        cutoff=args.cutoff,
        unswap_threshold=args.unswap_threshold,
        center_ratio=args.center_ratio,
        max_unswap_iterations=args.max_unswap_iterations,
        seed=args.seed,
        sabre_trials=args.sabre_trials,
        post_sabre_trials=args.post_sabre_trials,
        abort_after_no_progress_unswap_cycles=(
            None if args.allow_unbounded_no_progress else args.no_progress_limit
        ),
        max_unswap_cycles=args.max_unswap_cycles,
        max_work_gates=args.max_work_gates,
        swap_gate_representation=args.swap_gate_representation,
        reuse_full_swap_probe=not args.no_reuse_full_swap_probe,
        parallel_rewire=args.parallel_rewire,
        parallel_absorb_probes=args.parallel_absorb_probes,
    )

    def progress(row: dict[str, Any]) -> None:
        if row.get("stage") == "cycle_progress":
            print(
                f"[cycle {row.get('unswap_cycle')}] "
                f"{row.get('gates_consumed')}/{row.get('total_work_gates')} "
                f"work gates after {float(row.get('time', 0.0)):.0f}s",
                flush=True,
            )

    result = MidpointMPOSimulator(options).run(
        circuit,
        shots=args.shots,
        expected_bitstring=args.expected_bitstring,
        on_progress=progress,
    )
    result.diagnostics["platform"] = platform.platform()
    result.diagnostics["python"] = platform.python_version()
    _write_result(args.output_dir, result)
    print(json.dumps(result.to_summary(include_counts=False), indent=2, default=str))
    return 0


__all__ = [
    "PUBLISHED_P9_EXPECTED_BITSTRING",
    "MidpointMPODependencyError",
    "MidpointMPOError",
    "MidpointMPOWorkerError",
    "MidpointMPOOptions",
    "MidpointMPOResult",
    "MidpointMPOSimulator",
    "IsolatedMidpointMPOSimulator",
    "build_convergence_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
