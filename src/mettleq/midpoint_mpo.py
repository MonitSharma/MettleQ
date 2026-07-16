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
from pathlib import Path
import platform
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np


PUBLISHED_P9_EXPECTED_BITSTRING = (
    "01101110111001100000100000001010011100101101010111110111"
)
VENDORED_SOLVER_COMMIT = "3bcdc1e5bfd6abb9425f71bd43e560d2b27f45c1"


class MidpointMPOError(RuntimeError):
    """Raised when midpoint-MPO compression cannot produce a sampled state."""

    def __init__(self, message: str, *, diagnostics: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


class MidpointMPODependencyError(ImportError):
    """Raised when the optional tensor-network dependencies are unavailable."""


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


def build_convergence_report(
    results: Sequence[MidpointMPOResult],
    *,
    peak_fraction_atol: float = 0.03,
) -> dict[str, Any]:
    """Classify cutoff/bond convergence using expected-peak evidence."""

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
    converged = (
        len(results) >= 2
        and same_prediction
        and expected_recovered
        and spread is not None
        and spread <= peak_fraction_atol
    )
    return {
        "classification": "converged" if converged else "not_converged",
        "converged": converged,
        "required_points": 2,
        "peak_fraction_atol": peak_fraction_atol,
        "expected_peak_fraction_spread": spread,
        "same_predicted_bitstring": same_prediction,
        "expected_peak_recovered_all_points": expected_recovered,
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
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sabre-trials", type=int, default=90)
    parser.add_argument("--post-sabre-trials", type=int, default=50)
    parser.add_argument("--no-progress-limit", type=int, default=20)
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
        seed=args.seed,
        sabre_trials=args.sabre_trials,
        post_sabre_trials=args.post_sabre_trials,
        abort_after_no_progress_unswap_cycles=args.no_progress_limit,
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
    "MidpointMPOOptions",
    "MidpointMPOResult",
    "MidpointMPOSimulator",
    "build_convergence_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
