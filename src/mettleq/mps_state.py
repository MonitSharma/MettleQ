from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional
import math
import time
import numpy as np
from scipy.linalg import qr as _scipy_qr

# pyrefly: ignore [missing-import]
from ._mlx_compat import mx
from .mps_routing import plan_routing as _plan_routing

try:  # pragma: no cover - depends on the optional compiled extension
    from ._mps_native import (
        expectation_product as _native_expectation_product,
        probabilities as _native_probabilities,
        qr_move_left as _native_qr_move_left,
        qr_move_right as _native_qr_move_right,
        sample as _native_sample,
        statevector as _native_statevector,
        two_site_update as _native_two_site_update,
    )
except ImportError:  # pragma: no cover - portable source fallback
    _native_expectation_product = None
    _native_probabilities = None
    _native_qr_move_left = None
    _native_qr_move_right = None
    _native_sample = None
    _native_statevector = None
    _native_two_site_update = None


def _reduced_qr(matrix: np.ndarray):
    """Economic QR without LAPACK input validation.

    ~1.8x faster than numpy.linalg.qr on the small canonicalization matrices
    (checked on complex64 shapes up to 128x64). ``check_finite=False`` is safe
    here: MPS canonicalization keeps entries near unit scale and the split path
    already validates the spectrum downstream.
    """
    return _scipy_qr(matrix, mode="economic", overwrite_a=False,
                     check_finite=False)


@lru_cache(maxsize=1)
def _swap_gate():
    """Build the logical SWAP matrix only when an MPS operation needs it."""
    return mx.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        mx.complex64,
    )


@dataclass
class MPSOptions:
    dmax: int = 64
    eps: float = 1e-10
    svd_driver: str = "auto"
    routing_strategy: str = "lookahead"
    routing_lookahead: int = 8
    renormalize_splits: bool = True


class MPSNumericalError(RuntimeError):
    """A recoverable numerical failure in compact MPS execution."""


_SVD_DRIVERS = {"auto", "gesdd", "gesvd", "numpy"}
_ROUTING_STRATEGIES = {"lookahead", "restore"}
# Persistent routing avoids end-of-circuit restoration, but it can increase
# the entanglement carried across many MPS cuts. Require a meaningful swap
# reduction before accepting it; this keeps grid-like workloads on the stable
# restore-per-gate path while allowing large wins for rainbow/long-range work.
_PERSISTENT_ROUTE_MIN_REDUCTION = 0.25
# Direct row updates beat general contraction once the two bond dimensions
# create a sufficiently wide batch. Keep this explicit so CPU benchmark runs
# can tune it for different Apple CPU generations without changing semantics.
_SINGLE_GATE_DIRECT_PRODUCT_THRESHOLD = 2048
# Repeated routing SWAPs are exact, but complex64 SVDs can leave tiny
# roundoff-only Schmidt values behind. Apply this floor only when approximation
# is already enabled (eps > 0); eps=0 remains an exact/no-extra-truncation mode.
_SWAP_NUMERICAL_TRUNCATION_FLOOR = 1e-6


@lru_cache(maxsize=4)
def _scipy_lapack_driver(driver: str):
    """Resolve the complex64 LAPACK function once per process."""
    from scipy.linalg.lapack import get_lapack_funcs

    probe = np.empty((1, 1), dtype=np.complex64, order="F")
    return get_lapack_funcs(driver, (probe,))


def _scipy_lapack_svd(matrix: np.ndarray, driver: str):
    """Call LAPACK directly, avoiding repeated high-level SVD dispatch."""
    lapack_svd = _scipy_lapack_driver(driver)
    U, S, Vh, info = lapack_svd(
        matrix,
        compute_uv=1,
        full_matrices=0,
        overwrite_a=1,
    )
    if info < 0:
        raise ValueError(f"LAPACK {driver} received invalid argument {-info}")
    if info > 0:
        raise np.linalg.LinAlgError(
            f"LAPACK {driver} did not converge (info={info})"
        )
    return U, S, Vh


def _safe_cpu_svd(matrix: np.ndarray, driver: str):
    """Run a catchable CPU SVD with a deterministic fallback ladder.

    MLX 0.32's CPU ``sgesvdx`` can terminate the process before Python can
    catch an exception. SciPy and NumPy expose LAPACK failures as Python
    exceptions, which lets the SDK return a useful numerical error instead of
    losing the whole process.
    """
    array = np.asarray(matrix, dtype=np.complex64)
    scale = float(np.max(np.abs(array))) if array.size else 0.0
    if not np.isfinite(scale):
        raise MPSNumericalError("MPS SVD input contains NaN or infinity")
    if scale <= 0.0:
        raise MPSNumericalError("MPS SVD input has zero or invalid scale")
    # MPS canonicalization normally keeps entries near unit scale. Avoid an
    # extra full-matrix division in that common case, while preserving the
    # defensive rescaling needed for extreme intermediate values. The only
    # mandatory copy is the Fortran-order scratch array handed to LAPACK.
    rescale = scale < 2.0**-32 or scale > 2.0**32
    scaled = array / scale if rescale else array
    output_scale = scale if rescale else 1.0
    attempts = []
    if driver in ("auto", "gesdd", "gesvd"):
        try:
            scipy_drivers = (
                ("gesdd", "gesvd") if driver == "auto" else (driver,)
            )
            for scipy_driver in scipy_drivers:
                try:
                    U, S, Vh = _scipy_lapack_svd(
                        np.array(
                            scaled,
                            dtype=np.complex64,
                            order="F",
                            copy=True,
                        ),
                        scipy_driver,
                    )
                    return (
                        U,
                        S * output_scale,
                        Vh,
                        f"scipy_{scipy_driver}",
                        attempts,
                    )
                except Exception as error:
                    attempts.append(f"scipy_{scipy_driver}: {error}")
        except ImportError as error:
            attempts.append(f"scipy_unavailable: {error}")
    if driver in ("auto", "numpy") or attempts:
        for dtype in (np.complex64, np.complex128):
            try:
                U, S, Vh = np.linalg.svd(
                    scaled.astype(dtype, copy=False), full_matrices=False
                )
                return (
                    U,
                    S * output_scale,
                    Vh,
                    f"numpy_{dtype.__name__}",
                    attempts,
                )
            except Exception as error:
                attempts.append(f"numpy_{dtype.__name__}: {error}")
    detail = "; ".join(attempts) or "no SVD driver was attempted"
    raise MPSNumericalError(f"All recoverable MPS SVD drivers failed: {detail}")


def _safe_cpu_qr(matrix: np.ndarray):
    """Use SciPy's overwrite-capable LAPACK path, with NumPy as fallback."""
    scratch = np.array(matrix, dtype=np.complex64, order="F", copy=True)
    try:
        from scipy.linalg import qr
        return qr(scratch, mode="economic", overwrite_a=True,
                  check_finite=False)
    except (ImportError, ValueError, np.linalg.LinAlgError):
        return np.linalg.qr(scratch, mode="reduced")


def _svd_truncate_numpy(
    matrix: np.ndarray,
    dmax: int,
    eps: float,
    *,
    driver: str = "auto",
    renormalize: bool = True,
):
    start = time.perf_counter_ns()
    U_np, S_np, Vh_np, used_driver, failed_attempts = _safe_cpu_svd(
        np.asarray(matrix, dtype=np.complex64), driver
    )
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    singular_values = np.asarray(S_np, dtype=np.float64).reshape(-1)
    r = int(singular_values.size)
    # eps-based cutoff
    r_eps = r
    if r > 0:
        thresh = eps * singular_values[0]
        r_eps = int(np.count_nonzero(singular_values >= thresh))
    r_keep = min(r, max(1, min(dmax, r_eps)))
    squared_values = singular_values * singular_values
    total_weight = float(np.sum(squared_values, dtype=np.float64))
    discarded_weight = float(
        np.sum(squared_values[r_keep:], dtype=np.float64)
    )
    kept_weight = float(np.sum(squared_values[:r_keep], dtype=np.float64))
    if not np.isfinite(kept_weight) or kept_weight <= np.finfo(float).tiny:
        raise MPSNumericalError(
            "MPS truncation retained a zero or non-finite singular spectrum"
        )
    normalization_factor = kept_weight ** 0.5 if renormalize else 1.0
    metadata = {
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "matrix_elements": int(matrix.size),
        "rank_before": r,
        "rank_kept": r_keep,
        "local_discarded_weight": discarded_weight,
        "relative_discarded_weight": (
            discarded_weight / total_weight if total_weight > 0.0 else 0.0
        ),
        "limited_by_dmax": dmax < r,
        "limited_by_eps": r_eps < r,
        "svd_driver": used_driver,
        "svd_failed_attempts": list(failed_attempts),
        "svd_elapsed_ms": elapsed_ms,
        "pre_normalization_norm": kept_weight ** 0.5,
        "renormalized": bool(renormalize),
    }
    U_out = U_np[:, :r_keep].astype(np.complex64, copy=False)
    S_out = (S_np[:r_keep] / normalization_factor).astype(
        np.float32, copy=False
    )
    Vh_out = Vh_np[:r_keep, :].astype(np.complex64, copy=False)
    return U_out, S_out, Vh_out, metadata


class MPSState:
    """Minimal MPS engine with nearest-neighbor 2-qubit gates.

    Tensors stored as list A[i] with shape (Dl, 2, Dr).
    """

    def __init__(self, n_qubits: int, opts: Optional[MPSOptions] = None):
        self.n = int(n_qubits)
        if self.n <= 0:
            raise ValueError("n_qubits must be positive")
        self.opts = opts or MPSOptions()
        if self.opts.dmax < 1:
            raise ValueError("MPS dmax must be at least 1")
        if self.opts.eps < 0.0:
            raise ValueError("MPS eps must be non-negative")
        if self.opts.svd_driver not in _SVD_DRIVERS:
            raise ValueError(
                f"MPS svd_driver must be one of {sorted(_SVD_DRIVERS)}"
            )
        if self.opts.routing_strategy not in _ROUTING_STRATEGIES:
            raise ValueError(
                "MPS routing_strategy must be 'lookahead' or 'restore'"
            )
        if self.opts.routing_lookahead < 0:
            raise ValueError("MPS routing_lookahead must be non-negative")
        # Bond diagnostics
        self.bonds: List[int] = [1] * max(0, self.n - 1)
        self.max_bond_ever: int = 1
        # Truncation diagnostics
        self.truncated_any: bool = False
        self.trunc_events: int = 0
        # MPS is intentionally CPU-native. Its small, sequential SVD/QR
        # workload benefits from CPU LAPACK and avoids GPU launch/transfer
        # overhead; statevector simulation remains the GPU path.
        self.tensor_device = "cpu"
        self.svd_device = "cpu"
        self.reset()

    def reset(self):
        # |0..0> in right-canonical form: A[i] = [ [1,0] ] up to dimensions.
        # Site tensors stay as NumPy arrays end-to-end. This avoids a device
        # round-trip for every gate and keeps the CPU MPS execution explicit.
        self.A: List = []
        for i in range(self.n):
            self.A.append(
                np.array([1, 0], dtype=np.complex64).reshape(1, 2, 1)
            )
        # Reset bond diagnostics
        self.bonds = [1] * max(0, self.n - 1)
        self.max_bond_ever = 1
        self.truncated_any = False
        self.trunc_events = 0
        self.local_discarded_weight_sum = 0.0
        self.local_discarded_weight_max = 0.0
        self.relative_discarded_weight_sum = 0.0
        self.relative_discarded_weight_max = 0.0
        self.last_truncation = None
        self.canonical_center = 0
        self.renormalization_count = 0
        self.last_pre_normalization_norm = 1.0
        self.svd_calls = 0
        self.svd_total_ms = 0.0
        self.svd_fallback_count = 0
        self.svd_drivers_used: dict[str, int] = {}
        self.two_site_calls = 0
        self.two_site_contraction_total_ms = 0.0
        self.two_site_split_total_ms = 0.0
        self.swap_fast_path_calls = 0
        self.two_site_matrix_shape_calls: dict[str, int] = {}
        self.two_site_max_matrix_elements = 0
        self.site_to_logical = list(range(self.n))
        self.logical_to_site = list(range(self.n))
        self.routing_logical_gates = 0
        self.routing_swaps = 0
        self.routing_naive_restore_swaps = 0
        self.routing_final_restore_swaps = 0
        self.routing_effective_strategy = self.opts.routing_strategy
        self.routing_selection_reason = "configured_strategy"
        self.routing_planned_lookahead_swaps = None
        self.routing_planned_restore_swaps = None
        self.routing_plan = None
        self.routing_plan_index = 0
        self.routing_planner = "python"

    def _as_stored(self, value):
        """Coerce a site tensor to the CPU-native storage type."""
        return np.asarray(value, dtype=np.complex64)

    def _replace_tensor(self, index: int, value: np.ndarray) -> None:
        self.A[index] = self._as_stored(value)

    def _refresh_bonds(self) -> None:
        self.bonds = [int(tensor.shape[2]) for tensor in self.A[:-1]]
        self.max_bond_ever = max(
            self.max_bond_ever, max(self.bonds, default=1)
        )

    def _move_center_right(self, site: int) -> None:
        tensor = np.asarray(self.A[site], dtype=np.complex64)
        next_tensor = np.asarray(self.A[site + 1], dtype=np.complex64)
        if _native_qr_move_right is not None:
            try:
                q_tensor, absorbed = _native_qr_move_right(
                    tensor, next_tensor
                )
                self._replace_tensor(site, q_tensor)
                self._replace_tensor(site + 1, absorbed)
                self.canonical_center = site + 1
                return
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                pass
        dl, physical, dr = tensor.shape
        q, r = _reduced_qr(tensor.reshape(dl * physical, dr))
        absorbed = (r @ next_tensor.reshape(next_tensor.shape[0], -1)).reshape(
            r.shape[0], next_tensor.shape[1], next_tensor.shape[2]
        )
        self._replace_tensor(site, q.reshape(dl, physical, q.shape[1]))
        self._replace_tensor(site + 1, absorbed)
        self.canonical_center = site + 1

    def _move_center_left(self, site: int) -> None:
        tensor = np.asarray(self.A[site], dtype=np.complex64)
        previous = np.asarray(self.A[site - 1], dtype=np.complex64)
        if _native_qr_move_left is not None:
            try:
                absorbed, q_tensor = _native_qr_move_left(previous, tensor)
                self._replace_tensor(site - 1, absorbed)
                self._replace_tensor(site, q_tensor)
                self.canonical_center = site - 1
                return
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                pass
        dl, physical, dr = tensor.shape
        q, r = _reduced_qr(tensor.reshape(dl, physical * dr).T)
        absorbed = (previous.reshape(-1, previous.shape[2]) @ r.T).reshape(
            previous.shape[0], previous.shape[1], r.shape[0]
        )
        self._replace_tensor(site - 1, absorbed)
        self._replace_tensor(site, q.T.reshape(q.shape[1], physical, dr))
        self.canonical_center = site - 1

    def _move_center(self, target: int) -> None:
        target = int(target)
        if not 0 <= target < self.n:
            raise ValueError("MPS canonical center is out of range")
        while self.canonical_center < target:
            self._move_center_right(self.canonical_center)
        while self.canonical_center > target:
            self._move_center_left(self.canonical_center)
        self._refresh_bonds()

    def renormalize(self) -> float:
        """Normalize the mixed-canonical center without a full contraction."""
        center = np.asarray(
            self.A[self.canonical_center], dtype=np.complex128
        )
        norm = float(np.linalg.norm(center.reshape(-1)))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise MPSNumericalError(
                "Cannot renormalize an MPS with zero or non-finite norm"
            )
        self.A[self.canonical_center] = (
            self.A[self.canonical_center] / norm
        )
        self.renormalization_count += 1
        self.last_pre_normalization_norm = norm
        return norm

    def canonicalize(self, center: int = 0, *, normalize: bool = True) -> float:
        """Move the explicit orthogonality center and optionally normalize."""
        self._move_center(center)
        return self.renormalize() if normalize else self.norm()

    # -------------- internal helpers --------------
    def _record_split(self, bond: int, metadata: dict) -> None:
        self.svd_calls += 1
        self.svd_total_ms += float(metadata["svd_elapsed_ms"])
        self.svd_fallback_count += len(metadata["svd_failed_attempts"])
        driver = str(metadata["svd_driver"])
        self.svd_drivers_used[driver] = self.svd_drivers_used.get(driver, 0) + 1
        shape = tuple(int(value) for value in metadata["matrix_shape"])
        shape_key = f"{shape[0]}x{shape[1]}"
        self.two_site_matrix_shape_calls[shape_key] = (
            self.two_site_matrix_shape_calls.get(shape_key, 0) + 1
        )
        self.two_site_max_matrix_elements = max(
            self.two_site_max_matrix_elements,
            int(metadata["matrix_elements"]),
        )
        if metadata["renormalized"]:
            self.renormalization_count += 1
            self.last_pre_normalization_norm = float(
                metadata["pre_normalization_norm"]
            )
        # Truncation detection (due to dmax or eps)
        if metadata["rank_kept"] < metadata["rank_before"]:
            self.truncated_any = True
            self.trunc_events += 1
            discarded = float(metadata["local_discarded_weight"])
            self.local_discarded_weight_sum += discarded
            self.local_discarded_weight_max = max(
                self.local_discarded_weight_max, discarded
            )
            relative = float(metadata["relative_discarded_weight"])
            self.relative_discarded_weight_sum += relative
            self.relative_discarded_weight_max = max(
                self.relative_discarded_weight_max, relative
            )
            self.last_truncation = {"bond": int(bond), **metadata}

    def _split_two_site_numpy(
        self, T: np.ndarray, bond: int, *, eps_override: Optional[float] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        Dl, d1, d2, Dr2 = T.shape
        matrix = T.reshape(Dl * d1, d2 * Dr2)
        split_started = time.perf_counter_ns()
        U, S, Vh, metadata = _svd_truncate_numpy(
            matrix,
            self.opts.dmax,
            self.opts.eps if eps_override is None else float(eps_override),
            driver=self.opts.svd_driver,
            renormalize=self.opts.renormalize_splits,
        )
        self.two_site_split_total_ms += (
            time.perf_counter_ns() - split_started
        ) / 1e6
        self._record_split(bond, metadata)
        rank = int(U.shape[1])
        left = U.reshape(Dl, d1, rank)
        right = (S.reshape(rank, 1) * Vh).reshape(rank, d2, Dr2)
        return left, right

    def _native_two_site(
        self,
        bond: int,
        kind: str,
        payload: Optional[np.ndarray] = None,
        *,
        eps_override: Optional[float] = None,
    ) -> bool:
        """Run one merge/update/SVD/split entirely in the native CPU core."""
        if _native_two_site_update is None:
            return False
        left = np.asarray(self.A[bond], dtype=np.complex64)
        right = np.asarray(self.A[bond + 1], dtype=np.complex64)
        if payload is None:
            payload = np.empty((0,), dtype=np.complex64)
        started = time.perf_counter_ns()
        try:
            left_out, right_out, metadata = _native_two_site_update(
                left,
                right,
                np.asarray(payload, dtype=np.complex64),
                str(kind),
                int(self.opts.dmax),
                float(
                    self.opts.eps
                    if eps_override is None
                    else eps_override
                ),
                bool(self.opts.renormalize_splits),
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            # The native extension is an optimization boundary, not a new
            # numerical failure mode. Fall back to the recoverable SciPy/NumPy
            # path if Accelerate reports a singular or unsupported case.
            return False
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        matrix_shape = [int(left.shape[0] * 2), int(right.shape[2] * 2)]
        rank_before = int(metadata["rank_before"])
        metadata = {
            **dict(metadata),
            "matrix_shape": matrix_shape,
            "matrix_elements": int(np.prod(matrix_shape)),
            "limited_by_dmax": int(self.opts.dmax) < rank_before,
            "limited_by_eps": float(
                self.opts.eps if eps_override is None else eps_override
            ) > 0.0 and int(metadata["rank_kept"]) < rank_before,
            "svd_elapsed_ms": elapsed_ms,
            "renormalized": bool(self.opts.renormalize_splits),
        }
        self._record_split(bond, metadata)
        self.two_site_split_total_ms += elapsed_ms
        self._install_two_site(left_out, right_out, bond)
        return True

    def _install_two_site(self, left, right, bond: int) -> None:
        self.A[bond] = self._as_stored(left)
        self.A[bond + 1] = self._as_stored(right)
        rank = int(self.A[bond].shape[2])
        if 0 <= bond < len(self.bonds):
            self.bonds[bond] = rank
        self.max_bond_ever = max(self.max_bond_ever, rank)
        self.canonical_center = bond + 1

    # -------------- gate application --------------
    def apply_single(self, U: mx.array, q: int):
        if not (0 <= q < self.n):
            raise ValueError("Qubit index out of range")
        if U.shape != (2, 2):
            raise ValueError("Gate dimension does not match target qubit")
        # MPS is CPU-native. For larger bond products, direct 2x2 row updates
        # avoid a general tensordot dispatch while retaining the faster BLAS
        # path for the tiny product-state tensors.
        A = np.asarray(self.A[q], dtype=np.complex64)
        Um = np.asarray(U, dtype=np.complex64)
        if (
            A.shape[0] * A.shape[2]
            >= _SINGLE_GATE_DIRECT_PRODUCT_THRESHOLD
        ):
            B = np.empty_like(A)
            B[:, 0, :] = Um[0, 0] * A[:, 0, :] + Um[0, 1] * A[:, 1, :]
            B[:, 1, :] = Um[1, 0] * A[:, 0, :] + Um[1, 1] * A[:, 1, :]
            self.A[q] = B
        else:
            B = np.tensordot(Um, A, axes=([1], [1]))
            self.A[q] = np.transpose(B, (1, 0, 2))

    def apply_diagonal(self, diagonal: mx.array, wires: List[int]) -> None:
        """Apply a one- or two-qubit diagonal gate without dense matmul."""
        wires = [int(wire) for wire in wires]
        if len(wires) == 1:
            q = self.logical_to_site[wires[0]]
            values = np.asarray(diagonal, dtype=np.complex64).reshape(-1)
            if values.size != 2:
                raise ValueError("one-qubit diagonal gates need two entries")
            tensor = np.asarray(self.A[q], dtype=np.complex64).copy()
            tensor[:, 0, :] *= values[0]
            tensor[:, 1, :] *= values[1]
            self.A[q] = tensor
            return
        if len(wires) != 2:
            raise ValueError("MPS diagonal gates support one or two qubits")
        values = np.asarray(diagonal, dtype=np.complex64).reshape(-1)
        if values.size != 4:
            raise ValueError("two-qubit diagonal gates need four entries")
        self._apply_logical_diagonal(values, wires[0], wires[1])

    def _apply_two_adjacent_diagonal(
        self, diagonal: np.ndarray, bond: int
    ) -> None:
        self._move_center(bond)
        self.two_site_calls += 1
        if self._native_two_site(
            bond, "diagonal", np.asarray(diagonal, dtype=np.complex64).reshape(4)
        ):
            return
        contraction_started = time.perf_counter_ns()
        left = np.asarray(self.A[bond], dtype=np.complex64)
        right = np.asarray(self.A[bond + 1], dtype=np.complex64)
        if int(left.shape[2]) != int(right.shape[0]):
            raise ValueError("MPS bond mismatch")
        dl, _, bond_dim = left.shape
        dr2 = int(right.shape[2])
        tensor = (left.reshape(dl * 2, bond_dim)
                  @ right.reshape(bond_dim, 2 * dr2)).reshape(
                      dl, 2, 2, dr2
                  )
        tensor *= np.asarray(diagonal, dtype=np.complex64).reshape(1, 2, 2, 1)
        self.two_site_contraction_total_ms += (
            time.perf_counter_ns() - contraction_started
        ) / 1e6
        left_out, right_out = self._split_two_site_numpy(tensor, bond)
        self._install_two_site(left_out, right_out, bond)

    def _apply_two_adjacent_cnot(
        self, bond: int, *, control_on_right: bool = False
    ) -> None:
        """Apply CNOT by permuting physical legs before one SVD split."""
        self._move_center(bond)
        self.two_site_calls += 1
        if self._native_two_site(
            bond, "cnot_reverse" if control_on_right else "cnot"
        ):
            return
        contraction_started = time.perf_counter_ns()
        left = np.asarray(self.A[bond], dtype=np.complex64)
        right = np.asarray(self.A[bond + 1], dtype=np.complex64)
        if int(left.shape[2]) != int(right.shape[0]):
            raise ValueError("MPS bond mismatch")
        dl, _, bond_dim = left.shape
        dr2 = int(right.shape[2])
        tensor = (left.reshape(dl * 2, bond_dim)
                  @ right.reshape(bond_dim, 2 * dr2)).reshape(
                      dl, 2, 2, dr2
                  )
        transformed = np.empty_like(tensor)
        for first in (0, 1):
            for second in (0, 1):
                if control_on_right:
                    output = (first ^ second, second)
                else:
                    output = (first, second ^ first)
                transformed[:, output[0], output[1], :] = (
                    tensor[:, first, second, :]
                )
        self.two_site_contraction_total_ms += (
            time.perf_counter_ns() - contraction_started
        ) / 1e6
        left_out, right_out = self._split_two_site_numpy(transformed, bond)
        self._install_two_site(left_out, right_out, bond)

    def _apply_two_adjacent(self, U4: mx.array, i: int):
        self._move_center(i)
        self.two_site_calls += 1
        if self._native_two_site(
            i, "generic", np.asarray(U4, dtype=np.complex64).reshape(4, 4)
        ):
            return
        contraction_started = time.perf_counter_ns()
        left = np.asarray(self.A[i], dtype=np.complex64)
        right = np.asarray(self.A[i + 1], dtype=np.complex64)
        gate = np.asarray(U4, dtype=np.complex64).reshape(4, 4)
        Dl, _, bond = left.shape
        if int(right.shape[0]) != int(bond):
            raise ValueError("MPS bond mismatch")
        Dr2 = int(right.shape[2])
        tensor = (left.reshape(Dl * 2, bond)
                  @ right.reshape(bond, 2 * Dr2)).reshape(Dl, 2, 2, Dr2)
        merged = tensor.reshape(Dl, 4, Dr2)
        transformed = np.tensordot(
            gate, merged, axes=([1], [1])
        ).transpose(1, 0, 2)
        tensor = transformed.reshape(Dl, 2, 2, Dr2)
        self.two_site_contraction_total_ms += (
            time.perf_counter_ns() - contraction_started
        ) / 1e6
        left_out, right_out = self._split_two_site_numpy(tensor, i)
        self._install_two_site(left_out, right_out, i)

    def _apply_two_adjacent_zz(self, theta: float, i: int):
        """Apply ``exp(-i theta Z⊗Z)`` by broadcasting four phases."""
        self._move_center(i)
        self.two_site_calls += 1
        even = complex(math.cos(theta), -math.sin(theta))
        odd = complex(math.cos(theta), math.sin(theta))
        if self._native_two_site(
            i,
            "zz",
            np.asarray([even, odd, odd, even], dtype=np.complex64),
        ):
            return
        contraction_started = time.perf_counter_ns()
        left = np.asarray(self.A[i], dtype=np.complex64)
        right = np.asarray(self.A[i + 1], dtype=np.complex64)
        if int(left.shape[2]) != int(right.shape[0]):
            raise ValueError("MPS bond mismatch")
        dl, _, bond = left.shape
        dr2 = int(right.shape[2])
        T = (left.reshape(dl * 2, bond)
             @ right.reshape(bond, 2 * dr2)).reshape(dl, 2, 2, dr2)
        phases = np.array([even, odd, odd, even], dtype=np.complex64).reshape(
            1, 2, 2, 1
        )
        T *= phases
        self.two_site_contraction_total_ms += (
            time.perf_counter_ns() - contraction_started
        ) / 1e6
        left_out, right_out = self._split_two_site_numpy(T, i)
        self._install_two_site(left_out, right_out, i)

    def _swap_adjacent(self, i: int):
        """Swap adjacent physical legs without a dense 4x4 gate multiply."""
        self._move_center(i)
        self.two_site_calls += 1
        swap_eps = (
            max(self.opts.eps, _SWAP_NUMERICAL_TRUNCATION_FLOOR)
            if self.opts.eps > 0.0
            else self.opts.eps
        )
        if self._native_two_site(i, "swap", eps_override=swap_eps):
            self.swap_fast_path_calls += 1
            return
        contraction_started = time.perf_counter_ns()
        left = np.asarray(self.A[i], dtype=np.complex64)
        right = np.asarray(self.A[i + 1], dtype=np.complex64)
        if int(left.shape[2]) != int(right.shape[0]):
            raise ValueError("MPS bond mismatch")
        dl, _, bond = left.shape
        dr2 = int(right.shape[2])
        tensor = (left.reshape(dl * 2, bond)
                  @ right.reshape(bond, 2 * dr2)).reshape(dl, 2, 2, dr2)
        # SWAP only exchanges the two physical legs. The explicit contiguous
        # copy keeps the subsequent reshape predictable for LAPACK.
        tensor = np.ascontiguousarray(np.transpose(tensor, (0, 2, 1, 3)))
        self.two_site_contraction_total_ms += (
            time.perf_counter_ns() - contraction_started
        ) / 1e6
        self.swap_fast_path_calls += 1
        left_out, right_out = self._split_two_site_numpy(
            tensor, i, eps_override=swap_eps
        )
        self._install_two_site(left_out, right_out, i)

    def apply_two(self, U4: mx.array, c: int, t: int):
        if c == t:
            raise ValueError("Control and target must differ")
        if not (0 <= c < self.n and 0 <= t < self.n):
            raise ValueError("Qubit index out of range")
        if U4.shape != (4, 4):
            raise ValueError("Gate dimension does not match target qubits")
        i, j = sorted((c, t))
        # Swap network to bring i and j adjacent
        k = i
        while k < j - 1:
            self._swap_adjacent(k)
            k += 1
        # The swap network presents the local tensor in ascending site order.
        # When the caller supplied descending operands, conjugating by SWAP
        # preserves the gate's semantic first/second operand order.
        ordered_gate = U4
        if c > t:
            swap_gate = _swap_gate()
            ordered_gate = mx.matmul(swap_gate, mx.matmul(U4, swap_gate))
        self._apply_two_adjacent(ordered_gate, j-1)
        # Swap back to restore ordering
        while k > i:
            k -= 1
            self._swap_adjacent(k)

    def apply_zz_phase(self, theta: float, first: int, second: int) -> None:
        """Apply a ZZ phase through a restoring swap network."""
        first, second = int(first), int(second)
        if first == second:
            raise ValueError("ZZ phase operands must differ")
        if not (0 <= first < self.n and 0 <= second < self.n):
            raise ValueError("Qubit index out of range")
        left, right = sorted((first, second))
        site = left
        while site < right - 1:
            self._swap_adjacent(site)
            site += 1
        self._apply_two_adjacent_zz(float(theta), right - 1)
        while site > left:
            site -= 1
            self._swap_adjacent(site)

    def apply_logical_single(self, U: mx.array, logical_wire: int) -> None:
        """Apply a gate using the persistent routed logical-to-site layout."""
        logical_wire = int(logical_wire)
        if not 0 <= logical_wire < self.n:
            raise ValueError("Qubit index out of range")
        self.apply_single(U, self.logical_to_site[logical_wire])

    def _apply_logical_diagonal(
        self, diagonal: np.ndarray, first: int, second: int
    ) -> None:
        first, second = int(first), int(second)
        first_site = self.logical_to_site[first]
        second_site = self.logical_to_site[second]
        self.routing_logical_gates += 1
        if self.routing_effective_strategy == "restore":
            left, right = sorted((first_site, second_site))
            selected_swaps = list(range(left, right - 1))
            restore_swaps = list(range(right - 2, left - 1, -1))
        else:
            selected_swaps = self._route_swaps_for_gate(
                first_site, second_site, None
            )
            restore_swaps = []
        for site in selected_swaps:
            self._routing_swap(site)
        first_site = self.logical_to_site[first]
        second_site = self.logical_to_site[second]
        left_site = min(first_site, second_site)
        ordered = np.asarray(diagonal, dtype=np.complex64).reshape(2, 2)
        if self.site_to_logical[left_site] != first:
            ordered = ordered.T
        self._apply_two_adjacent_diagonal(ordered.reshape(-1), left_site)
        if restore_swaps:
            for site in restore_swaps:
                self._routing_swap(site, final_restore=True)

    def apply_logical_cnot(self, first: int, second: int) -> None:
        """Apply CNOT using a physical-leg permutation and one SVD."""
        first, second = int(first), int(second)
        first_site = self.logical_to_site[first]
        second_site = self.logical_to_site[second]
        self.routing_logical_gates += 1
        if self.routing_effective_strategy == "restore":
            left, right = sorted((first_site, second_site))
            selected_swaps = list(range(left, right - 1))
            restore_swaps = list(range(right - 2, left - 1, -1))
        else:
            selected_swaps = self._route_swaps_for_gate(
                first_site, second_site, None
            )
            restore_swaps = []
        for site in selected_swaps:
            self._routing_swap(site)
        first_site = self.logical_to_site[first]
        second_site = self.logical_to_site[second]
        left_site = min(first_site, second_site)
        self._apply_two_adjacent_cnot(
            left_site,
            control_on_right=self.site_to_logical[left_site] != first,
        )
        if restore_swaps:
            for site in restore_swaps:
                self._routing_swap(site, final_restore=True)

    def apply_swap_wires(self, first: int, second: int) -> None:
        """Apply a logical SWAP by exchanging the wire-to-site mapping."""
        first, second = int(first), int(second)
        if first == second:
            raise ValueError("SWAP wires must differ")
        if not (0 <= first < self.n and 0 <= second < self.n):
            raise ValueError("Qubit index out of range")
        first_site = self.logical_to_site[first]
        second_site = self.logical_to_site[second]
        self.logical_to_site[first], self.logical_to_site[second] = (
            second_site,
            first_site,
        )
        self.site_to_logical[first_site], self.site_to_logical[second_site] = (
            second,
            first,
        )
        # A circuit-level SWAP changes the future logical layout. Discard the
        # old precomputed plan rather than risking a stale route; subsequent
        # direct calls use the safe local fallback planner.
        self.routing_plan = None
        self.routing_effective_strategy = "restore"
        self.routing_selection_reason = "circuit_swap_invalidated_route_plan"

    def _routing_swap(self, site: int, *, final_restore: bool = False) -> None:
        self._swap_adjacent(site)
        first = self.site_to_logical[site]
        second = self.site_to_logical[site + 1]
        self.site_to_logical[site], self.site_to_logical[site + 1] = (
            second,
            first,
        )
        self.logical_to_site[first] = site + 1
        self.logical_to_site[second] = site
        self.routing_swaps += 1
        if final_restore:
            self.routing_final_restore_swaps += 1

    def prepare_routing(self, pairs: List[tuple[int, int]]) -> dict:
        """Precompute a persistent route without planning a final restore.

        The MPS tensors are now allowed to remain in their routed order. Readout
        maps logical wires through ``logical_to_site``, so a final SWAP network
        would only add numerical work and bond growth.
        """
        pairs = [(int(first), int(second)) for first, second in pairs]
        naive = sum(
            2 * max(0, abs(first - second) - 1)
            for first, second in pairs
        )
        self.routing_plan_index = 0
        self.routing_plan = None
        if naive == 0:
            self.routing_planned_lookahead_swaps = 0
            self.routing_planned_restore_swaps = 0
            self.routing_effective_strategy = "restore"
            self.routing_selection_reason = "all_two_qubit_gates_are_adjacent"
            return {
                "configured_strategy": self.opts.routing_strategy,
                "effective_strategy": self.routing_effective_strategy,
                "selection_reason": self.routing_selection_reason,
                "planned_lookahead_swaps": 0,
                "planned_restore_swaps": 0,
            }
        plan, _order, routed_swaps, planner = _plan_routing(
            self.n, pairs, self.opts.routing_lookahead
        )
        self.routing_plan = plan
        self.routing_planner = planner
        self.routing_planned_lookahead_swaps = routed_swaps
        self.routing_planned_restore_swaps = naive
        if self.opts.routing_strategy == "restore":
            self.routing_effective_strategy = "restore"
            self.routing_selection_reason = "explicit_restore_request"
        elif (
            routed_swaps <= naive
            and (
                naive == 0
                or routed_swaps / float(naive)
                < 1.0 - _PERSISTENT_ROUTE_MIN_REDUCTION
            )
        ):
            self.routing_effective_strategy = "lookahead"
            self.routing_selection_reason = (
                "whole_circuit_lookahead_not_more_swaps_than_restore"
            )
        elif routed_swaps <= naive:
            self.routing_effective_strategy = "restore"
            self.routing_selection_reason = (
                "persistent_route_swap_reduction_too_small_for_bond_stability"
            )
        else:
            self.routing_effective_strategy = "restore"
            self.routing_selection_reason = (
                "whole_circuit_lookahead_would_increase_swaps"
            )
        return {
            "configured_strategy": self.opts.routing_strategy,
            "effective_strategy": self.routing_effective_strategy,
            "selection_reason": self.routing_selection_reason,
            "planned_lookahead_swaps": routed_swaps,
            "planned_restore_swaps": naive,
            "planner": planner,
            "planned_final_restore_swaps": 0,
        }

    def _route_swaps_for_gate(
        self,
        first_site: int,
        second_site: int,
        lookahead: Optional[List[tuple[int, int]]],
    ) -> list[int]:
        """Return the next precomputed route, with a direct-call fallback."""
        if self.routing_effective_strategy == "lookahead" and self.routing_plan:
            index = self.routing_plan_index
            self.routing_plan_index += 1
            if index < len(self.routing_plan):
                return list(self.routing_plan[index])
        # Direct MPSState callers may not run prepare_routing first. Preserve
        # the old local planner for that API while normal Device execution uses
        # the native/portable whole-circuit plan above.
        if first_site > second_site:
            first_site, second_site = second_site, first_site
        move_second = list(range(second_site - 1, first_site, -1))
        move_first = list(range(first_site, second_site - 1))
        if lookahead:
            first_order = list(self.site_to_logical)
            for site in move_first:
                first_order[site], first_order[site + 1] = first_order[site + 1], first_order[site]
            second_order = list(self.site_to_logical)
            for site in move_second:
                second_order[site], second_order[site + 1] = second_order[site + 1], second_order[site]
            def cost(order):
                positions = {logical: site for site, logical in enumerate(order)}
                return sum(
                    max(0, abs(positions[first] - positions[second]) - 1)
                    / float(index + 1)
                    for index, (first, second) in enumerate(lookahead)
                )
            return move_first if cost(first_order) <= cost(second_order) else move_second
        return move_first

    def apply_logical_two(
        self,
        U4: mx.array,
        first: int,
        second: int,
        *,
        lookahead: Optional[List[tuple[int, int]]] = None,
    ) -> None:
        """Apply a logical two-qubit gate with persistent lookahead routing."""
        first, second = int(first), int(second)
        if first == second:
            raise ValueError("Control and target must differ")
        if not (0 <= first < self.n and 0 <= second < self.n):
            raise ValueError("Qubit index out of range")
        first_site = self.logical_to_site[first]
        second_site = self.logical_to_site[second]
        distance = abs(first_site - second_site)
        self.routing_logical_gates += 1
        naive_swaps = 2 * max(0, abs(first - second) - 1)
        self.routing_naive_restore_swaps += naive_swaps
        if self.routing_effective_strategy == "restore":
            self.apply_two(U4, first_site, second_site)
            self.routing_swaps += naive_swaps
            return

        selected_swaps = self._route_swaps_for_gate(
            first_site, second_site, list(lookahead or [])[: self.opts.routing_lookahead]
        )
        for site in selected_swaps:
            self._routing_swap(site)

        first_site = self.logical_to_site[first]
        second_site = self.logical_to_site[second]
        if abs(first_site - second_site) != 1:
            raise MPSNumericalError("MPS router failed to make operands adjacent")
        left_site = min(first_site, second_site)
        ordered_gate = U4
        if self.site_to_logical[left_site] != first:
            ordered_gate = mx.matmul(
                _swap_gate(), mx.matmul(U4, _swap_gate())
            )
        self._apply_two_adjacent(ordered_gate, left_site)

    def apply_logical_zz_phase(
        self,
        theta: float,
        first: int,
        second: int,
        *,
        lookahead: Optional[List[tuple[int, int]]] = None,
    ) -> None:
        """Apply a routed logical ZZ phase without constructing a dense gate."""
        first, second = int(first), int(second)
        if first == second:
            raise ValueError("ZZ phase operands must differ")
        if not (0 <= first < self.n and 0 <= second < self.n):
            raise ValueError("Qubit index out of range")
        first_site = self.logical_to_site[first]
        second_site = self.logical_to_site[second]
        self.routing_logical_gates += 1
        naive_swaps = 2 * max(0, abs(first - second) - 1)
        self.routing_naive_restore_swaps += naive_swaps
        if self.routing_effective_strategy == "restore":
            self.apply_zz_phase(theta, first_site, second_site)
            self.routing_swaps += naive_swaps
            return

        selected_swaps = self._route_swaps_for_gate(
            first_site, second_site, list(lookahead or [])[: self.opts.routing_lookahead]
        )
        for site in selected_swaps:
            self._routing_swap(site)

        first_site = self.logical_to_site[first]
        second_site = self.logical_to_site[second]
        if abs(first_site - second_site) != 1:
            raise MPSNumericalError("MPS router failed to make operands adjacent")
        self._apply_two_adjacent_zz(float(theta), min(first_site, second_site))

    def restore_logical_order(self) -> None:
        """Return routed tensors to logical wire order before measurement."""
        for target_site in range(self.n):
            current_site = self.logical_to_site[target_site]
            while current_site > target_site:
                self._routing_swap(current_site - 1, final_restore=True)
                current_site -= 1
            while current_site < target_site:
                self._routing_swap(current_site, final_restore=True)
                current_site += 1
        if self.site_to_logical != list(range(self.n)):
            raise MPSNumericalError("MPS router failed to restore logical order")

    # -------------- TEBD-style helpers --------------
    def apply_single_all(self, U2: mx.array):
        """Apply the same 1-qubit gate to all sites."""
        for q in range(self.n):
            self.apply_single(U2, q)

    def apply_two_sweep(self, U4: mx.array):
        """Apply a 4x4 two-qubit gate sequentially on all nearest-neighbor bonds.

        Order: (0,1), (1,2), ..., (n-2,n-1). Matches dense sweep ordering.
        """
        for i in range(self.n - 1):
            self._apply_two_adjacent(U4, i)

    def apply_zz_two_sweep(self, theta: float):
        """Apply exp(-i theta Z⊗Z) across nearest-neighbor bonds using diagonal MPO.

        This reduces the cost of forming and multiplying a dense 4x4 matrix by
        acting directly on the two-site tensor with Z⊗Z signs and combining with
        I⊗I contribution.
        """
        for i in range(self.n - 1):
            self._apply_two_adjacent_zz(theta, i)

    def apply_two_all_pairs(self, U4: mx.array, offset: int = 0):
        """Apply the same 4x4 two-qubit gate to disjoint adjacent pairs.

        offset=0 applies pairs (0,1), (2,3), ...; offset=1 applies (1,2), (3,4), ...
        """
        start = 0 if (offset % 2 == 0) else 1
        for i in range(start, self.n - 1, 2):
            self._apply_two_adjacent(U4, i)

    # -------------- diagnostics --------------
    def bond_dims(self) -> List[int]:
        return list(self.bonds)

    def bond_max(self) -> int:
        return int(self.max_bond_ever)

    def bond_mean(self) -> float:
        if not self.bonds:
            return 1.0
        s = float(sum(int(b) for b in self.bonds))
        return s / float(len(self.bonds))

    def truncated(self) -> bool:
        return bool(self.truncated_any)

    def trunc_count(self) -> int:
        return int(self.trunc_events)

    def truncation_diagnostics(self) -> dict:
        """Return local SVD truncation telemetry for this state.

        Discarded weights are sums of squared singular values at each local
        split. Their accumulated sum is useful telemetry, but is not claimed
        as a global fidelity bound when the MPS is not at that bond's
        orthogonality center.
        """
        current_bond_max = max(self.bonds, default=1)
        return {
            "events": int(self.trunc_events),
            "truncated": bool(self.truncated_any),
            "local_discarded_weight_sum": float(self.local_discarded_weight_sum),
            "local_discarded_weight_max": float(self.local_discarded_weight_max),
            "relative_discarded_weight_sum": float(
                self.relative_discarded_weight_sum
            ),
            "relative_discarded_weight_max": float(
                self.relative_discarded_weight_max
            ),
            "last_event": (
                dict(self.last_truncation) if self.last_truncation is not None else None
            ),
            "tensor_device": self.tensor_device,
            "svd_device": self.svd_device,
            "native_two_site_core": _native_two_site_update is not None,
            "native_qr_core": (
                _native_qr_move_left is not None
                and _native_qr_move_right is not None
            ),
            "native_readout_core": (
                _native_expectation_product is not None
                and _native_probabilities is not None
                and _native_statevector is not None
            ),
            "svd_requested_driver": self.opts.svd_driver,
            "svd_drivers_used": dict(self.svd_drivers_used),
            "svd_calls": int(self.svd_calls),
            "svd_total_ms": float(self.svd_total_ms),
            "svd_fallback_count": int(self.svd_fallback_count),
            "two_site_calls": int(self.two_site_calls),
            "swap_fast_path_calls": int(self.swap_fast_path_calls),
            "two_site_contraction_total_ms": float(
                self.two_site_contraction_total_ms
            ),
            "two_site_split_total_ms": float(self.two_site_split_total_ms),
            "two_site_matrix_shape_calls": dict(
                self.two_site_matrix_shape_calls
            ),
            "two_site_max_matrix_elements": int(
                self.two_site_max_matrix_elements
            ),
            "configured_max_bond_dimension": int(self.opts.dmax),
            "configured_truncation_threshold": float(self.opts.eps),
            "current_bond_dimension_max": int(current_bond_max),
            "current_bond_dimension_mean": float(self.bond_mean()),
            "maximum_bond_dimension_reached": int(self.max_bond_ever),
            "canonical_center": int(self.canonical_center),
            "renormalization_count": int(self.renormalization_count),
            "last_pre_normalization_norm": float(
                self.last_pre_normalization_norm
            ),
            "routing_strategy": self.opts.routing_strategy,
            "routing_effective_strategy": self.routing_effective_strategy,
            "routing_selection_reason": self.routing_selection_reason,
            "routing_planner": self.routing_planner,
            "logical_to_site": list(self.logical_to_site),
            "routing_planned_lookahead_swaps": (
                self.routing_planned_lookahead_swaps
            ),
            "routing_planned_restore_swaps": (
                self.routing_planned_restore_swaps
            ),
            "routing_lookahead": int(self.opts.routing_lookahead),
            "routing_logical_two_qubit_gates": int(
                self.routing_logical_gates
            ),
            "routing_swaps": int(self.routing_swaps),
            "routing_final_restore_swaps": int(
                self.routing_final_restore_swaps
            ),
            "routing_naive_restore_swaps": int(
                self.routing_naive_restore_swaps
            ),
            "routing_swap_reduction": int(
                self.routing_naive_restore_swaps - self.routing_swaps
            ),
            "state_norm": self.norm(),
            "approximation_warning": (
                "local discarded weights are telemetry, not a global fidelity bound"
                if self.truncated_any else None
            ),
        }

    # -------------- convenience MPO sweeps for XX/YY via basis transforms --------------
    def apply_xx_two_sweep(self, theta: float):
        from .gates import H
        for i in range(self.n - 1):
            self.apply_single(H(), i)
            self.apply_single(H(), i+1)
            self._apply_two_adjacent_zz(theta, i)
            self.apply_single(H(), i)
            self.apply_single(H(), i+1)

    def apply_yy_two_sweep(self, theta: float):
        from .gates import RX
        Rp = RX(math.pi/2.0)
        Rm = RX(-math.pi/2.0)
        for i in range(self.n - 1):
            self.apply_single(Rp, i)
            self.apply_single(Rp, i+1)
            self._apply_two_adjacent_zz(theta, i)
            self.apply_single(Rm, i)
            self.apply_single(Rm, i+1)

    # -------------- dense fallback APIs --------------
    def apply_dense_gate(self, gate: mx.array, qubits):
        qs = list(qubits)
        if len(qs) == 1:
            self.apply_single(gate, qs[0]); return
        if len(qs) == 2:
            self.apply_two(gate, qs[0], qs[1]); return
        raise ValueError("MPSState.apply_dense_gate only supports 1q/2q gates")

    @property
    def tensors(self):
        return self.A

    def _numpy_tensors(self) -> List[np.ndarray]:
        return [np.asarray(tensor, dtype=np.complex128) for tensor in self.A]

    @staticmethod
    def _transfer(
        environment: np.ndarray,
        tensor: np.ndarray,
        operator: np.ndarray,
    ) -> np.ndarray:
        # environment is indexed (ket-left, bra-left); operator is (bra, ket)
        return np.einsum(
            "ab,atr,st,bsq->rq",
            environment,
            tensor,
            operator,
            tensor.conj(),
            optimize=True,
        )

    def norm(self) -> float:
        center = np.asarray(
            self.A[self.canonical_center], dtype=np.complex128
        )
        return float(np.linalg.norm(center.reshape(-1)))

    def normalize(self) -> float:
        return self.renormalize()

    def _physical_wires(self, wires: List[int]) -> list[int]:
        logical = [int(wire) for wire in wires]
        if len(set(logical)) != len(logical):
            raise ValueError("Duplicate qubit indices")
        if any(wire < 0 or wire >= self.n for wire in logical):
            raise ValueError("Qubit index out of range")
        return [self.logical_to_site[wire] for wire in logical]

    def expectation_product(self, operators: dict[int, np.ndarray]) -> complex:
        """Expectation of a tensor product of local 2x2 operators."""
        if _native_expectation_product is not None:
            wires = [int(wire) for wire in operators]
            values = np.asarray(
                [operators[wire] for wire in wires], dtype=np.complex128
            )
            return complex(
                _native_expectation_product(
                    self.A, wires, values, self.logical_to_site, self.norm()
                )
            )
        physical_keys = self._physical_wires(list(operators))
        physical_operators = {
            physical_wire: operator
            for physical_wire, operator in zip(physical_keys, operators.values())
        }
        tensors = self._numpy_tensors()
        environment = np.ones((1, 1), dtype=np.complex128)
        identity = np.eye(2, dtype=np.complex128)
        for wire, tensor in enumerate(tensors):
            operator = np.asarray(
                physical_operators.get(wire, identity), dtype=np.complex128
            )
            if operator.shape != (2, 2):
                raise ValueError("MPS product operators must be 2x2")
            environment = self._transfer(environment, tensor, operator)
        numerator = complex(environment.reshape(-1)[0])
        norm = self.norm()
        return numerator / (norm * norm)

    def expectation_dense(
        self,
        wires: List[int],
        matrix: np.ndarray,
    ) -> complex:
        """Expectation of a small dense observable via product expansion."""
        wires = list(wires)
        self._physical_wires(wires)
        if len(wires) > 4:
            raise ValueError(
                "Dense MPS observables are limited to 4 wires; use Pauli sums "
                "for wider observables"
            )
        dimension = 1 << len(wires)
        matrix = np.asarray(matrix, dtype=np.complex128)
        if matrix.shape != (dimension, dimension):
            raise ValueError("Dense observable dimension does not match wires")
        value = 0.0 + 0.0j
        for bra in range(dimension):
            for ket in range(dimension):
                coefficient = matrix[bra, ket]
                if abs(coefficient) == 0.0:
                    continue
                operators = {}
                for index, wire in enumerate(wires):
                    bra_bit = (bra >> (len(wires) - 1 - index)) & 1
                    ket_bit = (ket >> (len(wires) - 1 - index)) & 1
                    local = np.zeros((2, 2), dtype=np.complex128)
                    local[bra_bit, ket_bit] = 1.0
                    operators[wire] = local
                value += coefficient * self.expectation_product(operators)
        return value

    def _projected_probability(
        self,
        tensors: List[np.ndarray],
        fixed_bits: dict[int, int],
        norm_squared: float,
    ) -> float:
        environment = np.ones((1, 1), dtype=np.complex128)
        identity = np.eye(2, dtype=np.complex128)
        for wire, tensor in enumerate(tensors):
            if wire in fixed_bits:
                bit = int(fixed_bits[wire])
                operator = np.zeros((2, 2), dtype=np.complex128)
                operator[bit, bit] = 1.0
            else:
                operator = identity
            environment = self._transfer(environment, tensor, operator)
        value = float(environment.reshape(-1)[0].real / norm_squared)
        return max(0.0, value)

    def probabilities_array(
        self, wires: Optional[List[int]] = None
    ) -> np.ndarray:
        selected = list(range(self.n)) if wires is None else list(wires)
        if _native_probabilities is not None:
            self._physical_wires(selected)
            return np.asarray(
                _native_probabilities(
                    self.A, selected, self.logical_to_site, self.norm()
                ),
                dtype=np.float64,
            )
        physical_selected = self._physical_wires(selected)
        tensors = self._numpy_tensors()
        norm = self.norm()
        norm_squared = norm * norm
        result = np.empty(1 << len(selected), dtype=np.float64)
        for outcome in range(len(result)):
            fixed = {
                wire: (outcome >> (len(selected) - 1 - index)) & 1
                for index, wire in enumerate(physical_selected)
            }
            result[outcome] = self._projected_probability(
                tensors, fixed, norm_squared
            )
        total = float(result.sum())
        if total <= 0.0:
            raise RuntimeError("MPS marginal has zero total probability")
        return result / total

    def probabilities(self, wires: Optional[List[int]] = None) -> List[float]:
        return self.probabilities_array(wires).tolist()

    def to_statevector(self) -> np.ndarray:
        """Materialize a dense state only when the caller explicitly asks."""
        if _native_statevector is not None:
            return np.asarray(
                _native_statevector(self.A, self.logical_to_site),
                dtype=np.complex64,
            )
        psi = np.asarray(self.A[0], dtype=np.complex64)
        for index in range(1, self.n):
            psi = np.tensordot(
                psi, np.asarray(self.A[index], dtype=np.complex64),
                axes=([psi.ndim - 1], [0]),
            )
        tensor = psi.reshape([2] * self.n)
        logical_axes = [self.logical_to_site[wire] for wire in range(self.n)]
        if logical_axes != list(range(self.n)):
            tensor = np.transpose(tensor, logical_axes)
        result = tensor.reshape(1 << self.n).astype(np.complex64)
        norm = np.sqrt(
            np.sum(np.abs(result.astype(np.complex128)) ** 2, dtype=np.float64)
        )
        if norm <= 0.0:
            raise RuntimeError("MPS produced a zero statevector")
        return result / norm

    def sample_array(
        self,
        shots: int,
        wires: Optional[List[int]] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample an arbitrary-size MPS without constructing 2**n amplitudes."""
        shots = int(shots)
        if shots < 0:
            raise ValueError("shots must be non-negative")
        selected = list(range(self.n)) if wires is None else list(wires)
        if _native_sample is not None:
            self._physical_wires(selected)
            rng = np.random.default_rng() if rng is None else rng
            random_values = rng.random((shots, self.n), dtype=np.float64)
            return np.asarray(
                _native_sample(
                    self.A,
                    selected,
                    self.logical_to_site,
                    random_values,
                ),
                dtype=np.int64,
            )
        physical_selected = self._physical_wires(selected)
        rng = np.random.default_rng() if rng is None else rng
        tensors = self._numpy_tensors()

        right = [None] * (self.n + 1)
        right[self.n] = np.ones((1, 1), dtype=np.complex128)
        for wire in range(self.n - 1, -1, -1):
            tensor = tensors[wire]
            right[wire] = np.einsum(
                "asr,bsq,rq->ab",
                tensor,
                tensor.conj(),
                right[wire + 1],
                optimize=True,
            )

        all_samples = np.empty((shots, self.n), dtype=np.int64)
        if shots == 0:
            return all_samples[:, physical_selected]

        # Each shot carries one density-like left environment. Vectorizing the
        # contractions removes the Python shots×wires loop, while an adaptive
        # batch ceiling prevents Dmax² temporary storage from becoming an
        # unbounded memory cost.
        maximum_bond = max(
            max(int(tensor.shape[0]), int(tensor.shape[2]))
            for tensor in tensors
        )
        temporary_bytes_per_shot = max(
            1, 4 * maximum_bond * maximum_bond * np.dtype(np.complex128).itemsize
        )
        batch_size = max(
            1,
            min(shots, (64 * 1024 * 1024) // temporary_bytes_per_shot),
        )
        tiny = np.finfo(float).tiny
        for batch_start in range(0, shots, batch_size):
            batch_stop = min(shots, batch_start + batch_size)
            batch_shots = batch_stop - batch_start
            left = np.ones((batch_shots, 1, 1), dtype=np.complex128)
            for wire, tensor in enumerate(tensors):
                updates = []
                weights = []
                for bit in (0, 1):
                    selected_tensor = tensor[:, bit, :]
                    update = np.einsum(
                        "xab,ar,bq->xrq",
                        left,
                        selected_tensor,
                        selected_tensor.conj(),
                        optimize=True,
                    )
                    weight = np.einsum(
                        "xrq,rq->x",
                        update,
                        right[wire + 1],
                        optimize=True,
                    ).real
                    updates.append(update)
                    weights.append(np.maximum(weight, 0.0))
                total = weights[0] + weights[1]
                if np.any(total <= 0.0):
                    raise RuntimeError("MPS conditional probability is zero")
                bits = rng.random(batch_shots) < (weights[1] / total)
                all_samples[batch_start:batch_stop, wire] = bits
                chosen_updates = np.where(
                    bits[:, None, None], updates[1], updates[0]
                )
                chosen_weights = np.where(bits, weights[1], weights[0])
                left = chosen_updates / np.maximum(
                    chosen_weights[:, None, None], tiny
                )
        return all_samples[:, physical_selected]

    def sample(self, shots: int, wires: Optional[List[int]] = None):
        return self.sample_array(shots, wires).tolist()

    def sample_counts(self, shots: int, wires: Optional[List[int]] = None):
        counts: dict[str,int] = {}
        for bits in self.sample(shots, wires):
            key = ''.join(str(b) for b in bits)
            counts[key] = counts.get(key, 0) + 1
        return counts
