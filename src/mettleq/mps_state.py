from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math
import time
import numpy as np

import mlx.core as mx


_SWAP_GATE = mx.array(
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


def _safe_cpu_svd(matrix: np.ndarray, driver: str):
    """Run a catchable CPU SVD with a deterministic fallback ladder.

    MLX 0.32's CPU ``sgesvdx`` can terminate the process before Python can
    catch an exception. SciPy and NumPy expose LAPACK failures as Python
    exceptions, which lets the SDK return a useful numerical error instead of
    losing the whole process.
    """
    array = np.asarray(matrix, dtype=np.complex64)
    if not np.all(np.isfinite(array)):
        raise MPSNumericalError("MPS SVD input contains NaN or infinity")
    scale = float(np.max(np.abs(array))) if array.size else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        raise MPSNumericalError("MPS SVD input has zero or invalid scale")
    scaled = array / scale
    attempts = []
    if driver in ("auto", "gesdd", "gesvd"):
        try:
            from scipy import linalg as scipy_linalg

            scipy_drivers = (
                ("gesdd", "gesvd") if driver == "auto" else (driver,)
            )
            for scipy_driver in scipy_drivers:
                try:
                    U, S, Vh = scipy_linalg.svd(
                        scaled,
                        full_matrices=False,
                        overwrite_a=True,
                        check_finite=False,
                        lapack_driver=scipy_driver,
                    )
                    return U, S * scale, Vh, f"scipy_{scipy_driver}", attempts
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
                return U, S * scale, Vh, f"numpy_{dtype.__name__}", attempts
            except Exception as error:
                attempts.append(f"numpy_{dtype.__name__}: {error}")
    detail = "; ".join(attempts) or "no SVD driver was attempted"
    raise MPSNumericalError(f"All recoverable MPS SVD drivers failed: {detail}")


def _svd_truncate(
    M: mx.array,
    dmax: int,
    eps: float,
    *,
    driver: str = "auto",
    renormalize: bool = True,
):
    # M shape: (a*2, 2*b) for two-site tensor; perform SVD and truncate
    start = time.perf_counter_ns()
    mx.eval(M)
    U_np, S_np, Vh_np, used_driver, failed_attempts = _safe_cpu_svd(
        np.asarray(M, dtype=np.complex64), driver
    )
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    s_list = [float(value) for value in np.asarray(S_np).reshape(-1)]
    r = len(s_list)
    # eps-based cutoff
    r_eps = r
    if r > 0:
        thresh = eps * s_list[0]
        r_eps = sum(1 for v in s_list if v >= thresh)
    r_keep = min(r, max(1, min(dmax, r_eps)))
    total_weight = sum(value * value for value in s_list)
    discarded_weight = sum(value * value for value in s_list[r_keep:])
    kept_weight = sum(value * value for value in s_list[:r_keep])
    if not np.isfinite(kept_weight) or kept_weight <= np.finfo(float).tiny:
        raise MPSNumericalError(
            "MPS truncation retained a zero or non-finite singular spectrum"
        )
    normalization_factor = kept_weight ** 0.5 if renormalize else 1.0
    metadata = {
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
    U_t = mx.array(U_np[:, :r_keep].astype(np.complex64), mx.complex64)
    S_t = mx.array(
        (S_np[:r_keep] / normalization_factor).astype(np.float32)
    )
    Vh_t = mx.array(Vh_np[:r_keep, :].astype(np.complex64), mx.complex64)
    return U_t, S_t, Vh_t, metadata


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
        self.tensor_device = (
            "cpu" if mx.default_device() == mx.Device(mx.cpu) else "gpu"
        )
        self.svd_device = "cpu"
        self.reset()

    def reset(self):
        # |0..0> in right-canonical form: A[i] = [ [1,0] ] up to dimensions
        self.A: List[mx.array] = []
        for i in range(self.n):
            v = mx.array([1+0j, 0+0j], mx.complex64)
            self.A.append(mx.reshape(v, (1, 2, 1)))  # (1,2,1)
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

    def _replace_tensor(self, index: int, value: np.ndarray) -> None:
        self.A[index] = mx.array(
            np.asarray(value, dtype=np.complex64), mx.complex64
        )

    def _refresh_bonds(self) -> None:
        self.bonds = [int(tensor.shape[2]) for tensor in self.A[:-1]]
        self.max_bond_ever = max(
            self.max_bond_ever, max(self.bonds, default=1)
        )

    def _move_center_right(self, site: int) -> None:
        tensor = np.asarray(self.A[site], dtype=np.complex64)
        dl, physical, dr = tensor.shape
        q, r = np.linalg.qr(
            tensor.reshape(dl * physical, dr), mode="reduced"
        )
        next_tensor = np.asarray(self.A[site + 1], dtype=np.complex64)
        absorbed = np.tensordot(r, next_tensor, axes=([1], [0]))
        self._replace_tensor(site, q.reshape(dl, physical, q.shape[1]))
        self._replace_tensor(site + 1, absorbed)
        self.canonical_center = site + 1

    def _move_center_left(self, site: int) -> None:
        tensor = np.asarray(self.A[site], dtype=np.complex64)
        dl, physical, dr = tensor.shape
        q, r = np.linalg.qr(
            tensor.reshape(dl, physical * dr).T, mode="reduced"
        )
        previous = np.asarray(self.A[site - 1], dtype=np.complex64)
        absorbed = np.tensordot(previous, r.T, axes=([2], [0]))
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
    def _two_site_tensor(self, i: int) -> mx.array:
        left = self.A[i]      # (Dl,2,Dr)
        right = self.A[i+1]   # (Dr,2,Dr2)
        Dl, _, Dr = left.shape
        Dr_, _, Dr2 = right.shape
        if Dr_ != Dr:
            # reshape to match bond
            raise ValueError("MPS bond mismatch")
        T = mx.tensordot(left, right, axes=([2],[0]))  # (Dl,2,2,Dr2)
        return T

    def _split_two_site(self, T: mx.array, bond: int) -> tuple[mx.array, mx.array]:
        Dl, d1, d2, Dr2 = T.shape
        M = mx.reshape(mx.transpose(T, (0,1,2,3)), (Dl*d1, d2*Dr2))
        U, S, Vh, metadata = _svd_truncate(
            M,
            self.opts.dmax,
            self.opts.eps,
            driver=self.opts.svd_driver,
            renormalize=self.opts.renormalize_splits,
        )
        self.svd_calls += 1
        self.svd_total_ms += float(metadata["svd_elapsed_ms"])
        self.svd_fallback_count += len(metadata["svd_failed_attempts"])
        driver = str(metadata["svd_driver"])
        self.svd_drivers_used[driver] = self.svd_drivers_used.get(driver, 0) + 1
        if metadata["renormalized"]:
            self.renormalization_count += 1
            self.last_pre_normalization_norm = float(
                metadata["pre_normalization_norm"]
            )
        r = int(U.shape[1])
        # Truncation detection (due to dmax or eps)
        if r < metadata["rank_before"]:
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
        # reshape back
        Aleft = mx.reshape(U, (Dl, d1, r))
        SVh = mx.reshape(S, (r,1)) * Vh  # (r, d2*Dr2)
        Aright = mx.reshape(SVh, (r, d2, Dr2))
        return Aleft, Aright

    # -------------- gate application --------------
    def apply_single(self, U: mx.array, q: int):
        if not (0 <= q < self.n):
            raise ValueError("Qubit index out of range")
        if U.shape != (2, 2):
            raise ValueError("Gate dimension does not match target qubit")
        A = self.A[q]
        # Contract U's input with the physical leg of A. Bond axes are batches,
        # not part of the 2-vector acted on by the gate.
        B = mx.tensordot(U, A, axes=([1], [1]))  # (2, Dl, Dr)
        self.A[q] = mx.transpose(B, (1, 0, 2))   # (Dl, 2, Dr)

    def _apply_two_adjacent(self, U4: mx.array, i: int):
        self._move_center(i)
        T = self._two_site_tensor(i)  # (Dl,2,2,Dr2)
        Dl, _, _, Dr2 = T.shape
        # merge physical legs (2,2)->4 and apply U
        Tm = mx.reshape(T, (Dl, 4, Dr2))
        Um = mx.reshape(U4, (4,4))
        # Apply Um along the merged physical leg (left-multiply on that axis)
        # Tm has axes (Dl, ab, Dr2); we compute over 'ab' using tensordot
        # Result shape: (4, Dl, Dr2) → transpose to (Dl, 4, Dr2)
        Tm2 = mx.tensordot(Um, Tm, axes=([1],[1]))  # (4, Dl, Dr2)
        Tm2 = mx.transpose(Tm2, (1, 0, 2))          # (Dl, 4, Dr2)
        T2 = mx.reshape(Tm2, (Dl, 2, 2, Dr2))
        Aleft, Aright = self._split_two_site(T2, i)
        self.A[i] = Aleft
        self.A[i+1] = Aright
        # Update bond diagnostics (bond between i and i+1 equals rank r)
        r = int(Aleft.shape[2])
        if 0 <= i < len(self.bonds):
            self.bonds[i] = r
        if r > self.max_bond_ever:
            self.max_bond_ever = r
        self.canonical_center = i + 1

    def _apply_two_adjacent_zz(self, theta: float, i: int):
        """Apply ``exp(-i theta Z⊗Z)`` by broadcasting four phases."""
        self._move_center(i)
        T = self._two_site_tensor(i)
        import math
        even = complex(math.cos(theta), -math.sin(theta))
        odd = complex(math.cos(theta), math.sin(theta))
        phases = mx.reshape(
            mx.array([even, odd, odd, even], mx.complex64),
            (1, 2, 2, 1),
        )
        T2 = T * phases
        Aleft, Aright = self._split_two_site(T2, i)
        self.A[i] = Aleft
        self.A[i+1] = Aright
        r = int(Aleft.shape[2])
        if 0 <= i < len(self.bonds):
            self.bonds[i] = r
        if r > self.max_bond_ever:
            self.max_bond_ever = r
        self.canonical_center = i + 1

    def _swap_adjacent(self, i: int):
        # Swap sites i and i+1 by applying SWAP gate U_swap to two-site tensor
        self._apply_two_adjacent(_SWAP_GATE, i)

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
            ordered_gate = mx.matmul(_SWAP_GATE, mx.matmul(U4, _SWAP_GATE))
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

    @staticmethod
    def _candidate_route_swaps(
        first_site: int, second_site: int, move_first: bool
    ) -> list[int]:
        if first_site < second_site:
            if move_first:
                return list(range(first_site, second_site - 1))
            return list(range(second_site - 1, first_site, -1))
        if move_first:
            return list(range(first_site - 1, second_site, -1))
        return list(range(second_site, first_site - 1))

    @staticmethod
    def _simulate_layout_swaps(order: list[int], swaps: list[int]) -> list[int]:
        candidate = list(order)
        for site in swaps:
            candidate[site], candidate[site + 1] = (
                candidate[site + 1],
                candidate[site],
            )
        return candidate

    @staticmethod
    def _lookahead_layout_cost(
        order: list[int], lookahead: list[tuple[int, int]]
    ) -> float:
        positions = {logical: site for site, logical in enumerate(order)}
        cost = 0.0
        for index, (first, second) in enumerate(lookahead):
            distance = abs(positions[int(first)] - positions[int(second)])
            cost += max(0, distance - 1) / float(index + 1)
        return cost

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
        """Select persistent routing only when whole-circuit swaps do not grow."""
        pairs = [(int(first), int(second)) for first, second in pairs]
        naive = sum(
            2 * max(0, abs(first - second) - 1)
            for first, second in pairs
        )
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
        order = list(range(self.n))
        routed_swaps = 0
        for index, (first, second) in enumerate(pairs):
            positions = {
                logical: site for site, logical in enumerate(order)
            }
            first_site = positions[first]
            second_site = positions[second]
            first_swaps = self._candidate_route_swaps(
                first_site, second_site, True
            )
            second_swaps = self._candidate_route_swaps(
                first_site, second_site, False
            )
            first_order = self._simulate_layout_swaps(order, first_swaps)
            second_order = self._simulate_layout_swaps(order, second_swaps)
            lookahead = pairs[
                index + 1:index + 1 + self.opts.routing_lookahead
            ]
            if self._lookahead_layout_cost(
                first_order, lookahead
            ) <= self._lookahead_layout_cost(second_order, lookahead):
                selected_swaps = first_swaps
                order = first_order
            else:
                selected_swaps = second_swaps
                order = second_order
            routed_swaps += len(selected_swaps)
        final_restore = sum(
            1
            for first in range(self.n)
            for second in range(first + 1, self.n)
            if order[first] > order[second]
        )
        routed_swaps += final_restore
        self.routing_planned_lookahead_swaps = routed_swaps
        self.routing_planned_restore_swaps = naive
        if self.opts.routing_strategy == "restore":
            self.routing_effective_strategy = "restore"
            self.routing_selection_reason = "explicit_restore_request"
        elif routed_swaps <= naive:
            self.routing_effective_strategy = "lookahead"
            self.routing_selection_reason = (
                "whole_circuit_lookahead_not_more_swaps_than_restore"
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
        }

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

        lookahead = list(lookahead or [])[: self.opts.routing_lookahead]
        first_swaps = self._candidate_route_swaps(
            first_site, second_site, True
        )
        second_swaps = self._candidate_route_swaps(
            first_site, second_site, False
        )
        first_order = self._simulate_layout_swaps(
            self.site_to_logical, first_swaps
        )
        second_order = self._simulate_layout_swaps(
            self.site_to_logical, second_swaps
        )
        first_score = self._lookahead_layout_cost(first_order, lookahead)
        second_score = self._lookahead_layout_cost(second_order, lookahead)
        selected_swaps = (
            first_swaps if first_score <= second_score else second_swaps
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
                _SWAP_GATE, mx.matmul(U4, _SWAP_GATE)
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

        lookahead = list(lookahead or [])[: self.opts.routing_lookahead]
        first_swaps = self._candidate_route_swaps(
            first_site, second_site, True
        )
        second_swaps = self._candidate_route_swaps(
            first_site, second_site, False
        )
        first_order = self._simulate_layout_swaps(
            self.site_to_logical, first_swaps
        )
        second_order = self._simulate_layout_swaps(
            self.site_to_logical, second_swaps
        )
        selected_swaps = (
            first_swaps
            if self._lookahead_layout_cost(first_order, lookahead)
            <= self._lookahead_layout_cost(second_order, lookahead)
            else second_swaps
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
            "svd_requested_driver": self.opts.svd_driver,
            "svd_drivers_used": dict(self.svd_drivers_used),
            "svd_calls": int(self.svd_calls),
            "svd_total_ms": float(self.svd_total_ms),
            "svd_fallback_count": int(self.svd_fallback_count),
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
        mx.eval(*self.A)
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

    def expectation_product(self, operators: dict[int, np.ndarray]) -> complex:
        """Expectation of a tensor product of local 2x2 operators."""
        tensors = self._numpy_tensors()
        environment = np.ones((1, 1), dtype=np.complex128)
        identity = np.eye(2, dtype=np.complex128)
        for wire, tensor in enumerate(tensors):
            operator = np.asarray(
                operators.get(wire, identity), dtype=np.complex128
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
        if len(set(selected)) != len(selected):
            raise ValueError("Duplicate qubit indices")
        if any(wire < 0 or wire >= self.n for wire in selected):
            raise ValueError("Qubit index out of range")
        tensors = self._numpy_tensors()
        norm = self.norm()
        norm_squared = norm * norm
        result = np.empty(1 << len(selected), dtype=np.float64)
        for outcome in range(len(result)):
            fixed = {
                wire: (outcome >> (len(selected) - 1 - index)) & 1
                for index, wire in enumerate(selected)
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
        psi = self.A[0]
        for index in range(1, self.n):
            psi = mx.tensordot(
                psi, self.A[index], axes=([psi.ndim - 1], [0])
            )
        psi = mx.reshape(psi, (1 << self.n,))
        mx.eval(psi)
        result = np.asarray(psi, dtype=np.complex64)
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
        if len(set(selected)) != len(selected):
            raise ValueError("Duplicate qubit indices")
        if any(wire < 0 or wire >= self.n for wire in selected):
            raise ValueError("Qubit index out of range")
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
            return all_samples[:, selected]

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
        return all_samples[:, selected]

    def sample(self, shots: int, wires: Optional[List[int]] = None):
        return self.sample_array(shots, wires).tolist()

    def sample_counts(self, shots: int, wires: Optional[List[int]] = None):
        counts: dict[str,int] = {}
        for bits in self.sample(shots, wires):
            key = ''.join(str(b) for b in bits)
            counts[key] = counts.get(key, 0) + 1
        return counts
