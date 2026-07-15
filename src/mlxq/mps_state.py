from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math
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


def _svd_truncate(M: mx.array, dmax: int, eps: float):
    # M shape: (a*2, 2*b) for two-site tensor; perform SVD and truncate
    # MLX currently exposes SVD on CPU. Apple unified memory lets a GPU tensor
    # be consumed by the CPU stream without first materializing a Python list;
    # keep this fallback visible in diagnostics rather than calling it GPU SVD.
    try:
        U, S, Vh = mx.linalg.svd(M, stream=mx.cpu)
        mx.eval(S)
    except Exception:
        # Last-resort compatibility path for older MLX releases.
        Mn = np.asarray(M, dtype=np.complex128)
        U_np, S_np, Vh_np = np.linalg.svd(Mn, full_matrices=False)
        U = mx.array(U_np.astype(np.complex64), mx.complex64)
        S = mx.array(S_np.astype(np.float32))
        Vh = mx.array(Vh_np.astype(np.complex64), mx.complex64)
    # Determine truncation rank
    s_vals = mx.reshape(S, (-1,))
    # Move to host to decide rank
    s_list = [float(v) for v in s_vals.tolist()]
    r = len(s_list)
    # eps-based cutoff
    r_eps = r
    if r > 0:
        thresh = eps * s_list[0]
        r_eps = sum(1 for v in s_list if v >= thresh)
    r_keep = min(r, max(1, min(dmax, r_eps)))
    total_weight = sum(value * value for value in s_list)
    discarded_weight = sum(value * value for value in s_list[r_keep:])
    metadata = {
        "rank_before": r,
        "rank_kept": r_keep,
        "local_discarded_weight": discarded_weight,
        "relative_discarded_weight": (
            discarded_weight / total_weight if total_weight > 0.0 else 0.0
        ),
        "limited_by_dmax": dmax < r,
        "limited_by_eps": r_eps < r,
    }
    U_t = U[:, :r_keep]
    S_t = S[:r_keep]
    Vh_t = Vh[:r_keep, :]
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
        self.last_truncation = None

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
        U, S, Vh, metadata = _svd_truncate(M, self.opts.dmax, self.opts.eps)
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

    def _apply_two_adjacent_zz(self, theta: float, i: int):
        """Apply exp(-i theta Z⊗Z) via MPO-like diagonal action without forming 4x4.

        Uses decomposition U = c0 I⊗I + c3 Z⊗Z, with c0=(a+b)/2, c3=(a-b)/2,
        where a=e^{-iθ}, b=e^{iθ}. This avoids a 4x4 matmul over merged legs.
        """
        # Two-site tensor T (Dl,2,2,Dr2)
        T = self._two_site_tensor(i)
        Dl, d1, d2, Dr2 = T.shape
        # Coefficients
        import math
        a = complex(math.cos(-theta), math.sin(-theta))
        b = complex(math.cos(theta), math.sin(theta))
        c0 = (a + b) * 0.5
        c3 = (a - b) * 0.5
        # Z action on physical legs (broadcast signs)
        z = mx.array([1.0+0j, -1.0+0j], mx.complex64)
        z1 = mx.reshape(z, (1, d1, 1, 1))
        z2 = mx.reshape(z, (1, 1, d2, 1))
        Tz = T * z1 * z2
        T2 = c0 * T + c3 * Tz
        # Split back via SVD
        Aleft, Aright = self._split_two_site(T2, i)
        self.A[i] = Aleft
        self.A[i+1] = Aright
        r = int(Aleft.shape[2])
        if 0 <= i < len(self.bonds):
            self.bonds[i] = r
        if r > self.max_bond_ever:
            self.max_bond_ever = r

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
        return {
            "events": int(self.trunc_events),
            "local_discarded_weight_sum": float(self.local_discarded_weight_sum),
            "local_discarded_weight_max": float(self.local_discarded_weight_max),
            "last_event": (
                dict(self.last_truncation) if self.last_truncation is not None else None
            ),
            "tensor_device": self.tensor_device,
            "svd_device": self.svd_device,
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
        tensors = self._numpy_tensors()
        environment = np.ones((1, 1), dtype=np.complex128)
        identity = np.eye(2, dtype=np.complex128)
        for tensor in tensors:
            environment = self._transfer(environment, tensor, identity)
        return float(max(0.0, environment.reshape(-1)[0].real) ** 0.5)

    def normalize(self) -> float:
        norm = self.norm()
        if norm <= 0.0:
            raise RuntimeError("Cannot normalize a zero MPS state")
        self.A[0] = self.A[0] / norm
        return norm

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
        for shot in range(shots):
            left = np.ones((1, 1), dtype=np.complex128)
            for wire, tensor in enumerate(tensors):
                weights = []
                updates = []
                for bit in (0, 1):
                    selected_tensor = tensor[:, bit, :]
                    update = np.einsum(
                        "ab,ar,bq->rq",
                        left,
                        selected_tensor,
                        selected_tensor.conj(),
                        optimize=True,
                    )
                    weight = float(
                        np.einsum(
                            "rq,rq->",
                            update,
                            right[wire + 1],
                            optimize=True,
                        ).real
                    )
                    weights.append(max(0.0, weight))
                    updates.append(update)
                total = weights[0] + weights[1]
                if total <= 0.0:
                    raise RuntimeError("MPS conditional probability is zero")
                probability_one = weights[1] / total
                bit = int(rng.random() < probability_one)
                all_samples[shot, wire] = bit
                chosen_weight = weights[bit]
                left = updates[bit] / max(chosen_weight, np.finfo(float).tiny)
        return all_samples[:, selected]

    def sample(self, shots: int, wires: Optional[List[int]] = None):
        return self.sample_array(shots, wires).tolist()

    def sample_counts(self, shots: int, wires: Optional[List[int]] = None):
        counts: dict[str,int] = {}
        for bits in self.sample(shots, wires):
            key = ''.join(str(b) for b in bits)
            counts[key] = counts.get(key, 0) + 1
        return counts
