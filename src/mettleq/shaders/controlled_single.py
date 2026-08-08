"""Native controlled-single-qubit Metal kernels.

The pair kernel shares one full-state traversal between two controlled gates
whose control/target wire sets are disjoint.  The single kernel covers the
remaining gate.  Both accept an arbitrary 2x2 target unitary, so CH and the
controlled rotation family use the same compiled kernels.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from .._mlx_compat import mx

__all__ = ["controlled_u_groups"]

_LaunchObserver = Optional[Callable[[mx.array], None]]

_CU_SINGLE_SRC = r"""
    uint p = thread_position_in_grid.x;
    if (p >= n_pairs) return;
    uint target_bit = 1u << target_shift;
    uint control_bit = 1u << control_shift;
    uint low = p & (target_bit - 1u);
    uint high = p >> target_shift;
    uint i0 = (high << (target_shift + 1u)) | low;
    uint i1 = i0 | target_bit;
    complex64_t a0 = state[i0], a1 = state[i1];
    if ((i0 & control_bit) == 0u) {
        out[i0] = a0; out[i1] = a1; return;
    }
    complex64_t u00=u[0], u01=u[1], u10=u[2], u11=u[3];
    out[i0] = complex64_t(
        u00.real*a0.real-u00.imag*a0.imag+u01.real*a1.real-u01.imag*a1.imag,
        u00.real*a0.imag+u00.imag*a0.real+u01.real*a1.imag+u01.imag*a1.real);
    out[i1] = complex64_t(
        u10.real*a0.real-u10.imag*a0.imag+u11.real*a1.real-u11.imag*a1.imag,
        u10.real*a0.imag+u10.imag*a0.real+u11.real*a1.imag+u11.imag*a1.real);
"""

_CU_PAIR_SRC = r"""
    uint p = thread_position_in_grid.x;
    if (p >= n_quads) return;
    uint lo_shift = min(target_a_shift, target_b_shift);
    uint hi_shift = max(target_a_shift, target_b_shift);
    uint low_mask = (1u << lo_shift) - 1u;
    uint low = p & low_mask;
    uint rest = p >> lo_shift;
    uint middle_width = hi_shift - lo_shift - 1u;
    uint middle_mask = (middle_width == 0u) ? 0u : ((1u << middle_width) - 1u);
    uint middle = rest & middle_mask;
    uint high = rest >> middle_width;
    uint base = low | (middle << (lo_shift + 1u)) | (high << (hi_shift + 1u));
    uint bit_a = 1u << target_a_shift, bit_b = 1u << target_b_shift;
    uint control_a = 1u << control_a_shift, control_b = 1u << control_b_shift;
    uint idx[4] = {base, base | bit_b, base | bit_a, base | bit_a | bit_b};
    complex64_t v[4] = {state[idx[0]], state[idx[1]], state[idx[2]], state[idx[3]]};

    // Apply gate A within the two target-bit subspace when its control is 1.
    for (uint b = 0u; b < 2u; ++b) {
        uint r0 = b, r1 = b + 2u;
        if ((idx[r0] & control_a) != 0u) {
            complex64_t x=v[r0], y=v[r1];
            complex64_t u00=ua[0],u01=ua[1],u10=ua[2],u11=ua[3];
            v[r0]=complex64_t(u00.real*x.real-u00.imag*x.imag+u01.real*y.real-u01.imag*y.imag,
                              u00.real*x.imag+u00.imag*x.real+u01.real*y.imag+u01.imag*y.real);
            v[r1]=complex64_t(u10.real*x.real-u10.imag*x.imag+u11.real*y.real-u11.imag*y.imag,
                              u10.real*x.imag+u10.imag*x.real+u11.real*y.imag+u11.imag*y.real);
        }
    }
    // Apply disjoint gate B. Its control bit is unchanged by gate A.
    for (uint a = 0u; a < 2u; ++a) {
        uint r0 = 2u*a, r1 = r0 + 1u;
        if ((idx[r0] & control_b) != 0u) {
            complex64_t x=v[r0], y=v[r1];
            complex64_t u00=ub[0],u01=ub[1],u10=ub[2],u11=ub[3];
            v[r0]=complex64_t(u00.real*x.real-u00.imag*x.imag+u01.real*y.real-u01.imag*y.imag,
                              u00.real*x.imag+u00.imag*x.real+u01.real*y.imag+u01.imag*y.real);
            v[r1]=complex64_t(u10.real*x.real-u10.imag*x.imag+u11.real*y.real-u11.imag*y.imag,
                              u10.real*x.imag+u10.imag*x.real+u11.real*y.imag+u11.imag*y.real);
        }
    }
    for (uint k=0u; k<4u; ++k) out[idx[k]]=v[k];
"""

_single_kernel = None
_pair_kernel = None


def controlled_u_groups(
    state: mx.array,
    n: int,
    groups: Sequence[Sequence[dict]],
    *,
    on_launch: _LaunchObserver = None,
) -> mx.array:
    """Execute groups of one or two disjoint controlled 2x2 operations."""
    global _single_kernel, _pair_kernel
    for group in groups:
        if len(group) == 1:
            if _single_kernel is None:
                _single_kernel = mx.fast.metal_kernel(
                    name="mettleq_controlled_u_single",
                    input_names=["state", "control_shift", "target_shift", "u", "n_pairs"],
                    output_names=["out"], source=_CU_SINGLE_SRC)
            gate = group[0]
            count = 1 << (n - 1)
            (state,) = _single_kernel(
                inputs=[state, mx.array(n - 1 - gate["control"], dtype=mx.uint32),
                        mx.array(n - 1 - gate["target"], dtype=mx.uint32), gate["matrix"],
                        mx.array(count, dtype=mx.uint32)],
                grid=(count, 1, 1), threadgroup=(min(256, count), 1, 1),
                output_shapes=[state.shape], output_dtypes=[mx.complex64])
        elif len(group) == 2:
            if _pair_kernel is None:
                _pair_kernel = mx.fast.metal_kernel(
                    name="mettleq_controlled_u_pair",
                    input_names=["state", "control_a_shift", "target_a_shift", "ua",
                                 "control_b_shift", "target_b_shift", "ub", "n_quads"],
                    output_names=["out"], source=_CU_PAIR_SRC)
            a, b = group
            count = 1 << (n - 2)
            (state,) = _pair_kernel(
                inputs=[state, mx.array(n - 1 - a["control"], dtype=mx.uint32),
                        mx.array(n - 1 - a["target"], dtype=mx.uint32), a["matrix"],
                        mx.array(n - 1 - b["control"], dtype=mx.uint32),
                        mx.array(n - 1 - b["target"], dtype=mx.uint32), b["matrix"],
                        mx.array(count, dtype=mx.uint32)],
                grid=(count, 1, 1), threadgroup=(min(256, count), 1, 1),
                output_shapes=[state.shape], output_dtypes=[mx.complex64])
        else:
            raise ValueError("controlled-unitary groups must contain one or two gates")
        if on_launch is not None:
            on_launch(state)
    return state
