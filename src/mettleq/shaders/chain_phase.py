"""Fuse a chain/ring diagonal phase with the first radix-16 U2 pass.

QAOA and TFIM alternate a diagonal nearest-neighbor interaction layer with a
uniform RX layer. The phase can be applied while each group of 16 input
amplitudes is loaded for the first radix-16 butterfly, preserving gate order
while removing one complete state read/write and one Metal launch per layer.
"""

from __future__ import annotations

from typing import Callable, Optional
import math

from .._mlx_compat import mx

from .single_qubit import u2_layer_all

__all__ = ["chain_phase_u2_layer_all"]


_CHAIN_PHASE_U2_SRC = """
    uint q = thread_position_in_grid.x;
    if (q >= n_hexads) return;
    uint bit_low = 1u << (shift_hi - 3u);
    uint low = q & (bit_low - 1u);
    uint high = q >> (shift_hi - 3u);
    uint base = (high << (shift_hi + 1u)) | low;

    complex64_t a[16];
    uint idx[16];
    for (uint c = 0u; c < 16u; ++c) {
        idx[c] = base | (c << (shift_hi - 3u));
        uint count;
        if (phase_kind == 0u) {
            count = metal::popcount((idx[c] & (idx[c] >> 1u)) & edge_mask);
            if (wrap_mask != 0u) {
                count += ((idx[c] & wrap_mask) == wrap_mask) ? 1u : 0u;
            }
        } else {
            count = metal::popcount((idx[c] ^ (idx[c] >> 1u)) & edge_mask);
        }
        complex64_t value = state[idx[c]];
        complex64_t phase = phase_lut[count];
        a[c] = complex64_t(
            value.real * phase.real - value.imag * phase.imag,
            value.real * phase.imag + value.imag * phase.real);
    }

    for (uint pass = 0u; pass < 4u; ++pass) {
        uint bit = 1u << pass;
        for (uint x = 0u; x < 16u; ++x) {
            if ((x & bit) != 0u) continue;
            uint y = x | bit;
            complex64_t v0 = a[x], v1 = a[y];
            a[x] = complex64_t(
                u[0].real*v0.real - u[0].imag*v0.imag +
                u[1].real*v1.real - u[1].imag*v1.imag,
                u[0].real*v0.imag + u[0].imag*v0.real +
                u[1].real*v1.imag + u[1].imag*v1.real);
            a[y] = complex64_t(
                u[2].real*v0.real - u[2].imag*v0.imag +
                u[3].real*v1.real - u[3].imag*v1.imag,
                u[2].real*v0.imag + u[2].imag*v0.real +
                u[3].real*v1.imag + u[3].imag*v1.real);
        }
    }
    for (uint r = 0u; r < 16u; ++r) out[idx[r]] = a[r];
"""


_chain_phase_u2_kernel = None
_LaunchObserver = Optional[Callable[[mx.array], None]]


def _phase_lut(kind: str, theta: float, n_bonds: int) -> mx.array:
    if kind == "cphase":
        values = [
            complex(math.cos(theta * count), math.sin(theta * count))
            for count in range(n_bonds + 1)
        ]
    elif kind == "zz":
        values = []
        for mismatches in range(n_bonds + 1):
            angle = -theta * (n_bonds - 2 * mismatches)
            values.append(complex(math.cos(angle), math.sin(angle)))
    else:
        raise ValueError("kind must be 'cphase' or 'zz'")
    return mx.array(values, dtype=mx.complex64)


def chain_phase_u2_layer_all(
    state: mx.array,
    n: int,
    u2x2: mx.array,
    *,
    kind: str,
    theta: float,
    n_bonds: int,
    edge_mask: int,
    wrap_mask: int = 0,
    on_launch: _LaunchObserver = None,
) -> mx.array:
    """Apply the diagonal layer, then the same U2 to every qubit."""
    global _chain_phase_u2_kernel
    if n < 4:
        raise ValueError("chain-phase radix-16 fusion requires at least 4 qubits")
    if _chain_phase_u2_kernel is None:
        _chain_phase_u2_kernel = mx.fast.metal_kernel(
            name="mettleq_chain_phase_u2_quad",
            input_names=[
                "state", "shift_hi", "u", "n_hexads", "phase_kind",
                "edge_mask", "wrap_mask", "phase_lut",
            ],
            output_names=["out"],
            source=_CHAIN_PHASE_U2_SRC,
        )
    n_hexads = 1 << (n - 4)
    (state,) = _chain_phase_u2_kernel(
        inputs=[
            state,
            mx.array(n - 1, dtype=mx.uint32),
            u2x2,
            mx.array(n_hexads, dtype=mx.uint32),
            mx.array(0 if kind == "cphase" else 1, dtype=mx.uint32),
            mx.array(edge_mask, dtype=mx.uint32),
            mx.array(wrap_mask, dtype=mx.uint32),
            _phase_lut(kind, theta, n_bonds),
        ],
        grid=(n_hexads, 1, 1),
        threadgroup=(min(256, n_hexads), 1, 1),
        output_shapes=[state.shape],
        output_dtypes=[mx.complex64],
    )
    if on_launch is not None:
        on_launch(state)
    return u2_layer_all(
        state,
        n,
        u2x2,
        start_shift=n - 5,
        on_launch=on_launch,
    )
