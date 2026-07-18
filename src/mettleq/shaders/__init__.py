"""Hand-written Metal shaders for structured gate layers.

This package is the single home for every hand-tuned Metal kernel in MettleQ.
The kernels quantify — and deliver — the headroom above the pure-MLX structured
dispatch path. Supported Apple GPUs select them automatically. Set
METTLEQ_METAL_KERNELS=0 to force the pure-MLX compatibility/ablation path.

Modules:
  zz            fused exp(-i*theta*sum Z_a Z_b) layers (parity + phase LUT)
  qft           radix-4 fused QFT stages (+ single-stage fallback)
  single_qubit  fused all-qubit single-qubit layers (RX-specialized and
                generic radix-16 tensor-product passes)

Design notes, derivations, measured numbers, and codex-review verdicts for
each shader live in shaders/README.md.
"""
from __future__ import annotations

from ..execution import metal_runtime_enabled, metal_runtime_status

from .zz import zz_chain_layer, zz_weighted_layer
from .qft import qft_stage_all, qft_stage_sub
from .single_qubit import (rx_layer_all, u2_layer_all, u2_list_layer_all,
                           hadamard_layer_all, s_phase_layer)
from .diag import diag_pair_layer, diag_weighted_layer
from .xor_affine import compose_inverse_affine, xor_affine_gather
from .pauli_pair import xx_layer, yy_layer

__all__ = [
    "metal_enabled",
    "metal_runtime_status",
    "zz_chain_layer",
    "zz_weighted_layer",
    "qft_stage_all",
    "qft_stage_sub",
    "rx_layer_all",
    "u2_layer_all",
    "u2_list_layer_all",
    "hadamard_layer_all",
    "s_phase_layer",
    "diag_pair_layer",
    "diag_weighted_layer",
    "compose_inverse_affine",
    "xor_affine_gather",
    "xx_layer",
    "yy_layer",
]


def metal_enabled(n_qubits=None, *, dtype: str = "complex64") -> bool:
    """Return true only when policy *and* runtime capabilities select Metal."""
    return metal_runtime_enabled(n_qubits, dtype=dtype)
