"""Regression tests for the CPU-native MPS optimization paths."""

import numpy as np

from mettleq import gates
from mettleq.mps_state import MPSOptions, MPSState
import mettleq.mps_state as mps_state


def test_single_gate_direct_update_matches_tensordot_at_threshold_boundary(
    monkeypatch,
):
    tensor = np.arange(32, dtype=np.float32).reshape(4, 2, 4).astype(
        np.complex64
    )
    tensor /= np.linalg.norm(tensor)
    direct = MPSState(3, MPSOptions(dmax=8))
    general = MPSState(3, MPSOptions(dmax=8))
    direct.A[1] = tensor.copy()
    general.A[1] = tensor.copy()

    monkeypatch.setattr(mps_state, "_SINGLE_GATE_DIRECT_PRODUCT_THRESHOLD", 16)
    direct.apply_single(gates.H(), 1)
    monkeypatch.setattr(
        mps_state, "_SINGLE_GATE_DIRECT_PRODUCT_THRESHOLD", 10**9
    )
    general.apply_single(gates.H(), 1)

    np.testing.assert_allclose(direct.A[1], general.A[1], rtol=1e-6, atol=1e-6)


def test_swap_fast_path_is_used_and_preserves_exact_expectation():
    state = MPSState(4, MPSOptions(dmax=16))
    state.apply_single(gates.H(), 0)
    state.apply_two(gates.CNOT(), 0, 3)

    assert state.swap_fast_path_calls == 4
    assert abs(state.norm() - 1.0) < 1e-5
