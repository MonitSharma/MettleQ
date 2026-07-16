import numpy as np
import pytest
import mlx.core as mx

from mlxq.gates import CNOT, CRY, H, RX, RY, RZ
from mlxq.mps_state import MPSOptions, MPSState
from mlxq.sim import StateVectorSimulator


EXACT_MPS = MPSOptions(dmax=64, eps=0.0)


def _statevector(sim: StateVectorSimulator) -> np.ndarray:
    mx.eval(sim.state)
    return np.asarray(sim.state.tolist(), dtype=np.complex128)


def _mps_statevector(sim: MPSState) -> np.ndarray:
    psi = sim.A[0]
    for tensor in sim.A[1:]:
        psi = mx.tensordot(psi, tensor, axes=([psi.ndim - 1], [0]))
    psi = mx.reshape(psi, (1 << sim.n,))
    mx.eval(psi)
    return np.asarray(psi.tolist(), dtype=np.complex128)


def _assert_same_state(reference: np.ndarray, actual: np.ndarray, atol: float = 5e-5):
    overlap = np.vdot(reference, actual)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-15 else 1.0 + 0.0j
    error = np.max(np.abs(actual - phase * reference))
    assert error < atol
    assert np.vdot(actual, actual).real == pytest.approx(1.0, abs=atol)


def _assert_bonds_consistent(sim: MPSState):
    actual = []
    for left, right in zip(sim.A, sim.A[1:]):
        assert left.shape[2] == right.shape[0]
        actual.append(int(left.shape[2]))
    assert sim.bond_dims() == actual


def _assert_mixed_canonical(sim: MPSState, center: int, atol: float = 2e-5):
    for q, tensor in enumerate(sim.A):
        if q == center:
            continue
        array = np.asarray(tensor.tolist(), dtype=np.complex128)
        dl, physical, dr = array.shape
        if q < center:
            matrix = array.reshape(dl * physical, dr)
            gram = matrix.conj().T @ matrix
            identity = np.eye(dr, dtype=np.complex128)
        else:
            matrix = array.reshape(dl, physical * dr)
            gram = matrix @ matrix.conj().T
            identity = np.eye(dl, dtype=np.complex128)
        np.testing.assert_allclose(gram, identity, atol=atol, rtol=0.0)


def _apply_both(sv, mps, gate, wires):
    if len(wires) == 1:
        sv.apply_single(gate, wires[0])
        mps.apply_single(gate, wires[0])
    else:
        sv.apply_two(gate, wires[0], wires[1])
        mps.apply_two(gate, wires[0], wires[1])


def test_single_qubit_gate_contracts_the_physical_axis_after_entanglement():
    sv = StateVectorSimulator(3)
    mps = MPSState(3, EXACT_MPS)

    for gate, wires in (
        (RY(0.73), (0,)),
        (CNOT(), (0, 1)),
        (RX(0.41), (0,)),
        (RZ(-0.29), (1,)),
    ):
        _apply_both(sv, mps, gate, wires)

    _assert_same_state(_statevector(sv), _mps_statevector(mps))
    _assert_bonds_consistent(mps)
    _assert_mixed_canonical(mps, center=1)
    assert not mps.truncated()


def test_asymmetric_two_qubit_gates_preserve_order_for_every_wire_layout():
    ordered_pairs = ((0, 1), (1, 0), (1, 2), (2, 1), (0, 3), (3, 0), (1, 3), (3, 1))
    asymmetric_gates = (CNOT(), CRY(0.47))

    for control, target in ordered_pairs:
        for two_qubit_gate in asymmetric_gates:
            sv = StateVectorSimulator(4)
            mps = MPSState(4, EXACT_MPS)
            for q, angle in enumerate((0.23, -0.51, 0.79, -1.07)):
                _apply_both(sv, mps, RY(angle), (q,))
                _apply_both(sv, mps, RZ(angle * 0.37), (q,))
            _apply_both(sv, mps, two_qubit_gate, (control, target))

            _assert_same_state(_statevector(sv), _mps_statevector(mps))
            _assert_bonds_consistent(mps)
            assert not mps.truncated()


def test_fixed_seed_random_circuits_match_complex_statevector():
    for seed in (7, 29, 101, 503):
        rng = np.random.default_rng(seed)
        sv = StateVectorSimulator(4)
        mps = MPSState(4, EXACT_MPS)

        for _ in range(24):
            if rng.random() < 0.58:
                q = int(rng.integers(0, 4))
                angle = float(rng.uniform(-np.pi, np.pi))
                gate = (RX, RY, RZ)[int(rng.integers(0, 3))](angle)
                _apply_both(sv, mps, gate, (q,))
            else:
                control, target = rng.choice(4, size=2, replace=False).tolist()
                if rng.random() < 0.5:
                    gate = CNOT()
                else:
                    gate = CRY(float(rng.uniform(-np.pi, np.pi)))
                _apply_both(sv, mps, gate, (int(control), int(target)))

        _assert_same_state(_statevector(sv), _mps_statevector(mps), atol=8e-5)
        _assert_bonds_consistent(mps)
        assert not mps.truncated()


def test_truncation_reports_local_discarded_weight():
    mps = MPSState(2, MPSOptions(dmax=1, eps=0.0))
    mps.apply_single(H(), 0)
    mps.apply_two(CNOT(), 0, 1)

    diagnostics = mps.truncation_diagnostics()
    event = diagnostics["last_event"]
    assert mps.truncated()
    assert mps.trunc_count() == 1
    assert diagnostics["events"] == 1
    assert diagnostics["local_discarded_weight_sum"] == pytest.approx(0.5, abs=1e-6)
    assert diagnostics["local_discarded_weight_max"] == pytest.approx(0.5, abs=1e-6)
    assert event["bond"] == 0
    assert event["rank_before"] == 2
    assert event["rank_kept"] == 1
    assert event["local_discarded_weight"] == pytest.approx(0.5, abs=1e-6)
    assert event["relative_discarded_weight"] == pytest.approx(0.5, abs=1e-6)
    assert event["limited_by_dmax"] is True
    assert event["limited_by_eps"] is False
    assert event["renormalized"] is True
    assert diagnostics["state_norm"] == pytest.approx(1.0, abs=2e-6)


def test_recoverable_svd_ladder_falls_back_without_losing_process(monkeypatch):
    from scipy import linalg as scipy_linalg

    monkeypatch.setattr(
        scipy_linalg,
        "svd",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            np.linalg.LinAlgError("forced LAPACK failure")
        ),
    )
    mps = MPSState(2, MPSOptions(dmax=4, eps=0.0, svd_driver="auto"))
    mps.apply_single(H(), 0)
    mps.apply_two(CNOT(), 0, 1)
    diagnostics = mps.truncation_diagnostics()

    _assert_same_state(
        np.array([2**-0.5, 0.0, 0.0, 2**-0.5], dtype=np.complex128),
        _mps_statevector(mps),
    )
    assert diagnostics["svd_fallback_count"] == 2
    assert diagnostics["svd_drivers_used"] == {"numpy_complex64": 1}


def test_explicit_canonicalization_moves_center_and_preserves_norm():
    mps = MPSState(5, MPSOptions(dmax=8, eps=1e-10))
    for wire in range(5):
        mps.apply_single(RY(0.13 * (wire + 1)), wire)
    for first, second in ((0, 4), (1, 3), (0, 2), (2, 4)):
        mps.apply_two(CNOT(), first, second)

    before = mps.to_statevector()
    mps.canonicalize(center=2)
    after = mps.to_statevector()

    _assert_mixed_canonical(mps, center=2)
    np.testing.assert_allclose(after, before, atol=5e-5, rtol=0.0)
    assert mps.norm() == pytest.approx(1.0, abs=2e-6)
