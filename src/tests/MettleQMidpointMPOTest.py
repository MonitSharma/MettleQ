import pytest

from mettleq.midpoint_mpo import (
    MidpointMPOOptions,
    MidpointMPOResult,
    MidpointMPOSimulator,
    build_convergence_report,
)


def _result(*, max_bond, cutoff, fraction, matches=True):
    shots = 100
    count = round(shots * fraction)
    expected = "101"
    return MidpointMPOResult(
        counts={expected: count, "000": shots - count},
        shots=shots,
        predicted_bitstring=expected if matches else "000",
        expected_bitstring=expected,
        expected_peak_count=count,
        expected_peak_fraction=count / shots,
        matches_expected_bitstring=matches,
        compression_time_s=1.0,
        materialize_time_s=0.1,
        sampling_time_s=0.1,
        measurement_permutation=[0, 1, 2],
        samples=[],
        raw_samples=[],
        diagnostics={"options": {"max_bond": max_bond, "cutoff": cutoff}},
        stats=[],
    )


def test_midpoint_mpo_options_reject_invalid_trust_controls():
    with pytest.raises(ValueError, match="max_bond"):
        MidpointMPOOptions(max_bond=0)
    with pytest.raises(ValueError, match="cutoff"):
        MidpointMPOOptions(cutoff=1.0)


def test_midpoint_mpo_convergence_requires_peak_recovery_and_stability():
    report = build_convergence_report(
        [
            _result(max_bond=64, cutoff=6e-4, fraction=0.10),
            _result(max_bond=128, cutoff=6e-4, fraction=0.12),
        ],
        peak_fraction_atol=0.03,
    )
    assert report["classification"] == "converged"
    assert report["expected_peak_fraction_spread"] == pytest.approx(0.02)
    assert report["convergence_axes"] == ["bond"]

    cutoff = build_convergence_report(
        [
            _result(max_bond=128, cutoff=1e-3, fraction=0.10),
            _result(max_bond=128, cutoff=6e-4, fraction=0.11),
        ]
    )
    assert cutoff["classification"] == "converged"
    assert cutoff["convergence_axes"] == ["cutoff"]

    confounded = build_convergence_report(
        [
            _result(max_bond=64, cutoff=1e-3, fraction=0.10),
            _result(max_bond=128, cutoff=6e-4, fraction=0.10),
        ]
    )
    assert confounded["classification"] == "not_converged"
    assert confounded["qualified_comparison_count"] == 0

    failed = build_convergence_report(
        [
            _result(max_bond=64, cutoff=6e-4, fraction=0.10),
            _result(max_bond=128, cutoff=6e-4, fraction=0.10, matches=False),
        ]
    )
    assert failed["classification"] == "not_converged"


def test_midpoint_mpo_exact_qiskit_smoke():
    pytest.importorskip("quimb")
    pytest.importorskip("qiskit_quimb")
    qiskit = pytest.importorskip("qiskit")
    circuit = qiskit.QuantumCircuit(3)
    circuit.x(0)
    circuit.barrier()
    circuit.x(2)
    simulator = MidpointMPOSimulator(
        MidpointMPOOptions(
            max_bond=8,
            cutoff=0.0,
            sabre_trials=2,
            post_sabre_trials=2,
        )
    )
    result = simulator.run(
        circuit, shots=16, expected_bitstring="101"
    )
    assert result.counts == {"101": 16}
    assert result.matches_expected_bitstring is True
    assert result.diagnostics["termination_reason"] == "completed"
    assert result.diagnostics["vendor_reference_commit"]
