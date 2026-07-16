import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from mettleq.midpoint_mpo import (
    IsolatedMidpointMPOSimulator,
    MidpointMPOError,
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


def test_isolated_midpoint_mpo_roundtrips_qiskit2_result_contract(
    tmp_path, monkeypatch
):
    qiskit = pytest.importorskip("qiskit")
    circuit = qiskit.QuantumCircuit(3)
    circuit.x(0)
    circuit.x(2)
    output_dir = tmp_path / "worker-output"

    def fake_run(command, **kwargs):
        assert Path(command[0]).samefile(sys.executable)
        assert kwargs["env"]["PYTHONPATH"]
        worker_output = command[command.index("--output-dir") + 1]
        assert str(worker_output) == str(output_dir)
        assert "OPENQASM 2.0" in (output_dir / "input.qasm").read_text()
        summary = {
            "method": "midpoint_mpo_unswapping",
            "shots": 4,
            "predicted_bitstring": "101",
            "expected_bitstring": "101",
            "expected_peak_count": 4,
            "expected_peak_fraction": 1.0,
            "matches_expected_bitstring": True,
            "compression_time_s": 1.0,
            "materialize_time_s": 0.1,
            "sampling_time_s": 0.2,
            "measurement_permutation": [0, 1, 2],
            "diagnostics": {
                "termination_reason": "completed",
                "options": {"max_bond": 8, "cutoff": 0.0},
            },
            "counts": {"101": 4},
        }
        (output_dir / "summary.json").write_text(json.dumps(summary))
        (output_dir / "stats.json").write_text(
            json.dumps([{"stage": "termination", "termination_reason": "completed"}])
        )
        (output_dir / "samples.tsv").write_text(
            "raw\tpermuted\n101\t101\n101\t101\n101\t101\n101\t101\n"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("mettleq.midpoint_mpo.subprocess.run", fake_run)
    simulator = IsolatedMidpointMPOSimulator(
        MidpointMPOOptions(max_bond=8, cutoff=0.0),
        worker_python=sys.executable,
    )
    result = simulator.run(
        circuit,
        shots=4,
        expected_bitstring="101",
        output_dir=output_dir,
    )
    assert result.counts == {"101": 4}
    assert result.matches_expected_bitstring is True
    assert result.diagnostics["execution_mode"] == "isolated_worker"
    assert result.diagnostics["caller_qiskit_version"] == qiskit.__version__


def test_isolated_midpoint_mpo_rejects_dynamic_circuit_before_worker(tmp_path):
    qiskit = pytest.importorskip("qiskit")
    circuit = qiskit.QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    simulator = IsolatedMidpointMPOSimulator(
        worker_python=sys.executable,
    )
    with pytest.raises(MidpointMPOError, match="unsupported operations: measure"):
        simulator.run(circuit, output_dir=tmp_path)


def test_priority_campaign_schedules_balance_every_position():
    from tools.benchmark_midpoint_mpo_priority_phase import (
        MAIN_ARMS,
        _cutoff_schedule,
        _main_schedule,
    )

    main = _main_schedule(3)
    for arm in MAIN_ARMS:
        assert sorted(row["position"] for row in main if row["arm"] == arm) == [0, 1, 2]
    cutoff = _cutoff_schedule(2)
    for arm in {row["arm"] for row in cutoff}:
        assert sorted(row["position"] for row in cutoff if row["arm"] == arm) == [0, 1]
