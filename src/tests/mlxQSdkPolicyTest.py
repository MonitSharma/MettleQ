import pytest

from mettleq.integrations.policy import profile_operations, recommend_sdk_engine


def _layer(n):
    return (
        [{"name": "U3", "wires": [q], "parameters": [0.1, 0.2, 0.3]} for q in range(n)]
        + [{"name": "CNOT", "wires": [q, q + 1]} for q in range(0, n - 1, 2)]
    )


def test_profile_reports_gate_mix_topology_and_fusion_coverage():
    profile = profile_operations(8, _layer(8))
    assert profile.operations == 12
    assert profile.single_qubit_operations == 8
    assert profile.topology_class == "nearest_neighbor"
    assert profile.estimated_fusion_coverage == pytest.approx(1.0)


def test_qiskit_policy_delegates_small_and_double_to_aer():
    small = recommend_sdk_engine("qiskit", 16, _layer(16))
    assert small.engine == "qiskit_aer_cpu_statevector"
    double = recommend_sdk_engine("qiskit", 24, _layer(24), precision="double")
    assert double.engine == "qiskit_aer_cpu_statevector"


def test_policy_selects_mettleq_gpu_above_circuit_aware_crossover():
    decision = recommend_sdk_engine(
        "qiskit", 22, _layer(22), output_contract="expectation"
    )
    assert decision.engine == "mettleq_gpu_statevector"
    assert decision.adjusted_gpu_crossover_qubits <= 20
    assert decision.profile.estimated_fusion_coverage == pytest.approx(1.0)


def test_policy_keeps_mps_on_strong_cpu_path():
    qiskit = recommend_sdk_engine("qiskit", 40, _layer(40), method="mps")
    pennylane = recommend_sdk_engine("pennylane", 40, _layer(40), method="mps")
    assert qiskit.engine == "qiskit_aer_cpu_mps"
    assert pennylane.engine == "mettleq_cpu_mps"
