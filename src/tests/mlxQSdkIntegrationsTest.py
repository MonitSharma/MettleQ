import math

import numpy as np
import pennylane as qml
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector

from mlxq.integrations.pennylane import QupertinoDevice
from mlxq.integrations.qiskit import QupertinoBackend


def test_qiskit_statevector_ordering_and_gate_parity():
    circuit = QuantumCircuit(3, name="ordering-and-gates")
    circuit.h(0)
    circuit.ry(0.37, 1)
    circuit.cx(0, 2)
    circuit.cp(-0.23, 2, 1)
    circuit.rxx(0.41, 0, 1)
    circuit.ryy(-0.19, 1, 2)
    circuit.rzz(0.29, 2, 0)
    circuit.global_phase = 0.17

    result = QupertinoBackend().run(
        circuit, shots=16, return_statevector=True
    ).result()
    actual = np.asarray(result.data(0)["statevector"])
    expected = np.asarray(Statevector.from_instruction(circuit).data)
    assert np.allclose(actual, expected, rtol=0.0, atol=2e-6)


def test_qiskit_classical_bit_mapping_seed_and_opt_in_state_readback():
    circuit = QuantumCircuit(3, 3)
    circuit.x(0)
    circuit.x(2)
    circuit.measure(0, 2)
    circuit.measure(1, 1)
    circuit.measure(2, 0)
    backend = QupertinoBackend()

    first = backend.run(circuit, shots=25, seed_simulator=9).result()
    second = backend.run(circuit, shots=25, seed_simulator=9).result()
    assert first.get_counts() == {"101": 25}
    assert first.get_counts() == second.get_counts()
    assert "statevector" not in first.data(0)


def test_qiskit_dynamic_target_and_execution_evidence():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    backend = QupertinoBackend()

    compiled = transpile(circuit, backend)
    assert compiled.num_qubits == circuit.num_qubits
    result = backend.run(
        compiled, shots=32, seed_simulator=4, execution_report=True
    ).result()
    data = result.data(0)
    assert data["qupertino_execution_plan"]["execution_status"] == "evaluated"
    assert data["qupertino_statevector_preflight"]["decision"].startswith("allowed")
    assert backend.last_execution_plans[0] == data["qupertino_execution_plan"]


def test_qiskit_rejects_mid_circuit_measurement():
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.x(0)
    with pytest.raises(QiskitError, match="final measurements only"):
        QupertinoBackend().run(circuit)


def test_pennylane_registered_device_analytic_parity_and_wire_order():
    device = qml.device("qupertino", wires=["left", "right"])
    reference = qml.device("default.qubit", wires=["left", "right"])

    def build(dev):
        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard("left")
            qml.CNOT(wires=["left", "right"])
            qml.Rot(theta, -0.21, 0.13, wires="right")
            qml.GlobalPhase(0.19)
            return (
                qml.state(),
                qml.probs(wires=["right", "left"]),
                qml.expval(qml.X("left") @ qml.Z("right")),
                qml.var(qml.Z("left")),
            )

        return circuit

    actual = build(device)(0.31)
    expected = build(reference)(0.31)
    for actual_value, expected_value in zip(actual, expected):
        assert np.allclose(actual_value, expected_value, rtol=0.0, atol=2e-6)


def test_pennylane_parameter_shift_gradient():
    device = QupertinoDevice(wires=1)

    @qml.qnode(device, diff_method="parameter-shift")
    def circuit(theta):
        qml.RX(theta, wires=0)
        return qml.expval(qml.Z(0))

    theta = qml.numpy.array(0.4, requires_grad=True)
    gradient = qml.grad(circuit)(theta)
    assert abs(float(gradient) + math.sin(0.4)) < 2e-6


def test_pennylane_finite_shots_counts_samples_and_shot_vector():
    device = QupertinoDevice(wires=2, seed=12)

    @qml.set_shots([(40, 2), 20])
    @qml.qnode(device)
    def circuit():
        qml.Hadamard(0)
        qml.CNOT(wires=[0, 1])
        return qml.counts(wires=[0, 1]), qml.sample(qml.Z(0) @ qml.Z(1))

    partitions = circuit()
    assert len(partitions) == 3
    for (counts, samples), shots in zip(partitions, (40, 40, 20)):
        assert sum(counts.values()) == shots
        assert set(counts) <= {"00", "11"}
        assert samples.shape == (shots,)
        assert np.all(samples == 1)


def test_pennylane_wide_pauli_sentence_avoids_dense_observable():
    device = QupertinoDevice(wires=10, execution_report=True)

    @qml.qnode(device)
    def circuit():
        return qml.expval(
            0.5 * qml.Z(0) @ qml.Z(9) + 0.25 * qml.Z(4)
        )

    assert abs(circuit() - 0.75) < 2e-6
    assert device.statevector_preflight["decision"].startswith("allowed")
    assert device.last_execution_plan["execution_status"] == "evaluated"
