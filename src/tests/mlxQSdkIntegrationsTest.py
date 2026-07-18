import math

import numpy as np
import pennylane as qml
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp

from mettleq.integrations.pennylane import AdaptivePennyLaneDevice, MettleQDevice
from mettleq.integrations.qiskit import (
    AdaptiveQiskitBackend,
    MettleQBackend,
    MettleQEstimatorV2,
    MettleQSamplerV2,
)


def test_adaptive_qiskit_backend_delegates_small_double_to_aer():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    backend = AdaptiveQiskitBackend(precision="double")
    result = backend.run(circuit, shots=32, seed_simulator=7).result()
    assert set(result.get_counts()) <= {"00", "11"}
    assert backend.last_adaptive_decisions[0]["engine"] == (
        "qiskit_aer_cpu_statevector"
    )


def test_adaptive_qiskit_backend_can_force_measured_gpu_side_of_policy():
    circuit = QuantumCircuit(3)
    circuit.h(range(3))
    backend = AdaptiveQiskitBackend(qiskit_gpu_crossover_qubits=1)
    result = backend.run(
        circuit, shots=1, return_statevector=True
    ).result()
    assert np.asarray(result.data(0)["statevector"]).shape == (8,)
    assert backend.last_adaptive_decisions[0]["engine"] == (
        "mettleq_gpu_statevector"
    )


def test_adaptive_qiskit_parameter_batch_preserves_mixed_engine_order():
    small = QuantumCircuit(2, 2)
    small.x(0)
    small.measure(range(2), range(2))
    large = QuantumCircuit(5, 5)
    large.x(4)
    large.measure(range(5), range(5))
    backend = AdaptiveQiskitBackend(qiskit_gpu_crossover_qubits=4)
    result = backend.run([small, large], shots=5).result()
    assert result.get_counts(0) == {"01": 5}
    assert result.get_counts(1) == {"10000": 5}
    assert [row["engine"] for row in backend.last_adaptive_decisions] == [
        "qiskit_aer_cpu_statevector", "mettleq_gpu_statevector"
    ]


def test_adaptive_pennylane_device_delegates_double_to_lightning():
    device = AdaptivePennyLaneDevice(wires=2, precision="double")

    @qml.qnode(device)
    def circuit():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        return qml.expval(qml.Z(0) @ qml.Z(1))

    assert circuit() == pytest.approx(1.0)
    assert device.last_adaptive_decisions[0]["engine"] == (
        "pennylane_lightning_cpu"
    )


def test_adaptive_pennylane_device_can_force_gpu_policy_side():
    device = AdaptivePennyLaneDevice(
        wires=3, pennylane_gpu_crossover_qubits=1
    )

    @qml.qnode(device)
    def circuit():
        for wire in range(3):
            qml.Hadamard(wire)
        return qml.state()

    assert circuit().shape == (8,)
    assert device.last_adaptive_decisions[0]["engine"] == (
        "mettleq_gpu_statevector"
    )


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

    result = MettleQBackend().run(
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
    backend = MettleQBackend()

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
    backend = MettleQBackend()

    compiled = transpile(circuit, backend)
    assert compiled.num_qubits == circuit.num_qubits
    result = backend.run(
        compiled, shots=32, seed_simulator=4, execution_report=True
    ).result()
    data = result.data(0)
    assert data["mettleq_execution_plan"]["execution_status"] == "evaluated"
    assert data["mettleq_statevector_preflight"]["decision"].startswith("allowed")
    assert backend.last_execution_plans[0] == data["mettleq_execution_plan"]


def test_qiskit_rejects_mid_circuit_measurement():
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.x(0)
    with pytest.raises(QiskitError, match="final measurements only"):
        MettleQBackend().run(circuit)


def test_pennylane_registered_device_analytic_parity_and_wire_order():
    device = qml.device("mettleq", wires=["left", "right"])
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
    device = MettleQDevice(wires=1)

    @qml.qnode(device, diff_method="parameter-shift")
    def circuit(theta):
        qml.RX(theta, wires=0)
        return qml.expval(qml.Z(0))

    theta = qml.numpy.array(0.4, requires_grad=True)
    gradient = qml.grad(circuit)(theta)
    assert abs(float(gradient) + math.sin(0.4)) < 2e-6


def test_pennylane_finite_shots_counts_samples_and_shot_vector():
    device = MettleQDevice(wires=2, seed=12)

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
    device = MettleQDevice(wires=10, execution_report=True)

    @qml.qnode(device)
    def circuit():
        return qml.expval(
            0.5 * qml.Z(0) @ qml.Z(9) + 0.25 * qml.Z(4)
        )

    assert abs(circuit() - 0.75) < 2e-6
    assert device.statevector_preflight["decision"].startswith("allowed")
    assert device.last_execution_plan["execution_status"] == "evaluated"


def test_qiskit_mps_method_counts_statevector_and_diagnostics():
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure(range(3), range(3))
    backend = MettleQBackend(
        method="matrix_product_state",
        device="cpu",
        mps_max_bond_dimension=16,
    )
    result = backend.run(
        circuit,
        shots=64,
        seed_simulator=11,
        return_statevector=True,
        execution_report=True,
    ).result()

    assert set(result.get_counts()) <= {"000", "111"}
    assert np.allclose(
        np.abs(np.asarray(result.data(0)["statevector"])) ** 2,
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        atol=2e-6,
    )
    selection = result.data(0)["mettleq_execution_selection"]
    diagnostics = result.data(0)["mettleq_mps_diagnostics"]
    assert selection["selected_method"] == "matrix_product_state"
    assert selection["selected_device"] == "cpu"
    assert diagnostics["tensor_device"] == "cpu"
    assert diagnostics["svd_device"] == "cpu"
    assert result.data(0)["mettleq_mps_accuracy"]["passed"] is True


def test_qiskit_sampler_v2_and_estimator_v2_native_contracts():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    estimator_result = MettleQEstimatorV2(device="cpu").run(
        [(circuit, [SparsePauliOp("ZZ"), SparsePauliOp("XX")])]
    ).result()[0]
    assert estimator_result.data.evs.shape == (2,)
    assert np.allclose(estimator_result.data.evs, [1.0, 1.0], atol=2e-6)
    assert np.all(estimator_result.data.stds == 0.0)

    measured = circuit.measure_all(inplace=False)
    sampler_result = MettleQSamplerV2(device="cpu").run(
        [measured], shots=64
    ).result()[0]
    counts = sampler_result.data.meas.get_counts()
    assert sum(counts.values()) == 64
    assert set(counts) <= {"00", "11"}


def test_estimator_batches_many_pauli_reductions_without_state_readback(monkeypatch):
    import mettleq.integrations._common as common

    calls = []
    original = common.mx.eval

    def recording_eval(*values):
        calls.append(len(values))
        return original(*values)

    monkeypatch.setattr(common.mx, "eval", recording_eval)
    circuit = QuantumCircuit(3)
    circuit.h(range(3))
    observable = SparsePauliOp.from_list(
        [("IIX", 0.25), ("IXI", 0.25), ("XII", 0.25), ("XXX", 0.25)]
    )
    value = MettleQEstimatorV2(device="gpu").run([(circuit, observable)]).result()[0]
    assert value.data.evs == pytest.approx(1.0, abs=2e-6)
    assert any(count >= 4 for count in calls)


def test_qiskit_estimator_reports_accuracy_and_automated_dmax_convergence():
    circuit = QuantumCircuit(6)
    for wire in range(6):
        circuit.h(wire)
    for first in range(6):
        for second in range(first + 1, 6):
            circuit.rzz(0.38, first, second)

    estimator = MettleQEstimatorV2(
        method="matrix_product_state",
        device="cpu",
        mps_max_bond_dimension=4,
        mps_convergence_bond_dimensions=(2, 4, 8),
        mps_convergence_atol=1e-3,
    )
    result = estimator.run([(circuit, SparsePauliOp("IIIIIZ"))]).result()[0]
    accuracy = result.metadata["mettleq_mps_accuracy"][0]
    convergence = result.metadata["mettleq_mps_convergence"]

    assert accuracy["classification"] == "threshold_exceeded"
    assert [run["dmax"] for run in convergence["runs"]] == [2, 4, 8]
    assert convergence["converged"] is True
    assert convergence["comparisons"][-1]["within_tolerance"] is True


def test_qiskit_backend_batch_reuses_one_same_width_device(monkeypatch):
    import mettleq.integrations._common as common

    original = common.Device
    constructed = []

    def recording_device(*args, **kwargs):
        constructed.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(common, "Device", recording_device)
    first = QuantumCircuit(2, 2)
    first.h(0)
    first.measure_all()
    second = QuantumCircuit(2, 2)
    second.x(1)
    second.measure_all()
    MettleQBackend(device="cpu").run(
        [first, second], shots=8, seed_simulator=3
    ).result()
    assert len(constructed) == 1


def test_pennylane_mps_analytic_finite_shots_gradient_and_tracking():
    device = MettleQDevice(
        wires=2,
        method="matrix_product_state",
        device="cpu",
        seed=5,
    )

    @qml.qnode(device, diff_method="parameter-shift")
    def analytic(theta):
        qml.RY(theta, 0)
        qml.CNOT([0, 1])
        return qml.expval(qml.Z(0) @ qml.Z(1))

    theta = qml.numpy.array(0.3, requires_grad=True)
    with device.tracker:
        value = analytic(theta)
    gradient = qml.grad(analytic)(theta)
    assert abs(float(value) - 1.0) < 2e-6
    assert abs(float(gradient)) < 2e-6
    assert device.tracker.totals["executions"] >= 1
    assert device.last_execution_selection["selected_method"] == (
        "matrix_product_state"
    )
    assert device.last_mps_diagnostics["svd_device"] == "cpu"

    @qml.set_shots(50)
    @qml.qnode(device)
    def sampled():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        return qml.counts(wires=[0, 1])

    counts = sampled()
    assert sum(counts.values()) == 50
    assert set(counts) <= {"00", "11"}


def test_pennylane_device_reports_accuracy_and_dmax_convergence():
    device = MettleQDevice(
        wires=6,
        method="matrix_product_state",
        device="cpu",
        mps_max_bond_dimension=4,
        mps_convergence_bond_dimensions=(2, 4, 8),
        mps_convergence_atol=1e-3,
    )

    @qml.qnode(device)
    def circuit():
        for wire in range(6):
            qml.Hadamard(wire)
        for first in range(6):
            for second in range(first + 1, 6):
                qml.IsingZZ(0.38, wires=[first, second])
        return qml.expval(qml.Z(0))

    value = circuit()
    report = device.last_mps_convergence_report
    assert np.isfinite(value)
    assert device.last_mps_accuracy_report["classification"] == (
        "threshold_exceeded"
    )
    assert [run["dmax"] for run in report["runs"]] == [2, 4, 8]
    assert report["converged"] is True


def test_pennylane_capabilities_declare_supported_contract():
    capabilities = MettleQDevice.capabilities
    assert "CNOT" in capabilities.operations
    assert "Hamiltonian" in capabilities.observables
    assert "StateMP" in capabilities.measurement_processes
    assert capabilities.dynamic_qubit_management is False
    assert capabilities.qjit_compatible is False
