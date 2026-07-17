#!/usr/bin/env python3
"""Build the reviewed MettleQ Qiskit and PennyLane tutorial notebooks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import textwrap

import nbformat


ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"


@dataclass(frozen=True)
class NotebookSpec:
    framework: str
    filename: str
    title: str
    summary: str
    source: str


QISKIT_SETUP = """
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from mettleq.integrations.qiskit import (
    MettleQBackend,
    MettleQEstimatorV2,
    MettleQSamplerV2,
)
from tutorials._support import (
    benchmark,
    emit_result,
    max_abs_error,
    phase_aligned_statevector_error,
    qiskit_selection,
    total_variation_distance,
)
"""


PENNYLANE_SETUP = """
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from mettleq.integrations.pennylane import MettleQDevice
from tutorials._support import (
    benchmark,
    emit_result,
    max_abs_error,
    pennylane_selection,
    phase_aligned_statevector_error,
    total_variation_distance,
)
"""


def _qiskit(filename: str, title: str, summary: str, source: str) -> NotebookSpec:
    return NotebookSpec("qiskit", filename, title, summary, source)


def _pennylane(filename: str, title: str, summary: str, source: str) -> NotebookSpec:
    return NotebookSpec("pennylane", filename, title, summary, source)


def qiskit_specs() -> list[NotebookSpec]:
    return [
        _qiskit(
            "01_backend_quickstart.ipynb",
            "Qiskit BackendV2 quickstart",
            "Build and transpile a Bell circuit, then compare Qiskit's exact state with MettleQ's native BackendV2 result and execution plan.",
            """
circuit = QuantumCircuit(2, name="bell")
circuit.h(0)
circuit.cx(0, 1)

reference, reference_ms, _ = benchmark(
    lambda: np.asarray(Statevector.from_instruction(circuit).data)
)
backend = MettleQBackend(method="statevector", device="auto")
compiled = transpile(circuit, backend, optimization_level=1)

def run_mettleq():
    return np.asarray(
        backend.run(
            compiled,
            shots=1,
            return_statevector=True,
            execution_report=True,
        ).result().data(0)["statevector"]
    )

candidate, mettleq_ms, _ = benchmark(run_mettleq)
error = phase_aligned_statevector_error(reference, candidate)
method, device = qiskit_selection(backend)
plan = backend.last_execution_plans[-1]
tutorial_result = emit_result(
    notebook="qiskit/01_backend_quickstart.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="phase-aligned statevector atol=2e-6",
    passed=error <= 2e-6 and compiled.num_qubits == 2,
    exact_match=bool(np.array_equal(reference, candidate)),
    selected_method=method,
    selected_device=device,
    metrics={"max_amplitude_error": error, "execution_status": plan["execution_status"]},
)
""",
        ),
        _qiskit(
            "02_statevectors_and_gates.ipynb",
            "Statevectors, gates, and global phase",
            "Exercise common one- and two-qubit gates, Qiskit's little-endian ordering, and global phase.",
            """
circuit = QuantumCircuit(3, name="gate-parity")
circuit.h(0)
circuit.ry(0.37, 1)
circuit.cx(0, 2)
circuit.cp(-0.23, 2, 1)
circuit.rxx(0.41, 0, 1)
circuit.ryy(-0.19, 1, 2)
circuit.rzz(0.29, 2, 0)
circuit.global_phase = 0.17

reference, reference_ms, _ = benchmark(
    lambda: np.asarray(Statevector.from_instruction(circuit).data)
)
backend = MettleQBackend(method="statevector", device="cpu")
compiled = transpile(circuit, backend, optimization_level=1)

def run_mettleq():
    return np.asarray(backend.run(compiled, shots=1, return_statevector=True).result().data(0)["statevector"])

candidate, mettleq_ms, _ = benchmark(run_mettleq)
error = phase_aligned_statevector_error(reference, candidate)
method, device = qiskit_selection(backend)
tutorial_result = emit_result(
    notebook="qiskit/02_statevectors_and_gates.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="phase-aligned statevector atol=2e-6",
    passed=error <= 2e-6,
    exact_match=bool(np.array_equal(reference, candidate)),
    selected_method=method,
    selected_device=device,
    metrics={"max_amplitude_error": error, "norm": float(np.linalg.norm(candidate))},
)
""",
        ),
        _qiskit(
            "03_sampler_and_counts.ipynb",
            "SamplerV2 and finite-shot counts",
            "Compare Qiskit's StatevectorSampler with MettleQSamplerV2 using the same Bell-state sampling contract.",
            """
circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.measure_all()
shots = 4096

def run_reference():
    result = StatevectorSampler(seed=19).run([circuit], shots=shots).result()[0]
    return result.data.meas.get_counts()

reference, reference_ms, _ = benchmark(run_reference)
backend = MettleQBackend(method="statevector", device="cpu")
compiled = transpile(circuit, backend, optimization_level=1)

def run_mettleq():
    result = MettleQSamplerV2(backend=backend).run([compiled], shots=shots).result()[0]
    return result.data.meas.get_counts()

candidate, mettleq_ms, _ = benchmark(run_mettleq)
tvd = total_variation_distance(reference, candidate)
support_ok = set(reference) <= {"000", "111"} and set(candidate) <= {"000", "111"}
method, device = qiskit_selection(backend)
tutorial_result = emit_result(
    notebook="qiskit/03_sampler_and_counts.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="finite-shot total-variation distance <= 0.05",
    passed=support_ok and tvd <= 0.05,
    exact_match=reference == candidate,
    selected_method=method,
    selected_device=device,
    metrics={"tvd": tvd, "reference_counts": reference, "mettleq_counts": candidate},
    notes="Independent sampler RNGs are compared statistically, not byte-for-byte.",
)
""",
        ),
        _qiskit(
            "04_estimator_chsh.ipynb",
            "CHSH-style EstimatorV2 sweep",
            "Evaluate two Bell-pair correlations across a measurement-basis sweep with Qiskit and MettleQ estimators.",
            """
angles = np.linspace(0.0, np.pi, 9)
observables = [SparsePauliOp("ZZ"), SparsePauliOp("ZX")]
circuits = []
for angle in angles:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(float(angle), 0)
    circuits.append(circuit)
pubs = [(circuit, observables) for circuit in circuits]

def run_reference():
    return np.asarray([StatevectorEstimator().run([pub]).result()[0].data.evs for pub in pubs])

reference, reference_ms, _ = benchmark(run_reference)
backend = MettleQBackend(method="statevector", device="cpu")
mettleq_pubs = [(transpile(circuit, backend, optimization_level=1), observables) for circuit in circuits]
estimator = MettleQEstimatorV2(backend=backend)

def run_mettleq():
    return np.asarray([estimator.run([pub]).result()[0].data.evs for pub in mettleq_pubs])

candidate, mettleq_ms, _ = benchmark(run_mettleq)
error = max_abs_error(reference, candidate)
method, device = qiskit_selection(estimator)
tutorial_result = emit_result(
    notebook="qiskit/04_estimator_chsh.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="EstimatorV2 correlation curve atol=2e-6",
    passed=error <= 2e-6,
    exact_match=bool(np.array_equal(reference, candidate)),
    selected_method=method,
    selected_device=device,
    metrics={"max_expectation_error": error, "angles": angles, "reference": reference, "mettleq": candidate},
)
""",
        ),
        _qiskit(
            "05_deutsch_jozsa.ipynb",
            "Deutsch–Jozsa classification",
            "Distinguish a balanced oracle from a constant oracle using exact output probabilities.",
            """
def deutsch_jozsa(balanced):
    n = 4
    circuit = QuantumCircuit(n + 1)
    circuit.x(n)
    circuit.h(range(n + 1))
    if balanced:
        for wire in range(n):
            circuit.cx(wire, n)
    circuit.h(range(n))
    return circuit

circuits = [deutsch_jozsa(False), deutsch_jozsa(True)]

def reference_probabilities():
    return np.asarray([Statevector.from_instruction(c).probabilities(qargs=range(4)) for c in circuits])

reference, reference_ms, _ = benchmark(reference_probabilities)
backend = MettleQBackend(method="statevector", device="cpu")
compiled = [transpile(c, backend, optimization_level=1) for c in circuits]

def mettleq_probabilities():
    values = []
    for circuit in compiled:
        state = backend.run(circuit, shots=1, return_statevector=True).result().data(0)["statevector"]
        values.append(Statevector(state).probabilities(qargs=range(4)))
    return np.asarray(values)

candidate, mettleq_ms, _ = benchmark(mettleq_probabilities)
error = max_abs_error(reference, candidate)
classes = [int(np.argmax(row) != 0) for row in candidate]
method, device = qiskit_selection(backend)
tutorial_result = emit_result(
    notebook="qiskit/05_deutsch_jozsa.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="probability vector atol=2e-6 and exact oracle class",
    passed=error <= 2e-6 and classes == [0, 1],
    exact_match=classes == [0, 1],
    selected_method=method,
    selected_device=device,
    metrics={"max_probability_error": error, "classifications": classes},
)
""",
        ),
        _qiskit(
            "06_bernstein_vazirani.ipynb",
            "Bernstein–Vazirani hidden string",
            "Recover a hidden binary string from one coherent oracle query.",
            """
secret = "101101"
n = len(secret)
circuit = QuantumCircuit(n + 1)
circuit.x(n)
circuit.h(range(n + 1))
for wire, bit in enumerate(reversed(secret)):
    if bit == "1":
        circuit.cx(wire, n)
circuit.h(range(n))

def get_reference():
    return Statevector.from_instruction(circuit).probabilities(qargs=range(n))

reference, reference_ms, _ = benchmark(get_reference)
backend = MettleQBackend(method="statevector", device="cpu")
compiled = transpile(circuit, backend, optimization_level=1)

def get_mettleq():
    state = backend.run(compiled, shots=1, return_statevector=True).result().data(0)["statevector"]
    return Statevector(state).probabilities(qargs=range(n))

candidate, mettleq_ms, _ = benchmark(get_mettleq)
error = max_abs_error(reference, candidate)
recovered = format(int(np.argmax(candidate)), f"0{n}b")
method, device = qiskit_selection(backend)
tutorial_result = emit_result(
    notebook="qiskit/06_bernstein_vazirani.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="probability vector atol=2e-6 and exact hidden string",
    passed=error <= 2e-6 and recovered == secret,
    exact_match=recovered == secret,
    selected_method=method,
    selected_device=device,
    metrics={"max_probability_error": error, "secret": secret, "recovered": recovered},
)
""",
        ),
        _qiskit(
            "07_grover_search.ipynb",
            "Grover search",
            "Amplify the marked two-qubit state and compare the complete ideal distribution.",
            """
circuit = QuantumCircuit(2)
circuit.h(range(2))
circuit.cz(0, 1)  # mark |11>
circuit.h(range(2))
circuit.x(range(2))
circuit.cz(0, 1)
circuit.x(range(2))
circuit.h(range(2))

def get_reference():
    return Statevector.from_instruction(circuit).probabilities()

reference, reference_ms, _ = benchmark(get_reference)
backend = MettleQBackend(method="statevector", device="cpu")
compiled = transpile(circuit, backend, optimization_level=1)

def get_mettleq():
    state = backend.run(compiled, shots=1, return_statevector=True).result().data(0)["statevector"]
    return np.abs(np.asarray(state)) ** 2

candidate, mettleq_ms, _ = benchmark(get_mettleq)
error = max_abs_error(reference, candidate)
marked = format(int(np.argmax(candidate)), "02b")
method, device = qiskit_selection(backend)
tutorial_result = emit_result(
    notebook="qiskit/07_grover_search.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="probability vector atol=2e-6 and exact marked item",
    passed=error <= 2e-6 and marked == "11",
    exact_match=marked == "11",
    selected_method=method,
    selected_device=device,
    metrics={"max_probability_error": error, "marked_item": marked, "marked_probability": candidate[-1]},
)
""",
        ),
        _qiskit(
            "08_qft_and_phase_estimation.ipynb",
            "QFT and phase estimation",
            "Estimate a phase exactly representable with three counting qubits and compare its full output distribution.",
            """
from qiskit.circuit.library import QFT

counting = 3
phase = 3 / 8
circuit = QuantumCircuit(counting + 1)
circuit.x(counting)
circuit.h(range(counting))
for wire in range(counting):
    circuit.cp(2 * np.pi * phase * (2 ** wire), wire, counting)
circuit.append(QFT(counting, inverse=True, do_swaps=True).to_gate(), range(counting))

def get_reference():
    return Statevector.from_instruction(circuit).probabilities(qargs=range(counting))

reference, reference_ms, _ = benchmark(get_reference)
backend = MettleQBackend(method="statevector", device="cpu")
compiled = transpile(circuit, backend, optimization_level=1)

def get_mettleq():
    state = backend.run(compiled, shots=1, return_statevector=True).result().data(0)["statevector"]
    return Statevector(state).probabilities(qargs=range(counting))

candidate, mettleq_ms, _ = benchmark(get_mettleq)
error = max_abs_error(reference, candidate)
reference_mode = int(np.argmax(reference))
candidate_mode = int(np.argmax(candidate))
method, device = qiskit_selection(backend)
tutorial_result = emit_result(
    notebook="qiskit/08_qft_and_phase_estimation.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="phase-register probabilities atol=2e-6 and identical mode",
    passed=error <= 2e-6 and candidate_mode == reference_mode,
    exact_match=candidate_mode == reference_mode,
    selected_method=method,
    selected_device=device,
    metrics={"max_probability_error": error, "expected_phase": phase, "reference_mode": reference_mode, "mettleq_mode": candidate_mode},
)
""",
        ),
        _qiskit(
            "09_shor_order_finding.ipynb",
            "Shor order finding for 15",
            "Run the compiled period-finding core used in the small N=15 Shor demonstration and recover non-trivial factors classically.",
            """
from fractions import Fraction
from math import gcd
from qiskit.circuit.library import QFT

# For a=2 mod 15, the modular order is r=4. The phase-register state below is
# the exact compiled order-finding core after modular exponentiation.
counting = 4
circuit = QuantumCircuit(counting)
circuit.h(range(counting))
for wire in range(counting):
    circuit.p(2 * np.pi * (2 ** wire) / 4, wire)
circuit.append(QFT(counting, inverse=True, do_swaps=True).to_gate(), range(counting))

def probabilities_from_qiskit():
    return Statevector.from_instruction(circuit).probabilities()

reference, reference_ms, _ = benchmark(probabilities_from_qiskit)
backend = MettleQBackend(method="statevector", device="cpu")
compiled = transpile(circuit, backend, optimization_level=1)

def probabilities_from_mettleq():
    state = backend.run(compiled, shots=1, return_statevector=True).result().data(0)["statevector"]
    return np.abs(np.asarray(state)) ** 2

candidate, mettleq_ms, _ = benchmark(probabilities_from_mettleq)
error = max_abs_error(reference, candidate)
phase_integer = int(np.argmax(candidate))
phase_fraction = Fraction(phase_integer, 2 ** counting).limit_denominator(15)
order = phase_fraction.denominator
factors = sorted({gcd(pow(2, order // 2) - 1, 15), gcd(pow(2, order // 2) + 1, 15)})
method, device = qiskit_selection(backend)
tutorial_result = emit_result(
    notebook="qiskit/09_shor_order_finding.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="compiled phase distribution atol=2e-6 and factors 3,5",
    passed=error <= 2e-6 and factors == [3, 5],
    exact_match=factors == [3, 5],
    selected_method=method,
    selected_device=device,
    metrics={"max_probability_error": error, "phase_integer": phase_integer, "order": order, "factors": factors},
    notes="This is the small compiled order-finding core, not a scalable modular-arithmetic implementation.",
)
""",
        ),
        _qiskit(
            "10_vqe.ipynb",
            "Estimator-based VQE energy scan",
            "Evaluate a compact two-qubit chemistry-style Hamiltonian over a variational ansatz parameter scan.",
            """
hamiltonian = SparsePauliOp.from_list([
    ("II", -1.05), ("ZI", 0.39), ("IZ", -0.39), ("ZZ", -0.01), ("XX", 0.18)
])
angles = np.linspace(-np.pi, np.pi, 25)
circuits = []
for angle in angles:
    circuit = QuantumCircuit(2)
    circuit.ry(float(angle), 0)
    circuit.cx(0, 1)
    circuit.ry(float(-0.37 * angle), 1)
    circuits.append(circuit)

def reference_energies():
    estimator = StatevectorEstimator()
    return np.asarray([estimator.run([(c, hamiltonian)]).result()[0].data.evs.item() for c in circuits])

reference, reference_ms, _ = benchmark(reference_energies)
backend = MettleQBackend(method="statevector", device="cpu")
compiled = [transpile(c, backend, optimization_level=1) for c in circuits]
estimator = MettleQEstimatorV2(backend=backend)

def mettleq_energies():
    return np.asarray([estimator.run([(c, hamiltonian)]).result()[0].data.evs.item() for c in compiled])

candidate, mettleq_ms, _ = benchmark(mettleq_energies)
error = max_abs_error(reference, candidate)
minima_match = int(np.argmin(reference)) == int(np.argmin(candidate))
method, device = qiskit_selection(estimator)
tutorial_result = emit_result(
    notebook="qiskit/10_vqe.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="VQE energy trace atol=3e-6",
    passed=error <= 3e-6 and minima_match,
    exact_match=minima_match,
    selected_method=method,
    selected_device=device,
    metrics={"max_energy_error": error, "reference_minimum": float(reference.min()), "mettleq_minimum": float(candidate.min()), "minimum_index": int(np.argmin(candidate))},
)
""",
        ),
        _qiskit(
            "11_qaoa_maxcut.ipynb",
            "QAOA for triangle MaxCut",
            "Compare the p=1 QAOA cost landscape for a three-node triangle graph.",
            """
cost = SparsePauliOp.from_list([
    ("III", 1.5), ("IZZ", -0.5), ("ZZI", -0.5), ("ZIZ", -0.5)
])
parameters = [(gamma, beta) for gamma in np.linspace(0.0, np.pi, 7) for beta in np.linspace(0.0, np.pi / 2, 5)]

def qaoa_circuit(gamma, beta):
    circuit = QuantumCircuit(3)
    circuit.h(range(3))
    for first, second in ((0, 1), (1, 2), (0, 2)):
        circuit.rzz(float(-gamma), first, second)
    for wire in range(3):
        circuit.rx(float(2 * beta), wire)
    return circuit

circuits = [qaoa_circuit(*values) for values in parameters]

def reference_costs():
    estimator = StatevectorEstimator()
    return np.asarray([estimator.run([(c, cost)]).result()[0].data.evs.item() for c in circuits])

reference, reference_ms, _ = benchmark(reference_costs)
backend = MettleQBackend(method="statevector", device="cpu")
compiled = [transpile(c, backend, optimization_level=1) for c in circuits]
estimator = MettleQEstimatorV2(backend=backend)

def mettleq_costs():
    return np.asarray([estimator.run([(c, cost)]).result()[0].data.evs.item() for c in compiled])

candidate, mettleq_ms, _ = benchmark(mettleq_costs)
error = max_abs_error(reference, candidate)
best_match = int(np.argmax(reference)) == int(np.argmax(candidate))
method, device = qiskit_selection(estimator)
tutorial_result = emit_result(
    notebook="qiskit/11_qaoa_maxcut.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="QAOA cost landscape atol=3e-6",
    passed=error <= 3e-6 and best_match,
    exact_match=best_match,
    selected_method=method,
    selected_device=device,
    metrics={"max_cost_error": error, "reference_best": float(reference.max()), "mettleq_best": float(candidate.max()), "best_parameters": parameters[int(np.argmax(candidate))]},
)
""",
        ),
        _qiskit(
            "12_hamiltonian_simulation.ipynb",
            "Hamiltonian simulation",
            "Trotterize a transverse-field Ising chain and compare an observable trajectory.",
            """
times = np.linspace(0.0, 1.2, 13)
observable = SparsePauliOp("IIZ")

def evolution_circuit(time_value):
    circuit = QuantumCircuit(3)
    circuit.x(0)
    steps = 6
    dt = float(time_value) / steps
    for _ in range(steps):
        circuit.rzz(1.1 * dt, 0, 1)
        circuit.rzz(1.1 * dt, 1, 2)
        for wire in range(3):
            circuit.rx(0.7 * dt, wire)
    return circuit

circuits = [evolution_circuit(value) for value in times]

def reference_trajectory():
    estimator = StatevectorEstimator()
    return np.asarray([estimator.run([(c, observable)]).result()[0].data.evs.item() for c in circuits])

reference, reference_ms, _ = benchmark(reference_trajectory)
backend = MettleQBackend(method="statevector", device="cpu")
compiled = [transpile(c, backend, optimization_level=1) for c in circuits]
estimator = MettleQEstimatorV2(backend=backend)

def mettleq_trajectory():
    return np.asarray([estimator.run([(c, observable)]).result()[0].data.evs.item() for c in compiled])

candidate, mettleq_ms, _ = benchmark(mettleq_trajectory)
error = max_abs_error(reference, candidate)
method, device = qiskit_selection(estimator)
tutorial_result = emit_result(
    notebook="qiskit/12_hamiltonian_simulation.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="observable trajectory atol=3e-6",
    passed=error <= 3e-6,
    exact_match=bool(np.array_equal(reference, candidate)),
    selected_method=method,
    selected_device=device,
    metrics={"max_observable_error": error, "times": times, "reference": reference, "mettleq": candidate},
)
""",
        ),
        _qiskit(
            "13_quantum_kernel.ipynb",
            "Fidelity quantum kernel",
            "Build a small feature-map kernel matrix from pairwise state fidelities.",
            """
data = np.asarray([[0.1, 0.2, -0.1], [0.7, -0.4, 0.3], [-0.5, 0.6, 0.8], [0.2, 0.9, -0.7]])

def feature_map(values):
    circuit = QuantumCircuit(3)
    for wire, value in enumerate(values):
        circuit.h(wire)
        circuit.rz(float(value), wire)
    for wire in range(2):
        circuit.rzz(float(values[wire] * values[wire + 1]), wire, wire + 1)
    return circuit

circuits = [feature_map(row) for row in data]

def kernel(states):
    return np.asarray([[abs(np.vdot(left, right)) ** 2 for right in states] for left in states])

def reference_kernel():
    return kernel([np.asarray(Statevector.from_instruction(c).data) for c in circuits])

reference, reference_ms, _ = benchmark(reference_kernel)
backend = MettleQBackend(method="statevector", device="cpu")
compiled = [transpile(c, backend, optimization_level=1) for c in circuits]

def mettleq_kernel():
    states = [np.asarray(backend.run(c, shots=1, return_statevector=True).result().data(0)["statevector"]) for c in compiled]
    return kernel(states)

candidate, mettleq_ms, _ = benchmark(mettleq_kernel)
error = max_abs_error(reference, candidate)
method, device = qiskit_selection(backend)
tutorial_result = emit_result(
    notebook="qiskit/13_quantum_kernel.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="fidelity kernel matrix atol=4e-6",
    passed=error <= 4e-6,
    exact_match=bool(np.array_equal(reference, candidate)),
    selected_method=method,
    selected_device=device,
    metrics={"max_kernel_error": error, "reference": reference, "mettleq": candidate},
)
""",
        ),
        _qiskit(
            "14_mps_topology_and_convergence.ipynb",
            "MPS topology and Dmax convergence",
            "Compare routed CPU MPS with an exact 10-qubit long-range circuit and inspect automated convergence evidence.",
            """
rng = np.random.default_rng(41)
circuit = QuantumCircuit(10)
for layer in range(3):
    for wire in range(10):
        circuit.ry(float(rng.uniform(-1, 1)), wire)
    order = rng.permutation(10)
    for index in range(0, 10, 2):
        circuit.rzz(float(rng.uniform(-0.8, 0.8)), int(order[index]), int(order[index + 1]))

reference, reference_ms, _ = benchmark(lambda: np.asarray(Statevector.from_instruction(circuit).data))
backend = MettleQBackend(
    method="matrix_product_state",
    device="cpu",
    mps_max_bond_dimension=32,
    mps_truncation_threshold=1e-12,
    mps_routing_strategy="lookahead",
)
compiled = transpile(circuit, backend, optimization_level=1)

def run_mps():
    return np.asarray(backend.run(compiled, shots=1, return_statevector=True, execution_report=True).result().data(0)["statevector"])

candidate, mettleq_ms, _ = benchmark(run_mps)
error = phase_aligned_statevector_error(reference, candidate)
diagnostics = backend.last_mps_diagnostics[-1]
accuracy = backend.last_mps_accuracy_reports[-1]
convergence_estimator = MettleQEstimatorV2(
    method="matrix_product_state",
    device="cpu",
    mps_max_bond_dimension=32,
    mps_truncation_threshold=1e-12,
    mps_convergence_bond_dimensions=(8, 16, 32),
    mps_convergence_atol=5e-5,
)
convergence_result = convergence_estimator.run([(circuit, SparsePauliOp("IIIIIIIIIZ"))]).result()[0]
convergence = convergence_result.metadata["mettleq_mps_convergence"]
method, device = qiskit_selection(backend)
tutorial_result = emit_result(
    notebook="qiskit/14_mps_topology_and_convergence.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="phase-aligned MPS state atol=8e-5 and convergence report",
    passed=error <= 8e-5 and convergence["converged"] and accuracy["passed"],
    exact_match=bool(np.array_equal(reference, candidate)),
    selected_method=method,
    selected_device=device,
    metrics={"max_amplitude_error": error, "peak_bond": diagnostics["maximum_bond_dimension_reached"], "accuracy": accuracy, "convergence": convergence},
)
""",
        ),
        _qiskit(
            "15_apple_gpu_scaling.ipynb",
            "Apple Silicon CPU/GPU selection",
            "Sweep exact statevector widths around MettleQ's measured GPU crossover and verify parity at each width.",
            """
import statistics

widths = [12, 14, 16]
rows = []
for width in widths:
    circuit = QuantumCircuit(width)
    for wire in range(width):
        circuit.ry(0.03 * (wire + 1), wire)
    for wire in range(width - 1):
        circuit.cx(wire, wire + 1)
    reference, reference_ms, _ = benchmark(lambda c=circuit: np.asarray(Statevector.from_instruction(c).data), repeats=2)
    backend = MettleQBackend(method="statevector", device="auto")
    compiled = transpile(circuit, backend, optimization_level=1)
    def run_mettleq(c=compiled, b=backend):
        return np.asarray(b.run(c, shots=1, return_statevector=True).result().data(0)["statevector"])
    candidate, mettleq_ms, _ = benchmark(run_mettleq, repeats=2)
    method, device = qiskit_selection(backend)
    rows.append({
        "width": width,
        "reference_ms": reference_ms,
        "mettleq_ms": mettleq_ms,
        "error": phase_aligned_statevector_error(reference, candidate),
        "method": method,
        "device": device,
    })

passed = all(row["error"] <= 3e-6 for row in rows) and rows[-1]["device"] == "gpu"
tutorial_result = emit_result(
    notebook="qiskit/15_apple_gpu_scaling.ipynb",
    framework="qiskit",
    reference_ms=statistics.median(row["reference_ms"] for row in rows),
    mettleq_ms=statistics.median(row["mettleq_ms"] for row in rows),
    check="per-width statevector atol=3e-6 and policy-selected GPU",
    passed=passed,
    exact_match=all(row["error"] == 0.0 for row in rows),
    selected_method=rows[-1]["method"],
    selected_device=rows[-1]["device"],
    metrics={"widths": rows},
    notes="The aggregate medians summarize different widths; use the per-width rows for timing interpretation.",
)
""",
        ),
        _qiskit(
            "16_peaked_circuit_smoke.ipynb",
            "Peaked-circuit MPS smoke test",
            "Run the deterministic mirrored peaked family through Qiskit Aer MPS and MettleQ routed MPS with the same shots and known mode.",
            """
from qiskit_aer import AerSimulator
from mettleq.peaked import build_mirrored_peaked_circuit

circuit = build_mirrored_peaked_circuit(8, depth=2, topology="long_range", seed=153, measure=True)
peak = circuit.metadata["peak_bitstring"]
expected_probability = circuit.metadata["expected_peak_probability"]
shots = 4096
aer = AerSimulator(method="matrix_product_state")
aer_circuit = transpile(circuit, aer, optimization_level=1)

def run_reference():
    return aer.run(aer_circuit, shots=shots, seed_simulator=153).result().get_counts()

reference, reference_ms, _ = benchmark(run_reference)
backend = MettleQBackend(
    method="matrix_product_state",
    device="cpu",
    mps_max_bond_dimension=32,
    mps_truncation_threshold=1e-12,
)
compiled = transpile(circuit, backend, optimization_level=1)

def run_mettleq():
    return backend.run(compiled, shots=shots, seed_simulator=153).result().get_counts()

candidate, mettleq_ms, _ = benchmark(run_mettleq)
tvd = total_variation_distance(reference, candidate)
observed = candidate.get(peak, 0) / shots
mode = max(candidate, key=candidate.get)
method, device = qiskit_selection(backend)
tutorial_result = emit_result(
    notebook="qiskit/16_peaked_circuit_smoke.ipynb",
    framework="qiskit",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="expected mode, peak probability atol=0.025, and TVD<=0.06",
    passed=mode == peak and abs(observed - expected_probability) <= 0.025 and tvd <= 0.06,
    exact_match=reference == candidate,
    selected_method=method,
    selected_device=device,
    metrics={"tvd": tvd, "expected_peak": peak, "mettleq_mode": mode, "expected_probability": expected_probability, "observed_probability": observed},
    notes="This mirrored regression is not the full 56-qubit P9 quantum-advantage instance.",
)
""",
        ),
    ]


def pennylane_specs() -> list[NotebookSpec]:
    return [
        _pennylane(
            "01_qnodes_and_measurements.ipynb",
            "PennyLane QNodes and analytic measurements",
            "Bind one quantum function to default.qubit and MettleQ, then compare state, probabilities, expectation, and variance.",
            """
def make_qnode(device):
    @qml.qnode(device)
    def circuit(theta):
        qml.Hadamard(0)
        qml.CNOT(wires=[0, 1])
        qml.Rot(theta, -0.21, 0.13, wires=1)
        return qml.state(), qml.probs(wires=[1, 0]), qml.expval(qml.X(0) @ qml.Z(1)), qml.var(qml.Z(0))
    return circuit

reference_device = qml.device("default.qubit", wires=2)
reference_qnode = make_qnode(reference_device)
reference, reference_ms, _ = benchmark(lambda: reference_qnode(0.31))
mettleq_device = MettleQDevice(wires=2, method="statevector", device="cpu")
mettleq_qnode = make_qnode(mettleq_device)
candidate, mettleq_ms, _ = benchmark(lambda: mettleq_qnode(0.31))
errors = [max_abs_error(left, right) for left, right in zip(reference, candidate)]
method, device = pennylane_selection(mettleq_device)
tutorial_result = emit_result(
    notebook="pennylane/01_qnodes_and_measurements.ipynb",
    framework="pennylane",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="all analytic measurements atol=2e-6",
    passed=max(errors) <= 2e-6,
    exact_match=all(np.array_equal(np.asarray(left), np.asarray(right)) for left, right in zip(reference, candidate)),
    selected_method=method,
    selected_device=device,
    metrics={"per_measurement_max_errors": errors},
)
""",
        ),
        _pennylane(
            "02_finite_shots.ipynb",
            "PennyLane finite shots",
            "Compare finite-shot Bell counts from default.qubit and MettleQ while keeping exact equality separate from statistical agreement.",
            """
shots = 4096
def make_counts(device):
    @qml.qnode(device)
    def circuit():
        qml.Hadamard(0)
        qml.CNOT(wires=[0, 1])
        return qml.counts(wires=[0, 1])
    return circuit

reference_device = qml.device("default.qubit", wires=2, shots=shots, seed=27)
reference_qnode = make_counts(reference_device)
reference, reference_ms, _ = benchmark(reference_qnode)
mettleq_device = MettleQDevice(wires=2, shots=shots, seed=27, method="statevector", device="cpu")
mettleq_qnode = make_counts(mettleq_device)
candidate, mettleq_ms, _ = benchmark(mettleq_qnode)
tvd = total_variation_distance(reference, candidate)
support_ok = set(reference) <= {"00", "11"} and set(candidate) <= {"00", "11"}
method, device = pennylane_selection(mettleq_device)
tutorial_result = emit_result(
    notebook="pennylane/02_finite_shots.ipynb",
    framework="pennylane",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="finite-shot total-variation distance <= 0.05",
    passed=support_ok and tvd <= 0.05,
    exact_match=reference == candidate,
    selected_method=method,
    selected_device=device,
    metrics={"tvd": tvd, "reference_counts": reference, "mettleq_counts": candidate},
    notes="Independent device RNG implementations need not return identical count dictionaries.",
)
""",
        ),
        _pennylane(
            "03_parameter_shift_gradients.ipynb",
            "Parameter-shift gradients",
            "Differentiate the same QNode with the hardware-compatible parameter-shift rule on both devices.",
            """
def make_qnode(device):
    @qml.qnode(device, diff_method="parameter-shift")
    def circuit(theta):
        qml.RX(theta, wires=0)
        qml.RY(-0.23, wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Z(1))
    return circuit

theta = pnp.array(0.41, requires_grad=True)
reference_qnode = make_qnode(qml.device("default.qubit", wires=2))
def reference_value_gradient():
    return float(reference_qnode(theta)), float(qml.grad(reference_qnode)(theta))
reference, reference_ms, _ = benchmark(reference_value_gradient)
mettleq_device = MettleQDevice(wires=2, method="statevector", device="cpu")
mettleq_qnode = make_qnode(mettleq_device)
def mettleq_value_gradient():
    return float(mettleq_qnode(theta)), float(qml.grad(mettleq_qnode)(theta))
candidate, mettleq_ms, _ = benchmark(mettleq_value_gradient)
error = max_abs_error(reference, candidate)
method, device = pennylane_selection(mettleq_device)
tutorial_result = emit_result(
    notebook="pennylane/03_parameter_shift_gradients.ipynb",
    framework="pennylane",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="value and parameter-shift gradient atol=3e-6",
    passed=error <= 3e-6,
    exact_match=reference == candidate,
    selected_method=method,
    selected_device=device,
    metrics={"max_value_or_gradient_error": error, "reference": reference, "mettleq": candidate},
)
""",
        ),
        _pennylane(
            "04_variational_optimization.ipynb",
            "Variational optimization loop",
            "Run the same fixed-step gradient-descent loop with a reference QNode and the MettleQ device.",
            """
def make_qnode(device):
    @qml.qnode(device, diff_method="parameter-shift")
    def circuit(weights):
        qml.RY(weights[0], wires=0)
        qml.RX(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[2], wires=1)
        return qml.expval(qml.Z(0) @ qml.Z(1))
    return circuit

def train(qnode):
    weights = pnp.array([0.2, -0.4, 0.7], requires_grad=True)
    trace = []
    for _ in range(8):
        value = qnode(weights)
        trace.append(float(value))
        weights = weights - 0.15 * qml.grad(qnode)(weights)
    trace.append(float(qnode(weights)))
    return np.asarray(trace)

reference_qnode = make_qnode(qml.device("default.qubit", wires=2))
reference, reference_ms, _ = benchmark(lambda: train(reference_qnode), repeats=2)
mettleq_device = MettleQDevice(wires=2, method="statevector", device="cpu")
mettleq_qnode = make_qnode(mettleq_device)
candidate, mettleq_ms, _ = benchmark(lambda: train(mettleq_qnode), repeats=2)
error = max_abs_error(reference, candidate)
method, device = pennylane_selection(mettleq_device)
tutorial_result = emit_result(
    notebook="pennylane/04_variational_optimization.ipynb",
    framework="pennylane",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="optimization trace atol=4e-5",
    passed=error <= 4e-5 and candidate[-1] <= candidate[0],
    exact_match=bool(np.array_equal(reference, candidate)),
    selected_method=method,
    selected_device=device,
    metrics={"max_trace_error": error, "reference_trace": reference, "mettleq_trace": candidate},
)
""",
        ),
        _pennylane(
            "05_vqe.ipynb",
            "PennyLane VQE",
            "Minimize a two-qubit Hamiltonian with parameter-shift gradients and compare the complete energy trace.",
            """
hamiltonian = -1.05 * qml.I(0) + 0.39 * qml.Z(0) - 0.39 * qml.Z(1) - 0.01 * (qml.Z(0) @ qml.Z(1)) + 0.18 * (qml.X(0) @ qml.X(1))

def make_energy(device):
    @qml.qnode(device, diff_method="parameter-shift")
    def energy(weights):
        qml.RY(weights[0], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[1], wires=1)
        return qml.expval(hamiltonian)
    return energy

def optimize(energy):
    weights = pnp.array([0.2, -0.3], requires_grad=True)
    trace = []
    for _ in range(10):
        trace.append(float(energy(weights)))
        weights = weights - 0.12 * qml.grad(energy)(weights)
    trace.append(float(energy(weights)))
    return np.asarray(trace)

reference_energy = make_energy(qml.device("default.qubit", wires=2))
reference, reference_ms, _ = benchmark(lambda: optimize(reference_energy), repeats=2)
mettleq_device = MettleQDevice(wires=2, method="statevector", device="cpu")
mettleq_energy = make_energy(mettleq_device)
candidate, mettleq_ms, _ = benchmark(lambda: optimize(mettleq_energy), repeats=2)
error = max_abs_error(reference, candidate)
method, device = pennylane_selection(mettleq_device)
tutorial_result = emit_result(
    notebook="pennylane/05_vqe.ipynb",
    framework="pennylane",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="VQE energy trace atol=5e-5",
    passed=error <= 5e-5 and candidate[-1] < candidate[0],
    exact_match=bool(np.array_equal(reference, candidate)),
    selected_method=method,
    selected_device=device,
    metrics={"max_energy_error": error, "reference_final": reference[-1], "mettleq_final": candidate[-1]},
)
""",
        ),
        _pennylane(
            "06_qaoa_maxcut.ipynb",
            "PennyLane QAOA for MaxCut",
            "Evaluate a p=1 triangle-graph QAOA landscape with the same Hamiltonian on both devices.",
            """
cost = 1.5 * qml.I(0) - 0.5 * (qml.Z(0) @ qml.Z(1)) - 0.5 * (qml.Z(1) @ qml.Z(2)) - 0.5 * (qml.Z(0) @ qml.Z(2))
parameters = [(gamma, beta) for gamma in np.linspace(0.0, np.pi, 7) for beta in np.linspace(0.0, np.pi / 2, 5)]

def make_qnode(device):
    @qml.qnode(device)
    def circuit(gamma, beta):
        for wire in range(3):
            qml.Hadamard(wire)
        for wires in ((0, 1), (1, 2), (0, 2)):
            qml.IsingZZ(-gamma, wires=wires)
        for wire in range(3):
            qml.RX(2 * beta, wires=wire)
        return qml.expval(cost)
    return circuit

reference_qnode = make_qnode(qml.device("default.qubit", wires=3))
reference, reference_ms, _ = benchmark(lambda: np.asarray([reference_qnode(*values) for values in parameters]))
mettleq_device = MettleQDevice(wires=3, method="statevector", device="cpu")
mettleq_qnode = make_qnode(mettleq_device)
candidate, mettleq_ms, _ = benchmark(lambda: np.asarray([mettleq_qnode(*values) for values in parameters]))
error = max_abs_error(reference, candidate)
best_match = int(np.argmax(reference)) == int(np.argmax(candidate))
method, device = pennylane_selection(mettleq_device)
tutorial_result = emit_result(
    notebook="pennylane/06_qaoa_maxcut.ipynb",
    framework="pennylane",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="QAOA landscape atol=4e-6",
    passed=error <= 4e-6 and best_match,
    exact_match=best_match,
    selected_method=method,
    selected_device=device,
    metrics={"max_cost_error": error, "best_parameters": parameters[int(np.argmax(candidate))], "best_cost": candidate.max()},
)
""",
        ),
        _pennylane(
            "07_qft_and_qpe.ipynb",
            "PennyLane QFT and phase estimation",
            "Estimate a representable phase with three counting wires and compare the probability distribution.",
            """
counting = 3
phase = 3 / 8

def make_qnode(device):
    @qml.qnode(device)
    def circuit():
        qml.PauliX(counting)
        for wire in range(counting):
            qml.Hadamard(wire)
            qml.ControlledPhaseShift(2 * np.pi * phase * (2 ** wire), wires=[wire, counting])
        qml.adjoint(qml.QFT)(wires=range(counting))
        return qml.probs(wires=range(counting))
    return circuit

reference_qnode = make_qnode(qml.device("default.qubit", wires=counting + 1))
reference, reference_ms, _ = benchmark(reference_qnode)
mettleq_device = MettleQDevice(wires=counting + 1, method="statevector", device="cpu")
mettleq_qnode = make_qnode(mettleq_device)
candidate, mettleq_ms, _ = benchmark(mettleq_qnode)
error = max_abs_error(reference, candidate)
mode_match = int(np.argmax(reference)) == int(np.argmax(candidate))
method, device = pennylane_selection(mettleq_device)
tutorial_result = emit_result(
    notebook="pennylane/07_qft_and_qpe.ipynb",
    framework="pennylane",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="phase probabilities atol=3e-6 and identical mode",
    passed=error <= 3e-6 and mode_match,
    exact_match=mode_match,
    selected_method=method,
    selected_device=device,
    metrics={"max_probability_error": error, "expected_phase": phase, "mode": int(np.argmax(candidate))},
)
""",
        ),
        _pennylane(
            "08_variational_classifier.ipynb",
            "Variational classifier",
            "Train a tiny parity classifier with parameter-shift gradients and compare predictions and loss.",
            """
features = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
labels = np.asarray([1.0, -1.0, -1.0, 1.0])

def make_model(device):
    @qml.qnode(device, diff_method="parameter-shift")
    def circuit(x, weights):
        qml.RX(np.pi * x[0], wires=0)
        qml.RX(np.pi * x[1], wires=1)
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[2], wires=1)
        return qml.expval(qml.Z(1))
    return circuit

def train(model):
    weights = pnp.array([0.2, -0.1, 0.3], requires_grad=True)
    def loss(current):
        predictions = pnp.stack([model(row, current) for row in features])
        return pnp.mean((predictions - labels) ** 2)
    trace = []
    for _ in range(6):
        trace.append(float(loss(weights)))
        weights = weights - 0.18 * qml.grad(loss)(weights)
    predictions = np.asarray([model(row, weights) for row in features], dtype=float)
    return np.asarray(trace), predictions

reference_model = make_model(qml.device("default.qubit", wires=2))
reference, reference_ms, _ = benchmark(lambda: train(reference_model), repeats=2)
mettleq_device = MettleQDevice(wires=2, method="statevector", device="cpu")
mettleq_model = make_model(mettleq_device)
candidate, mettleq_ms, _ = benchmark(lambda: train(mettleq_model), repeats=2)
trace_error = max_abs_error(reference[0], candidate[0])
prediction_error = max_abs_error(reference[1], candidate[1])
method, device = pennylane_selection(mettleq_device)
tutorial_result = emit_result(
    notebook="pennylane/08_variational_classifier.ipynb",
    framework="pennylane",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="training trace and predictions atol=8e-5",
    passed=max(trace_error, prediction_error) <= 8e-5,
    exact_match=bool(np.array_equal(reference[1], candidate[1])),
    selected_method=method,
    selected_device=device,
    metrics={"trace_error": trace_error, "prediction_error": prediction_error, "predictions": candidate[1]},
)
""",
        ),
        _pennylane(
            "09_quantum_kernel.ipynb",
            "PennyLane fidelity quantum kernel",
            "Construct a feature-map kernel from QNode state overlaps on default.qubit and MettleQ.",
            """
data = np.asarray([[0.1, 0.2], [0.7, -0.4], [-0.5, 0.6], [0.2, 0.9]])

def make_state_qnode(device):
    @qml.qnode(device)
    def circuit(values):
        for wire in range(2):
            qml.Hadamard(wire)
            qml.RZ(values[wire], wires=wire)
        qml.IsingZZ(values[0] * values[1], wires=[0, 1])
        return qml.state()
    return circuit

def kernel(qnode):
    states = [np.asarray(qnode(row)) for row in data]
    return np.asarray([[abs(np.vdot(left, right)) ** 2 for right in states] for left in states])

reference_qnode = make_state_qnode(qml.device("default.qubit", wires=2))
reference, reference_ms, _ = benchmark(lambda: kernel(reference_qnode))
mettleq_device = MettleQDevice(wires=2, method="statevector", device="cpu")
mettleq_qnode = make_state_qnode(mettleq_device)
candidate, mettleq_ms, _ = benchmark(lambda: kernel(mettleq_qnode))
error = max_abs_error(reference, candidate)
method, device = pennylane_selection(mettleq_device)
tutorial_result = emit_result(
    notebook="pennylane/09_quantum_kernel.ipynb",
    framework="pennylane",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="kernel matrix atol=4e-6",
    passed=error <= 4e-6,
    exact_match=bool(np.array_equal(reference, candidate)),
    selected_method=method,
    selected_device=device,
    metrics={"max_kernel_error": error, "reference": reference, "mettleq": candidate},
)
""",
        ),
        _pennylane(
            "10_teleportation_deferred.ipynb",
            "Coherent deferred-measurement teleportation",
            "Use controlled corrections instead of native mid-circuit control and compare receiver observables.",
            """
def make_qnode(device):
    @qml.qnode(device)
    def teleport(theta):
        qml.RY(theta, wires=0)
        qml.Hadamard(1)
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[0, 1])
        qml.Hadamard(0)
        qml.CNOT(wires=[1, 2])
        qml.CZ(wires=[0, 2])
        return qml.expval(qml.X(2)), qml.expval(qml.Z(2))
    return teleport

angles = np.linspace(-1.1, 1.1, 9)
reference_qnode = make_qnode(qml.device("default.qubit", wires=3))
reference, reference_ms, _ = benchmark(lambda: np.asarray([reference_qnode(value) for value in angles]))
mettleq_device = MettleQDevice(wires=3, method="statevector", device="cpu")
mettleq_qnode = make_qnode(mettleq_device)
candidate, mettleq_ms, _ = benchmark(lambda: np.asarray([mettleq_qnode(value) for value in angles]))
error = max_abs_error(reference, candidate)
analytic = np.column_stack([np.sin(angles), np.cos(angles)])
analytic_error = max_abs_error(analytic, candidate)
method, device = pennylane_selection(mettleq_device)
tutorial_result = emit_result(
    notebook="pennylane/10_teleportation_deferred.ipynb",
    framework="pennylane",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="receiver observables atol=3e-6",
    passed=max(error, analytic_error) <= 3e-6,
    exact_match=bool(np.array_equal(reference, candidate)),
    selected_method=method,
    selected_device=device,
    metrics={"reference_error": error, "analytic_receiver_error": analytic_error},
    notes="MettleQ does not claim native mid-circuit control; this notebook applies the deferred coherent circuit explicitly.",
)
""",
        ),
        _pennylane(
            "11_hamiltonian_simulation.ipynb",
            "PennyLane Hamiltonian simulation",
            "Trotterize transverse-field Ising dynamics and compare an expectation-value trajectory.",
            """
times = np.linspace(0.0, 1.2, 13)

def make_qnode(device):
    @qml.qnode(device)
    def evolution(time_value):
        qml.PauliX(0)
        steps = 6
        dt = time_value / steps
        for _ in range(steps):
            qml.IsingZZ(1.1 * dt, wires=[0, 1])
            qml.IsingZZ(1.1 * dt, wires=[1, 2])
            for wire in range(3):
                qml.RX(0.7 * dt, wires=wire)
        return qml.expval(qml.Z(0))
    return evolution

reference_qnode = make_qnode(qml.device("default.qubit", wires=3))
reference, reference_ms, _ = benchmark(lambda: np.asarray([reference_qnode(value) for value in times]))
mettleq_device = MettleQDevice(wires=3, method="statevector", device="cpu")
mettleq_qnode = make_qnode(mettleq_device)
candidate, mettleq_ms, _ = benchmark(lambda: np.asarray([mettleq_qnode(value) for value in times]))
error = max_abs_error(reference, candidate)
method, device = pennylane_selection(mettleq_device)
tutorial_result = emit_result(
    notebook="pennylane/11_hamiltonian_simulation.ipynb",
    framework="pennylane",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="observable trajectory atol=3e-6",
    passed=error <= 3e-6,
    exact_match=bool(np.array_equal(reference, candidate)),
    selected_method=method,
    selected_device=device,
    metrics={"max_observable_error": error, "times": times, "reference": reference, "mettleq": candidate},
)
""",
        ),
        _pennylane(
            "12_mps_convergence.ipynb",
            "PennyLane MPS convergence evidence",
            "Compare a long-range 10-wire QNode with default.qubit and inspect MettleQ Dmax convergence metadata.",
            """
rng = np.random.default_rng(52)
angles = rng.uniform(-0.8, 0.8, size=(3, 10))
pairs = [[tuple(map(int, pair)) for pair in rng.permutation(10).reshape(5, 2)] for _ in range(3)]

def make_qnode(device):
    @qml.qnode(device)
    def circuit():
        for layer in range(3):
            for wire in range(10):
                qml.RY(angles[layer, wire], wires=wire)
            for first, second in pairs[layer]:
                qml.IsingZZ(0.43, wires=[first, second])
        return qml.state(), qml.expval(qml.Z(0))
    return circuit

reference_qnode = make_qnode(qml.device("default.qubit", wires=10))
reference, reference_ms, _ = benchmark(reference_qnode)
mettleq_device = MettleQDevice(
    wires=10,
    method="matrix_product_state",
    device="cpu",
    mps_max_bond_dimension=32,
    mps_truncation_threshold=1e-12,
    mps_convergence_bond_dimensions=(8, 16, 32),
    mps_convergence_atol=5e-5,
)
mettleq_qnode = make_qnode(mettleq_device)
candidate, mettleq_ms, _ = benchmark(mettleq_qnode)
state_error = phase_aligned_statevector_error(reference[0], candidate[0])
expectation_error = abs(float(reference[1]) - float(candidate[1]))
convergence = mettleq_device.last_mps_convergence_report
accuracy = mettleq_device.last_mps_accuracy_report
method, device = pennylane_selection(mettleq_device)
tutorial_result = emit_result(
    notebook="pennylane/12_mps_convergence.ipynb",
    framework="pennylane",
    reference_ms=reference_ms,
    mettleq_ms=mettleq_ms,
    check="MPS state atol=8e-5 and automated Dmax convergence",
    passed=state_error <= 8e-5 and expectation_error <= 5e-5 and convergence["converged"] and accuracy["passed"],
    exact_match=bool(np.array_equal(reference[0], candidate[0])),
    selected_method=method,
    selected_device=device,
    metrics={"max_amplitude_error": state_error, "expectation_error": expectation_error, "accuracy": accuracy, "convergence": convergence},
)
""",
        ),
        _pennylane(
            "13_apple_gpu_scaling.ipynb",
            "PennyLane Apple Silicon scaling",
            "Sweep statevector widths around MettleQ's automatic GPU crossover and verify complete-state parity.",
            """
import statistics

def make_qnode(device, width):
    @qml.qnode(device)
    def circuit():
        for wire in range(width):
            qml.RY(0.03 * (wire + 1), wires=wire)
        for wire in range(width - 1):
            qml.CNOT(wires=[wire, wire + 1])
        return qml.state()
    return circuit

rows = []
for width in (12, 14, 16):
    reference_qnode = make_qnode(qml.device("default.qubit", wires=width), width)
    reference, reference_ms, _ = benchmark(reference_qnode, repeats=2)
    mettleq_device = MettleQDevice(wires=width, method="statevector", device="auto")
    mettleq_qnode = make_qnode(mettleq_device, width)
    candidate, mettleq_ms, _ = benchmark(mettleq_qnode, repeats=2)
    method, device = pennylane_selection(mettleq_device)
    rows.append({
        "width": width,
        "reference_ms": reference_ms,
        "mettleq_ms": mettleq_ms,
        "error": phase_aligned_statevector_error(reference, candidate),
        "method": method,
        "device": device,
    })

tutorial_result = emit_result(
    notebook="pennylane/13_apple_gpu_scaling.ipynb",
    framework="pennylane",
    reference_ms=statistics.median(row["reference_ms"] for row in rows),
    mettleq_ms=statistics.median(row["mettleq_ms"] for row in rows),
    check="per-width statevector atol=3e-6 and policy-selected GPU",
    passed=all(row["error"] <= 3e-6 for row in rows) and rows[-1]["device"] == "gpu",
    exact_match=all(row["error"] == 0.0 for row in rows),
    selected_method=rows[-1]["method"],
    selected_device=rows[-1]["device"],
    metrics={"widths": rows},
    notes="The aggregate medians summarize different widths; use per-width timings for interpretation.",
)
""",
        ),
    ]


def build_notebook(spec: NotebookSpec):
    setup = QISKIT_SETUP if spec.framework == "qiskit" else PENNYLANE_SETUP
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3 (MettleQ)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "mettleq": {"generated": True, "framework": spec.framework},
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            f"# {spec.title}\n\n{spec.summary}\n\n"
            "The SDK reference and MettleQ calls below use the same circuit and "
            "result contract. Timing includes the complete call shown."
        ),
        nbformat.v4.new_code_cell(textwrap.dedent(setup).strip()),
        nbformat.v4.new_code_cell(textwrap.dedent(spec.source).strip()),
    ]
    stem = Path(spec.filename).stem.replace("_", "-")
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"{spec.framework}-{stem}-{index}"
    return notebook


def _matches_generated(existing_text: str, generated_notebook) -> bool:
    try:
        existing = nbformat.reads(existing_text, as_version=4)
    except Exception:
        return False
    if existing.metadata.get("mettleq") != generated_notebook.metadata.get("mettleq"):
        return False
    if len(existing.cells) != len(generated_notebook.cells):
        return False
    return all(
        existing_cell.get("cell_type") == generated_cell.get("cell_type")
        and existing_cell.get("source") == generated_cell.get("source")
        and existing_cell.get("id") == generated_cell.get("id")
        for existing_cell, generated_cell in zip(
            existing.cells, generated_notebook.cells
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    specs = qiskit_specs() + pennylane_specs()
    changed = []
    for spec in specs:
        path = TUTORIALS / spec.framework / spec.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        generated_notebook = build_notebook(spec)
        generated = nbformat.writes(generated_notebook)
        existing = path.read_text() if path.exists() else None
        needs_change = (
            existing is None
            or not _matches_generated(existing, generated_notebook)
            if args.check
            else existing != generated
        )
        if needs_change:
            changed.append(str(path.relative_to(ROOT)))
            if not args.check:
                path.write_text(generated)
    if args.check and changed:
        raise SystemExit("tutorial notebooks need rebuilding: " + ", ".join(changed))
    print(f"built {len(specs)} notebooks; changed {len(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
