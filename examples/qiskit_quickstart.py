"""Run a Bell-state sampling job through MettleQ's adaptive Qiskit backend."""

from qiskit import QuantumCircuit, transpile

from mettleq.integrations.qiskit import AdaptiveQiskitBackend


backend = AdaptiveQiskitBackend(precision="single")
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

compiled = transpile(circuit, backend)
result = backend.run(compiled, shots=1_000, seed_simulator=7).result()
counts = result.get_counts()

assert set(counts) <= {"00", "11"}, counts
print("counts:", counts)
decision = backend.last_adaptive_decisions[-1]
print("selected engine:", decision["engine"])
print("selection reason:", decision["reason"])
