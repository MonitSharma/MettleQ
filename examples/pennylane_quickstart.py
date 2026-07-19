"""Evaluate a Bell-state observable through MettleQ's adaptive PennyLane device."""

import pennylane as qml


device = qml.device("mettleq.adaptive", wires=2, precision="single")


@qml.qnode(device)
def bell_circuit():
    qml.Hadamard(0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.Z(0) @ qml.Z(1)), qml.probs(wires=[0, 1])


correlation, probabilities = bell_circuit()
assert abs(float(correlation) - 1.0) < 1e-6, correlation
print("<Z0 Z1>:", correlation)
print("probabilities:", probabilities)
