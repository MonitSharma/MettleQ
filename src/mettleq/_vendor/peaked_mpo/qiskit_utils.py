"""Small Qiskit and tensor-network utilities used by the P9 solver."""

import os

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit


def iter_layers(qc):
    """Yield a circuit as Qiskit's DAG layers converted back to circuits."""
    for layer in circuit_to_dag(qc).layers():
        yield dag_to_circuit(layer["graph"])


def merge_layers(layers, barrier=False):
    """Compose a list of layer circuits into one circuit."""
    layers = iter(layers)
    first = next(layers)
    qc = QuantumCircuit(first.num_qubits, first.num_clbits)
    qc.compose(first, inplace=True)
    for layer in layers:
        if barrier:
            qc.barrier()
        qc.compose(layer, inplace=True)
    return qc


def merge_gates(gates, num_qubits=None):
    """Build a circuit from a Qiskit instruction slice."""
    if num_qubits is None:
        num_qubits = gates[0].qubits[0]._register.size
    qc = QuantumCircuit(num_qubits, num_qubits)
    for gate in gates:
        qc.append(
            gate.operation,
            qargs=[qubit._index for qubit in gate.qubits],
            cargs=[clbit._index for clbit in gate.clbits],
        )
    return qc


def elem_counts(tensor_network):
    """Total scalar entries stored across all tensors."""
    return sum(np.prod(tensor.shape).item() for tensor in tensor_network)


def get_tn_info(tensor_network):
    """Compact shape summary for logging and run metadata."""
    shapes_flat = [dim for tensor in tensor_network for dim in tensor.shape]
    link_counts = [len(tensor.shape) for tensor in tensor_network]
    tensor_sizes = [np.prod(tensor.shape).item() for tensor in tensor_network]
    info = {
        "max_bond": tensor_network.max_bond(),
        "max_links": max(link_counts),
        "total_elems": sum(tensor_sizes),
        "total_shapes": np.sum(shapes_flat).item(),
        "total_links": sum(link_counts),
        "num_tensors": tensor_network.num_tensors,
    }

    if os.environ.get("P9SOLVER_BOND_SIZES") == "1":
        sites = list(getattr(tensor_network, "sites", ()))
        bond_sizes = []
        for left, right in zip(sites, sites[1:]):
            try:
                bond_sizes.append(int(tensor_network.bond_size(left, right)))
            except Exception:
                bond_sizes.append(0)
        info["bond_sizes"] = bond_sizes

    return info

