"""Shared execution and measurement helpers for native SDK adapters."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional, Sequence

import mlx.core as mx
import numpy as np

from ..device import Device


_ARITIES = {
    "H": 1,
    "I": 1,
    "X": 1,
    "Y": 1,
    "Z": 1,
    "S": 1,
    "SDG": 1,
    "T": 1,
    "TDG": 1,
    "SX": 1,
    "RX": 1,
    "RY": 1,
    "RZ": 1,
    "U1": 1,
    "U2": 1,
    "U3": 1,
    "CNOT": 2,
    "CH": 2,
    "CZ": 2,
    "CPHASE": 2,
    "CRX": 2,
    "CRY": 2,
    "CRZ": 2,
    "SWAP": 2,
    "ISWAP": 2,
    "XXPHASE": 2,
    "YYPHASE": 2,
    "ZZPHASE": 2,
    "TOFFOLI": 3,
    "FREDKIN": 3,
}

_PARAMETER_COUNTS = {
    "RX": 1,
    "RY": 1,
    "RZ": 1,
    "U1": 1,
    "U2": 2,
    "U3": 3,
    "CPHASE": 1,
    "CRX": 1,
    "CRY": 1,
    "CRZ": 1,
    "XXPHASE": 1,
    "YYPHASE": 1,
    "ZZPHASE": 1,
}

_ALIASES = {
    "CX": "CNOT",
    "CP": "CPHASE",
    "CCX": "TOFFOLI",
    "CSWAP": "FREDKIN",
}


def core_operation(
    name: str,
    wires: Iterable[int],
    parameters: Sequence[float] = (),
) -> dict:
    """Build and validate one operation in Qupertino's canonical format."""
    canonical_name = _ALIASES.get(str(name).upper(), str(name).upper())
    canonical_wires = [int(wire) for wire in wires]
    canonical_parameters = [float(value) for value in parameters]
    if canonical_name == "I":
        return {"name": "I", "wires": canonical_wires, "parameters": []}
    if canonical_name not in _ARITIES:
        raise ValueError(f"Unsupported operation: {name}")
    expected_arity = _ARITIES[canonical_name]
    if len(canonical_wires) != expected_arity:
        raise ValueError(
            f"{canonical_name} expects {expected_arity} wire(s), got "
            f"{len(canonical_wires)}"
        )
    expected_parameters = _PARAMETER_COUNTS.get(canonical_name, 0)
    if len(canonical_parameters) != expected_parameters:
        raise ValueError(
            f"{canonical_name} expects {expected_parameters} parameter(s), got "
            f"{len(canonical_parameters)}"
        )
    return {
        "name": canonical_name,
        "wires": canonical_wires,
        "parameters": canonical_parameters,
    }


def execute_operations(
    n_qubits: int,
    operations: Sequence[dict],
    *,
    shots: int = 1000,
    report: bool = False,
    metal_checkpoint_budget_bytes: Optional[int] = None,
    allow_unsafe_statevector: Optional[bool] = None,
) -> Device:
    """Execute canonical operations through the validated core device."""
    device = Device(
        n_qubits,
        shots=shots,
        metal_checkpoint_budget_bytes=metal_checkpoint_budget_bytes,
        allow_unsafe_statevector=allow_unsafe_statevector,
    )
    device.execute(list(operations), report=report)
    return device


def apply_global_phase(device: Device, phase: float) -> None:
    """Apply an SDK circuit's global phase without a host state readback."""
    if phase:
        device.sim.state = device.sim.state * complex(
            np.cos(float(phase)), np.sin(float(phase))
        )


def statevector_numpy(device: Device) -> np.ndarray:
    """Synchronize and materialize the state only when an SDK requests it."""
    state = device.synchronize()
    return np.asarray(state, dtype=np.complex64)


def marginal_probabilities(device: Device, wires: Sequence[int]) -> np.ndarray:
    """Read back only the requested marginal probability vector."""
    device.synchronize()
    probabilities = device.sim.probabilities_array(list(wires))
    mx.eval(probabilities)
    result = np.asarray(probabilities, dtype=np.float64)
    total = float(result.sum())
    if total <= 0.0:
        raise RuntimeError("Simulator returned a zero-probability state")
    return result / total


def sample_bits(
    device: Device,
    shots: int,
    wires: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample requested wires in MSB-first order from their GPU marginal."""
    wires = list(wires)
    if not wires:
        return np.empty((int(shots), 0), dtype=np.int64)
    probabilities = marginal_probabilities(device, wires)
    indices = rng.choice(len(probabilities), size=int(shots), p=probabilities)
    shifts = np.arange(len(wires) - 1, -1, -1, dtype=np.int64)
    return ((indices[:, None] >> shifts) & 1).astype(np.int64, copy=False)


def bit_counts(samples: np.ndarray) -> dict[str, int]:
    """Convert an MSB-first bit sample matrix to string-keyed counts."""
    return dict(Counter("".join(str(int(bit)) for bit in row) for row in samples))


def _apply_local_matrix(
    state: mx.array,
    n_qubits: int,
    wires: Sequence[int],
    matrix: np.ndarray,
) -> mx.array:
    """Apply a small local matrix functionally, without mutating the device."""
    wires = [int(wire) for wire in wires]
    if not wires:
        scalar = complex(np.asarray(matrix).reshape(-1)[0])
        return state * scalar
    if len(set(wires)) != len(wires):
        raise ValueError("Observable wires must be unique")
    if any(wire < 0 or wire >= n_qubits for wire in wires):
        raise ValueError("Observable wire is outside the device")
    dimension = 1 << len(wires)
    local_matrix = np.asarray(matrix, dtype=np.complex64)
    if local_matrix.shape != (dimension, dimension):
        raise ValueError(
            f"Observable matrix must have shape {(dimension, dimension)}, "
            f"got {local_matrix.shape}"
        )

    tensor = mx.reshape(state, [2] * n_qubits)
    remaining = [wire for wire in range(n_qubits) if wire not in wires]
    permutation = remaining + wires
    inverse = [0] * n_qubits
    for index, axis in enumerate(permutation):
        inverse[axis] = index
    if n_qubits > 1 and permutation != list(range(n_qubits)):
        tensor = mx.transpose(tensor, permutation)
    outer_dimension = 1 << (n_qubits - len(wires))
    flattened = mx.reshape(tensor, (outer_dimension, dimension))
    updated = mx.matmul(flattened, mx.transpose(mx.array(local_matrix)))
    updated = mx.reshape(updated, [2] * n_qubits)
    if n_qubits > 1 and inverse != list(range(n_qubits)):
        updated = mx.transpose(updated, inverse)
    return mx.reshape(updated, (1 << n_qubits,))


def local_expectation(
    device: Device,
    wires: Sequence[int],
    matrix: np.ndarray,
) -> complex:
    """Evaluate a local observable while keeping the full state on-device."""
    device.synchronize()
    acted = _apply_local_matrix(device.sim.state, device.wires, wires, matrix)
    value = mx.sum(mx.conj(device.sim.state) * acted)
    mx.eval(value)
    return complex(value.item())


def observable_samples(
    device: Device,
    wires: Sequence[int],
    matrix: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an arbitrary small Hermitian observable from its eigenbasis."""
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.asarray(matrix, dtype=np.complex128)
    )
    rotated = _apply_local_matrix(
        device.sim.state,
        device.wires,
        wires,
        eigenvectors.conj().T,
    )
    probabilities = mx.abs(rotated) ** 2
    tensor = mx.reshape(probabilities, [2] * device.wires)
    wires = list(wires)
    remaining = [wire for wire in range(device.wires) if wire not in wires]
    permutation = wires + remaining
    if device.wires > 1 and permutation != list(range(device.wires)):
        tensor = mx.transpose(tensor, permutation)
    probabilities = mx.sum(
        mx.reshape(tensor, (1 << len(wires), -1)), axis=1
    )
    mx.eval(probabilities)
    host_probabilities = np.asarray(probabilities, dtype=np.float64)
    host_probabilities = np.clip(host_probabilities, 0.0, None)
    host_probabilities /= host_probabilities.sum()
    indices = rng.choice(
        len(eigenvalues), size=int(shots), p=host_probabilities
    )
    return np.real_if_close(eigenvalues[indices])
