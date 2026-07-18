"""Shared execution and measurement helpers for native SDK adapters."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional, Sequence

import mlx.core as mx
import numpy as np

from ..device import Device
from ..execution import require_statevector_preflight
from ..mps_accuracy import assess_mps_accuracy, enforce_mps_accuracy
from ..mps_state import MPSOptions
from ..planning import select_execution
from ..planning import (
    DEFAULT_AUTOMATIC_MPS_MIN_QUBITS,
    DEFAULT_MPS_GPU_MIN_QUBITS,
    DEFAULT_STATEVECTOR_GPU_MIN_QUBITS,
)


_PAULI_MATRICES = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex64),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex64),
}


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
    """Build and validate one operation in MettleQ's canonical format."""
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
    method: str = "automatic",
    execution_device: str = "auto",
    allow_approximation: bool = False,
    mps_max_bond_dimension: int = 64,
    mps_truncation_threshold: float = 1e-10,
    mps_svd_driver: str = "auto",
    mps_routing_strategy: str = "lookahead",
    mps_routing_lookahead: int = 8,
    mps_accuracy_policy: str = "report",
    mps_max_relative_discarded_weight: Optional[float] = 1e-6,
    mps_max_norm_error: Optional[float] = 1e-5,
    statevector_gpu_min_qubits: int = DEFAULT_STATEVECTOR_GPU_MIN_QUBITS,
    mps_gpu_min_qubits: Optional[int] = DEFAULT_MPS_GPU_MIN_QUBITS,
    automatic_mps_min_qubits: int = DEFAULT_AUTOMATIC_MPS_MIN_QUBITS,
    execution_cache: Optional[dict] = None,
) -> Device:
    """Execute canonical operations through the validated core device."""
    selection = select_execution(
        n_qubits,
        operations,
        method=method,
        device=execution_device,
        allow_approximation=allow_approximation,
        allow_unsafe_statevector=allow_unsafe_statevector,
        statevector_gpu_min_qubits=statevector_gpu_min_qubits,
        mps_gpu_min_qubits=mps_gpu_min_qubits,
        automatic_mps_min_qubits=automatic_mps_min_qubits,
    )
    backend = (
        "sv" if selection.selected_method == "statevector" else "mps"
    )
    mps_options = (
        MPSOptions(
            dmax=int(mps_max_bond_dimension),
            eps=float(mps_truncation_threshold),
            svd_driver=str(mps_svd_driver),
            routing_strategy=str(mps_routing_strategy),
            routing_lookahead=int(mps_routing_lookahead),
        )
        if backend == "mps"
        else None
    )
    cache_key = (
        int(n_qubits),
        backend,
        selection.selected_device,
        int(mps_max_bond_dimension) if backend == "mps" else None,
        float(mps_truncation_threshold) if backend == "mps" else None,
        str(mps_svd_driver) if backend == "mps" else None,
        str(mps_routing_strategy) if backend == "mps" else None,
        int(mps_routing_lookahead) if backend == "mps" else None,
    )
    device = (
        execution_cache.get(cache_key)
        if execution_cache is not None
        else None
    )
    if device is None:
        device = Device(
            n_qubits,
            shots=shots,
            backend=backend,
            mps_opts=mps_options,
            metal_checkpoint_budget_bytes=metal_checkpoint_budget_bytes,
            allow_unsafe_statevector=allow_unsafe_statevector,
            execution_device=selection.selected_device,
        )
        if execution_cache is not None:
            execution_cache[cache_key] = device
    else:
        device.shots = int(shots)
        device.reset()
    device.execute(list(operations), report=report)
    device.mps_accuracy_report = None
    if backend == "mps":
        diagnostics = device.sim.truncation_diagnostics()
        device.mps_accuracy_report = assess_mps_accuracy(
            diagnostics,
            policy=mps_accuracy_policy,
            max_relative_discarded_weight=(
                None
                if mps_max_relative_discarded_weight is None
                else float(mps_max_relative_discarded_weight)
            ),
            max_norm_error=(
                None
                if mps_max_norm_error is None
                else float(mps_max_norm_error)
            ),
        )
        enforce_mps_accuracy(device.mps_accuracy_report)
    device.execution_selection = selection.to_dict()
    if device.last_execution_plan is not None:
        device.last_execution_plan["sdk_execution_selection"] = (
            device.execution_selection
        )
        device.last_execution_plan["mps_accuracy"] = (
            device.mps_accuracy_report
        )
        device.last_execution_plan["mps_diagnostics"] = (
            device.sim.truncation_diagnostics()
            if backend == "mps"
            else None
        )
    return device


def apply_global_phase(device: Device, phase: float) -> None:
    """Apply an SDK circuit's global phase without a host state readback."""
    if phase:
        factor = complex(np.cos(float(phase)), np.sin(float(phase)))
        if hasattr(device.sim, "state"):
            device.sim.state = device.sim.state * factor
        else:
            device.sim.A[0] = device.sim.A[0] * factor


def statevector_numpy(device: Device) -> np.ndarray:
    """Synchronize and materialize the state only when an SDK requests it."""
    state = device.synchronize()
    if state is not None:
        return np.asarray(state, dtype=np.complex64)
    # MPS execution itself is compact, but converting it to an SDK statevector
    # still allocates 2**n amplitudes. Apply the same safety gate as the exact
    # statevector constructor before that dense materialization.
    require_statevector_preflight(
        device.wires,
        allow_unsafe=device._allow_unsafe_statevector,
    )
    return device.sim.to_statevector()


def marginal_probabilities(device: Device, wires: Sequence[int]) -> np.ndarray:
    """Read back only the requested marginal probability vector."""
    device.synchronize()
    probabilities = device.sim.probabilities_array(list(wires))
    if isinstance(probabilities, mx.array):
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
    """Sample requested wires without a full probability-vector readback."""
    wires = list(wires)
    if not wires:
        return np.empty((int(shots), 0), dtype=np.int64)
    device.synchronize()
    if hasattr(device.sim, "sample_array"):
        return np.asarray(
            device.sim.sample_array(int(shots), wires, rng=rng),
            dtype=np.int64,
        )
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
    if device.backend == "mps":
        return device.sim.expectation_dense(list(wires), matrix)
    acted = _apply_local_matrix(device.sim.state, device.wires, wires, matrix)
    value = mx.sum(mx.conj(device.sim.state) * acted)
    mx.eval(value)
    return complex(value.item())


def pauli_product_expectation(
    device: Device,
    paulis: dict[int, str],
) -> complex:
    """Evaluate one Pauli word without constructing a dense register matrix."""
    normalized = {
        int(wire): str(pauli).upper()
        for wire, pauli in paulis.items()
        if str(pauli).upper() != "I"
    }
    if any(pauli not in _PAULI_MATRICES for pauli in normalized.values()):
        raise ValueError("Pauli products support only I, X, Y, and Z")
    device.synchronize()
    if device.backend == "mps":
        operators = {
            wire: _PAULI_MATRICES[pauli]
            for wire, pauli in normalized.items()
        }
        return device.sim.expectation_product(operators)

    acted = device.sim.state
    for wire, pauli in normalized.items():
        acted = _apply_local_matrix(
            acted,
            device.wires,
            [wire],
            _PAULI_MATRICES[pauli],
        )
    value = mx.sum(mx.conj(device.sim.state) * acted)
    mx.eval(value)
    return complex(value.item())


def pauli_product_expectations(
    device: Device,
    words: Sequence[dict[int, str]],
) -> list[complex]:
    """Evaluate many Pauli words with one synchronization/readback boundary.

    Each statevector reduction remains on the selected MLX device. Only the
    final scalar per word crosses to the host, and all lazy reductions are
    evaluated together so Estimator Hamiltonians do not synchronize once per
    Pauli term.
    """
    normalized_words = [
        {
            int(wire): str(pauli).upper()
            for wire, pauli in word.items()
            if str(pauli).upper() != "I"
        }
        for word in words
    ]
    if any(
        pauli not in _PAULI_MATRICES
        for word in normalized_words
        for pauli in word.values()
    ):
        raise ValueError("Pauli products support only I, X, Y, and Z")
    device.synchronize()
    if device.backend == "mps":
        return [
            device.sim.expectation_product(
                {wire: _PAULI_MATRICES[pauli] for wire, pauli in word.items()}
            )
            for word in normalized_words
        ]
    values = []
    for word in normalized_words:
        acted = device.sim.state
        for wire, pauli in word.items():
            acted = _apply_local_matrix(
                acted, device.wires, [wire], _PAULI_MATRICES[pauli]
            )
        values.append(mx.sum(mx.conj(device.sim.state) * acted))
    if values:
        mx.eval(*values)
    return [complex(value.item()) for value in values]


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
    if device.backend == "mps":
        probabilities = []
        for eigenvector in eigenvectors.T:
            projector = np.outer(eigenvector, eigenvector.conj())
            probabilities.append(
                device.sim.expectation_dense(list(wires), projector).real
            )
        host_probabilities = np.clip(
            np.asarray(probabilities, dtype=np.float64), 0.0, None
        )
        host_probabilities /= host_probabilities.sum()
        indices = rng.choice(
            len(eigenvalues), size=int(shots), p=host_probabilities
        )
        return np.real_if_close(eigenvalues[indices])
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
