"""Peaked-circuit benchmark fixtures with explicit scope and provenance."""

from __future__ import annotations

import hashlib
import math
from importlib import resources
from pathlib import Path
from typing import Optional


PUBLISHED_P9_FILENAME = "peaked_circuit_P9_Hqap_56x1917.qasm"
PUBLISHED_P9_SHA256 = (
    "cff3496c45d9133c1f1693f1d3b0cf1fc2da338f13cd7b339db330a4762d0f35"
)
PUBLISHED_P9_EXPECTED_BITSTRING = (
    "01101110111001100000100000001010011100101101010111110111"
)


def published_p9_qasm_path() -> Path:
    """Return the installed path of the published 56-qubit P9 QASM."""
    resource = resources.files("mettleq").joinpath(
        "datasets", PUBLISHED_P9_FILENAME
    )
    return Path(str(resource))


def published_p9_manifest() -> dict:
    """Validate and summarize the vendored published benchmark input."""
    path = published_p9_qasm_path()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    text = payload.decode("utf-8")
    return {
        "path": str(path),
        "sha256": digest,
        "integrity_ok": digest == PUBLISHED_P9_SHA256,
        "num_qubits": 56,
        "u_gates": sum(line.startswith("u(") for line in text.splitlines()),
        "rzz_gates": sum(
            line.startswith("rzz(") for line in text.splitlines()
        ),
        "operation_count": sum(
            line.startswith(("u(", "rzz(")) for line in text.splitlines()
        ),
        "expected_bitstring": PUBLISHED_P9_EXPECTED_BITSTRING,
        "benchmark_contract": "published_midpoint_mpo_with_unswapping",
    }


def _topology_pairs(n_qubits: int, topology: str, layer: int, rng):
    if topology == "linear":
        return [
            (wire, wire + 1)
            for wire in range(layer % 2, n_qubits - 1, 2)
        ]
    if topology == "grid":
        columns = max(2, int(math.ceil(math.sqrt(n_qubits))))
        rows = int(math.ceil(n_qubits / columns))
        pairs = []
        if layer % 2 == 0:
            for row in range(rows):
                start = row * columns + ((layer // 2) % 2)
                for first in range(start, min((row + 1) * columns, n_qubits) - 1, 2):
                    pairs.append((first, first + 1))
        else:
            for row in range((layer // 2) % 2, rows - 1, 2):
                for column in range(columns):
                    first = row * columns + column
                    second = first + columns
                    if second < n_qubits:
                        pairs.append((first, second))
        return pairs
    if topology == "long_range":
        order = list(map(int, rng.permutation(n_qubits)))
        return [tuple(order[index:index + 2]) for index in range(0, n_qubits - 1, 2)]
    if topology == "all_to_all":
        return [
            (first, second)
            for first in range(n_qubits)
            for second in range(first + 1, n_qubits)
        ]
    raise ValueError(
        "topology must be one of: linear, grid, long_range, all_to_all"
    )


def build_mirrored_peaked_circuit(
    n_qubits: int,
    *,
    depth: int = 2,
    topology: str = "long_range",
    seed: int = 153,
    peak_bitstring: Optional[str] = None,
    noise_angle: float = 0.16,
    measure: bool = True,
):
    """Build a deterministic peaked regression circuit with a known mode.

    A seeded entangling body containing U, RZZ, and permutation gates is
    followed by its exact inverse. Small final RY rotations make the requested
    computational-basis string the unique most likely result with analytically
    known probability. This exercises bond growth and recovery, but is not a
    substitute for the published P9 midpoint-MPO benchmark.
    """
    try:
        import numpy as np
        from qiskit import QuantumCircuit
    except ImportError as exc:  # pragma: no cover - optional SDK dependency
        raise ImportError(
            "Mirrored peaked circuits require the optional Qiskit dependency"
        ) from exc

    n_qubits = int(n_qubits)
    depth = int(depth)
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least 2")
    if depth < 1:
        raise ValueError("depth must be positive")
    if not 0.0 <= abs(float(noise_angle)) < math.pi / 2.0:
        raise ValueError("abs(noise_angle) must be smaller than pi/2")
    if peak_bitstring is None:
        peak_bitstring = "".join(
            "1" if ((index * 5 + 3) % 7) < 3 else "0"
            for index in range(n_qubits)
        )
    if len(peak_bitstring) != n_qubits or set(peak_bitstring) - {"0", "1"}:
        raise ValueError("peak_bitstring must contain one binary digit per qubit")

    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(n_qubits)
    # Qiskit displays count strings from highest to lowest classical bit.
    for wire, bit in enumerate(reversed(peak_bitstring)):
        if bit == "1":
            circuit.x(wire)

    body = QuantumCircuit(n_qubits, name="seeded_entangling_body")
    for layer in range(depth):
        for wire in range(n_qubits):
            angles = rng.uniform(-math.pi, math.pi, size=3)
            body.u(*map(float, angles), wire)
        for first, second in _topology_pairs(
            n_qubits, topology, layer, rng
        ):
            body.rzz(float(rng.uniform(-math.pi, math.pi)), first, second)
        if n_qubits >= 4:
            first = layer % (n_qubits - 1)
            second = n_qubits - 1 - first
            if first != second:
                body.swap(first, second)

    circuit.compose(body, inplace=True)
    circuit.compose(body.inverse(), inplace=True)
    for wire in range(n_qubits):
        circuit.ry(float(noise_angle), wire)
    if measure:
        circuit.measure_all()
    circuit.metadata = {
        "benchmark": "mettleq_mirrored_peaked_regression",
        "scope": "ci_regression_not_quantum_advantage_claim",
        "topology": topology,
        "depth": depth,
        "seed": int(seed),
        "peak_bitstring": peak_bitstring,
        "expected_peak_probability": math.cos(noise_angle / 2.0) ** (
            2 * n_qubits
        ),
    }
    return circuit
