#!/usr/bin/env python3
"""Benchmark native Qiskit and PennyLane paths against their CPU references.

The two SDK comparisons intentionally use different native result contracts:
Qiskit requests a full statevector from both backends, while PennyLane requests
a local expectation from both devices.  Speedups are reported only within each
SDK pair, never across the two measurement modes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from mlxq.integrations.pennylane import QupertinoDevice
from mlxq.integrations.qiskit import QupertinoBackend


def _command_output(*command: str) -> str | None:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _qiskit_circuit(n_qubits: int, steps: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, name="sdk_tfim")
    for wire in range(n_qubits):
        circuit.h(wire)
    for step in range(steps):
        zz_angle = 0.17 + 0.01 * step
        rx_angle = 0.23 - 0.01 * step
        for wire in range(n_qubits - 1):
            circuit.rzz(zz_angle, wire, wire + 1)
        for wire in range(n_qubits):
            circuit.rx(rx_angle, wire)
    return circuit


def _pennylane_qnode(device, n_qubits: int, steps: int):
    @qml.qnode(device)
    def circuit():
        for wire in range(n_qubits):
            qml.Hadamard(wire)
        for step in range(steps):
            zz_angle = 0.17 + 0.01 * step
            rx_angle = 0.23 - 0.01 * step
            for wire in range(n_qubits - 1):
                qml.IsingZZ(zz_angle, wires=[wire, wire + 1])
            for wire in range(n_qubits):
                qml.RX(rx_angle, wires=wire)
        return qml.expval(qml.Z(0))

    return circuit


def _time_pairs(functions, warmups: int, repeats: int, sdk: str):
    latest = {}
    for function in functions.values():
        for _ in range(warmups):
            function()

    rows = []
    names = list(functions)
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else list(reversed(names))
        for order_index, name in enumerate(order):
            start = time.perf_counter_ns()
            latest[name] = functions[name]()
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            rows.append(
                {
                    "sdk": sdk,
                    "implementation": name,
                    "repeat": repeat,
                    "order_index": order_index,
                    "elapsed_ms": elapsed_ms,
                }
            )
    return rows, latest


def _median(rows, implementation: str) -> float:
    return statistics.median(
        row["elapsed_ms"]
        for row in rows
        if row["implementation"] == implementation
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--qubits", type=int, default=20)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.qubits <= 0 or args.steps <= 0:
        parser.error("--qubits and --steps must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("--warmups must be non-negative and --repeats positive")

    args.outdir.mkdir(parents=True, exist_ok=True)
    qiskit_circuit = _qiskit_circuit(args.qubits, args.steps)
    aer_circuit = qiskit_circuit.copy()
    aer_circuit.save_statevector()
    qupertino_backend = QupertinoBackend()
    aer_backend = AerSimulator(method="statevector")

    def run_qupertino_qiskit():
        result = qupertino_backend.run(
            qiskit_circuit, shots=1, return_statevector=True
        ).result()
        return np.asarray(result.data(0)["statevector"])

    def run_aer():
        result = aer_backend.run(aer_circuit, shots=1).result()
        return np.asarray(result.data(0)["statevector"])

    qiskit_rows, qiskit_values = _time_pairs(
        {"qupertino": run_qupertino_qiskit, "qiskit_aer_cpu": run_aer},
        args.warmups,
        args.repeats,
        "qiskit",
    )

    qupertino_qnode = _pennylane_qnode(
        QupertinoDevice(wires=args.qubits), args.qubits, args.steps
    )
    default_qnode = _pennylane_qnode(
        qml.device("default.qubit", wires=args.qubits),
        args.qubits,
        args.steps,
    )
    pennylane_rows, pennylane_values = _time_pairs(
        {"qupertino": qupertino_qnode, "pennylane_default_qubit": default_qnode},
        args.warmups,
        args.repeats,
        "pennylane",
    )

    raw_rows = qiskit_rows + pennylane_rows
    qiskit_qupertino_ms = _median(qiskit_rows, "qupertino")
    qiskit_reference_ms = _median(qiskit_rows, "qiskit_aer_cpu")
    pennylane_qupertino_ms = _median(pennylane_rows, "qupertino")
    pennylane_reference_ms = _median(
        pennylane_rows, "pennylane_default_qubit"
    )
    qiskit_actual = qiskit_values["qupertino"].astype(np.complex128)
    qiskit_expected = qiskit_values["qiskit_aer_cpu"].astype(np.complex128)

    summary = {
        "schema_version": 1,
        "workload": "open-chain TFIM-style H + nearest-neighbor ZZ + RX",
        "qubits": args.qubits,
        "steps": args.steps,
        "warmups_per_implementation": args.warmups,
        "paired_rotating_repeats": args.repeats,
        "qiskit": {
            "result_contract": "full statevector readback",
            "qupertino_median_ms": qiskit_qupertino_ms,
            "reference": "Qiskit Aer CPU statevector",
            "reference_median_ms": qiskit_reference_ms,
            "reference_over_qupertino_speedup": (
                qiskit_reference_ms / qiskit_qupertino_ms
            ),
            "max_amplitude_error": float(
                np.max(np.abs(qiskit_actual - qiskit_expected))
            ),
            "l2_state_error": float(
                np.linalg.norm(qiskit_actual - qiskit_expected)
            ),
            "qupertino_norm_error": float(
                abs(
                    np.sqrt(
                        np.sum(np.abs(qiskit_actual) ** 2, dtype=np.float64)
                    )
                    - 1.0
                )
            ),
        },
        "pennylane": {
            "result_contract": "analytic local Pauli-Z expectation",
            "qupertino_median_ms": pennylane_qupertino_ms,
            "reference": "PennyLane default.qubit",
            "reference_median_ms": pennylane_reference_ms,
            "reference_over_qupertino_speedup": (
                pennylane_reference_ms / pennylane_qupertino_ms
            ),
            "absolute_expectation_error": float(
                abs(
                    complex(pennylane_values["qupertino"])
                    - complex(pennylane_values["pennylane_default_qubit"])
                )
            ),
        },
        "interpretation": (
            "Speedups are comparable only within each SDK pair because the "
            "Qiskit and PennyLane result contracts differ. Timings include "
            "SDK translation, execution, synchronization, and requested result "
            "materialization."
        ),
    }
    validation = {
        "qiskit_max_amplitude_atol": 5e-6,
        "qiskit_norm_error_atol": 1e-5,
        "pennylane_expectation_atol": 5e-6,
    }
    validation["passed"] = bool(
        summary["qiskit"]["max_amplitude_error"]
        <= validation["qiskit_max_amplitude_atol"]
        and summary["qiskit"]["qupertino_norm_error"]
        <= validation["qiskit_norm_error_atol"]
        and summary["pennylane"]["absolute_expectation_error"]
        <= validation["pennylane_expectation_atol"]
    )
    summary["validation"] = validation

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(os.sys.argv),
        "git_commit": _command_output("git", "rev-parse", "HEAD"),
        "tracked_worktree_clean": not bool(
            _command_output(
                "git", "status", "--porcelain", "--untracked-files=no"
            )
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": _command_output("sysctl", "-n", "machdep.cpu.brand_string"),
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "qiskit": importlib.metadata.version("qiskit"),
        "qiskit_aer": importlib.metadata.version("qiskit-aer"),
        "pennylane": importlib.metadata.version("pennylane"),
        "mlx_default_device": str(mx.default_device()),
        "metal_available": bool(mx.metal.is_available()),
        "mlxq_metal_kernels": os.environ.get("MLXQ_METAL_KERNELS"),
    }

    raw_path = args.outdir / "sdk_adapter_timings.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    (args.outdir / "sdk_adapter_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.outdir / "sdk_adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
