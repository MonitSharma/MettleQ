#!/usr/bin/env python3
"""Benchmark Qiskit and PennyLane expectation paths across Qupertino methods."""

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
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from mlxq.integrations.pennylane import QupertinoDevice
from mlxq.integrations.qiskit import QupertinoEstimatorV2


def _command_output(*command: str) -> str | None:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _qiskit_circuit(n_qubits: int, steps: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, name="sdk_method_tfim")
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


def _time_rotating(functions: dict, warmups: int, repeats: int, sdk: str):
    latest = {}
    for name, function in functions.items():
        for _ in range(warmups):
            latest[name] = function()

    rows = []
    names = list(functions)
    for repeat in range(repeats):
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        if repeat % 2:
            order.reverse()
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
                    "expectation_z0": latest[name],
                }
            )
    return rows, latest


def _plot(summary_rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for axis, sdk in zip(axes, ("qiskit", "pennylane")):
        rows = [row for row in summary_rows if row["sdk"] == sdk]
        labels = [row["implementation"].replace("_", " ") for row in rows]
        values = [float(row["median_ms"]) for row in rows]
        colors = ["#6b7280" if row["is_reference"] else "#2563eb" for row in rows]
        axis.barh(labels, values, color=colors)
        axis.set_xscale("log")
        axis.set_xlabel("Median end-to-end time (ms, log scale)")
        axis.set_title(sdk.capitalize())
        axis.grid(True, axis="x", which="both", alpha=0.25)
        axis.invert_yaxis()
    figure.suptitle("Qupertino SDK method/device matrix")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--qubits", type=int, default=20)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--mps-dmax", type=int, default=32)
    parser.add_argument("--mps-eps", type=float, default=1e-10)
    args = parser.parse_args()
    if args.qubits <= 0 or args.steps <= 0 or args.repeats <= 0:
        parser.error("qubits, steps, and repeats must be positive")
    if args.warmups < 0 or args.mps_dmax <= 0 or args.mps_eps < 0.0:
        parser.error("invalid warmup or MPS configuration")
    if not mx.metal.is_available():
        parser.error("Apple Metal is unavailable")
    args.outdir.mkdir(parents=True, exist_ok=True)

    qiskit_circuit = _qiskit_circuit(args.qubits, args.steps)
    observable = SparsePauliOp("I" * (args.qubits - 1) + "Z")
    qiskit_reference = StatevectorEstimator()
    qiskit_estimators = {
        f"qupertino_{method_name}_{device}": QupertinoEstimatorV2(
            method=method,
            device=device,
            mps_max_bond_dimension=args.mps_dmax,
            mps_truncation_threshold=args.mps_eps,
        )
        for method_name, method in (
            ("statevector", "statevector"),
            ("mps", "matrix_product_state"),
        )
        for device in ("cpu", "gpu")
    }

    def run_qiskit_reference():
        result = qiskit_reference.run(
            [(qiskit_circuit, observable)]
        ).result()[0]
        return float(np.asarray(result.data.evs))

    qiskit_functions = {"qiskit_statevector_reference": run_qiskit_reference}
    for name, estimator in qiskit_estimators.items():
        qiskit_functions[name] = (
            lambda estimator=estimator: float(
                np.asarray(
                    estimator.run([(qiskit_circuit, observable)])
                    .result()[0]
                    .data.evs
                )
            )
        )

    qiskit_rows, qiskit_values = _time_rotating(
        qiskit_functions, args.warmups, args.repeats, "qiskit"
    )

    pennylane_functions = {
        "pennylane_default_qubit_reference": _pennylane_qnode(
            qml.device("default.qubit", wires=args.qubits),
            args.qubits,
            args.steps,
        )
    }
    pennylane_devices = {}
    for method_name, method in (
        ("statevector", "statevector"),
        ("mps", "matrix_product_state"),
    ):
        for device_name in ("cpu", "gpu"):
            name = f"qupertino_{method_name}_{device_name}"
            device = QupertinoDevice(
                wires=args.qubits,
                method=method,
                device=device_name,
                mps_max_bond_dimension=args.mps_dmax,
                mps_truncation_threshold=args.mps_eps,
            )
            pennylane_devices[name] = device
            pennylane_functions[name] = _pennylane_qnode(
                device, args.qubits, args.steps
            )

    pennylane_rows, pennylane_values = _time_rotating(
        pennylane_functions, args.warmups, args.repeats, "pennylane"
    )
    raw_rows = qiskit_rows + pennylane_rows
    latest = {"qiskit": qiskit_values, "pennylane": pennylane_values}
    references = {
        "qiskit": "qiskit_statevector_reference",
        "pennylane": "pennylane_default_qubit_reference",
    }
    summary_rows = []
    for sdk in ("qiskit", "pennylane"):
        sdk_rows = [row for row in raw_rows if row["sdk"] == sdk]
        reference_name = references[sdk]
        reference_median = statistics.median(
            row["elapsed_ms"]
            for row in sdk_rows
            if row["implementation"] == reference_name
        )
        reference_value = latest[sdk][reference_name]
        for implementation in dict.fromkeys(
            row["implementation"] for row in sdk_rows
        ):
            median = statistics.median(
                row["elapsed_ms"]
                for row in sdk_rows
                if row["implementation"] == implementation
            )
            summary_rows.append(
                {
                    "sdk": sdk,
                    "implementation": implementation,
                    "is_reference": implementation == reference_name,
                    "median_ms": median,
                    "reference_over_implementation_speedup": (
                        reference_median / median
                    ),
                    "absolute_expectation_error": abs(
                        latest[sdk][implementation] - reference_value
                    ),
                }
            )

    with (args.outdir / "sdk_method_matrix_raw.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(raw_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(raw_rows)
    with (args.outdir / "sdk_method_matrix_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    selections = {
        "qiskit": {
            name: estimator.last_execution_selections
            for name, estimator in qiskit_estimators.items()
        },
        "pennylane": {
            name: device.last_execution_selection
            for name, device in pennylane_devices.items()
        },
    }
    diagnostics = {
        "qiskit": {
            name: estimator.last_mps_diagnostics
            for name, estimator in qiskit_estimators.items()
            if "_mps_" in name
        },
        "pennylane": {
            name: device.last_mps_diagnostics
            for name, device in pennylane_devices.items()
            if "_mps_" in name
        },
    }
    payload = {
        "schema_version": 1,
        "workload": "open-chain TFIM-style H + nearest-neighbor ZZ + RX",
        "result_contract": "analytic local Pauli-Z expectation",
        "qubits": args.qubits,
        "steps": args.steps,
        "warmups_per_implementation": args.warmups,
        "rotating_repeats": args.repeats,
        "mps": {"dmax": args.mps_dmax, "eps": args.mps_eps},
        "interpretation": (
            "Each Qiskit speedup is relative to Qiskit StatevectorEstimator; "
            "each PennyLane speedup is relative to default.qubit. CPU and GPU "
            "arms execute independently and are not combined."
        ),
        "maximum_absolute_expectation_error": max(
            float(row["absolute_expectation_error"])
            for row in summary_rows
        ),
        "rows": summary_rows,
        "execution_selections": selections,
        "mps_diagnostics": diagnostics,
    }
    (args.outdir / "sdk_method_matrix_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(os.sys.argv),
        "git_commit": _command_output("git", "rev-parse", "HEAD"),
        "git_status_porcelain": _command_output("git", "status", "--porcelain"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "qiskit": importlib.metadata.version("qiskit"),
        "pennylane": importlib.metadata.version("pennylane"),
        "metal_available": bool(mx.metal.is_available()),
        "mlxq_metal_kernels": os.environ.get("MLXQ_METAL_KERNELS"),
    }
    (args.outdir / "sdk_method_matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    _plot(summary_rows, args.outdir / "sdk_method_matrix.png")
    print(json.dumps(summary_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
