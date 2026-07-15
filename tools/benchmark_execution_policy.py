#!/usr/bin/env python3
"""Calibrate Qupertino's CPU/GPU policy for statevector and MPS execution.

The benchmark times one complete canonical-operation execution followed by a
local expectation value. CPU and GPU arms are run independently in rotating
order; their times are never added together or described as cooperative work.
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

from mlxq.integrations._common import (
    execute_operations,
    pauli_product_expectation,
)


def _command_output(*command: str) -> str | None:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _operations(n_qubits: int, steps: int) -> list[dict]:
    operations = [
        {"name": "H", "wires": [wire], "parameters": []}
        for wire in range(n_qubits)
    ]
    for step in range(steps):
        zz_angle = 0.085 + 0.005 * step
        rx_angle = 0.21 - 0.005 * step
        operations.extend(
            {
                "name": "ZZPHASE",
                "wires": [wire, wire + 1],
                "parameters": [zz_angle],
            }
            for wire in range(n_qubits - 1)
        )
        operations.extend(
            {"name": "RX", "wires": [wire], "parameters": [rx_angle]}
            for wire in range(n_qubits)
        )
    return operations


def _run_once(
    n_qubits: int,
    operations: list[dict],
    method: str,
    device: str,
    dmax: int,
    eps: float,
) -> tuple[float, dict | None]:
    simulator = execute_operations(
        n_qubits,
        operations,
        shots=1,
        method=method,
        execution_device=device,
        mps_max_bond_dimension=dmax,
        mps_truncation_threshold=eps,
    )
    value = pauli_product_expectation(simulator, {0: "Z"})
    diagnostics = (
        simulator.sim.truncation_diagnostics()
        if simulator.backend == "mps"
        else None
    )
    return float(np.real(value)), diagnostics


def _sustained_crossover(
    summaries: list[dict], *, minimum_speedup: float, consecutive: int
) -> int | None:
    ordered = sorted(summaries, key=lambda row: int(row["qubits"]))
    for index in range(0, len(ordered) - consecutive + 1):
        window = ordered[index : index + consecutive]
        if all(
            float(row["cpu_over_gpu_speedup"]) >= minimum_speedup
            for row in window
        ):
            return int(window[0]["qubits"])
    return None


def _plot(summary_rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    labels = {
        "statevector": "Exact statevector",
        "matrix_product_state": "MPS (Dmax-limited)",
    }
    for axis, method in zip(axes, labels):
        rows = [row for row in summary_rows if row["method"] == method]
        qubits = [int(row["qubits"]) for row in rows]
        axis.plot(
            qubits,
            [float(row["cpu_median_ms"]) for row in rows],
            "o-",
            label="CPU",
        )
        axis.plot(
            qubits,
            [float(row["gpu_median_ms"]) for row in rows],
            "o-",
            label="GPU",
        )
        axis.set_title(labels[method])
        axis.set_xlabel("Qubits")
        axis.set_ylabel("End-to-end median (ms, log scale)")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.suptitle("Qupertino CPU/GPU execution-policy calibration")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--statevector-qubits",
        nargs="+",
        type=int,
        default=[4, 6, 8, 10, 12, 14, 16, 18, 20],
    )
    parser.add_argument(
        "--mps-qubits",
        nargs="+",
        type=int,
        default=[4, 8, 12, 16, 20, 24, 28, 32],
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--mps-dmax", type=int, default=32)
    parser.add_argument("--mps-eps", type=float, default=1e-10)
    parser.add_argument("--minimum-gpu-speedup", type=float, default=1.10)
    parser.add_argument("--consecutive-sizes", type=int, default=2)
    args = parser.parse_args()
    if not mx.metal.is_available():
        parser.error("Apple Metal is unavailable; CPU/GPU calibration requires it")
    all_qubits = args.statevector_qubits + args.mps_qubits
    if any(value <= 0 for value in all_qubits):
        parser.error("qubit counts must be positive")
    if args.steps <= 0 or args.repeats <= 0 or args.warmups < 0:
        parser.error("steps/repeats must be positive and warmups non-negative")
    if args.mps_dmax <= 0 or args.mps_eps < 0.0:
        parser.error("MPS Dmax must be positive and epsilon non-negative")
    if args.minimum_gpu_speedup <= 1.0 or args.consecutive_sizes <= 0:
        parser.error("GPU speedup must exceed 1 and consecutive sizes be positive")

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    summary_rows = []
    for method, qubit_counts in (
        ("statevector", args.statevector_qubits),
        ("matrix_product_state", args.mps_qubits),
    ):
        for n_qubits in qubit_counts:
            operations = _operations(n_qubits, args.steps)
            latest = {}
            for device in ("cpu", "gpu"):
                for _ in range(args.warmups):
                    latest[device] = _run_once(
                        n_qubits,
                        operations,
                        method,
                        device,
                        args.mps_dmax,
                        args.mps_eps,
                    )
            for repeat in range(args.repeats):
                order = ("cpu", "gpu") if repeat % 2 == 0 else ("gpu", "cpu")
                for order_index, device in enumerate(order):
                    start = time.perf_counter_ns()
                    latest[device] = _run_once(
                        n_qubits,
                        operations,
                        method,
                        device,
                        args.mps_dmax,
                        args.mps_eps,
                    )
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                    raw_rows.append(
                        {
                            "method": method,
                            "qubits": n_qubits,
                            "device": device,
                            "repeat": repeat,
                            "order_index": order_index,
                            "elapsed_ms": elapsed_ms,
                            "expectation_z0": latest[device][0],
                        }
                    )
            cpu_times = [
                row["elapsed_ms"]
                for row in raw_rows
                if row["method"] == method
                and row["qubits"] == n_qubits
                and row["device"] == "cpu"
            ]
            gpu_times = [
                row["elapsed_ms"]
                for row in raw_rows
                if row["method"] == method
                and row["qubits"] == n_qubits
                and row["device"] == "gpu"
            ]
            diagnostics = latest["gpu"][1]
            summary_rows.append(
                {
                    "method": method,
                    "qubits": n_qubits,
                    "cpu_median_ms": statistics.median(cpu_times),
                    "gpu_median_ms": statistics.median(gpu_times),
                    "cpu_over_gpu_speedup": (
                        statistics.median(cpu_times)
                        / statistics.median(gpu_times)
                    ),
                    "cpu_gpu_expectation_delta": abs(
                        latest["cpu"][0] - latest["gpu"][0]
                    ),
                    "mps_truncation_events": (
                        diagnostics["events"] if diagnostics else None
                    ),
                    "mps_local_discarded_weight_sum": (
                        diagnostics["local_discarded_weight_sum"]
                        if diagnostics
                        else None
                    ),
                    "mps_tensor_device_for_gpu_arm": (
                        diagnostics["tensor_device"] if diagnostics else None
                    ),
                    "mps_svd_device_for_gpu_arm": (
                        diagnostics["svd_device"] if diagnostics else None
                    ),
                }
            )

    crossovers = {}
    for method in ("statevector", "matrix_product_state"):
        rows = [row for row in summary_rows if row["method"] == method]
        crossovers[method] = _sustained_crossover(
            rows,
            minimum_speedup=args.minimum_gpu_speedup,
            consecutive=args.consecutive_sizes,
        )

    raw_path = args.outdir / "execution_policy_raw.csv"
    summary_csv_path = args.outdir / "execution_policy_summary.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(raw_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(raw_rows)
    with summary_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "schema_version": 1,
        "workload": "H + nearest-neighbor ZZ + RX; local Pauli-Z expectation",
        "timing_scope": (
            "canonical translation output through method/device selection, "
            "allocation, execution, synchronization, and local expectation"
        ),
        "policy": (
            "CPU and GPU arms are independent. One numerical device is "
            "selected per execution; their times are not combined."
        ),
        "warmups_per_arm": args.warmups,
        "paired_rotating_repeats": args.repeats,
        "minimum_gpu_speedup": args.minimum_gpu_speedup,
        "consecutive_sizes_required": args.consecutive_sizes,
        "mps": {"dmax": args.mps_dmax, "eps": args.mps_eps},
        "sustained_gpu_crossover_qubits": crossovers,
        "recommended_defaults": {
            "statevector_gpu_min_qubits": crossovers["statevector"],
            "mps_gpu_min_qubits": crossovers["matrix_product_state"],
            "mps_none_interpretation": (
                "automatic MPS should stay on CPU over the measured range"
            ),
        },
        "maximum_cpu_gpu_expectation_delta": max(
            float(row["cpu_gpu_expectation_delta"]) for row in summary_rows
        ),
        "rows": summary_rows,
    }
    (args.outdir / "execution_policy_summary.json").write_text(
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
        "metal_available": bool(mx.metal.is_available()),
        "default_mlx_device": str(mx.default_device()),
        "mlxq_metal_kernels": os.environ.get("MLXQ_METAL_KERNELS"),
    }
    (args.outdir / "execution_policy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    _plot(summary_rows, args.outdir / "execution_policy.png")
    print(json.dumps(payload["recommended_defaults"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
