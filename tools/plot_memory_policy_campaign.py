#!/usr/bin/env python3
"""Plot cross-workload peak-memory and runtime changes by qubit count."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


LABELS = {
    "adaptive_balanced": "Balanced: max(256 MiB, 4× state)",
    "adaptive_minimum": "Minimum: max(128 MiB, 2× state)",
}
COLORS = {
    "adaptive_balanced": "#168cf7",
    "adaptive_minimum": "#35d06f",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--title", default="Adaptive Metal memory policy across workloads"
    )
    args = parser.parse_args()

    with args.summary.open(newline="") as file:
        rows = list(csv.DictReader(file))
    candidates = [row for row in rows if row["policy"] in LABELS]
    if not candidates:
        parser.error("summary has no adaptive-policy rows")
    qubits = sorted({int(row["qubits"]) for row in candidates})

    plt.style.use("dark_background")
    figure, (memory_axis, runtime_axis) = plt.subplots(
        2, 1, figsize=(11.5, 8), sharex=True, constrained_layout=True
    )
    figure.patch.set_facecolor("#0b1017")
    figure.suptitle(args.title, fontsize=19, fontweight="bold")
    for axis in (memory_axis, runtime_axis):
        axis.set_facecolor("#0b1017")
        axis.axhline(0, color="#94a3b8", linewidth=1)
        axis.grid(color="#334155", alpha=0.65)
        axis.set_axisbelow(True)

    for policy in LABELS:
        peak_values = []
        runtime_values = []
        for n in qubits:
            selected = [
                row for row in candidates
                if row["policy"] == policy and int(row["qubits"]) == n
            ]
            peak_values.append(statistics.median(
                float(row["peak_reduction_percent"]) for row in selected
            ))
            runtime_values.append(statistics.median(
                float(row.get(
                    "paired_runtime_change_percent_median",
                    row["runtime_change_percent"],
                )) for row in selected
            ))
        memory_axis.plot(
            qubits, peak_values, marker="o", linewidth=2.3,
            color=COLORS[policy], label=LABELS[policy],
        )
        runtime_axis.plot(
            qubits, runtime_values, marker="o", linewidth=2.3,
            color=COLORS[policy], label=LABELS[policy],
        )

    memory_axis.set_ylabel("Median peak reduction vs lazy (%)")
    memory_axis.legend(frameon=False, ncol=2)
    runtime_axis.set_ylabel("Median paired runtime change vs lazy (%)")
    runtime_axis.set_xlabel("Qubits (median across workloads)")
    runtime_axis.set_xticks(qubits)
    runtime_axis.text(
        0.01, 0.04, "Below zero is faster",
        transform=runtime_axis.transAxes, color="#94a3b8",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
