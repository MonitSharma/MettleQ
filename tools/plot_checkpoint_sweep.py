#!/usr/bin/env python3
"""Plot peak memory and runtime from a checkpoint-sweep summary CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _label(raw: str) -> str:
    if raw == "lazy":
        return "Lazy"
    value = float(raw.removesuffix("_mib"))
    if value >= 1024 and value % 1024 == 0:
        return f"{value / 1024:g} GiB"
    return f"{value:g} MiB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--title", default="Memory-budgeted Metal execution crossover"
    )
    args = parser.parse_args()

    with args.summary.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        parser.error("summary CSV is empty")

    labels = [_label(row["label"]) for row in rows]
    peaks = [float(row["peak_bytes_median"]) / (1024 * 1024) for row in rows]
    runtimes = [float(row["total_ms_median"]) for row in rows]
    changes = [float(row["runtime_change_percent"]) for row in rows]

    plt.style.use("dark_background")
    figure, (memory_axis, runtime_axis) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, constrained_layout=True
    )
    figure.patch.set_facecolor("#0b1017")
    figure.suptitle(args.title, fontsize=19, fontweight="bold")
    for axis in (memory_axis, runtime_axis):
        axis.set_facecolor("#0b1017")
        axis.grid(axis="y", color="#334155", alpha=0.65)
        axis.set_axisbelow(True)

    memory_bars = memory_axis.bar(labels, peaks, color="#168cf7")
    memory_axis.set_ylabel("Median allocator peak (MiB)")
    memory_axis.bar_label(
        memory_bars, labels=[f"{value:,.0f}" for value in peaks], padding=4
    )
    memory_axis.set_ylim(0, max(peaks) * 1.18)

    colors = ["#94a3b8" if index == 0 else "#35d06f"
              for index in range(len(rows))]
    runtime_bars = runtime_axis.bar(labels, runtimes, color=colors)
    runtime_axis.set_ylabel("Median synchronized time (ms)")
    runtime_axis.set_xlabel("Checkpoint scheduling policy")
    runtime_axis.bar_label(
        runtime_bars,
        labels=[
            f"{runtime:.1f} ms" if index == 0
            else f"{runtime:.1f} ms ({change:+.1f}%)"
            for index, (runtime, change) in enumerate(zip(runtimes, changes))
        ],
        padding=4,
    )
    runtime_axis.set_ylim(0, max(runtimes) * 1.22)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
