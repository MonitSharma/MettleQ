#!/usr/bin/env python3
"""Plot matched MettleQ MPS sampling evidence before and after batching."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def _rows(path):
    with path.open(newline="") as handle:
        return {
            row["case"]: row
            for row in csv.DictReader(handle)
            if row["implementation"] == "mettleq_mps_cpu"
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()
    before = _rows(args.before)
    after = _rows(args.after)
    cases = sorted(set(before) & set(after))
    if not cases:
        parser.error("no matched MettleQ cases")

    rows = []
    for case in cases:
        before_ms = float(before[case]["median_time_ms"])
        after_ms = float(after[case]["median_time_ms"])
        rows.append(
            {
                "case": case,
                "before_median_ms": before_ms,
                "after_median_ms": after_ms,
                "batched_sampling_speedup": before_ms / after_ms,
                "after_aer_over_mettleq_speedup": float(
                    after[case]["aer_over_mettleq_speedup"]
                ),
            }
        )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(rows))
    labels = [row["case"].replace("_", " ") for row in rows]
    width = 0.38
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    axes[0].bar(
        x - width / 2,
        [row["before_median_ms"] for row in rows],
        width,
        label="Before: per-shot Python loop",
        color="#94a3b8",
    )
    axes[0].bar(
        x + width / 2,
        [row["after_median_ms"] for row in rows],
        width,
        label="Current: memory-bounded batched sampling",
        color="#7557ff",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median end-to-end time (ms, log scale)")
    axes[0].set_title("Matched MettleQ MPS timings")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", which="both", alpha=0.22)

    speedups = [row["batched_sampling_speedup"] for row in rows]
    axes[1].bar(x, speedups, color="#0f9d82")
    axes[1].axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Before / current speedup")
    axes[1].set_title(
        f"Median improvement: {statistics.median(speedups):.1f}×"
    )
    axes[1].grid(axis="y", alpha=0.22)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=38, ha="right", fontsize=8)
    figure.suptitle("MettleQ batched MPS sampling improvement")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
