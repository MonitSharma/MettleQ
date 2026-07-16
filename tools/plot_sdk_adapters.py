#!/usr/bin/env python3
"""Render the frozen native-SDK timing comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())

    def renamed(section, field):
        legacy = field.replace("mettleq", "qupertino")
        return section[field] if field in section else section[legacy]

    panels = [
        (
            "Qiskit · full statevector",
            renamed(summary["qiskit"], "mettleq_median_ms"),
            summary["qiskit"]["reference_median_ms"],
            "Aer CPU",
            renamed(summary["qiskit"], "reference_over_mettleq_speedup"),
        ),
        (
            "PennyLane · local ⟨Z⟩",
            renamed(summary["pennylane"], "mettleq_median_ms"),
            summary["pennylane"]["reference_median_ms"],
            "default.qubit",
            renamed(summary["pennylane"], "reference_over_mettleq_speedup"),
        ),
    ]
    colors = ("#7C3AED", "#94A3B8")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for axis, (title, mettleq_ms, reference_ms, reference, speedup) in zip(
        axes, panels
    ):
        values = [mettleq_ms, reference_ms]
        bars = axis.bar(["MettleQ", reference], values, color=colors, width=0.62)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_ylabel("Median end-to-end time (ms)")
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
        axis.set_ylim(0, max(values) * 1.24)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                f"{value:.2f} ms",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.text(
            0.5,
            0.89,
            f"{speedup:.2f}× faster",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#5B21B6",
        )

    figure.suptitle(
        f"Native SDK paths · {summary['qubits']}-qubit TFIM-style circuit",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Apple M3 Pro · 7 rotating repeats · lower is better · result contracts differ",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
