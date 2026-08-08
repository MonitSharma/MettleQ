"""Plot the persisted native CPU-MPS benchmark artifact."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_rows(args.artifact_dir / "mps_native_summary.csv")
    labels = [row["workload"] for row in rows]
    routed = [float(row["mettleq_routed_ms"]) for row in rows]
    restore = [float(row["mettleq_restore_ms"]) for row in rows]
    aer = [float(row["aer_ms"]) for row in rows]
    speedup = [float(row["aer_over_mettleq"]) for row in rows]

    fig, (runtime_ax, ratio_ax) = plt.subplots(
        1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.7, 1]}
    )
    x = list(range(len(labels)))
    width = 0.25
    runtime_ax.bar([i - width for i in x], routed, width, label="MettleQ routed")
    runtime_ax.bar(x, restore, width, label="MettleQ restore")
    runtime_ax.bar([i + width for i in x], aer, width, label="Qiskit Aer CPU MPS")
    runtime_ax.set_yscale("log")
    runtime_ax.set_ylabel("Median runtime (ms, log scale)")
    runtime_ax.set_xticks(x, labels, rotation=35, ha="right")
    runtime_ax.grid(axis="y", alpha=0.25)
    runtime_ax.legend(frameon=False)
    runtime_ax.set_title("Native CPU MPS runtime")

    colors = ["#2e7d32" if value >= 1 else "#c62828" for value in speedup]
    ratio_ax.barh(x, speedup, color=colors)
    ratio_ax.axvline(1.0, color="black", linewidth=1)
    ratio_ax.set_xscale("log")
    ratio_ax.set_xlabel("Aer / MettleQ routed\n(>1 means MettleQ is faster)")
    ratio_ax.set_yticks(x, labels)
    ratio_ax.invert_yaxis()
    ratio_ax.grid(axis="x", alpha=0.25)
    ratio_ax.set_title("Relative performance")

    fig.suptitle("MettleQ native Accelerate MPS vs Qiskit Aer — Apple M3 Pro")
    fig.text(
        0.5,
        0.01,
        "dmax=64 · eps=1e-10 · 2 warmups + 5 repeats · CPU-only",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
