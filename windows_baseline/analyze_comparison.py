#!/usr/bin/env python3
"""Build the exact-width Windows/WSL versus MettleQ comparison artifacts."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_RESULTS = ROOT / "windows_baseline" / "results"
METTLEQ_RESULTS = (
    ROOT / "assets" / "benchmarks-frozen"
    / "fork-m3pro-20260719-windows-matched-metal"
)
OUTPUT = WINDOWS_RESULTS / "mettleq_windows_matched_widths.csv"
PLOT = ROOT / "windows_baseline" / "plots" / "mettleq_vs_windows_matched_widths.png"
WIDTHS = (15, 20, 24, 26, 28)
WORKLOADS = (
    "qft", "qaoa_ring", "ghz", "grover_proxy",
    "phase_estimation", "tfim_trotter",
)
BACKENDS = {
    "CUDA-Q NVIDIA GPU": "cudaq_nvidia_summary.csv",
    "PennyLane Lightning GPU": "pennylane_lightning_gpu_summary.csv",
    "Qiskit Aer statevector CPU": "qiskit_aer_statevector_cpu_summary.csv",
    "PennyLane Lightning CPU": "pennylane_lightning_qubit_summary.csv",
}


def _read(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["benchmark"], int(row["qubits"])): row
            for row in csv.DictReader(handle)
        }


def _read_mettleq() -> dict[tuple[str, int], dict[str, str]]:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    for workload in WORKLOADS:
        rows.update(_read(METTLEQ_RESULTS / workload / "mettleq_metal_summary.csv"))
    return rows


def build_rows() -> list[dict[str, object]]:
    sources = {"MettleQ Metal": _read_mettleq()}
    sources.update({
        label: _read(WINDOWS_RESULTS / filename)
        for label, filename in BACKENDS.items()
    })
    rows: list[dict[str, object]] = []
    for workload in WORKLOADS:
        for qubits in WIDTHS:
            mettleq_ms = float(sources["MettleQ Metal"][(workload, qubits)]["mean_ms"])
            for backend, values in sources.items():
                source = values[(workload, qubits)]
                mean_ms = float(source["mean_ms"])
                rows.append({
                    "workload": workload,
                    "qubits": qubits,
                    "backend": backend,
                    "mean_ms": mean_ms,
                    "ci95_ms": float(source["ci95_ms"]),
                    "backend_over_mettleq": mean_ms / mettleq_ms,
                    "result_contract": "synchronized full state",
                })
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "MettleQ Metal": "#167D8D",
        "CUDA-Q NVIDIA GPU": "#76B900",
        "PennyLane Lightning GPU": "#7B4AB5",
        "Qiskit Aer statevector CPU": "#D87822",
        "PennyLane Lightning CPU": "#61758A",
    }
    lookup = {
        (str(row["workload"]), int(row["qubits"]), str(row["backend"])): row
        for row in rows
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.4), sharex=True)
    for ax, workload in zip(axes.flat, WORKLOADS):
        for backend, color in colors.items():
            values = [
                float(lookup[(workload, width, backend)]["mean_ms"])
                for width in WIDTHS
            ]
            ax.plot(WIDTHS, values, marker="o", linewidth=2, markersize=4,
                    label=backend, color=color)
        ax.set_yscale("log")
        ax.set_title(workload.replace("_", " ").title())
        ax.grid(which="both", linestyle=":", alpha=0.4)
        ax.set_xticks(WIDTHS)
    axes[0, 0].set_ylabel("Mean time (ms, log scale)")
    axes[1, 0].set_ylabel("Mean time (ms, log scale)")
    for ax in axes[1, :]:
        ax.set_xlabel("Qubits")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
        ncol=3, frameon=False,
    )
    fig.suptitle(
        "Matched-width full-state simulation: Apple M3 Pro Metal vs Windows/WSL",
        y=0.995, fontsize=15,
    )
    fig.text(
        0.5, 0.012,
        "Same circuits, widths, warm-up/repeat count, and full-state contract; different host systems.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.865))
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=180)
    plt.close(fig)


def main() -> int:
    rows = build_rows()
    write_csv(rows)
    write_plot(rows)
    print(f"wrote {OUTPUT}")
    print(f"wrote {PLOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
