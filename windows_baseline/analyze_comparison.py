#!/usr/bin/env python3
"""Build the exact-width Windows/WSL versus MettleQ comparison artifacts."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_RESULTS = ROOT / "windows_baseline" / "results"
METTLEQ_RESULTS = (
    ROOT / "assets" / "benchmarks-frozen"
    / "fork-m3pro-20260719-cudaq-competitive-metal"
)
OUTPUT = WINDOWS_RESULTS / "mettleq_windows_matched_widths.csv"
PLOT = ROOT / "windows_baseline" / "plots" / "mettleq_vs_windows_matched_widths.png"
CUDAQ_PLOT = ROOT / "windows_baseline" / "plots" / "mettleq_vs_cudaq_matched_widths.png"
WIDTHS = (15, 20, 24, 26, 28)
WORKLOADS = (
    "qft", "qaoa_ring", "ghz", "grover_proxy",
    "phase_estimation", "tfim_trotter",
)
BACKENDS = {
    "CUDA-Q NVIDIA GPU": "cudaq_nvidia_summary.csv",
    "PennyLane Lightning GPU": "pennylane_lightning_gpu_summary.csv",
    "Qiskit Aer statevector GPU": "qiskit_aer_statevector_gpu_summary.csv",
    "Qiskit Aer MPS GPU": "qiskit_aer_matrix_product_state_gpu_summary.csv",
    "Qiskit Aer statevector CPU": "qiskit_aer_statevector_cpu_summary.csv",
    "Qiskit Aer MPS CPU": "qiskit_aer_matrix_product_state_cpu_summary.csv",
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
        "Qiskit Aer statevector GPU": "#B53F4E",
        "Qiskit Aer MPS GPU": "#D05A9E",
        "Qiskit Aer statevector CPU": "#D87822",
        "Qiskit Aer MPS CPU": "#8B8D16",
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
    for ax in axes.flat:
        ax.set_xlabel("Qubits")
        ax.tick_params(axis="x", labelbottom=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
        ncol=4, frameon=False,
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


def write_cudaq_plot(rows: list[dict[str, object]]) -> None:
    """Focused scaling figure for the Apple Metal versus CUDA-Q question."""
    import matplotlib.pyplot as plt

    lookup = {
        (str(row["workload"]), int(row["qubits"]), str(row["backend"])): row
        for row in rows
    }
    series = {
        "MettleQ Metal · Apple M3 Pro": ("#167D8D", "o"),
        "CUDA-Q NVIDIA GPU · RTX 3070": ("#76B900", "s"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.5), sharex=False)
    for ax, workload in zip(axes.flat, WORKLOADS):
        for backend, (color, marker) in series.items():
            values = [
                float(lookup[(workload, width, backend.split(" · ")[0])]["mean_ms"])
                for width in WIDTHS
            ]
            ax.plot(WIDTHS, values, marker=marker, linewidth=2.4,
                    markersize=5, label=backend, color=color)
        mettleq_28 = float(lookup[(workload, 28, "MettleQ Metal")]["mean_ms"])
        cudaq_28 = float(lookup[(workload, 28, "CUDA-Q NVIDIA GPU")]["mean_ms"])
        ratio = mettleq_28 / cudaq_28
        ax.annotate(
            f"CUDA-Q {ratio:.1f}× faster at 28q",
            xy=(28, cudaq_28), xytext=(-7, 9), textcoords="offset points",
            ha="right", fontsize=8,
        )
        ax.set_yscale("log")
        ax.set_title(workload.replace("_", " ").title())
        ax.set_xlabel("Number of qubits")
        ax.set_xticks(WIDTHS, [str(width) for width in WIDTHS])
        ax.set_ylabel("Mean full-state time (ms, log scale)")
        ax.grid(which="both", linestyle=":", alpha=0.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=2, frameon=False)
    fig.suptitle("Exact-width Apple Metal versus CUDA-Q NVIDIA scaling",
                 y=0.995, fontsize=15)
    fig.text(
        0.5, 0.012,
        "Widths: 15, 20, 24, 26, 28 qubits · one warm-up + three measured full-state runs",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.88))
    CUDAQ_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CUDAQ_PLOT, dpi=180)
    plt.close(fig)


def main() -> int:
    rows = build_rows()
    write_csv(rows)
    write_plot(rows)
    write_cudaq_plot(rows)
    print(f"wrote {OUTPUT}")
    print(f"wrote {PLOT}")
    print(f"wrote {CUDAQ_PLOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
