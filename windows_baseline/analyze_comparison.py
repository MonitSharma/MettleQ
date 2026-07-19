#!/usr/bin/env python3
"""Build the honest adjacent-width Windows/WSL versus MettleQ comparison.

The Windows campaign did not measure 25 qubits, while the frozen MettleQ
workload campaign did.  This script therefore retains the measured 24q and
26q Windows values instead of interpolating a fictional 25q result.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_RESULTS = ROOT / "windows_baseline" / "results"
METTLEQ_RESULTS = (
    ROOT
    / "assets"
    / "benchmarks-frozen"
    / "fork-m3pro-20260715"
    / "shader_sweep_summary.csv"
)
OUTPUT = WINDOWS_RESULTS / "mettleq_windows_adjacent_widths.csv"
PLOT = ROOT / "windows_baseline" / "plots" / "mettleq_vs_windows_adjacent_widths.png"

WORKLOADS = {
    "qft": "qft",
    "qaoa_ring": "qaoa",
    "ghz": "ghz",
    "grover_proxy": "grover",
    "phase_estimation": "phase_estimation",
    "tfim_trotter": "tfim_trotter2",
}
BACKENDS = {
    "CUDA-Q NVIDIA GPU": "cudaq_nvidia_summary.csv",
    "PennyLane Lightning GPU": "pennylane_lightning_gpu_summary.csv",
    "Qiskit Aer statevector CPU": "qiskit_aer_statevector_cpu_summary.csv",
    "PennyLane Lightning CPU": "pennylane_lightning_qubit_summary.csv",
}


def _read_mettleq() -> dict[str, float]:
    with METTLEQ_RESULTS.open(newline="", encoding="utf-8") as handle:
        rows = {row["benchmark"]: float(row["metal_mean_ms"]) for row in csv.DictReader(handle)}
    return {workload: rows[source] for workload, source in WORKLOADS.items()}


def _read_windows(path: Path) -> dict[tuple[str, int], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["benchmark"], int(row["qubits"])): float(row["mean_ms"])
            for row in csv.DictReader(handle)
        }


def build_rows() -> list[dict[str, object]]:
    mettleq = _read_mettleq()
    windows = {
        label: _read_windows(WINDOWS_RESULTS / filename)
        for label, filename in BACKENDS.items()
    }
    rows: list[dict[str, object]] = []
    for workload, mettleq_ms in mettleq.items():
        row: dict[str, object] = {
            "workload": workload,
            "mettleq_m3_pro_metal_25q_ms": mettleq_ms,
        }
        for label, values in windows.items():
            key = label.lower().replace(" ", "_").replace("-", "_")
            for width in (24, 26):
                value = values[(workload, width)]
                row[f"{key}_{width}q_ms"] = value
                row[f"{key}_{width}q_over_mettleq_25q"] = value / mettleq_ms
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [str(row["workload"]).replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    series = [
        ("MettleQ Metal, M3 Pro, 25q", "mettleq_m3_pro_metal_25q_ms"),
        ("CUDA-Q NVIDIA, RTX 3070, 24q", "cuda_q_nvidia_gpu_24q_ms"),
        ("CUDA-Q NVIDIA, RTX 3070, 26q", "cuda_q_nvidia_gpu_26q_ms"),
        ("Lightning GPU, RTX 3070, 24q", "pennylane_lightning_gpu_24q_ms"),
        ("Lightning GPU, RTX 3070, 26q", "pennylane_lightning_gpu_26q_ms"),
    ]
    width = 0.16
    fig, ax = plt.subplots(figsize=(13, 6.8))
    for index, (label, key) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        ax.bar(x + offset, [float(row[key]) for row in rows], width, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("Mean full-state execution time (ms, log scale; lower is better)")
    ax.set_title("MettleQ Apple Metal versus Windows/WSL NVIDIA baselines")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.45)
    ax.legend(ncol=2, fontsize=9)
    fig.text(
        0.5,
        0.01,
        "Adjacent widths are shown because Windows did not record 25q. No interpolation or same-device claim is made.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
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
