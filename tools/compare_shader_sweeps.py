#!/usr/bin/env python3
"""Compare two 29-workload shader sweeps and generate decision-useful plots.

The historical input only needs ``benchmark`` and ``paired_ratio_mean``.
The current input is the normal ``shader_sweep_summary.csv`` produced by
``tools/shader_suite_sweep.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gen_perf_charts import BG, C_METAL, C_MLX, FG, GRID, SWEEP_LABELS, _style


def _read(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["benchmark"]: row for row in rows}


def _write_csv(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _comparison_plot(rows, historical_label: str, current_label: str, out: Path):
    ordered = sorted(rows, key=lambda row: row["speedup_delta"])
    labels = [SWEEP_LABELS.get(row["benchmark"], row["benchmark"]) for row in ordered]
    deltas = [row["speedup_delta"] for row in ordered]
    y = np.arange(len(ordered))
    colors = [C_MLX if value >= 0 else "#ff453a" for value in deltas]

    fig, ax = plt.subplots(figsize=(9.4, 9.4), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.barh(y, deltas, color=colors, edgecolor=BG, linewidth=0.4)
    ax.axvline(0.0, color=FG, linewidth=0.9)
    for yi, row in zip(y, ordered):
        delta = row["speedup_delta"]
        old = row["historical_ratio"]
        new = row["current_ratio"]
        align = "left" if delta >= 0 else "right"
        offset = 3 if delta >= 0 else -3
        ax.annotate(
            f"{old:.1f}→{new:.1f}x",
            (delta, yi),
            ha=align,
            va="center",
            color=FG,
            fontsize=7.5,
            xytext=(offset, 0),
            textcoords="offset points",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    span = max(abs(min(deltas)), abs(max(deltas)))
    ax.set_xlim(-span * 1.3, span * 1.3)
    _style(
        ax,
        f"Change in Metal speedup: {historical_label} → {current_label}",
        "Change in pure-MLX / Metal speedup (×)",
        None,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def _runtime_plot(rows, current_label: str, out: Path):
    ordered = sorted(rows, key=lambda row: row["current_ratio"])
    labels = [SWEEP_LABELS.get(row["benchmark"], row["benchmark"]) for row in ordered]
    pure = np.array([row["pure_mean_ms"] for row in ordered])
    metal = np.array([row["metal_mean_ms"] for row in ordered])
    y = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(9.4, 9.4), dpi=150)
    fig.patch.set_facecolor(BG)
    for yi, p, m in zip(y, pure, metal):
        ax.plot([m, p], [yi, yi], color=GRID, linewidth=1.2, zorder=1)
    ax.scatter(pure, y, color=C_MLX, s=27, label="Pure MLX", zorder=2)
    ax.scatter(metal, y, color=C_METAL, s=27, label="Custom Metal", zorder=3)
    for yi, row in zip(y, ordered):
        ax.annotate(
            f"{row['current_ratio']:.1f}x",
            (row["metal_mean_ms"], yi),
            ha="right",
            va="center",
            color=FG,
            fontsize=7.2,
            xytext=(-4, 0),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_xlim(float(metal.min()) * 0.35, float(pure.max()) * 1.35)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    _style(
        ax,
        f"25-qubit synchronized runtime — {current_label}",
        "Mean wall time (ms, log scale; lower is better)",
        None,
    )
    ax.legend(
        loc="lower right", fontsize=8.5, facecolor="#121820",
        edgecolor=GRID, labelcolor=FG,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--historical-label", default="historical")
    parser.add_argument("--current-label", default="current")
    parser.add_argument(
        "--noise-band", type=float, default=0.25,
        help="Speedup-delta magnitude treated as unchanged (default: 0.25x)",
    )
    args = parser.parse_args()

    historical = _read(args.historical)
    current = _read(args.current)
    missing = sorted(set(historical) ^ set(current))
    if missing:
        raise SystemExit(f"workload sets differ: {missing}")

    rows = []
    for benchmark in sorted(current):
        old = float(historical[benchmark]["paired_ratio_mean"])
        new = float(current[benchmark]["paired_ratio_mean"])
        delta = new - old
        if delta > args.noise_band:
            verdict = "improved"
        elif delta < -args.noise_band:
            verdict = "regressed"
        else:
            verdict = "unchanged"
        rows.append({
            "benchmark": benchmark,
            "historical_ratio": old,
            "current_ratio": new,
            "speedup_delta": delta,
            "relative_speedup_change_percent": 100.0 * delta / old,
            "verdict": verdict,
            "pure_mean_ms": float(current[benchmark]["pure_mean_ms"]),
            "metal_mean_ms": float(current[benchmark]["metal_mean_ms"]),
        })

    args.outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.outdir / "historical_vs_current.csv", rows)
    _comparison_plot(
        rows, args.historical_label, args.current_label,
        args.outdir / "chart_historical_vs_current_speedup.png",
    )
    _runtime_plot(
        rows, args.current_label,
        args.outdir / "chart_current_runtime_dumbbell.png",
    )
    counts = {
        verdict: sum(row["verdict"] == verdict for row in rows)
        for verdict in ("improved", "unchanged", "regressed")
    }
    payload = {
        "historical_label": args.historical_label,
        "current_label": args.current_label,
        "noise_band": args.noise_band,
        "counts": counts,
        "largest_improvements": sorted(
            rows, key=lambda row: row["speedup_delta"], reverse=True
        )[:5],
        "largest_regressions": sorted(
            rows, key=lambda row: row["speedup_delta"]
        )[:5],
    }
    (args.outdir / "comparison_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
