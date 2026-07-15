#!/usr/bin/env python3
"""Compare two revisions using drift-balanced A/B campaign pairs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Dict, List


def _load(path: Path) -> Dict[str, dict]:
    with path.open(newline="") as file:
        return {row["benchmark"]: row for row in csv.DictReader(file)}


def _balanced(a: dict, b: dict, field: str) -> float:
    return math.sqrt(float(a[field]) * float(b[field]))


def _geometric_percent(changes: List[float]) -> float:
    return 100.0 * (
        math.exp(sum(math.log1p(value / 100.0) for value in changes)
                 / len(changes)) - 1.0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-a", type=Path, required=True)
    parser.add_argument("--baseline-b", type=Path, required=True)
    parser.add_argument("--current-a", type=Path, required=True)
    parser.add_argument("--current-b", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--current-label", default="current")
    parser.add_argument("--noise-band-percent", type=float, default=3.0)
    args = parser.parse_args()

    campaigns = [
        _load(args.baseline_a),
        _load(args.baseline_b),
        _load(args.current_a),
        _load(args.current_b),
    ]
    benchmark_sets = [set(campaign) for campaign in campaigns]
    if not benchmark_sets[0] or any(
        items != benchmark_sets[0] for items in benchmark_sets[1:]
    ):
        parser.error("all four summaries must contain the same workloads")
    baseline_a, baseline_b, current_a, current_b = campaigns

    rows = []
    for benchmark in sorted(benchmark_sets[0]):
        base_pure = _balanced(
            baseline_a[benchmark], baseline_b[benchmark], "pure_mean_ms"
        )
        base_metal = _balanced(
            baseline_a[benchmark], baseline_b[benchmark], "metal_mean_ms"
        )
        current_pure = _balanced(
            current_a[benchmark], current_b[benchmark], "pure_mean_ms"
        )
        current_metal = _balanced(
            current_a[benchmark], current_b[benchmark], "metal_mean_ms"
        )
        metal_change = 100.0 * (current_metal / base_metal - 1.0)
        pure_change = 100.0 * (current_pure / base_pure - 1.0)
        normalized_change = 100.0 * (
            (current_metal / current_pure) / (base_metal / base_pure) - 1.0
        )
        if metal_change < -args.noise_band_percent:
            verdict = "faster"
        elif metal_change > args.noise_band_percent:
            verdict = "slower"
        else:
            verdict = "within_band"
        rows.append({
            "benchmark": benchmark,
            "baseline_pure_ms_balanced": base_pure,
            "current_pure_ms_balanced": current_pure,
            "pure_change_percent": pure_change,
            "baseline_metal_ms_balanced": base_metal,
            "current_metal_ms_balanced": current_metal,
            "metal_change_percent": metal_change,
            "pure_normalized_metal_change_percent": normalized_change,
            "verdict": verdict,
        })

    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "revision_comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metal_changes = [row["metal_change_percent"] for row in rows]
    pure_changes = [row["pure_change_percent"] for row in rows]
    normalized = [row["pure_normalized_metal_change_percent"] for row in rows]
    summary = {
        "schema_version": 1,
        "method": (
            "geometric mean of campaign A and B mean wall times for each "
            "revision and workload"
        ),
        "campaign_order": "current / baseline / baseline / current",
        "baseline_label": args.baseline_label,
        "current_label": args.current_label,
        "workloads": len(rows),
        "noise_band_percent": args.noise_band_percent,
        "metal_change_percent": {
            "median": statistics.median(metal_changes),
            "geometric_mean": _geometric_percent(metal_changes),
        },
        "pure_change_percent": {
            "median": statistics.median(pure_changes),
            "geometric_mean": _geometric_percent(pure_changes),
        },
        "pure_normalized_metal_change_percent": {
            "median": statistics.median(normalized),
            "geometric_mean": _geometric_percent(normalized),
        },
        "counts": {
            verdict: sum(row["verdict"] == verdict for row in rows)
            for verdict in ("faster", "within_band", "slower")
        },
        "largest_slowdowns": sorted(
            rows, key=lambda row: row["metal_change_percent"], reverse=True
        )[:5],
        "largest_speedups": sorted(
            rows, key=lambda row: row["metal_change_percent"]
        )[:5],
    }
    (args.outdir / "revision_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
