#!/usr/bin/env python3
"""Compare matched MettleQ GPU medians while keeping accuracy a hard gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--maximum-regression", type=float, default=0.15)
    parser.add_argument("--accuracy-atol", type=float, default=5e-6)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    current = json.loads(args.current.read_text())
    base_rows = {int(row["width"]): row for row in baseline["records"]}
    failures = []
    comparisons = []
    for row in current["records"]:
        width = int(row["width"])
        if width not in base_rows:
            continue
        old = base_rows[width]
        current_ms = float(row["qiskit"]["mettleq_gpu_ms"])
        baseline_ms = float(old["qiskit"]["mettleq_gpu_ms"])
        change = current_ms / baseline_ms - 1.0
        error = max(float(row["qiskit"]["max_amplitude_error"]),
                    float(row["pennylane"]["max_amplitude_error"]))
        comparisons.append({"width": width, "baseline_ms": baseline_ms,
                            "current_ms": current_ms, "change": change,
                            "maximum_error": error})
        if change > args.maximum_regression:
            failures.append(f"{width}q latency regressed {change:.1%}")
        if error > args.accuracy_atol:
            failures.append(f"{width}q error {error:.3e} exceeds {args.accuracy_atol:.3e}")
    print(json.dumps({"comparisons": comparisons, "failures": failures}, indent=2))
    if not comparisons:
        print("No matched widths; regression comparison is inconclusive.")
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
