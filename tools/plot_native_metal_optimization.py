#!/usr/bin/env python3
"""Plot the matched generic-MLX versus optimized native-Metal crossover."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--before-label", default="generic MLX GPU (before)")
    parser.add_argument("--after-label", default="native Metal + scheduling (current)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    before_by_width = {row["width"]: row for row in before["records"]}
    after_by_width = {row["width"]: row for row in after["records"]}
    widths = sorted(set(before_by_width) & set(after_by_width))

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    for axis, framework, title in (
        (axes[0], "qiskit", "Qiskit Aer CPU / MettleQ GPU"),
        (axes[1], "pennylane", "Lightning CPU / MettleQ GPU"),
    ):
        historical = [
            before_by_width[width][framework]["cpu_over_mettleq"]
            for width in widths
        ]
        current = [
            after_by_width[width][framework]["cpu_over_mettleq"]
            for width in widths
        ]
        axis.plot(
            widths, historical, marker="o", linewidth=2,
            color="#9aa4b2", label=args.before_label,
        )
        axis.plot(
            widths, current, marker="o", linewidth=2.5,
            color="#6558f5", label=args.after_label,
        )
        axis.axhline(1.0, color="#26354a", linestyle="--", linewidth=1)
        axis.fill_between(
            widths, 1.0, current, where=[value >= 1 for value in current],
            color="#61bd88", alpha=0.16,
        )
        axis.set_title(title)
        axis.set_xlabel("Qubits")
        axis.set_ylabel("CPU / MettleQ (above 1 means MettleQ wins)")
        axis.set_xticks(widths)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "MettleQ optimization impact on the same Apple M3 Pro full-state contract"
    )
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "schema_version": 1,
        "before": str(Path(args.before)),
        "after": str(Path(args.after)),
        "common_widths": widths,
        "interpretation": (
            "Ratios compare frozen runs on the same Mac and full-state contract; "
            "above one favors MettleQ. Small-width native dispatch can cost more, "
            "while fused large states amortize it."
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
