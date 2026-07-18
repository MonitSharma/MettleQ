#!/usr/bin/env python3
"""Turn tutorial validation records into a concise correctness audit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _declared_tolerance(check: str) -> float | None:
    match = re.search(r"atol\s*=\s*([0-9.eE+-]+)", check)
    if match:
        return float(match.group(1))
    match = re.search(r"(?:distance|TVD)\s*<=\s*([0-9.eE+-]+)", check, re.I)
    return float(match.group(1)) if match else None


def _primary_error(record: dict[str, Any]) -> float | None:
    metrics = record["metrics"]
    if "widths" in metrics:
        return max(float(row["error"]) for row in metrics["widths"])
    if "per_measurement_max_errors" in metrics:
        return max(float(value) for value in metrics["per_measurement_max_errors"])
    preferred = (
        "max_amplitude_error",
        "max_expectation_error",
        "max_probability_error",
        "max_energy_error",
        "max_cost_error",
        "max_observable_error",
        "max_kernel_error",
        "max_value_or_gradient_error",
        "max_trace_error",
        "trace_error",
        "prediction_error",
        "analytic_receiver_error",
        "expectation_error",
        "tvd",
    )
    values = [float(metrics[key]) for key in preferred if key in metrics]
    return max(values) if values else None


def _classification(record: dict[str, Any]) -> str:
    check = record["check"].lower()
    if "finite-shot" in check or "expected mode" in check:
        return "statistical agreement (independent random samples)"
    if "mps" in check:
        return "approximate MPS, independently checked with convergence evidence"
    if record["exact_match"]:
        return "bit-for-bit match"
    return "numerically equivalent within an explicit tolerance"


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for record in payload["results"]:
        tolerance = _declared_tolerance(record["check"])
        error = _primary_error(record)
        ratio = error / tolerance if error is not None and tolerance else None
        rows.append(
            {
                "notebook": record["notebook"],
                "passed": bool(record["passed"]),
                "exact_match": bool(record["exact_match"]),
                "classification": _classification(record),
                "primary_error": error,
                "declared_tolerance": tolerance,
                "error_over_tolerance": ratio,
                "check": record["check"],
            }
        )
    ratios = [row["error_over_tolerance"] for row in rows if row["error_over_tolerance"]]
    return {
        "schema_version": 1,
        "all_passed": all(row["passed"] for row in rows),
        "notebooks": len(rows),
        "bit_for_bit_matches": sum(row["exact_match"] for row in rows),
        "tolerance_validated_matches": sum(not row["exact_match"] for row in rows),
        "largest_primary_error_over_tolerance": max(ratios, default=None),
        "rows": rows,
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Tutorial correctness audit",
        "",
        f"**Result: {'PASS' if result['all_passed'] else 'FAIL'} — "
        f"{result['notebooks']} notebooks validated.**",
        "",
        "`exact_match` means bit-for-bit equality, which is stricter than numerical "
        "correctness. MettleQ's Apple GPU statevector commonly uses complex64 while "
        "the CPU references use complex128, so harmless last-bit differences are "
        "expected. Finite-shot results use independent random samples and therefore "
        "must be compared statistically. MPS is approximate and additionally needs "
        "bond-dimension convergence evidence.",
        "",
        f"- Bit-for-bit matches: {result['bit_for_bit_matches']}",
        f"- Tolerance/statistically validated matches: {result['tolerance_validated_matches']}",
        f"- Largest primary-error/tolerance fraction: {result['largest_primary_error_over_tolerance']:.3f}",
        "",
        "| Notebook | Validation class | Exact | Primary error | Tolerance | Fraction used |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        error = "—" if row["primary_error"] is None else f"{row['primary_error']:.2e}"
        tolerance = "—" if row["declared_tolerance"] is None else f"{row['declared_tolerance']:.2e}"
        ratio = "—" if row["error_over_tolerance"] is None else f"{row['error_over_tolerance']:.3f}"
        lines.append(
            f"| `{row['notebook']}` | {row['classification']} | "
            f"{'yes' if row['exact_match'] else 'no'} | {error} | {tolerance} | {ratio} |"
        )
    lines.extend(
        [
            "",
            "A passing tolerance is evidence for the tested circuits and methods, not "
            "a proof that every possible circuit is correct. The project test suite, "
            "full-state comparisons, MPS convergence runs, and benchmark accuracy gates "
            "provide complementary coverage.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tutorials/results.json")
    parser.add_argument("--json-output", default="tutorials/correctness_audit.json")
    parser.add_argument("--markdown-output", default="tutorials/correctness_audit.md")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    result = audit(payload)
    Path(args.json_output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_markdown(Path(args.markdown_output), result)
    print(json.dumps({key: result[key] for key in result if key != "rows"}, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
