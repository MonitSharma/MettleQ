"""Accuracy policy and convergence evidence for bounded MPS execution."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


ACCURACY_POLICIES = {"report", "warn", "error"}


class MPSAccuracyError(RuntimeError):
    """Configured MPS accuracy thresholds were exceeded."""


def normalize_accuracy_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized not in ACCURACY_POLICIES:
        raise ValueError("MPS accuracy policy must be report, warn, or error")
    return normalized


def assess_mps_accuracy(
    diagnostics: dict,
    *,
    policy: str = "report",
    max_relative_discarded_weight: Optional[float] = 1e-6,
    max_norm_error: Optional[float] = 1e-5,
) -> dict:
    """Classify MPS telemetry against explicit, local trust thresholds."""
    policy = normalize_accuracy_policy(policy)
    if max_relative_discarded_weight is not None and (
        max_relative_discarded_weight < 0.0
    ):
        raise ValueError("MPS discarded-weight threshold must be non-negative")
    if max_norm_error is not None and max_norm_error < 0.0:
        raise ValueError("MPS norm-error threshold must be non-negative")

    relative_max = float(diagnostics["relative_discarded_weight_max"])
    norm_error = abs(float(diagnostics["state_norm"]) - 1.0)
    violations = []
    if (
        max_relative_discarded_weight is not None
        and relative_max > max_relative_discarded_weight
    ):
        violations.append(
            {
                "metric": "relative_discarded_weight_max",
                "observed": relative_max,
                "threshold": float(max_relative_discarded_weight),
            }
        )
    if max_norm_error is not None and norm_error > max_norm_error:
        violations.append(
            {
                "metric": "state_norm_error",
                "observed": norm_error,
                "threshold": float(max_norm_error),
            }
        )
    if violations:
        classification = "threshold_exceeded"
    elif diagnostics["truncated"]:
        classification = "within_configured_local_thresholds"
    else:
        classification = "no_local_truncation_observed"
    return {
        "schema_version": 1,
        "policy": policy,
        "passed": not violations,
        "classification": classification,
        "violations": violations,
        "thresholds": {
            "max_relative_discarded_weight": max_relative_discarded_weight,
            "max_norm_error": max_norm_error,
        },
        "observed": {
            "truncated": bool(diagnostics["truncated"]),
            "relative_discarded_weight_max": relative_max,
            "relative_discarded_weight_sum": float(
                diagnostics["relative_discarded_weight_sum"]
            ),
            "state_norm_error": norm_error,
            "maximum_bond_dimension_reached": int(
                diagnostics["maximum_bond_dimension_reached"]
            ),
        },
        "scope_warning": (
            "Passing local telemetry thresholds is not a global fidelity "
            "proof; use Dmax convergence or an independent reference."
        ),
    }


def enforce_mps_accuracy(report: dict) -> None:
    if report["passed"] or report["policy"] == "report":
        return
    metrics = ", ".join(
        violation["metric"] for violation in report["violations"]
    )
    message = f"MPS accuracy threshold exceeded: {metrics}"
    if report["policy"] == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return
    raise MPSAccuracyError(message)


def _flatten_numeric(value) -> np.ndarray:
    if isinstance(value, (tuple, list)):
        pieces = [_flatten_numeric(item) for item in value]
        return np.concatenate(pieces) if pieces else np.empty(0)
    return np.asarray(value, dtype=np.complex128).reshape(-1)


def _serialize_numeric(value):
    if isinstance(value, tuple):
        return [_serialize_numeric(item) for item in value]
    if isinstance(value, list):
        return [_serialize_numeric(item) for item in value]
    array = np.real_if_close(np.asarray(value))
    return array.tolist()


def build_convergence_report(
    runs: list[dict], *, atol: float = 5e-5
) -> dict:
    """Compare ordered Dmax result arrays and retain per-run telemetry."""
    if atol <= 0.0:
        raise ValueError("MPS convergence tolerance must be positive")
    ordered = sorted(runs, key=lambda run: int(run["dmax"]))
    comparisons = []
    for lower, upper in zip(ordered, ordered[1:]):
        lower_value = _flatten_numeric(lower["value"])
        upper_value = _flatten_numeric(upper["value"])
        if lower_value.shape != upper_value.shape:
            raise ValueError("MPS convergence result shapes do not match")
        delta = float(np.max(np.abs(upper_value - lower_value)))
        comparisons.append(
            {
                "lower_dmax": int(lower["dmax"]),
                "upper_dmax": int(upper["dmax"]),
                "maximum_absolute_delta": delta,
                "within_tolerance": delta <= atol,
            }
        )
    converged = bool(comparisons) and comparisons[-1]["within_tolerance"]
    serialized_runs = []
    for run in ordered:
        serialized_runs.append(
            {
                "dmax": int(run["dmax"]),
                "value": _serialize_numeric(run["value"]),
                "diagnostics": run.get("diagnostics"),
                "accuracy": run.get("accuracy"),
            }
        )
    return {
        "schema_version": 1,
        "atol": float(atol),
        "converged": converged,
        "comparisons": comparisons,
        "runs": serialized_runs,
        "scope_warning": (
            "Successive Dmax agreement is convergence evidence, not an "
            "independent exactness proof."
        ),
    }
