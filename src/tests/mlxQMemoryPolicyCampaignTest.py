"""Regression tests for reviewed memory-policy campaign aggregation."""

import pytest

from tools.memory_policy_campaign import _acceptance, _summarize


def _row(policy, repeat, total_ms, peak_bytes, checkpoints):
    return {
        "workload": "test",
        "qubits": 22,
        "policy": policy,
        "repeat": repeat,
        "budget_bytes": None if policy == "lazy" else 1024,
        "total_ms": total_ms,
        "peak_bytes": peak_bytes,
        "observed_checkpoints": checkpoints,
        "predicted_checkpoints": checkpoints,
    }


def test_campaign_runtime_summary_pairs_arms_before_aggregation():
    rows = [
        _row("lazy", 1, 100.0, 1000, 0),
        _row("adaptive_balanced", 1, 80.0, 500, 1),
        _row("adaptive_minimum", 1, 70.0, 400, 2),
        _row("lazy", 2, 200.0, 1000, 0),
        _row("adaptive_balanced", 2, 220.0, 500, 1),
        _row("adaptive_minimum", 2, 140.0, 400, 2),
    ]
    summary, aggregate = _summarize(rows)
    balanced = next(
        row for row in summary if row["policy"] == "adaptive_balanced"
    )
    # Separate medians are both 150 ms and would incorrectly report 0%.
    assert balanced["runtime_change_percent"] == pytest.approx(0.0)
    # Paired changes are -20% and +10%, whose median is -5%.
    assert balanced["paired_runtime_change_percent_median"] == pytest.approx(-5.0)
    balanced_aggregate = next(
        row for row in aggregate if row["policy"] == "adaptive_balanced"
    )
    assert balanced_aggregate[
        "maximum_cell_paired_runtime_change_percent"
    ] == pytest.approx(-5.0)


def test_campaign_acceptance_gates_amplitude_and_norm_separately():
    rows = [{"predicted_checkpoints": 3, "observed_checkpoints": 3}]
    validation = [{
        "max_amplitude_error": 1e-6,
        "norm_error": 7e-6,
    }]
    accepted = _acceptance(
        rows, validation, 5e-6, 1e-5, unsafe_override_used=False
    )
    assert accepted["amplitude_parity_passed"]
    assert accepted["norm_parity_passed"]
    assert accepted["numerical_parity_passed"]

    rejected = _acceptance(
        rows, validation, 5e-6, 5e-6, unsafe_override_used=False
    )
    assert rejected["amplitude_parity_passed"]
    assert not rejected["norm_parity_passed"]
    assert not rejected["numerical_parity_passed"]
