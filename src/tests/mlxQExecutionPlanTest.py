"""Capability, execution-plan, and synchronized-dispatch evidence tests."""
import math

import pytest

from mlxq.device import Device
from mlxq.execution import metal_runtime_status, state_memory_estimate


def _qft_ops(n):
    ops = []
    for j in range(n):
        ops.append({"name": "H", "wires": [j]})
        for k in range(j + 1, n):
            ops.append({
                "name": "CPHASE",
                "wires": [k, j],
                "parameters": [math.pi / (2 ** (k - j))],
            })
    return ops


def test_metal_policy_is_explicit_and_invalid_values_fail_closed(monkeypatch):
    monkeypatch.delenv("MLXQ_METAL_KERNELS", raising=False)
    off = metal_runtime_status(4)
    assert off["policy"] == "off_by_default"
    assert not off["enabled"]

    monkeypatch.setenv("MLXQ_METAL_KERNELS", "not-a-policy")
    invalid = metal_runtime_status(4)
    assert invalid["policy"] == "invalid"
    assert not invalid["enabled"]
    assert "invalid" in invalid["reason"]


def test_capability_report_exposes_index_and_memory_limits(monkeypatch):
    monkeypatch.setenv("MLXQ_METAL_KERNELS", "1")
    report = metal_runtime_status(32)
    assert report["kernel_constraints"]["index_bits"] == 32
    assert report["kernel_constraints"]["index_qubit_limit"] == 31
    assert not report["checks"]["index_width"]
    assert not report["enabled"]
    memory = state_memory_estimate(10)
    assert memory["state_bytes"] == 8 * (1 << 10)
    assert memory["minimum_input_plus_output_bytes"] == 16 * (1 << 10)


def test_disabled_plan_reports_fallback_without_claiming_dispatch(monkeypatch):
    monkeypatch.setenv("MLXQ_METAL_KERNELS", "0")
    dev = Device(4)
    plan = dev.explain(_qft_ops(4))
    assert plan["selected_custom_kernels"] == []
    assert plan["expected_custom_kernel_launches"] == 0
    assert plan["fallback_reasons"]
    assert plan["dtype"] == "complex64"
    assert plan["selected_device"]
    dev.execute([{"name": "H", "wires": [0]}])
    assert dev.last_execution_plan is None


def test_metal_plan_observes_qft_dispatch_and_synchronized_evaluation(monkeypatch):
    monkeypatch.setenv("MLXQ_METAL_KERNELS", "1")
    capability = metal_runtime_status(4)
    if not capability["enabled"]:
        pytest.skip(capability["reason"])

    dev = Device(4)
    dev.execute(_qft_ops(4), report=True)
    plan = dev.last_execution_plan
    assert plan is not None
    assert plan["matched_structured_patterns"] == {"qft_stage": 3}
    assert plan["selected_concrete_kernels"] == ["mlxq_qft_stage_gen"]
    assert plan["expected_custom_kernel_launches"] == 3
    assert len(plan["observed_custom_dispatches"]) == 3
    assert plan["execution_status"] == "lazy_graph_built"

    dev.synchronize()
    assert plan["execution_status"] == "evaluated"
    assert plan["synchronized"] is True
    assert plan["compilation_cache"]["status"] == (
        "compiled_or_loaded_and_synchronously_executed"
    )


def test_dense_ablation_and_mps_explain_why_custom_kernels_are_not_used(monkeypatch):
    monkeypatch.setenv("MLXQ_METAL_KERNELS", "1")
    monkeypatch.setenv("MLXQ_DENSE_ONLY", "1")
    dense_plan = Device(3).explain(_qft_ops(3))
    assert dense_plan["selected_custom_kernels"] == []
    assert "dense_ablation_disabled" in dense_plan["capabilities"]["failed_checks"]

    monkeypatch.delenv("MLXQ_DENSE_ONLY")
    mps_plan = Device(3, backend="mps").explain([{"name": "H", "wires": [0]}])
    assert mps_plan["backend"] == "mps"
    assert mps_plan["selected_custom_kernels"] == []
    assert "statevector_backend" in mps_plan["capabilities"]["failed_checks"]
