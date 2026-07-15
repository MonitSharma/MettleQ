"""Capability, execution-plan, and synchronized-dispatch evidence tests."""
import math

import mlx.core as mx
import pytest

from mlxq.device import Device
from mlxq.sim import StateVectorSimulator
import mlxq.execution as execution
from mlxq.execution import (
    METAL_CHECKPOINT_BUDGET_ENV,
    STATEVECTOR_UNSAFE_OVERRIDE_ENV,
    StatevectorMemoryError,
    clear_metal_capability_cache,
    metal_checkpoint_policy,
    metal_runtime_enabled,
    metal_runtime_status,
    state_memory_estimate,
    statevector_preflight,
)


@pytest.fixture(autouse=True)
def _reset_metal_capability_cache():
    clear_metal_capability_cache()
    yield
    clear_metal_capability_cache()


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


def _set_memory_limits(monkeypatch, *, max_buffer, working_set):
    monkeypatch.setattr(
        execution,
        "_device_info",
        lambda: {
            "device_name": "Constrained test GPU",
            "architecture": "test",
            "max_buffer_length": max_buffer,
            "max_recommended_working_set_size": working_set,
        },
    )
    clear_metal_capability_cache()


def test_statevector_preflight_refuses_before_array_allocation(monkeypatch):
    # A 10-qubit complex64 state is 8192 bytes; an out-of-place operation has
    # a 16384-byte lower bound. Both synthetic limits reject it.
    _set_memory_limits(monkeypatch, max_buffer=4096, working_set=12000)
    report = statevector_preflight(10)
    assert not report["allowed"]
    assert report["decision"] == "refused"
    assert report["failure_reasons"] == [
        "single_state_exceeds_max_buffer_length",
        "two_state_lower_bound_exceeds_recommended_working_set",
    ]
    assert report["cost_model"]["state_bytes"] == 8192
    assert report["cost_model"]["minimum_input_plus_output_bytes"] == 16384

    with pytest.raises(StatevectorMemoryError) as caught:
        StateVectorSimulator(10)
    assert caught.value.report == report
    assert "before allocation" in str(caught.value)
    assert STATEVECTOR_UNSAFE_OVERRIDE_ENV in str(caught.value)


def test_statevector_preflight_override_is_explicit_and_observable(monkeypatch):
    _set_memory_limits(monkeypatch, max_buffer=4096, working_set=12000)
    report = statevector_preflight(10, allow_unsafe=True)
    assert report["allowed"]
    assert report["overridden"]
    assert report["decision"] == "allowed_with_unsafe_override"
    assert report["override"]["source"] == "constructor_argument"
    assert report["failure_reasons"]

    simulator = StateVectorSimulator(10, allow_unsafe_statevector=True)
    assert simulator.preflight["overridden"]
    assert simulator.state.shape == (1 << 10,)


def test_statevector_preflight_environment_fails_closed(monkeypatch):
    _set_memory_limits(monkeypatch, max_buffer=4096, working_set=12000)
    monkeypatch.setenv(STATEVECTOR_UNSAFE_OVERRIDE_ENV, "typo")
    with pytest.raises(ValueError, match=STATEVECTOR_UNSAFE_OVERRIDE_ENV):
        statevector_preflight(10)

    monkeypatch.setenv(STATEVECTOR_UNSAFE_OVERRIDE_ENV, "1")
    assert statevector_preflight(10)["overridden"]
    # An explicit False takes precedence over an enabled environment override.
    assert not statevector_preflight(10, allow_unsafe=False)["allowed"]


def test_statevector_preflight_reports_unverified_missing_limits(monkeypatch):
    _set_memory_limits(monkeypatch, max_buffer=0, working_set=0)
    report = statevector_preflight(20)
    assert report["allowed"]
    assert report["decision"] == "allowed_with_unverified_device_limits"
    assert report["checks"] == {
        "single_state_within_max_buffer_length": None,
        "two_state_lower_bound_within_recommended_working_set": None,
    }
    with pytest.raises(ValueError, match="complex64"):
        statevector_preflight(4, dtype="complex128")


def test_static_capabilities_are_cached_but_policy_remains_dynamic(monkeypatch):
    calls = {"version": 0, "device": 0, "metal": 0}

    def package_version(name):
        calls["version"] += 1
        return "test-version"

    def device_info():
        calls["device"] += 1
        return {
            "device_name": "Test GPU",
            "architecture": "test-arch",
            "max_buffer_length": 1 << 34,
            "max_recommended_working_set_size": 1 << 35,
        }

    def metal_available():
        calls["metal"] += 1
        return True

    monkeypatch.setattr(execution, "_package_version", package_version)
    monkeypatch.setattr(execution, "_device_info", device_info)
    monkeypatch.setattr(execution, "_metal_available", metal_available)
    monkeypatch.setenv("MLXQ_METAL_KERNELS", "1")

    first = metal_runtime_status(20)
    monkeypatch.setenv("MLXQ_METAL_KERNELS", "0")
    second = metal_runtime_status(20)

    assert calls == {"version": 1, "device": 1, "metal": 1}
    assert first["policy"] == "enabled"
    assert second["policy"] == "disabled"
    assert first["capability_cache"]["info"]["misses"] == 1
    assert second["capability_cache"]["info"]["hits"] >= 1

    clear_metal_capability_cache()
    metal_runtime_status(20)
    assert calls == {"version": 2, "device": 2, "metal": 2}


def test_fast_selector_keeps_selected_device_and_other_checks_live(monkeypatch):
    monkeypatch.setenv("MLXQ_METAL_KERNELS", "1")
    monkeypatch.setattr(execution, "_default_device", lambda: "Device(gpu, 0)")
    report = metal_runtime_status(4)
    assert metal_runtime_enabled(4) == report["enabled"]

    monkeypatch.setattr(execution, "_default_device", lambda: "Device(cpu, 0)")
    assert not metal_runtime_enabled(4)
    assert not metal_runtime_status(4)["enabled"]

    monkeypatch.setattr(execution, "_default_device", lambda: "Device(gpu, 0)")
    monkeypatch.setenv("MLXQ_DENSE_ONLY", "1")
    assert not metal_runtime_enabled(4)
    monkeypatch.delenv("MLXQ_DENSE_ONLY")
    assert not metal_runtime_enabled(4, dtype="complex128")
    assert not metal_runtime_enabled(32)


def test_checkpoint_budget_policy_is_opt_in_and_rejects_invalid_values(monkeypatch):
    monkeypatch.delenv(METAL_CHECKPOINT_BUDGET_ENV, raising=False)
    assert not metal_checkpoint_policy()["configured"]

    monkeypatch.setenv(METAL_CHECKPOINT_BUDGET_ENV, "1.5")
    policy = metal_checkpoint_policy()
    assert policy["configured"]
    assert policy["budget_bytes"] == int(1.5 * 1024 * 1024)
    assert policy["source"] == "environment"

    monkeypatch.setenv(METAL_CHECKPOINT_BUDGET_ENV, "not-a-size")
    with pytest.raises(ValueError, match=METAL_CHECKPOINT_BUDGET_ENV):
        metal_checkpoint_policy()
    with pytest.raises(ValueError, match="non-negative"):
        metal_checkpoint_policy(-1)
    with pytest.raises(TypeError, match="integer byte count"):
        metal_checkpoint_policy(1.5)


def test_checkpointing_occurs_between_fused_layers_and_preserves_state(monkeypatch):
    monkeypatch.setenv("MLXQ_METAL_KERNELS", "1")
    capability = metal_runtime_status(4)
    if not capability["enabled"]:
        pytest.skip(capability["reason"])

    ops = _qft_ops(4)
    reference = Device(4)
    reference.execute(ops)
    reference.synchronize()

    checkpointed = Device(4, metal_checkpoint_budget_bytes=512)
    checkpointed.execute(ops, report=True)
    plan = checkpointed.last_execution_plan
    policy = plan["checkpointing"]
    assert policy["enabled"]
    assert policy["predicted_checkpoint_count"] == 1
    assert policy["actual_checkpoint_count"] == 1
    assert policy["actual_checkpoints"][0]["boundary"] == (
        "before_optimized_operation"
    )
    assert policy["actual_checkpoints"][0]["optimized_operation_index"] == 2
    assert policy["actual_checkpoints"][0]["estimated_passes_evaluated"] == 2
    assert policy["pending_custom_passes_after_graph_build"] == 1
    assert plan["execution_status"] == "checkpointed_lazy_graph_built"

    checkpointed.synchronize()
    error = mx.max(mx.abs(reference.sim.state - checkpointed.sim.state))
    mx.eval(error)
    assert float(error.item()) <= 5e-6
    assert policy["pending_custom_passes_after_synchronize"] == 0


def test_oversized_fused_layer_streams_between_custom_launches(monkeypatch):
    monkeypatch.setenv("MLXQ_METAL_KERNELS", "1")
    capability = metal_runtime_status(4)
    if not capability["enabled"]:
        pytest.skip(capability["reason"])

    ops = [
        {"name": "RX", "wires": [wire], "parameters": [0.2]}
        for wire in range(4)
    ]
    dev = Device(4, metal_checkpoint_budget_bytes=256)
    dev.execute(ops, report=True)
    plan = dev.last_execution_plan
    checkpoints = plan["checkpointing"]["actual_checkpoints"]
    assert plan["matched_structured_patterns"] == {"uniform_rx_layer": 1}
    assert plan["expected_custom_kernel_launches"] == 2
    assert plan["schema_version"] == 4
    assert plan["checkpointing"][
        "max_pending_custom_passes_per_streamed_chunk"
    ] == 1
    assert plan["checkpointing"]["predicted_checkpoint_count"] == 1
    assert len(checkpoints) == 1
    assert checkpoints[0]["boundary"] == "within_optimized_operation"
    assert checkpoints[0]["optimized_operation_index"] == 0
    assert checkpoints[0]["within_layer_launch_index"] == 1
    assert checkpoints[0]["within_layer_launch_count"] == 2
    assert checkpoints[0]["estimated_passes_evaluated"] == 1
    assert checkpoints[0]["reason"] == "multi_launch_layer_streaming_budget"
    assert plan["checkpointing"]["pending_custom_passes_after_graph_build"] == 1

    unreported = Device(4, metal_checkpoint_budget_bytes=256)
    unreported.execute(ops, report=False)
    assert unreported.last_execution_plan is None
    assert unreported._pending_custom_passes == 1
    assert unreported._pending_custom_io_bytes == 256


def _streaming_ops(kind, n):
    if kind == "uniform_u2":
        return [{"name": "H", "wires": [wire]} for wire in range(n)]
    if kind == "per_qubit_u2":
        ops = []
        for wire in range(n):
            ops.append({
                "name": "RY",
                "wires": [wire],
                "parameters": [0.1 + 0.03 * wire],
            })
            ops.append({
                "name": "RZ",
                "wires": [wire],
                "parameters": [-0.2 + 0.02 * wire],
            })
        return ops
    gate = "XXPHASE" if kind == "xx" else "YYPHASE"
    return [
        {
            "name": gate,
            "wires": [wire, wire + 1],
            "parameters": [0.07],
        }
        for wire in range(n - 1)
    ]


@pytest.mark.parametrize(
    ("kind", "family", "expected_launches"),
    [
        ("uniform_u2", "uniform_single_qubit_layer", 3),
        ("per_qubit_u2", "per_qubit_single_qubit_layer", 3),
        ("xx", "xx_layer", 5),
        ("yy", "yy_layer", 7),
    ],
)
def test_streamable_layer_families_match_pure_mlx_and_plan(
    monkeypatch, kind, family, expected_launches
):
    n = 5
    ops = _streaming_ops(kind, n)
    monkeypatch.setenv("MLXQ_METAL_KERNELS", "0")
    reference = Device(n)
    reference.execute(ops)
    reference.synchronize()

    monkeypatch.setenv("MLXQ_METAL_KERNELS", "1")
    capability = metal_runtime_status(n)
    if not capability["enabled"]:
        pytest.skip(capability["reason"])
    # A complex64 n=5 input/output pass is exactly 512 bytes.  This budget
    # therefore evaluates after every launch except the final pending launch.
    candidate = Device(n, metal_checkpoint_budget_bytes=512)
    candidate.execute(ops, report=True)
    plan = candidate.last_execution_plan
    policy = plan["checkpointing"]
    assert plan["matched_structured_patterns"] == {family: 1}
    assert plan["expected_custom_kernel_launches"] == expected_launches
    assert policy["predicted_checkpoint_count"] == expected_launches - 1
    assert policy["actual_checkpoint_count"] == expected_launches - 1
    assert [
        event["within_layer_launch_index"]
        for event in policy["actual_checkpoints"]
    ] == list(range(1, expected_launches))
    assert all(
        event["boundary"] == "within_optimized_operation"
        for event in policy["actual_checkpoints"]
    )
    assert policy["pending_custom_passes_after_graph_build"] == 1

    candidate.synchronize()
    error = mx.max(mx.abs(reference.sim.state - candidate.sim.state))
    mx.eval(error)
    assert float(error.item()) <= 5e-6


def test_configured_checkpoint_budget_stays_inactive_without_metal(monkeypatch):
    monkeypatch.setenv("MLXQ_METAL_KERNELS", "0")
    plan = Device(4, metal_checkpoint_budget_bytes=512).explain(_qft_ops(4))
    assert plan["checkpointing"]["configured"]
    assert not plan["checkpointing"]["enabled"]
    assert plan["checkpointing"]["predicted_checkpoint_count"] == 0
    assert "custom Metal is not selected" in plan["checkpointing"]["reason"]


def test_disabled_plan_reports_fallback_without_claiming_dispatch(monkeypatch):
    monkeypatch.setenv("MLXQ_METAL_KERNELS", "0")
    dev = Device(4)
    plan = dev.explain(_qft_ops(4))
    assert plan["selected_custom_kernels"] == []
    assert plan["expected_custom_kernel_launches"] == 0
    assert plan["fallback_reasons"]
    assert plan["dtype"] == "complex64"
    assert plan["selected_device"]
    assert plan["statevector_preflight"] == dev.statevector_preflight
    assert plan["execution_cost_model"]["state_bytes"] == 128
    assert plan["execution_cost_model"]["predicted_custom_launches"] == 0
    assert not plan["execution_cost_model"]["pure_mlx_operations_modeled"]
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
