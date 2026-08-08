import numpy as np
import pytest

import mettleq.planning as planning
from mettleq.device import Device
from mettleq.gates import CNOT, H
from mettleq.integrations import _common
from mettleq.mps_accuracy import MPSAccuracyError
from mettleq.mps_state import MPSOptions


def _local_chain(n_qubits):
    return [
        {"name": "CNOT", "wires": [wire, wire + 1], "parameters": []}
        for wire in range(n_qubits - 1)
    ]


def test_automatic_planner_uses_cpu_below_and_gpu_above_crossover(monkeypatch):
    monkeypatch.setattr(planning, "_gpu_available", lambda: True)
    small = planning.select_execution(
        5, [], statevector_gpu_min_qubits=8
    )
    large = planning.select_execution(
        8, [], statevector_gpu_min_qubits=8
    )

    assert small.selected_method == "statevector"
    assert small.selected_device == "cpu"
    assert large.selected_device == "gpu"
    assert "not combined into a speed claim" in large.cpu_gpu_policy


def test_default_statevector_crossover_matches_complete_sdk_evidence(monkeypatch):
    monkeypatch.setattr(planning, "_gpu_available", lambda: True)

    assert planning.DEFAULT_STATEVECTOR_GPU_MIN_QUBITS == 16
    assert planning.select_execution(14, []).selected_device == "cpu"
    assert planning.select_execution(16, []).selected_device == "gpu"


def test_automatic_mps_requires_opt_in_and_conservative_compatibility(monkeypatch):
    monkeypatch.setattr(planning, "_gpu_available", lambda: False)
    compatible = planning.select_execution(
        12,
        _local_chain(12),
        allow_approximation=True,
        automatic_mps_min_qubits=12,
    )
    exact = planning.select_execution(
        12,
        _local_chain(12),
        allow_approximation=False,
        automatic_mps_min_qubits=12,
    )
    nonlocal_circuit = planning.select_execution(
        12,
        [{"name": "CNOT", "wires": [0, 11], "parameters": []}],
        allow_approximation=True,
        automatic_mps_min_qubits=12,
    )

    assert compatible.selected_method == "matrix_product_state"
    assert compatible.mps_svd_device == "cpu"
    assert exact.selected_method == "statevector"
    assert nonlocal_circuit.selected_method == "statevector"
    assert "contains_non_adjacent_two_qubit_operation" in (
        nonlocal_circuit.mps_compatibility_reasons
    )


def test_explicit_method_and_device_alias_validation(monkeypatch):
    monkeypatch.setattr(planning, "_gpu_available", lambda: False)
    selection = planning.select_execution(
        4, [], method="mps", device="cpu"
    )
    assert selection.selected_method == "matrix_product_state"
    assert selection.selected_device == "cpu"
    with pytest.raises(ValueError, match="method must be"):
        planning.select_execution(4, [], method="hybrid")
    with pytest.raises(ValueError, match="device must be"):
        planning.select_execution(4, [], device="neural-engine")


def test_statevector_cpu_and_gpu_have_numerical_parity():
    if not planning._gpu_available():
        pytest.skip("Apple Metal GPU is unavailable")
    operations = [
        {"name": "H", "wires": [0], "parameters": []},
        {"name": "CNOT", "wires": [0, 1], "parameters": []},
        {"name": "RY", "wires": [2], "parameters": [0.37]},
    ]
    cpu = _common.execute_operations(
        3, operations, method="statevector", execution_device="cpu"
    )
    gpu = _common.execute_operations(
        3, operations, method="statevector", execution_device="gpu"
    )
    assert np.allclose(
        _common.statevector_numpy(cpu),
        _common.statevector_numpy(gpu),
        rtol=0.0,
        atol=2e-6,
    )


def test_gpu_request_for_mps_is_rejected():
    with pytest.raises(ValueError, match="MPS is CPU-only"):
        planning.select_execution(3, [], method="mps", device="gpu")

    with pytest.raises(ValueError, match="MPS is CPU-only"):
        Device(3, backend="mps", execution_device="gpu")


def test_mps_sampling_and_marginals_do_not_materialize_dense_state(monkeypatch):
    simulator = Device(
        24,
        backend="mps",
        execution_device="cpu",
        mps_opts=MPSOptions(dmax=8, eps=1e-12),
    ).sim
    simulator.apply_single(H(), 0)
    for wire in range(23):
        simulator.apply_two(CNOT(), wire, wire + 1)
    monkeypatch.setattr(
        simulator,
        "to_statevector",
        lambda: (_ for _ in ()).throw(AssertionError("dense state requested")),
    )

    probabilities = simulator.probabilities_array([0, 23])
    samples = simulator.sample_array(
        64, [0, 23], rng=np.random.default_rng(7)
    )
    assert np.allclose(probabilities, [0.5, 0.0, 0.0, 0.5], atol=2e-6)
    assert samples.shape == (64, 2)
    assert np.all(samples[:, 0] == samples[:, 1])


def test_mps_truncation_diagnostics_make_approximation_visible():
    simulator = Device(
        2,
        backend="mps",
        execution_device="cpu",
        mps_opts=MPSOptions(dmax=1, eps=0.0),
    ).sim
    simulator.apply_single(H(), 0)
    simulator.apply_two(CNOT(), 0, 1)
    diagnostics = simulator.truncation_diagnostics()

    assert diagnostics["events"] == 1
    assert diagnostics["truncated"] is True
    assert diagnostics["configured_max_bond_dimension"] == 1
    assert diagnostics["current_bond_dimension_max"] == 1
    assert diagnostics["maximum_bond_dimension_reached"] == 1
    assert diagnostics["local_discarded_weight_sum"] > 0.0
    assert diagnostics["approximation_warning"] is not None


def test_lookahead_routing_reduces_nonlocal_swaps_and_preserves_state():
    operations = [
        {"name": "H", "wires": [wire], "parameters": []}
        for wire in range(8)
    ]
    operations.extend(
        {
            "name": "ZZPHASE",
            "wires": [first, second],
            "parameters": [0.19],
        }
        for first in range(8)
        for second in range(first + 1, 8)
    )
    restored = _common.execute_operations(
        8,
        operations,
        method="matrix_product_state",
        execution_device="cpu",
        mps_max_bond_dimension=256,
        mps_truncation_threshold=0.0,
        mps_routing_strategy="restore",
    )
    routed = _common.execute_operations(
        8,
        operations,
        method="matrix_product_state",
        execution_device="cpu",
        mps_max_bond_dimension=256,
        mps_truncation_threshold=0.0,
        mps_routing_strategy="lookahead",
    )
    restored_diagnostics = restored.sim.truncation_diagnostics()
    routed_diagnostics = routed.sim.truncation_diagnostics()

    np.testing.assert_allclose(
        routed.sim.to_statevector(),
        restored.sim.to_statevector(),
        atol=8e-5,
        rtol=0.0,
    )
    assert restored_diagnostics["routing_swaps"] == (
        restored_diagnostics["routing_naive_restore_swaps"]
    )
    assert routed_diagnostics["routing_swaps"] < (
        routed_diagnostics["routing_naive_restore_swaps"]
    )
    assert routed_diagnostics["svd_calls"] < restored_diagnostics["svd_calls"]


def test_routing_preflight_refuses_a_swap_increase_for_grid_order():
    side = 6
    operations = []
    for row in range(side):
        for column in range(side - 1):
            first = row * side + column
            operations.append(
                {
                    "name": "ZZPHASE",
                    "wires": [first, first + 1],
                    "parameters": [0.19],
                }
            )
    for row in range(side - 1):
        for column in range(side):
            first = row * side + column
            operations.append(
                {
                    "name": "ZZPHASE",
                    "wires": [first, first + side],
                    "parameters": [0.2],
                }
            )
    device = _common.execute_operations(
        side * side,
        operations,
        method="matrix_product_state",
        execution_device="cpu",
        mps_max_bond_dimension=8,
        mps_routing_strategy="lookahead",
    )
    diagnostics = device.sim.truncation_diagnostics()
    assert diagnostics["routing_effective_strategy"] == "restore"
    assert diagnostics["routing_planned_lookahead_swaps"] < (
        diagnostics["routing_planned_restore_swaps"]
    )
    assert diagnostics["routing_swaps"] == (
        diagnostics["routing_naive_restore_swaps"]
    )
    assert diagnostics["routing_final_restore_swaps"] == 0


def test_routing_preflight_short_circuits_an_all_adjacent_schedule():
    simulator = Device(
        100,
        backend="mps",
        execution_device="cpu",
        mps_opts=MPSOptions(dmax=4),
    ).sim
    report = simulator.prepare_routing(
        [(wire, wire + 1) for wire in range(99)]
    )
    assert report == {
        "configured_strategy": "lookahead",
        "effective_strategy": "restore",
        "selection_reason": "all_two_qubit_gates_are_adjacent",
        "planned_lookahead_swaps": 0,
        "planned_restore_swaps": 0,
    }


def test_mps_accuracy_error_policy_rejects_excessive_local_loss():
    operations = [
        {"name": "H", "wires": [0], "parameters": []},
        {"name": "CNOT", "wires": [0, 1], "parameters": []},
    ]
    with pytest.raises(MPSAccuracyError, match="threshold exceeded"):
        _common.execute_operations(
            2,
            operations,
            method="matrix_product_state",
            execution_device="cpu",
            mps_max_bond_dimension=1,
            mps_accuracy_policy="error",
            mps_max_relative_discarded_weight=1e-3,
        )


def test_mps_dense_state_request_applies_statevector_preflight(monkeypatch):
    device = Device(2, backend="mps", execution_device="cpu")

    def refuse(*args, **kwargs):
        raise MemoryError("preflight sentinel")

    monkeypatch.setattr(_common, "require_statevector_preflight", refuse)
    with pytest.raises(MemoryError, match="preflight sentinel"):
        _common.statevector_numpy(device)
