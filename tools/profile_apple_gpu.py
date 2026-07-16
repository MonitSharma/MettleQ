#!/usr/bin/env python3
"""Synchronized MettleQ custom-Metal inspection campaign.

This is intentionally a profiler, not a benchmark-claim generator.  It records
the first execution in the current process, synchronized warm executions,
MLX allocator memory, host graph-building time, full-state readback time,
dispatch evidence, and pure-MLX numerical parity.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List

import mlx.core as mx

from mettleq.device import Device
from mettleq.execution import metal_memory_snapshot, metal_runtime_status


def _qft_ops(n: int) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for j in range(n):
        ops.append({"name": "H", "wires": [j]})
        for k in range(j + 1, n):
            ops.append({
                "name": "CPHASE",
                "wires": [k, j],
                "parameters": [math.pi / (2 ** (k - j))],
            })
    return ops


def _tfim_ops(n: int, steps: int = 6) -> List[Dict[str, Any]]:
    ops = [{"name": "H", "wires": [q]} for q in range(n)]
    for step in range(steps):
        theta = -0.03 - 0.002 * step
        ops.extend({
            "name": "ZZPHASE", "wires": [q, q + 1], "parameters": [theta]
        } for q in range(n - 1))
        ops.extend({
            "name": "RX", "wires": [q], "parameters": [0.07 + 0.001 * step]
        } for q in range(n))
    return ops


def _affine_ops(n: int) -> List[Dict[str, Any]]:
    ops = [{"name": "H", "wires": [0]}]
    ops.extend({"name": "CNOT", "wires": [q, q + 1]} for q in range(n - 1))
    ops.extend([
        {"name": "X", "wires": [n // 2]},
        {"name": "SWAP", "wires": [1, n - 2]},
    ])
    return ops


WORKLOADS = {
    "qft": _qft_ops,
    "tfim": _tfim_ops,
    "affine": _affine_ops,
}


def _reset_peak_memory() -> None:
    fn = getattr(mx, "reset_peak_memory", None)
    if not callable(fn):
        fn = getattr(getattr(mx, "metal", None), "reset_peak_memory", None)
    if callable(fn):
        fn()


def _clear_allocator_cache() -> None:
    fn = getattr(mx, "clear_cache", None)
    if not callable(fn):
        fn = getattr(getattr(mx, "metal", None), "clear_cache", None)
    if callable(fn):
        fn()


def _run_once(n: int, ops: List[Dict[str, Any]], *, metal: bool) -> Dict[str, Any]:
    os.environ["METTLEQ_METAL_KERNELS"] = "1" if metal else "0"
    dev = Device(n)
    _reset_peak_memory()
    memory_before = metal_memory_snapshot()

    t0 = time.perf_counter_ns()
    dev.execute(ops, report=True)
    t1 = time.perf_counter_ns()
    dev.synchronize()
    t2 = time.perf_counter_ns()

    # Full-state readback is measured separately and is not part of execution.
    readback_start = time.perf_counter_ns()
    host_state = list(dev.sim.state.tolist())
    readback_end = time.perf_counter_ns()
    if len(host_state) != 1 << n:
        raise RuntimeError("state readback returned the wrong length")

    plan = dev.last_execution_plan
    return {
        "graph_build_ms": (t1 - t0) / 1e6,
        "synchronized_evaluation_ms": (t2 - t1) / 1e6,
        "total_execution_ms": (t2 - t0) / 1e6,
        "full_state_readback_ms": (readback_end - readback_start) / 1e6,
        "memory_before": memory_before,
        "memory_after": metal_memory_snapshot(),
        "execution_plan": plan,
    }


def _median(runs: List[Dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(run[key]) for run in runs))


def _validate(n: int, ops: List[Dict[str, Any]]) -> float:
    os.environ["METTLEQ_METAL_KERNELS"] = "0"
    pure = Device(n)
    pure.execute(ops)
    pure.synchronize()
    os.environ["METTLEQ_METAL_KERNELS"] = "1"
    metal = Device(n)
    metal.execute(ops)
    metal.synchronize()
    error = mx.max(mx.abs(pure.sim.state - metal.sim.state))
    mx.eval(error)
    return float(error.item())


def _profile_workload(
    name: str,
    n: int,
    ops: List[Dict[str, Any]],
    repeats: int,
) -> Dict[str, Any]:
    arms: Dict[str, Any] = {}
    for label, metal in (("pure_mlx", False), ("custom_metal", True)):
        gc.collect()
        _clear_allocator_cache()
        first = _run_once(n, ops, metal=metal)
        warm = [_run_once(n, ops, metal=metal) for _ in range(repeats)]
        warm_eval = _median(warm, "synchronized_evaluation_ms")
        plan = first["execution_plan"]
        launches = int(plan.get("expected_custom_kernel_launches", 0))
        state_bytes = int(plan["memory"]["state_bytes"])
        traffic_bytes = 2 * state_bytes * launches
        first_memory = first["memory_after"]
        allocator_peak = int(first_memory.get("peak_bytes") or 0)
        minimum_peak = int(plan["memory"]["minimum_input_plus_output_bytes"])
        bandwidth = (
            traffic_bytes / (warm_eval / 1000.0) / 1e9
            if traffic_bytes and warm_eval > 0 else None
        )
        arms[label] = {
            "first_process_execution": first,
            "warm_runs": warm,
            "summary": {
                "warm_repeats": repeats,
                "warm_graph_build_median_ms": _median(warm, "graph_build_ms"),
                "warm_evaluation_median_ms": warm_eval,
                "warm_total_median_ms": _median(warm, "total_execution_ms"),
                "warm_readback_median_ms": _median(warm, "full_state_readback_ms"),
                "first_minus_warm_evaluation_ms": max(
                    0.0, first["synchronized_evaluation_ms"] - warm_eval
                ),
                "compile_overhead_note": (
                    "upper-bound estimate only: first-process minus warm; MLX "
                    "does not expose per-kernel compiler-cache hit state"
                ),
                "custom_kernel_launches": launches,
                "estimated_statevector_traffic_bytes": traffic_bytes,
                "estimated_effective_bandwidth_gb_s": bandwidth,
                "bandwidth_note": (
                    "algorithmic input+output traffic divided by synchronized "
                    "evaluation time; not a hardware-counter measurement"
                ),
                "first_allocator_peak_bytes": allocator_peak,
                "first_active_after_evaluation_bytes": int(
                    first_memory.get("active_bytes") or 0
                ),
                "first_allocator_cache_bytes": int(
                    first_memory.get("cache_bytes") or 0
                ),
                "allocator_peak_over_state_x": (
                    allocator_peak / state_bytes if state_bytes else None
                ),
                "allocator_peak_over_minimum_input_output_x": (
                    allocator_peak / minimum_peak if minimum_peak else None
                ),
                "memory_note": (
                    "MLX allocator peak includes active and cached GPU buffers; "
                    "the amplification quantifies allocation/cache pressure, not "
                    "simultaneously live statevectors"
                ),
            },
        }
    max_error = _validate(n, ops)
    pure_ms = arms["pure_mlx"]["summary"]["warm_total_median_ms"]
    metal_ms = arms["custom_metal"]["summary"]["warm_total_median_ms"]
    return {
        "workload": name,
        "qubits": n,
        "gate_count": len(ops),
        "arms": arms,
        "validation": {
            "reference": "pure MLX statevector",
            "max_absolute_amplitude_error": max_error,
        },
        "warm_speedup_pure_over_custom": pure_ms / metal_ms if metal_ms else None,
    }


def _rank_bottlenecks(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for result in results:
        name = result["workload"]
        summary = result["arms"]["custom_metal"]["summary"]
        candidates.extend([
            {
                "kind": "synchronized_gpu_evaluation",
                "workload": name,
                "milliseconds": summary["warm_evaluation_median_ms"],
                "evidence": {
                    "custom_kernel_launches": summary["custom_kernel_launches"],
                    "estimated_effective_bandwidth_gb_s": summary[
                        "estimated_effective_bandwidth_gb_s"
                    ],
                },
            },
            {
                "kind": "first_process_compile_or_cache_overhead",
                "workload": name,
                "milliseconds": summary["first_minus_warm_evaluation_ms"],
                "evidence": {"estimate": "first evaluation minus warm median"},
            },
            {
                "kind": "host_graph_build_and_dispatch",
                "workload": name,
                "milliseconds": summary["warm_graph_build_median_ms"],
                "evidence": {"timing_scope": "Device.execute before mx.eval"},
            },
            {
                "kind": "full_state_cpu_readback",
                "workload": name,
                "milliseconds": summary["warm_readback_median_ms"],
                "evidence": {"timing_scope": "state.tolist after synchronized execution"},
            },
        ])
    # Rank distinct pressure-point categories so one category repeated across
    # workloads cannot crowd all other measured costs out of the top three.
    worst_by_kind: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        current = worst_by_kind.get(candidate["kind"])
        if current is None or candidate["milliseconds"] > current["milliseconds"]:
            worst_by_kind[candidate["kind"]] = candidate
    timed = sorted(
        worst_by_kind.values(),
        key=lambda item: item["milliseconds"],
        reverse=True,
    )
    memory_result = max(
        results,
        key=lambda result: result["arms"]["custom_metal"]["summary"][
            "allocator_peak_over_minimum_input_output_x"
        ],
    )
    memory_summary = memory_result["arms"]["custom_metal"]["summary"]
    ranked = [{
        "kind": "mlx_allocator_and_temporary_memory_pressure",
        "workload": memory_result["workload"],
        "allocator_peak_bytes": memory_summary["first_allocator_peak_bytes"],
        "allocator_peak_over_minimum_input_output_x": memory_summary[
            "allocator_peak_over_minimum_input_output_x"
        ],
        "evidence": memory_summary["memory_note"],
    }]
    ranked.extend(timed[:2])
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def _capture_metal_trace(
    n: int, ops: List[Dict[str, Any]], path: Path
) -> Dict[str, Any]:
    start = getattr(getattr(mx, "metal", None), "start_capture", None)
    stop = getattr(getattr(mx, "metal", None), "stop_capture", None)
    if not callable(start) or not callable(stop):
        raise RuntimeError("this MLX build does not expose Metal capture")
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["METTLEQ_METAL_KERNELS"] = "1"
    dev = Device(n)
    try:
        start(str(path))
    except RuntimeError as exc:
        if "Capture layer is not inserted" in str(exc):
            raise RuntimeError(
                "Metal capture requires MTL_CAPTURE_ENABLED=1 when Python starts"
            ) from exc
        raise
    try:
        dev.execute(ops, report=True)
        dev.synchronize()
    finally:
        stop()
    return {
        "path": str(path),
        "execution_plan": dev.last_execution_plan,
        "note": "open the .gputrace bundle in Xcode Instruments",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--workloads", nargs="+", choices=sorted(WORKLOADS), default=sorted(WORKLOADS)
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture-workload", choices=sorted(WORKLOADS))
    parser.add_argument("--capture-output", type=Path)
    args = parser.parse_args()
    if args.qubits < 4:
        parser.error("--qubits must be at least 4")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if bool(args.capture_workload) != bool(args.capture_output):
        parser.error("--capture-workload and --capture-output must be used together")

    previous_policy = os.environ.get("METTLEQ_METAL_KERNELS")
    capability = metal_runtime_status(args.qubits)
    os.environ["METTLEQ_METAL_KERNELS"] = "1"
    capability = metal_runtime_status(args.qubits)
    if not capability["enabled"]:
        raise SystemExit(capability["reason"])
    results = []
    capture = None
    try:
        for name in args.workloads:
            results.append(_profile_workload(
                name, args.qubits, WORKLOADS[name](args.qubits), args.repeats
            ))
        if args.capture_workload:
            capture = _capture_metal_trace(
                args.qubits,
                WORKLOADS[args.capture_workload](args.qubits),
                args.capture_output,
            )
    finally:
        if previous_policy is None:
            os.environ.pop("METTLEQ_METAL_KERNELS", None)
        else:
            os.environ["METTLEQ_METAL_KERNELS"] = previous_policy

    payload = {
        "schema_version": 1,
        "purpose": "Phase D inspection/profiling; not publication evidence",
        "timing": {
            "clock": "time.perf_counter_ns",
            "synchronization": "Device.synchronize -> mx.eval(final state)",
            "cold_definition": (
                "first execution in this process after clearing the MLX allocator cache; "
                "persistent compiler-cache state remains opaque"
            ),
            "warm_definition": "fresh Device and identical operations after first execution",
        },
        "capability": capability,
        "results": results,
        "ranked_measured_pressure_points": _rank_bottlenecks(results),
        "metal_system_trace": capture,
    }
    encoded = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
