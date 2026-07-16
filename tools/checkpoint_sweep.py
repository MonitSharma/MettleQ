#!/usr/bin/env python3
"""Measure Metal lazy-graph checkpoint memory/runtime crossover.

The lazy baseline and every configured budget are rotated through each repeat
so thermal or session drift does not consistently favor one arm. Each run
records synchronized wall time, MLX allocator counters, predicted and observed
checkpoint counts, and the execution-plan accounting used for the decision.
"""
from __future__ import annotations

import argparse
import csv
import gc
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402

from mettleq.device import Device  # noqa: E402
from mettleq.execution import (  # noqa: E402
    METAL_CHECKPOINT_BUDGET_ENV,
    metal_memory_snapshot,
    metal_runtime_status,
)


def _tfim_ops(n: int, steps: int) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = [
        {"name": "H", "wires": [wire]} for wire in range(n)
    ]
    for step in range(steps):
        theta = -0.03 - 0.002 * step
        ops.extend({
            "name": "ZZPHASE",
            "wires": [wire, wire + 1],
            "parameters": [theta],
        } for wire in range(n - 1))
        ops.extend({
            "name": "RX",
            "wires": [wire],
            "parameters": [0.07 + 0.001 * step],
        } for wire in range(n))
    return ops


def _allocator_call(name: str) -> None:
    fn = getattr(mx, name, None)
    if not callable(fn):
        fn = getattr(getattr(mx, "metal", None), name, None)
    if callable(fn):
        fn()


def _run_once(
    n: int,
    operations: List[Dict[str, Any]],
    budget_bytes: Optional[int],
) -> Dict[str, Any]:
    gc.collect()
    _allocator_call("clear_cache")
    dev = Device(n, metal_checkpoint_budget_bytes=budget_bytes)
    _allocator_call("reset_peak_memory")

    start = time.perf_counter_ns()
    dev.execute(operations, report=True)
    graph_built = time.perf_counter_ns()
    dev.synchronize()
    finished = time.perf_counter_ns()

    plan = dev.last_execution_plan
    checkpointing = plan["checkpointing"]
    checkpoint_events = checkpointing["actual_checkpoints"]
    intra_layer_events = [
        event for event in checkpoint_events
        if event["boundary"] == "within_optimized_operation"
    ]
    memory = metal_memory_snapshot()
    result = {
        "graph_build_ms": (graph_built - start) / 1e6,
        "final_synchronize_ms": (finished - graph_built) / 1e6,
        "total_ms": (finished - start) / 1e6,
        "peak_bytes": memory["peak_bytes"],
        "active_bytes": memory["active_bytes"],
        "cache_bytes": memory["cache_bytes"],
        "predicted_checkpoints": checkpointing["predicted_checkpoint_count"],
        "observed_checkpoints": checkpointing["actual_checkpoint_count"],
        "intra_layer_checkpoints": len(intra_layer_events),
        "inter_layer_checkpoints": (
            len(checkpoint_events) - len(intra_layer_events)
        ),
        "streamed_custom_passes": sum(
            int(event["estimated_passes_evaluated"])
            for event in intra_layer_events
        ),
        "checkpoint_evaluation_ms": sum(
            event["evaluation_ms"]
            for event in checkpoint_events
        ),
        "pending_passes_before_final_synchronize": checkpointing[
            "pending_custom_passes_after_graph_build"
        ],
        "expected_custom_kernel_launches": plan[
            "expected_custom_kernel_launches"
        ],
    }
    del dev
    gc.collect()
    _allocator_call("clear_cache")
    return result


def _validate(
    n: int,
    operations: List[Dict[str, Any]],
    budgets: Dict[str, Optional[int]],
) -> Dict[str, float]:
    os.environ["METTLEQ_METAL_KERNELS"] = "0"
    reference = Device(n)
    reference.execute(operations)
    reference.synchronize()
    errors: Dict[str, float] = {}
    os.environ["METTLEQ_METAL_KERNELS"] = "1"
    for label, budget in budgets.items():
        candidate = Device(n, metal_checkpoint_budget_bytes=budget)
        candidate.execute(operations)
        candidate.synchronize()
        error = mx.max(mx.abs(reference.sim.state - candidate.sim.state))
        mx.eval(error)
        errors[label] = float(error.item())
    return errors


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--qubits", type=int, default=20)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--budgets-mib", type=float, nargs="+", default=[512, 256, 128]
    )
    args = parser.parse_args()
    if args.qubits < 1 or args.steps < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error("qubits, steps, and repeats must be positive; warmups non-negative")
    if any(not math.isfinite(value) or value <= 0 for value in args.budgets_mib):
        parser.error("every checkpoint budget must be a finite positive MiB value")

    os.environ.pop(METAL_CHECKPOINT_BUDGET_ENV, None)
    os.environ["METTLEQ_METAL_KERNELS"] = "1"
    capability = metal_runtime_status(args.qubits)
    if not capability["enabled"]:
        raise SystemExit(capability["reason"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    operations = _tfim_ops(args.qubits, args.steps)
    budgets: Dict[str, Optional[int]] = {"lazy": None}
    for value in args.budgets_mib:
        label = f"{value:g}_mib"
        if label in budgets:
            parser.error(f"duplicate budget label: {label}")
        budgets[label] = int(value * 1024 * 1024)

    for budget in budgets.values():
        for _ in range(args.warmups):
            _run_once(args.qubits, operations, budget)

    rows: List[Dict[str, Any]] = []
    labels = list(budgets)
    for repeat in range(1, args.repeats + 1):
        offset = (repeat - 1) % len(labels)
        order = labels[offset:] + labels[:offset]
        for order_index, label in enumerate(order):
            result = _run_once(args.qubits, operations, budgets[label])
            row = {
                "label": label,
                "budget_bytes": budgets[label],
                "repeat": repeat,
                "order_index": order_index,
                **result,
            }
            rows.append(row)
            print(
                f"repeat {repeat} {label}: {result['total_ms']:.3f} ms, "
                f"peak {(result['peak_bytes'] or 0) / (1024 * 1024):.1f} MiB, "
                f"checkpoints {result['observed_checkpoints']}",
                flush=True,
            )

    with (outdir / "checkpoint_sweep_raw.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries: List[Dict[str, Any]] = []
    baseline_peak = statistics.median(
        float(row["peak_bytes"]) for row in rows if row["label"] == "lazy"
    )
    baseline_time = statistics.median(
        float(row["total_ms"]) for row in rows if row["label"] == "lazy"
    )
    for label, budget in budgets.items():
        selected = [row for row in rows if row["label"] == label]
        peak = statistics.median(float(row["peak_bytes"]) for row in selected)
        total = statistics.median(float(row["total_ms"]) for row in selected)
        summaries.append({
            "label": label,
            "budget_bytes": budget,
            "repeats": len(selected),
            "peak_bytes_median": peak,
            "peak_reduction_percent": 100 * (1 - peak / baseline_peak),
            "total_ms_median": total,
            "runtime_change_percent": 100 * (total / baseline_time - 1),
            "observed_checkpoints_median": statistics.median(
                int(row["observed_checkpoints"]) for row in selected
            ),
            "predicted_checkpoints_median": statistics.median(
                int(row["predicted_checkpoints"]) for row in selected
            ),
            "intra_layer_checkpoints_median": statistics.median(
                int(row["intra_layer_checkpoints"]) for row in selected
            ),
            "inter_layer_checkpoints_median": statistics.median(
                int(row["inter_layer_checkpoints"]) for row in selected
            ),
            "streamed_custom_passes_median": statistics.median(
                int(row["streamed_custom_passes"]) for row in selected
            ),
            "checkpoint_evaluation_ms_median": statistics.median(
                float(row["checkpoint_evaluation_ms"]) for row in selected
            ),
        })
    with (outdir / "checkpoint_sweep_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    validation = _validate(args.qubits, operations, budgets)
    (outdir / "checkpoint_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload": "tfim",
        "qubits": args.qubits,
        "trotter_steps": args.steps,
        "input_operation_count": len(operations),
        "repeats": args.repeats,
        "warmups_per_arm": args.warmups,
        "order": "rotating lazy and budget arms once per repeat",
        "budgets_mib": args.budgets_mib,
        "platform": platform.platform(),
        "python": sys.version,
        "mlx_version": importlib_metadata.version("mlx"),
        "device_info": dict(mx.device_info()),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_porcelain": _git_value("status", "--porcelain"),
        "capability": capability,
        "validation_reference": "pure MLX complex64 statevector",
    }
    (outdir / "checkpoint_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    os.environ.pop("METTLEQ_METAL_KERNELS", None)
    print(f"wrote {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
