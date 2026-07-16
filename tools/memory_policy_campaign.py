#!/usr/bin/env python3
"""Cross-workload evidence for adaptive Metal checkpoint policies.

The campaign rotates a fully lazy arm and two state-size-aware checkpoint
policies across deterministic circuits and qubit counts. It records allocator
peak, synchronized runtime, planner/runtime checkpoint agreement, execution
cost estimates, and full-state parity. Pure MLX is the validation reference at
the configured smaller sizes; larger sizes compare checkpointed execution to
the same fully lazy Metal kernels to keep validation tractable and explicit.
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
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402

from mettleq.device import Device  # noqa: E402
from mettleq.execution import (  # noqa: E402
    METAL_CHECKPOINT_BUDGET_ENV,
    STATEVECTOR_UNSAFE_OVERRIDE_ENV,
    metal_memory_snapshot,
    metal_runtime_status,
    state_memory_estimate,
    statevector_preflight,
)


MIB = 1024 * 1024
POLICIES = ("lazy", "adaptive_balanced", "adaptive_minimum")
DEFAULT_WORKLOADS = ("tfim", "qft", "qaoa", "qcbm", "heisenberg", "su2")


def _tfim_ops(n: int) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = [
        {"name": "H", "wires": [wire]} for wire in range(n)
    ]
    for step in range(6):
        ops.extend({
            "name": "ZZPHASE",
            "wires": [wire, wire + 1],
            "parameters": [-0.03 - 0.002 * step],
        } for wire in range(n - 1))
        ops.extend({
            "name": "RX",
            "wires": [wire],
            "parameters": [0.07 + 0.001 * step],
        } for wire in range(n))
    return ops


def _qft_ops(n: int) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for target in range(n):
        ops.append({"name": "H", "wires": [target]})
        for control in range(target + 1, n):
            ops.append({
                "name": "CPHASE",
                "wires": [control, target],
                "parameters": [math.pi / (2 ** (control - target))],
            })
    return ops


def _qaoa_ops(n: int) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = [
        {"name": "H", "wires": [wire]} for wire in range(n)
    ]
    for layer in range(4):
        gamma = 0.6 + 0.1 * layer
        beta = 0.4 + 0.05 * layer
        ops.extend({
            "name": "CPHASE",
            "wires": [wire, (wire + 1) % n],
            "parameters": [gamma],
        } for wire in range(n))
        ops.extend({
            "name": "RX",
            "wires": [wire],
            "parameters": [2.0 * beta],
        } for wire in range(n))
    return ops


def _qcbm_ops(n: int) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for layer in range(3):
        ops.extend({
            "name": "RY",
            "wires": [wire],
            "parameters": [0.2 * (layer + 1)],
        } for wire in range(n))
        ops.extend({
            "name": "RZ",
            "wires": [wire],
            "parameters": [0.1 * (layer + 1)],
        } for wire in range(n))
        ops.extend({
            "name": "CNOT",
            "wires": [wire, (wire + 1) % n],
        } for wire in range(n))
    return ops


def _heisenberg_ops(n: int) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for step in range(2):
        theta = -0.025 - 0.003 * step
        for gate in ("XXPHASE", "YYPHASE", "ZZPHASE"):
            ops.extend({
                "name": gate,
                "wires": [wire, wire + 1],
                "parameters": [theta],
            } for wire in range(n - 1))
    return ops


def _su2_ops(n: int) -> List[Dict[str, Any]]:
    rng = random.Random(5678 + 3 * n)
    ops: List[Dict[str, Any]] = []
    for layer in range(3):
        for wire in range(n):
            for gate in ("RX", "RZ", "RX"):
                ops.append({
                    "name": gate,
                    "wires": [wire],
                    "parameters": [(rng.random() - 0.5) * 2.0],
                })
        ops.extend({
            "name": "CZ",
            "wires": [wire, wire + 1],
        } for wire in range(layer % 2, n - 1, 2))
    return ops


WORKLOAD_BUILDERS: Dict[str, Callable[[int], List[Dict[str, Any]]]] = {
    "tfim": _tfim_ops,
    "qft": _qft_ops,
    "qaoa": _qaoa_ops,
    "qcbm": _qcbm_ops,
    "heisenberg": _heisenberg_ops,
    "su2": _su2_ops,
}


def _budget_bytes(policy: str, n: int) -> Optional[int]:
    state_bytes = int(state_memory_estimate(n)["state_bytes"])
    if policy == "lazy":
        return None
    if policy == "adaptive_balanced":
        return max(256 * MIB, 4 * state_bytes)
    if policy == "adaptive_minimum":
        return max(128 * MIB, 2 * state_bytes)
    raise ValueError(f"unknown policy: {policy}")


def _allocator_call(name: str) -> None:
    fn = getattr(mx, name, None)
    if not callable(fn):
        fn = getattr(getattr(mx, "metal", None), name, None)
    if callable(fn):
        fn()


def _run_once(
    n: int,
    operations: List[Dict[str, Any]],
    policy: str,
) -> Dict[str, Any]:
    gc.collect()
    _allocator_call("clear_cache")
    budget = _budget_bytes(policy, n)
    dev = Device(n, metal_checkpoint_budget_bytes=budget)
    _allocator_call("reset_peak_memory")

    started = time.perf_counter_ns()
    dev.execute(operations, report=True)
    graph_built = time.perf_counter_ns()
    dev.synchronize()
    finished = time.perf_counter_ns()

    plan = dev.last_execution_plan
    checkpointing = plan["checkpointing"]
    memory = metal_memory_snapshot()
    result = {
        "budget_bytes": budget,
        "graph_build_ms": (graph_built - started) / 1e6,
        "final_synchronize_ms": (finished - graph_built) / 1e6,
        "total_ms": (finished - started) / 1e6,
        "peak_bytes": memory["peak_bytes"],
        "active_bytes": memory["active_bytes"],
        "cache_bytes": memory["cache_bytes"],
        "predicted_checkpoints": checkpointing["predicted_checkpoint_count"],
        "observed_checkpoints": checkpointing["actual_checkpoint_count"],
        "expected_custom_kernel_launches": plan[
            "expected_custom_kernel_launches"
        ],
        "predicted_custom_io_bytes": plan["execution_cost_model"][
            "predicted_custom_input_output_bytes"
        ],
        "preflight_decision": plan["statevector_preflight"]["decision"],
    }
    del dev
    gc.collect()
    _allocator_call("clear_cache")
    return result


def _validate(
    workload: str,
    n: int,
    operations: List[Dict[str, Any]],
    pure_reference_max_qubits: int,
) -> List[Dict[str, Any]]:
    use_pure = n <= pure_reference_max_qubits
    if use_pure:
        os.environ["METTLEQ_METAL_KERNELS"] = "0"
        reference_label = "pure_mlx"
    else:
        os.environ["METTLEQ_METAL_KERNELS"] = "1"
        reference_label = "metal_lazy"
    reference = Device(n)
    reference.execute(operations)
    reference.synchronize()

    labels = POLICIES if use_pure else POLICIES[1:]
    rows: List[Dict[str, Any]] = []
    os.environ["METTLEQ_METAL_KERNELS"] = "1"
    for policy in labels:
        candidate = Device(
            n, metal_checkpoint_budget_bytes=_budget_bytes(policy, n)
        )
        candidate.execute(operations)
        candidate.synchronize()
        error = mx.max(mx.abs(reference.sim.state - candidate.sim.state))
        norm_error = mx.abs(
            mx.sum(mx.abs(candidate.sim.state) ** 2) - mx.array(1.0)
        )
        mx.eval(error, norm_error)
        rows.append({
            "workload": workload,
            "qubits": n,
            "policy": policy,
            "reference": reference_label,
            "max_amplitude_error": float(error.item()),
            "norm_error": float(norm_error.item()),
        })
        del candidate, error, norm_error
        gc.collect()
        _allocator_call("clear_cache")
    del reference
    gc.collect()
    _allocator_call("clear_cache")
    return rows


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(index, 0)]


def _summarize(
    rows: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    repeat_timings: Dict[tuple[str, int, int], Dict[str, float]] = {}
    for row in rows:
        repeat_key = (
            str(row["workload"]), int(row["qubits"]), int(row["repeat"])
        )
        repeat_timings.setdefault(repeat_key, {})[str(row["policy"])] = float(
            row["total_ms"]
        )
    paired_changes: Dict[tuple[str, int, str], List[float]] = {}
    for (workload, qubits, _), timings in repeat_timings.items():
        lazy = timings["lazy"]
        for policy in POLICIES:
            paired_changes.setdefault((workload, qubits, policy), []).append(
                100 * (timings[policy] / lazy - 1)
            )

    grouped: Dict[tuple[str, int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["workload"]), int(row["qubits"]), str(row["policy"]))
        grouped.setdefault(key, []).append(row)

    cells: List[Dict[str, Any]] = []
    for (workload, qubits, policy), selected in grouped.items():
        cells.append({
            "workload": workload,
            "qubits": qubits,
            "policy": policy,
            "budget_bytes": selected[0]["budget_bytes"],
            "repeats": len(selected),
            "total_ms_median": statistics.median(
                float(row["total_ms"]) for row in selected
            ),
            "peak_bytes_median": statistics.median(
                float(row["peak_bytes"]) for row in selected
            ),
            "observed_checkpoints_median": statistics.median(
                int(row["observed_checkpoints"]) for row in selected
            ),
            "predicted_observed_checkpoints_match": all(
                int(row["predicted_checkpoints"])
                == int(row["observed_checkpoints"])
                for row in selected
            ),
        })

    lookup = {
        (row["workload"], row["qubits"], row["policy"]): row for row in cells
    }
    for row in cells:
        baseline = lookup[(row["workload"], row["qubits"], "lazy")]
        row["runtime_change_percent"] = 100 * (
            row["total_ms_median"] / baseline["total_ms_median"] - 1
        )
        row["peak_reduction_percent"] = 100 * (
            1 - row["peak_bytes_median"] / baseline["peak_bytes_median"]
        )
        changes = paired_changes[
            (row["workload"], row["qubits"], row["policy"])
        ]
        row["paired_runtime_change_percent_median"] = statistics.median(changes)
        row["paired_runtime_change_percent_geomean"] = 100 * (
            math.exp(statistics.fmean(
                math.log(1 + change / 100) for change in changes
            )) - 1
        )

    aggregate: List[Dict[str, Any]] = []
    for policy in POLICIES[1:]:
        selected = [row for row in cells if row["policy"] == policy]
        applicable = [
            row for row in selected if row["observed_checkpoints_median"] > 0
        ]
        runtime_ratios = [1 + row["runtime_change_percent"] / 100 for row in selected]
        paired_medians = [
            row["paired_runtime_change_percent_median"] for row in selected
        ]
        paired_geomean_ratios = [
            1 + row["paired_runtime_change_percent_geomean"] / 100
            for row in selected
        ]
        aggregate.append({
            "policy": policy,
            "cells": len(selected),
            "checkpointed_cells": len(applicable),
            "median_peak_reduction_percent_all_cells": statistics.median(
                row["peak_reduction_percent"] for row in selected
            ),
            "median_peak_reduction_percent_checkpointed_cells": (
                statistics.median(
                    row["peak_reduction_percent"] for row in applicable
                ) if applicable else 0.0
            ),
            "geomean_runtime_change_percent": 100 * (
                math.exp(statistics.fmean(math.log(value) for value in runtime_ratios))
                - 1
            ),
            "p90_runtime_change_percent": _percentile(
                [row["runtime_change_percent"] for row in selected], 0.9
            ),
            "maximum_runtime_change_percent": max(
                row["runtime_change_percent"] for row in selected
            ),
            "cells_over_10_percent_slower": sum(
                row["runtime_change_percent"] > 10 for row in selected
            ),
            "cells_with_peak_increase": sum(
                row["peak_reduction_percent"] < -0.01 for row in selected
            ),
            "planner_runtime_checkpoint_agreement": all(
                row["predicted_observed_checkpoints_match"] for row in selected
            ),
            "median_cell_paired_runtime_change_percent": statistics.median(
                paired_medians
            ),
            "geomean_paired_runtime_change_percent": 100 * (
                math.exp(statistics.fmean(
                    math.log(value) for value in paired_geomean_ratios
                )) - 1
            ),
            "p90_cell_paired_runtime_change_percent": _percentile(
                paired_medians, 0.9
            ),
            "maximum_cell_paired_runtime_change_percent": max(paired_medians),
            "paired_cells_over_10_percent_slower": sum(
                change > 10 for change in paired_medians
            ),
            "paired_cells_faster": sum(change < 0 for change in paired_medians),
        })
    return cells, aggregate


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _acceptance(
    rows: List[Dict[str, Any]],
    validation: List[Dict[str, Any]],
    tolerance: float,
    norm_tolerance: float,
    *,
    unsafe_override_used: bool,
) -> Dict[str, Any]:
    maximum_error = max(float(row["max_amplitude_error"]) for row in validation)
    maximum_norm_error = max(float(row["norm_error"]) for row in validation)
    return {
        "amplitude_tolerance": tolerance,
        "norm_tolerance": norm_tolerance,
        "maximum_amplitude_error": maximum_error,
        "maximum_norm_error": maximum_norm_error,
        "amplitude_parity_passed": maximum_error <= tolerance,
        "norm_parity_passed": maximum_norm_error <= norm_tolerance,
        "numerical_parity_passed": (
            maximum_error <= tolerance and maximum_norm_error <= norm_tolerance
        ),
        "planner_runtime_checkpoint_agreement": all(
            int(row["predicted_checkpoints"])
            == int(row["observed_checkpoints"])
            for row in rows
        ),
        "unsafe_override_used": unsafe_override_used,
    }


def _run_isolated(args: argparse.Namespace) -> int:
    """Run every workload/qubit cell in a fresh Python and MLX process."""
    outdir = Path(args.outdir)
    cells_dir = outdir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    validation: List[Dict[str, Any]] = []
    child_manifests: List[Dict[str, Any]] = []
    script = Path(__file__).resolve()
    for qubits in args.qubits:
        for workload in args.workloads:
            child_dir = cells_dir / f"{qubits}q_{workload}"
            command = [
                sys.executable,
                str(script),
                "--worker",
                "--outdir", str(child_dir),
                "--qubits", str(qubits),
                "--workloads", workload,
                "--repeats", str(args.repeats),
                "--warmups", str(args.warmups),
                "--pure-reference-max-qubits",
                str(args.pure_reference_max_qubits),
                "--tolerance", str(args.tolerance),
                "--norm-tolerance", str(args.norm_tolerance),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            with (child_dir / "memory_policy_raw.csv").open(newline="") as file:
                rows.extend(csv.DictReader(file))
            with (child_dir / "memory_policy_validation.csv").open(
                newline=""
            ) as file:
                validation.extend(csv.DictReader(file))
            child_manifests.append(json.loads(
                (child_dir / "memory_policy_manifest.json").read_text()
            ))

    summary, aggregate = _summarize(rows)
    _write_csv(outdir / "memory_policy_raw.csv", rows)
    _write_csv(outdir / "memory_policy_summary.csv", summary)
    _write_csv(outdir / "memory_policy_aggregate.csv", aggregate)
    _write_csv(outdir / "memory_policy_validation.csv", validation)
    acceptance = _acceptance(
        rows,
        validation,
        args.tolerance,
        args.norm_tolerance,
        unsafe_override_used=any(
            report["overridden"]
            for manifest in child_manifests
            for report in manifest["preflights"].values()
        ),
    )
    (outdir / "memory_policy_acceptance.json").write_text(
        json.dumps(acceptance, indent=2) + "\n"
    )
    manifest = {
        "schema_version": 2,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workloads": args.workloads,
        "qubits": args.qubits,
        "repeats": args.repeats,
        "warmups_per_arm": args.warmups,
        "process_isolation": (
            "fresh Python and MLX process for every workload/qubit cell"
        ),
        "rotation": "policy order rotates by repeat, qubit count, and workload",
        "policies": {
            "lazy": "no checkpoint budget",
            "adaptive_balanced": "max(256 MiB, 4 * state bytes)",
            "adaptive_minimum": "max(128 MiB, 2 * state bytes)",
        },
        "validation": {
            "pure_mlx_through_qubits": args.pure_reference_max_qubits,
            "larger_reference": "fully lazy execution using the same Metal kernels",
            "comparison": "full-state maximum amplitude error and state norm",
            "amplitude_tolerance": args.tolerance,
            "norm_tolerance": args.norm_tolerance,
        },
        "platform": platform.platform(),
        "python": sys.version,
        "mlx_version": importlib_metadata.version("mlx"),
        "device_info": dict(mx.device_info()),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_porcelain": _git_value("status", "--porcelain"),
        "child_manifest_count": len(child_manifests),
        "child_manifests": child_manifests,
    }
    (outdir / "memory_policy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"merged {len(child_manifests)} isolated cells into {outdir}", flush=True)
    if not acceptance["numerical_parity_passed"]:
        return 2
    if not acceptance["planner_runtime_checkpoint_agreement"]:
        return 3
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--qubits", type=int, nargs="+", default=[22, 23, 24, 25, 26, 27]
    )
    parser.add_argument(
        "--workloads", nargs="+", choices=sorted(WORKLOAD_BUILDERS),
        default=list(DEFAULT_WORKLOADS),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--pure-reference-max-qubits", type=int, default=24)
    parser.add_argument("--tolerance", type=float, default=5e-6)
    parser.add_argument("--norm-tolerance", type=float, default=1e-5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(n < 1 for n in args.qubits):
        parser.error("all qubit counts must be positive")
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups non-negative")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("tolerance must be finite and positive")
    if not math.isfinite(args.norm_tolerance) or args.norm_tolerance <= 0:
        parser.error("norm tolerance must be finite and positive")
    if not args.worker:
        return _run_isolated(args)

    os.environ.pop(METAL_CHECKPOINT_BUDGET_ENV, None)
    os.environ.pop(STATEVECTOR_UNSAFE_OVERRIDE_ENV, None)
    os.environ["METTLEQ_METAL_KERNELS"] = "1"
    capabilities = {str(n): metal_runtime_status(n) for n in args.qubits}
    unusable = [report["reason"] for report in capabilities.values()
                if not report["enabled"]]
    if unusable:
        raise SystemExit(unusable[0])
    preflights = {str(n): statevector_preflight(n) for n in args.qubits}
    refused = [report for report in preflights.values() if not report["allowed"]]
    if refused:
        raise SystemExit(refused[0]["reason"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    validation: List[Dict[str, Any]] = []
    for qubits in args.qubits:
        for workload in args.workloads:
            operations = WORKLOAD_BUILDERS[workload](qubits)
            for policy in POLICIES:
                for _ in range(args.warmups):
                    _run_once(qubits, operations, policy)
            for repeat in range(1, args.repeats + 1):
                offset = (repeat + qubits + args.workloads.index(workload)) % len(POLICIES)
                order = POLICIES[offset:] + POLICIES[:offset]
                for order_index, policy in enumerate(order):
                    result = _run_once(qubits, operations, policy)
                    row = {
                        "workload": workload,
                        "qubits": qubits,
                        "input_operation_count": len(operations),
                        "policy": policy,
                        "repeat": repeat,
                        "order_index": order_index,
                        **result,
                    }
                    rows.append(row)
                    print(
                        f"{qubits}q {workload} r{repeat} {policy}: "
                        f"{result['total_ms']:.2f} ms, "
                        f"{float(result['peak_bytes']) / MIB:.0f} MiB, "
                        f"checkpoints {result['observed_checkpoints']}",
                        flush=True,
                    )
            validation.extend(_validate(
                workload, qubits, operations, args.pure_reference_max_qubits
            ))
            worst = max(
                row["max_amplitude_error"] for row in validation
                if row["workload"] == workload and row["qubits"] == qubits
            )
            print(
                f"validated {qubits}q {workload}: max error {worst:.3e}",
                flush=True,
            )

    cells, aggregate = _summarize(rows)
    _write_csv(outdir / "memory_policy_raw.csv", rows)
    _write_csv(outdir / "memory_policy_summary.csv", cells)
    _write_csv(outdir / "memory_policy_aggregate.csv", aggregate)
    _write_csv(outdir / "memory_policy_validation.csv", validation)
    acceptance = _acceptance(
        rows,
        validation,
        args.tolerance,
        args.norm_tolerance,
        unsafe_override_used=any(
            report["overridden"] for report in preflights.values()
        ),
    )
    (outdir / "memory_policy_acceptance.json").write_text(
        json.dumps(acceptance, indent=2) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workloads": args.workloads,
        "qubits": args.qubits,
        "repeats": args.repeats,
        "warmups_per_arm": args.warmups,
        "rotation": "policy order rotates by repeat, qubit count, and workload",
        "policies": {
            "lazy": "no checkpoint budget",
            "adaptive_balanced": "max(256 MiB, 4 * state bytes)",
            "adaptive_minimum": "max(128 MiB, 2 * state bytes)",
        },
        "validation": {
            "pure_mlx_through_qubits": args.pure_reference_max_qubits,
            "larger_reference": "fully lazy execution using the same Metal kernels",
            "comparison": "full-state maximum amplitude error and state norm",
        },
        "platform": platform.platform(),
        "python": sys.version,
        "mlx_version": importlib_metadata.version("mlx"),
        "device_info": dict(mx.device_info()),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_porcelain": _git_value("status", "--porcelain"),
        "capabilities": capabilities,
        "preflights": preflights,
    }
    (outdir / "memory_policy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    os.environ.pop("METTLEQ_METAL_KERNELS", None)
    print(f"wrote {outdir}", flush=True)
    if not acceptance["numerical_parity_passed"]:
        return 2
    if not acceptance["planner_runtime_checkpoint_agreement"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
