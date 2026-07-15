#!/usr/bin/env python3
"""Probe Qupertino MPS limits across qubit count and entanglement topology.

Each circuit runs through ``QupertinoEstimatorV2`` in a fresh process. The
campaign distinguishes mere completion from trustworthy evidence by recording
bond growth, local SVD truncation, discarded-weight telemetry, norm drift,
runtime, and process peak RSS. Wide cases are not described as exact unless a
small statevector validation or a stronger independent bound exists.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from mlxq.integrations.qiskit import QupertinoEstimatorV2


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (
    "ghz_chain",
    "line_brickwork",
    "ring_brickwork",
    "grid_2d",
    "rainbow",
    "random_long_range",
    "all_to_all",
)


STANDARD_CASES = (
    ("ghz_chain", 100, 1),
    ("ghz_chain", 250, 1),
    ("ghz_chain", 500, 1),
    ("ghz_chain", 1000, 1),
    ("line_brickwork", 16, 4),
    ("line_brickwork", 100, 2),
    ("line_brickwork", 100, 4),
    ("line_brickwork", 100, 8),
    ("line_brickwork", 200, 4),
    ("ring_brickwork", 16, 2),
    ("ring_brickwork", 50, 2),
    ("ring_brickwork", 100, 2),
    ("ring_brickwork", 100, 4),
    ("grid_2d", 16, 2),
    ("grid_2d", 36, 2),
    ("grid_2d", 64, 2),
    ("rainbow", 16, 1),
    ("rainbow", 32, 1),
    ("rainbow", 48, 1),
    ("random_long_range", 16, 1),
    ("random_long_range", 32, 1),
    ("random_long_range", 48, 1),
    ("all_to_all", 12, 1),
    ("all_to_all", 16, 1),
    ("all_to_all", 20, 1),
    ("all_to_all", 24, 1),
)


SMOKE_CASES = (
    ("ghz_chain", 20, 1),
    ("line_brickwork", 12, 2),
    ("ring_brickwork", 12, 1),
    ("grid_2d", 16, 1),
    ("rainbow", 12, 1),
    ("random_long_range", 12, 1),
    ("all_to_all", 10, 1),
)


def topology_pairs(family: str, n_qubits: int, layer: int) -> list[tuple[int, int]]:
    """Return the deterministic two-qubit topology for one layer."""
    if family == "line_brickwork":
        return [
            (wire, wire + 1)
            for wire in range(layer % 2, n_qubits - 1, 2)
        ]
    if family == "ring_brickwork":
        pairs = topology_pairs("line_brickwork", n_qubits, layer)
        if n_qubits > 2:
            pairs.append((n_qubits - 1, 0))
        return pairs
    if family == "grid_2d":
        side = math.isqrt(n_qubits)
        if side * side != n_qubits:
            raise ValueError("grid_2d requires a perfect-square qubit count")
        pairs = []
        if layer % 2 == 0:
            for row in range(side):
                for column in range(side - 1):
                    pairs.append((row * side + column, row * side + column + 1))
        else:
            for row in range(side - 1):
                for column in range(side):
                    pairs.append((row * side + column, (row + 1) * side + column))
        return pairs
    if family == "rainbow":
        return [(wire, n_qubits - 1 - wire) for wire in range(n_qubits // 2)]
    if family == "random_long_range":
        wires = list(range(n_qubits))
        random.Random(7919 * n_qubits + layer).shuffle(wires)
        pairs = []
        for index in range(0, n_qubits - 1, 2):
            first, second = wires[index : index + 2]
            pairs.append((first, second))
        return pairs
    if family == "all_to_all":
        return [
            (first, second)
            for first in range(n_qubits)
            for second in range(first + 1, n_qubits)
        ]
    if family == "ghz_chain":
        return [(wire, wire + 1) for wire in range(n_qubits - 1)]
    raise ValueError(f"unknown MPS topology family: {family}")


def build_circuit(family: str, n_qubits: int, depth: int) -> QuantumCircuit:
    if family not in FAMILIES:
        raise ValueError(f"unknown MPS topology family: {family}")
    if n_qubits < 2 or depth < 1:
        raise ValueError("MPS limit cases need at least two qubits and depth one")
    circuit = QuantumCircuit(n_qubits, name=f"mps_limit_{family}")
    if family == "ghz_chain":
        circuit.h(0)
        for _ in range(depth):
            for first, second in topology_pairs(family, n_qubits, 0):
                circuit.cx(first, second)
        return circuit

    for wire in range(n_qubits):
        circuit.h(wire)
        circuit.ry(0.013 * (wire + 1), wire)
    for layer in range(depth):
        for wire in range(n_qubits):
            circuit.ry(0.071 + 0.003 * layer + 0.0001 * wire, wire)
            circuit.rz(-0.043 + 0.002 * layer, wire)
        for first, second in topology_pairs(family, n_qubits, layer):
            circuit.rzz(0.19 + 0.01 * layer, first, second)
    return circuit


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _worker(args: argparse.Namespace) -> int:
    output = Path(args.worker_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_start = time.perf_counter_ns()
    circuit = build_circuit(args.family, args.worker_qubits, args.worker_depth)
    build_ms = (time.perf_counter_ns() - build_start) / 1e6
    two_qubit_operations = sum(
        len(topology_pairs(args.family, args.worker_qubits, layer))
        for layer in range(args.worker_depth)
    )
    observable = SparsePauliOp("I" * (args.worker_qubits - 1) + "Z")
    estimator = QupertinoEstimatorV2(
        method="matrix_product_state",
        device=args.device,
        mps_max_bond_dimension=args.dmax,
        mps_truncation_threshold=args.eps,
    )
    start = time.perf_counter_ns()
    result = estimator.run([(circuit, observable)]).result()[0]
    execution_ms = (time.perf_counter_ns() - start) / 1e6
    expectation = float(np.asarray(result.data.evs))
    diagnostics = estimator.last_mps_diagnostics[0]

    exact_expectation = None
    exact_error = None
    exact_ms = None
    if args.worker_qubits <= args.exact_max_qubits:
        exact_start = time.perf_counter_ns()
        exact_result = StatevectorEstimator().run(
            [(circuit, observable)]
        ).result()[0]
        exact_ms = (time.perf_counter_ns() - exact_start) / 1e6
        exact_expectation = float(np.asarray(exact_result.data.evs))
        exact_error = abs(expectation - exact_expectation)

    row = {
        "family": args.family,
        "qubits": args.worker_qubits,
        "depth": args.worker_depth,
        "status": "completed",
        "device": args.device,
        "dmax": args.dmax,
        "eps": args.eps,
        "circuit_build_ms": build_ms,
        "execution_ms": execution_ms,
        "operation_count": len(circuit.data),
        "two_qubit_operation_count": two_qubit_operations,
        "expectation_z0": expectation,
        "exact_expectation_z0": exact_expectation,
        "exact_absolute_error": exact_error,
        "exact_reference_ms": exact_ms,
        "truncated": diagnostics["truncated"],
        "truncation_events": diagnostics["events"],
        "local_discarded_weight_sum": diagnostics[
            "local_discarded_weight_sum"
        ],
        "local_discarded_weight_max": diagnostics[
            "local_discarded_weight_max"
        ],
        "current_bond_dimension_max": diagnostics[
            "current_bond_dimension_max"
        ],
        "current_bond_dimension_mean": diagnostics[
            "current_bond_dimension_mean"
        ],
        "maximum_bond_dimension_reached": diagnostics[
            "maximum_bond_dimension_reached"
        ],
        "state_norm": diagnostics["state_norm"],
        "tensor_device": diagnostics["tensor_device"],
        "svd_device": diagnostics["svd_device"],
        "peak_rss_bytes": _peak_rss_bytes(),
        "error": None,
        "trust_classification": (
            "small_exact_reference_passed"
            if exact_error is not None and exact_error <= args.exact_atol
            else "small_exact_reference_failed"
            if exact_error is not None
            else "wide_no_local_truncation_observed"
            if not diagnostics["truncated"]
            else "wide_local_truncation_observed"
        ),
    }
    output.write_text(json.dumps(row, indent=2) + "\n")
    return 0


def _git_value(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_case(value: str) -> tuple[str, int, int]:
    """Parse a ``family:qubits:depth`` campaign case."""
    try:
        family, qubits_text, depth_text = value.split(":")
        qubits = int(qubits_text)
        depth = int(depth_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "case must use family:qubits:depth"
        ) from error
    if family not in FAMILIES:
        raise argparse.ArgumentTypeError(
            f"unknown family {family!r}; choose one of {', '.join(FAMILIES)}"
        )
    if qubits < 2 or depth < 1:
        raise argparse.ArgumentTypeError(
            "case needs at least two qubits and depth one"
        )
    if family == "grid_2d" and math.isqrt(qubits) ** 2 != qubits:
        raise argparse.ArgumentTypeError(
            "grid_2d case requires a perfect-square qubit count"
        )
    return family, qubits, depth


def _empty_failure_row(family: str, qubits: int, depth: int, args) -> dict:
    return {
        "family": family,
        "qubits": qubits,
        "depth": depth,
        "status": "not_run",
        "device": args.device,
        "dmax": args.dmax,
        "eps": args.eps,
        "circuit_build_ms": None,
        "execution_ms": None,
        "operation_count": None,
        "two_qubit_operation_count": None,
        "expectation_z0": None,
        "exact_expectation_z0": None,
        "exact_absolute_error": None,
        "exact_reference_ms": None,
        "truncated": None,
        "truncation_events": None,
        "local_discarded_weight_sum": None,
        "local_discarded_weight_max": None,
        "current_bond_dimension_max": None,
        "current_bond_dimension_mean": None,
        "maximum_bond_dimension_reached": None,
        "state_norm": None,
        "tensor_device": None,
        "svd_device": None,
        "peak_rss_bytes": None,
        "error": None,
        "trust_classification": "not_completed",
    }


def _summarize(rows: list[dict], timeout_seconds: float) -> dict:
    families = {}
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        completed = [row for row in selected if row["status"] == "completed"]
        families[family] = {
            "cases": len(selected),
            "completed": len(completed),
            "timeouts": sum(row["status"] == "timeout" for row in selected),
            "errors": sum(row["status"] == "error" for row in selected),
            "maximum_completed_qubits": max(
                (int(row["qubits"]) for row in completed), default=None
            ),
            "maximum_completed_depth": max(
                (int(row["depth"]) for row in completed), default=None
            ),
            "maximum_bond_dimension_reached": max(
                (
                    int(row["maximum_bond_dimension_reached"])
                    for row in completed
                ),
                default=None,
            ),
            "cases_with_truncation": sum(
                bool(row["truncated"]) for row in completed
            ),
            "largest_no_truncation_case": max(
                (
                    (int(row["qubits"]), int(row["depth"]))
                    for row in completed
                    if not row["truncated"]
                ),
                default=None,
            ),
        }
    exact_rows = [
        row for row in rows if row.get("exact_absolute_error") is not None
    ]
    return {
        "schema_version": 1,
        "interpretation": (
            "A completed wide circuit is not automatically exact. Trust wide "
            "results using bond growth, truncation telemetry, convergence in "
            "Dmax/epsilon, and problem-specific validation."
        ),
        "per_case_timeout_seconds": timeout_seconds,
        "cases": len(rows),
        "completed": sum(row["status"] == "completed" for row in rows),
        "timeouts": sum(row["status"] == "timeout" for row in rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "maximum_small_exact_absolute_error": max(
            (float(row["exact_absolute_error"]) for row in exact_rows),
            default=None,
        ),
        "families": families,
    }


def _plot(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    completed = [row for row in rows if row["status"] == "completed"]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    colors = dict(zip(FAMILIES, plt.cm.tab10.colors))
    for family in FAMILIES:
        selected = [row for row in completed if row["family"] == family]
        if not selected:
            continue
        axes[0].scatter(
            [row["qubits"] for row in selected],
            [row["execution_ms"] for row in selected],
            s=[35 + 10 * row["depth"] for row in selected],
            color=colors[family],
            label=family.replace("_", " "),
        )
        axes[1].scatter(
            [row["qubits"] for row in selected],
            [row["maximum_bond_dimension_reached"] for row in selected],
            s=[35 + 10 * row["depth"] for row in selected],
            color=colors[family],
            label=family.replace("_", " "),
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Qubits")
    axes[0].set_ylabel("Qiskit Estimator execution (ms, log scale)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].set_yscale("log", base=2)
    axes[1].set_xlabel("Qubits")
    axes[1].set_ylabel("Maximum MPS bond dimension (log₂ scale)")
    axes[1].grid(True, which="both", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4)
    figure.suptitle("Qupertino MPS scaling by entanglement topology")
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _campaign(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    cells_dir = outdir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    cases: Iterable[tuple[str, int, int]] = args.cases or (
        SMOKE_CASES if args.profile == "smoke" else STANDARD_CASES
    )
    rows = []
    script = Path(__file__).resolve()
    for family, qubits, depth in cases:
        output = cells_dir / f"{family}_{qubits}q_d{depth}.json"
        command = [
            sys.executable,
            str(script),
            "--worker",
            "--worker-output",
            str(output),
            "--family",
            family,
            "--worker-qubits",
            str(qubits),
            "--worker-depth",
            str(depth),
            "--device",
            args.device,
            "--dmax",
            str(args.dmax),
            "--eps",
            str(args.eps),
            "--exact-max-qubits",
            str(args.exact_max_qubits),
            "--exact-atol",
            str(args.exact_atol),
        ]
        row = _empty_failure_row(family, qubits, depth, args)
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=args.timeout_seconds,
            )
            if completed.returncode == 0 and output.exists():
                row = json.loads(output.read_text())
            else:
                row["status"] = "error"
                row["error"] = (completed.stderr or completed.stdout)[-2000:]
        except subprocess.TimeoutExpired:
            row["status"] = "timeout"
            row["error"] = f"exceeded {args.timeout_seconds:.1f} seconds"
        rows.append(row)
        detail = (
            f"{row['execution_ms']:.2f} ms, bond "
            f"{row['maximum_bond_dimension_reached']}, "
            f"truncations {row['truncation_events']}"
            if row["status"] == "completed"
            else row["error"]
        )
        print(
            f"{family} {qubits}q depth {depth}: {row['status']} — {detail}",
            flush=True,
        )

    with (outdir / "mps_limit_raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows, args.timeout_seconds)
    (outdir / "mps_limit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_porcelain": _git_value("status", "--porcelain"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "qiskit": importlib.metadata.version("qiskit"),
        "profile": args.profile,
        "custom_cases": args.cases,
        "device": args.device,
        "dmax": args.dmax,
        "eps": args.eps,
        "exact_max_qubits": args.exact_max_qubits,
        "exact_atol": args.exact_atol,
        "timeout_seconds_per_case": args.timeout_seconds,
        "process_isolation": "fresh process for every topology/qubit/depth case",
    }
    (outdir / "mps_limit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    _plot(rows, outdir / "mps_limit_scaling.png")
    print(json.dumps(summary, indent=2))
    return 0 if summary["errors"] == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir")
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        type=parse_case,
        help=(
            "run only this family:qubits:depth case; repeat for multiple "
            "boundary probes"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--dmax", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--exact-max-qubits", type=int, default=20)
    parser.add_argument("--exact-atol", type=float, default=5e-5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--family", choices=FAMILIES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-qubits", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-depth", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dmax < 1 or args.eps < 0.0:
        parser.error("Dmax must be positive and epsilon non-negative")
    if args.timeout_seconds <= 0 or args.exact_max_qubits < 0:
        parser.error("timeout must be positive and exact limit non-negative")
    if args.exact_atol <= 0.0:
        parser.error("exact tolerance must be positive")
    if args.worker:
        if not all(
            value is not None
            for value in (
                args.worker_output,
                args.family,
                args.worker_qubits,
                args.worker_depth,
            )
        ):
            parser.error("worker arguments are incomplete")
        return _worker(args)
    if not args.outdir:
        parser.error("--outdir is required")
    return _campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
