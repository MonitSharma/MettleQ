#!/usr/bin/env python3
"""Matched MPS routing, CPU/GPU, and Qiskit Aer comparison campaign."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from mettleq.integrations.qiskit import MettleQEstimatorV2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_mps_limits import build_circuit, parse_case


IMPLEMENTATIONS = (
    "mettleq_cpu_routed",
    "mettleq_cpu_restore",
    "mettleq_gpu_routed",
    "qiskit_aer_cpu_mps",
)
STANDARD_CASES = (
    ("ghz_chain", 1000, 1),
    ("line_brickwork", 100, 8),
    ("ring_brickwork", 50, 2),
    ("grid_2d", 36, 2),
    ("rainbow", 32, 1),
    ("random_long_range", 32, 1),
    ("all_to_all", 20, 1),
)
SMOKE_CASES = (
    ("ghz_chain", 20, 1),
    ("line_brickwork", 12, 2),
    ("all_to_all", 10, 1),
)


def _git_value(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _observable(n_qubits: int) -> SparsePauliOp:
    return SparsePauliOp("I" * (n_qubits - 1) + "Z")


def _make_estimator(implementation: str, args):
    if implementation == "qiskit_aer_cpu_mps":
        from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

        return AerEstimatorV2(
            options={
                "default_precision": 0.0,
                "backend_options": {
                    "method": "matrix_product_state",
                    "device": "CPU",
                    "matrix_product_state_max_bond_dimension": args.dmax,
                    "matrix_product_state_truncation_threshold": args.eps,
                    "mps_log_data": True,
                },
            }
        )
    device = "gpu" if "gpu" in implementation else "cpu"
    routing = "restore" if implementation.endswith("restore") else "lookahead"
    return MettleQEstimatorV2(
        method="matrix_product_state",
        device=device,
        mps_max_bond_dimension=args.dmax,
        mps_truncation_threshold=args.eps,
        mps_svd_driver=args.svd_driver,
        mps_routing_strategy=routing,
        mps_routing_lookahead=args.routing_lookahead,
    )


def _rotate(values: list[str], offset: int) -> list[str]:
    if not values:
        return []
    offset %= len(values)
    return values[offset:] + values[:offset]


def _worker(args) -> int:
    output = Path(args.worker_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    circuit = build_circuit(args.family, args.worker_qubits, args.worker_depth)
    observable = _observable(args.worker_qubits)
    implementations = list(args.implementations)
    estimators = {}
    construction_errors = {}
    for implementation in implementations:
        try:
            estimators[implementation] = _make_estimator(implementation, args)
        except Exception as error:
            construction_errors[implementation] = repr(error)

    rows = []
    for repeat in range(args.warmups + args.repeats):
        warmup = repeat < args.warmups
        for order_index, implementation in enumerate(
            _rotate(implementations, repeat)
        ):
            row = {
                "family": args.family,
                "qubits": args.worker_qubits,
                "depth": args.worker_depth,
                "implementation": implementation,
                "repeat": repeat,
                "warmup": warmup,
                "order_index": order_index,
                "status": "completed",
                "execution_ms": None,
                "expectation_z0": None,
                "exact_expectation_z0": None,
                "exact_absolute_error": None,
                "maximum_bond_dimension_reached": None,
                "truncation_events": None,
                "relative_discarded_weight_max": None,
                "state_norm": None,
                "svd_calls": None,
                "svd_total_ms": None,
                "routing_swaps": None,
                "routing_naive_restore_swaps": None,
                "accuracy_classification": None,
                "error": None,
            }
            if implementation in construction_errors:
                row["status"] = "unavailable"
                row["error"] = construction_errors[implementation]
                rows.append(row)
                continue
            estimator = estimators[implementation]
            try:
                start = time.perf_counter_ns()
                result = estimator.run([(circuit, observable)]).result()[0]
                row["execution_ms"] = (
                    time.perf_counter_ns() - start
                ) / 1e6
                row["expectation_z0"] = float(np.asarray(result.data.evs))
                if implementation.startswith("mettleq"):
                    diagnostics = estimator.last_mps_diagnostics[0]
                    accuracy = estimator.last_mps_accuracy_reports[0]
                    for field in (
                        "maximum_bond_dimension_reached",
                        "events",
                        "relative_discarded_weight_max",
                        "state_norm",
                        "svd_calls",
                        "svd_total_ms",
                        "routing_swaps",
                        "routing_naive_restore_swaps",
                    ):
                        target = (
                            "truncation_events" if field == "events" else field
                        )
                        row[target] = diagnostics[field]
                    row["accuracy_classification"] = accuracy["classification"]
            except Exception as error:
                row["status"] = "error"
                row["error"] = repr(error)
            rows.append(row)

    exact_expectation = None
    if args.worker_qubits <= args.exact_max_qubits:
        exact = StatevectorEstimator().run([(circuit, observable)]).result()[0]
        exact_expectation = float(np.asarray(exact.data.evs))
    for row in rows:
        row["exact_expectation_z0"] = exact_expectation
        if exact_expectation is not None and row["expectation_z0"] is not None:
            row["exact_absolute_error"] = abs(
                row["expectation_z0"] - exact_expectation
            )
    output.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


def _median(rows: list[dict], implementation: str) -> float | None:
    values = [
        float(row["execution_ms"])
        for row in rows
        if row["implementation"] == implementation
        and not row["warmup"]
        and row["status"] == "completed"
    ]
    return statistics.median(values) if values else None


def _summaries(rows: list[dict], cases) -> list[dict]:
    summaries = []
    for family, qubits, depth in cases:
        selected = [
            row
            for row in rows
            if row["family"] == family
            and int(row["qubits"]) == qubits
            and int(row["depth"]) == depth
        ]
        medians = {
            implementation: _median(selected, implementation)
            for implementation in IMPLEMENTATIONS
        }
        routed = medians["mettleq_cpu_routed"]
        restore = medians["mettleq_cpu_restore"]
        gpu = medians["mettleq_gpu_routed"]
        aer = medians["qiskit_aer_cpu_mps"]
        routed_rows = [
            row
            for row in selected
            if row["implementation"] == "mettleq_cpu_routed"
            and not row["warmup"]
            and row["status"] == "completed"
        ]
        summaries.append(
            {
                "family": family,
                "qubits": qubits,
                "depth": depth,
                **{f"{key}_median_ms": value for key, value in medians.items()},
                "routing_speedup": (
                    restore / routed if routed and restore else None
                ),
                "cpu_over_gpu_speedup": gpu / routed if routed and gpu else None,
                "mettleq_over_aer_speedup": aer / routed if routed and aer else None,
                "maximum_mettleq_exact_error": max(
                    (
                        float(row["exact_absolute_error"])
                        for row in routed_rows
                        if row["exact_absolute_error"] is not None
                    ),
                    default=None,
                ),
                "routed_swaps": (
                    routed_rows[-1]["routing_swaps"] if routed_rows else None
                ),
                "restore_baseline_swaps": (
                    routed_rows[-1]["routing_naive_restore_swaps"]
                    if routed_rows
                    else None
                ),
                "accuracy_classification": (
                    routed_rows[-1]["accuracy_classification"]
                    if routed_rows
                    else None
                ),
            }
        )
    return summaries


def _plot(summaries: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    # Frozen pre-rename summaries retain their original schema. Plot them with
    # the canonical MettleQ labels without rewriting the measured source data.
    normalized = []
    for source in summaries:
        row = dict(source)
        for old, new in (
            ("qupertino_cpu_routed_median_ms", "mettleq_cpu_routed_median_ms"),
            ("qupertino_cpu_restore_median_ms", "mettleq_cpu_restore_median_ms"),
            ("qupertino_gpu_routed_median_ms", "mettleq_gpu_routed_median_ms"),
            ("qupertino_over_aer_speedup", "mettleq_over_aer_speedup"),
            ("maximum_qupertino_exact_error", "maximum_mettleq_exact_error"),
        ):
            if new not in row and old in row:
                row[new] = row[old]
        normalized.append(row)
    summaries = normalized

    labels = [
        f"{row['family'].replace('_', ' ')}\n{row['qubits']}q d{row['depth']}"
        for row in summaries
    ]
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    x = np.arange(len(labels))
    styles = (
        ("mettleq_cpu_routed_median_ms", "MettleQ CPU routed", "#2563eb"),
        ("mettleq_cpu_restore_median_ms", "MettleQ CPU restore", "#60a5fa"),
        ("mettleq_gpu_routed_median_ms", "MettleQ GPU tensors", "#7c3aed"),
        ("qiskit_aer_cpu_mps_median_ms", "Qiskit Aer CPU MPS", "#ea580c"),
    )
    width = 0.19
    for offset, (field, label, color) in enumerate(styles):
        values = [row[field] if row[field] is not None else np.nan for row in summaries]
        axes[0].bar(x + (offset - 1.5) * width, values, width, label=label, color=color)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median Qiskit Estimator execution (ms, log scale)")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].grid(True, axis="y", which="both", alpha=0.22)
    axes[0].legend(fontsize=8)

    comparisons = (
        ("routing_speedup", "routing vs restore", "#2563eb"),
        ("cpu_over_gpu_speedup", "CPU vs GPU tensors", "#7c3aed"),
        ("mettleq_over_aer_speedup", "MettleQ vs Aer", "#ea580c"),
    )
    for field, label, color in comparisons:
        axes[1].plot(
            x,
            [row[field] if row[field] is not None else np.nan for row in summaries],
            marker="o",
            label=label,
            color=color,
        )
    axes[1].axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Speedup (>1 favors routed MettleQ CPU)")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(fontsize=8)
    figure.suptitle("Matched MPS paths through Qiskit EstimatorV2")
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _campaign(args) -> int:
    outdir = Path(args.outdir)
    cells = outdir / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    cases = args.cases or (
        SMOKE_CASES if args.profile == "smoke" else STANDARD_CASES
    )
    rows = []
    for family, qubits, depth in cases:
        output = cells / f"{family}_{qubits}q_d{depth}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-output",
            str(output),
            "--family",
            family,
            "--worker-qubits",
            str(qubits),
            "--worker-depth",
            str(depth),
            "--dmax",
            str(args.dmax),
            "--eps",
            str(args.eps),
            "--svd-driver",
            args.svd_driver,
            "--routing-lookahead",
            str(args.routing_lookahead),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--exact-max-qubits",
            str(args.exact_max_qubits),
            "--implementations",
            *args.implementations,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=args.timeout_seconds,
            )
            if completed.returncode == 0 and output.exists():
                case_rows = json.loads(output.read_text())
                rows.extend(case_rows)
                statuses = sorted({row["status"] for row in case_rows})
                print(
                    f"{family} {qubits}q d{depth}: {', '.join(statuses)}",
                    flush=True,
                )
            else:
                print(
                    f"{family} {qubits}q d{depth}: worker error — "
                    f"{(completed.stderr or completed.stdout)[-1000:]}",
                    flush=True,
                )
        except subprocess.TimeoutExpired:
            print(
                f"{family} {qubits}q d{depth}: exceeded "
                f"{args.timeout_seconds:.1f} s",
                flush=True,
            )
    if not rows:
        return 2
    with (outdir / "mps_phase8_raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summaries = _summaries(rows, cases)
    with (outdir / "mps_phase8_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summaries[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summaries)
    (outdir / "mps_phase8_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
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
        "qiskit_aer": importlib.metadata.version("qiskit-aer"),
        "cases": cases,
        "implementations": args.implementations,
        "dmax": args.dmax,
        "eps": args.eps,
        "svd_driver": args.svd_driver,
        "routing_lookahead": args.routing_lookahead,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "timeout_seconds_per_case": args.timeout_seconds,
        "protocol": (
            "same Qiskit circuit and analytic Z0 EstimatorV2 contract; "
            "implementation order rotates each repeat; fresh process per case"
        ),
    }
    (outdir / "mps_phase8_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    _plot(summaries, outdir / "mps_phase8_comparison.png")
    print(json.dumps(summaries, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir")
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--case", dest="cases", action="append", type=parse_case)
    parser.add_argument("--dmax", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--svd-driver", choices=("auto", "gesdd", "gesvd", "numpy"), default="auto")
    parser.add_argument("--routing-lookahead", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--exact-max-qubits", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=IMPLEMENTATIONS,
        default=list(IMPLEMENTATIONS),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--family", help=argparse.SUPPRESS)
    parser.add_argument("--worker-qubits", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-depth", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return _worker(args)
    if not args.outdir:
        parser.error("--outdir is required")
    if args.dmax < 1 or args.eps < 0 or args.repeats < 1 or args.warmups < 0:
        parser.error("invalid MPS or repeat configuration")
    return _campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
