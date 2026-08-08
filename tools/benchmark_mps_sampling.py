#!/usr/bin/env python3
"""Compare CPU-native MPS sampling with Qiskit Aer MPS.

This is intentionally a separate protocol from the Estimator benchmark. It
measures the end-to-end final-measurement contract, including sampling, and
keeps every case in a fresh worker process so allocator and BLAS state do not
leak between cases.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from mettleq.integrations.qiskit import MettleQBackend

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_mps_limits import build_circuit


IMPLEMENTATIONS = ("mettleq_cpu_mps", "qiskit_aer_cpu_mps")
STANDARD_CASES = (
    ("ghz_chain", 32, 1),
    ("line_brickwork", 24, 8),
    ("random_long_range", 24, 2),
    ("rainbow", 24, 1),
    ("all_to_all", 16, 1),
    ("vqe_layered", 20, 4),
)
SMOKE_CASES = (
    ("ghz_chain", 12, 1),
    ("line_brickwork", 12, 2),
    ("vqe_layered", 10, 2),
)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _thread_environment() -> dict[str, str | None]:
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {name: os.environ.get(name) for name in names}


def _vqe_circuit(n_qubits: int, depth: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, name="mps_limit_vqe_layered")
    for wire in range(n_qubits):
        circuit.ry(0.11 * (wire + 1), wire)
        circuit.rz(-0.07 * (wire + 1), wire)
    for layer in range(depth):
        for wire in range(n_qubits - 1):
            circuit.cx(wire, wire + 1)
        for wire in range(n_qubits):
            circuit.ry(0.031 + 0.004 * layer + 0.0007 * wire, wire)
            circuit.rz(-0.021 + 0.002 * layer, wire)
    return circuit


def _build(family: str, n_qubits: int, depth: int) -> QuantumCircuit:
    if family == "vqe_layered":
        return _vqe_circuit(n_qubits, depth)
    return build_circuit(family, n_qubits, depth)


def _measured(circuit: QuantumCircuit) -> QuantumCircuit:
    result = circuit.copy()
    result.measure_all()
    return result


def _p_one_q0(statevector: Statevector) -> float:
    probabilities = np.asarray(statevector.probabilities(), dtype=np.float64)
    return float(probabilities[1::2].sum())


def _p_one_from_counts(counts: dict[str, int], shots: int) -> float:
    ones = 0
    for key, count in counts.items():
        text = str(key).replace(" ", "")
        value = int(text, 16) if text.startswith("0x") else int(text, 2)
        ones += int(count) * (value & 1)
    return ones / shots


def _backends(args):
    return {
        "mettleq_cpu_mps": MettleQBackend(
            method="matrix_product_state",
            device="cpu",
            allow_approximation=True,
            mps_max_bond_dimension=args.dmax,
            mps_truncation_threshold=args.eps,
            mps_svd_driver=args.svd_driver,
            mps_routing_strategy="lookahead",
            mps_routing_lookahead=args.routing_lookahead,
        ),
        "qiskit_aer_cpu_mps": __import__(
            "qiskit_aer", fromlist=["AerSimulator"]
        ).AerSimulator(
            method="matrix_product_state",
            device="CPU",
            matrix_product_state_max_bond_dimension=args.dmax,
            matrix_product_state_truncation_threshold=args.eps,
        ),
    }


def _run_backend(implementation, backend, circuit, *, shots: int, seed: int):
    measured = _measured(circuit)
    if implementation == "mettleq_cpu_mps":
        result = backend.run(
            measured, shots=shots, seed_simulator=seed
        ).result()
        counts = result.get_counts()
        diagnostics = (backend.last_mps_diagnostics or [None])[0]
    else:
        result = backend.run(
            measured, shots=shots, seed_simulator=seed
        ).result()
        counts = result.get_counts()
        diagnostics = None
    return counts, diagnostics


def _worker(args) -> int:
    output = Path(args.worker_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    circuit = _build(args.family, args.worker_qubits, args.worker_depth)
    exact_p_one = None
    if args.worker_qubits <= args.exact_max_qubits:
        exact_p_one = _p_one_q0(Statevector(circuit))
    backends = {}
    construction_errors = {}
    available_backends = None
    try:
        available_backends = _backends(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        construction_errors = {
            implementation: repr(error)
            for implementation in args.implementations
        }
    for implementation in args.implementations:
        if available_backends is not None:
            backends[implementation] = available_backends[implementation]

    rows = []
    for repeat in range(args.warmups + args.repeats):
        warmup = repeat < args.warmups
        offset = repeat % len(args.implementations)
        order = args.implementations[offset:] + args.implementations[:offset]
        for order_index, implementation in enumerate(order):
            row = {
                "family": args.family,
                "qubits": args.worker_qubits,
                "depth": args.worker_depth,
                "implementation": implementation,
                "repeat": repeat,
                "warmup": warmup,
                "order_index": order_index,
                "shots": args.shots,
                "status": "completed",
                "execution_ms": None,
                "peak_rss_bytes": None,
                "p_one_q0": None,
                "exact_p_one_q0": exact_p_one,
                "sampling_absolute_error": None,
                "maximum_bond_dimension_reached": None,
                "truncation_events": None,
                "routing_swaps": None,
                "error": None,
            }
            if implementation in construction_errors:
                row["status"] = "unavailable"
                row["error"] = construction_errors[implementation]
                rows.append(row)
                continue
            try:
                start = time.perf_counter_ns()
                counts, diagnostics = _run_backend(
                    implementation,
                    backends[implementation],
                    circuit,
                    shots=args.shots,
                    seed=args.seed + repeat,
                )
                row["execution_ms"] = (time.perf_counter_ns() - start) / 1e6
                row["peak_rss_bytes"] = _peak_rss_bytes()
                row["p_one_q0"] = _p_one_from_counts(counts, args.shots)
                if exact_p_one is not None:
                    row["sampling_absolute_error"] = abs(
                        row["p_one_q0"] - exact_p_one
                    )
                if diagnostics is not None:
                    row["maximum_bond_dimension_reached"] = diagnostics.get(
                        "maximum_bond_dimension_reached"
                    )
                    row["truncation_events"] = diagnostics.get("events")
                    row["routing_swaps"] = diagnostics.get("routing_swaps")
            except (OSError, RuntimeError, ValueError, TypeError) as error:
                row["status"] = "error"
                row["error"] = repr(error)
            rows.append(row)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


def _git_value(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _median(rows, implementation):
    values = [
        float(row["execution_ms"])
        for row in rows
        if row["implementation"] == implementation
        and not row["warmup"]
        and row["status"] == "completed"
    ]
    return statistics.median(values) if values else None


def _campaign(args) -> int:
    outdir = Path(args.outdir)
    cells = outdir / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    cases = SMOKE_CASES if args.profile == "smoke" else STANDARD_CASES
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
            "--shots",
            str(args.shots),
            "--seed",
            str(args.seed),
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
        except subprocess.TimeoutExpired:
            print(f"{family} {qubits}q d{depth}: timeout", flush=True)
            continue
        if completed.returncode != 0 or not output.exists():
            print(
                f"{family} {qubits}q d{depth}: worker error — "
                f"{(completed.stderr or completed.stdout)[-1000:]}",
                flush=True,
            )
            continue
        case_rows = json.loads(output.read_text())
        rows.extend(case_rows)
        print(
            f"{family} {qubits}q d{depth}: "
            f"{', '.join(sorted({row['status'] for row in case_rows}))}",
            flush=True,
        )
    if not rows:
        return 2
    fields = list(rows[0])
    with (outdir / "mps_sampling_raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for family, qubits, depth in cases:
        selected = [
            row for row in rows
            if row["family"] == family and int(row["qubits"]) == qubits
            and int(row["depth"]) == depth
        ]
        medians = {
            implementation: _median(selected, implementation)
            for implementation in args.implementations
        }
        summaries.append({
            "family": family,
            "qubits": qubits,
            "depth": depth,
            **{f"{key}_median_ms": value for key, value in medians.items()},
            "aer_over_mettleq_speedup": (
                medians["mettleq_cpu_mps"] / medians["qiskit_aer_cpu_mps"]
                if medians.get("mettleq_cpu_mps")
                and medians.get("qiskit_aer_cpu_mps") else None
            ),
            "maximum_sampling_absolute_error": max(
                (float(row["sampling_absolute_error"]) for row in selected
                 if row["sampling_absolute_error"] is not None),
                default=None,
            ),
        })
    (outdir / "mps_sampling_summary.json").write_text(
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
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "qiskit": _package_version("qiskit"),
        "qiskit_aer": _package_version("qiskit-aer"),
        "thread_environment": _thread_environment(),
        "machine_note": args.machine_note,
        "thermal_state": args.thermal_state,
        "cases": cases,
        "implementations": args.implementations,
        "shots": args.shots,
        "seed": args.seed,
        "dmax": args.dmax,
        "eps": args.eps,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "protocol": (
            "same final-measurement circuit and shot count; order rotates each "
            "repeat; one fresh process per case; exact p(q0=1) only for small cases"
        ),
    }
    (outdir / "mps_sampling_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir")
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--dmax", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--svd-driver", choices=("auto", "gesdd", "gesvd", "numpy"), default="auto")
    parser.add_argument("--routing-lookahead", type=int, default=8)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7919)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--exact-max-qubits", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--machine-note", default="")
    parser.add_argument("--thermal-state", default="unspecified")
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
    if args.dmax < 1 or args.eps < 0 or args.shots < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error("invalid benchmark configuration")
    return _campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
