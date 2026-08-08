"""Measure CPU-thread scaling for MettleQ MPS and Qiskit Aer MPS.

Each thread setting runs in a fresh Python process because Accelerate and
OpenMP read their thread limits at import/runtime boundaries. MettleQ uses
the process environment for the Apple Accelerate path; Aer receives the
corresponding ``mps_omp_threads`` backend option.

Example:
    PYTHONPATH=src .venv/bin/python tools/benchmark_mps_cpu_threads.py \
        --out /tmp/mettleq-mps-thread-scaling.json --repeats 5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_mps_limits import parse_case


DEFAULT_CASES = [
    "rainbow:32:1",
    "random_long_range:32:1",
    "all_to_all:20:1",
    "grid_2d:36:2",
    "ghz_chain:1000:1",
]


def _worker(args: argparse.Namespace) -> dict:
    import numpy as np
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

    from mettleq.integrations.qiskit import MettleQEstimatorV2
    from tools.benchmark_mps_limits import build_circuit

    def observable(n_qubits: int) -> SparsePauliOp:
        label = ["I"] * n_qubits
        label[0] = "Z"
        return SparsePauliOp.from_list([("".join(label), 1.0)])

    rows = []
    for case in args.cases:
        family, qubits, depth = parse_case(case)
        circuit = build_circuit(family, qubits, depth)
        obs = observable(qubits)
        estimators = {
            "mettleq_cpu_mps": MettleQEstimatorV2(
                method="mps",
                allow_approximation=True,
                mps_max_bond_dimension=64,
                mps_truncation_threshold=1e-10,
            ),
            "qiskit_aer_cpu_mps": AerEstimatorV2(
                options={
                    "default_precision": 0.0,
                    "backend_options": {
                        "method": "matrix_product_state",
                        "device": "CPU",
                        "matrix_product_state_max_bond_dimension": 64,
                        "matrix_product_state_truncation_threshold": 1e-10,
                        "mps_omp_threads": args.threads,
                        "mps_parallel_threshold": 14,
                        "max_parallel_experiments": 1,
                        "max_parallel_shots": 1,
                    },
                }
            ),
        }
        for name, estimator in estimators.items():
            estimator.run([(circuit, obs)]).result()
            samples = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = estimator.run([(circuit, obs)]).result()[0]
                samples.append((time.perf_counter() - started) * 1000.0)
            row = {
                "case": case,
                "implementation": name,
                "threads": args.threads,
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
                "expectation": float(np.asarray(result.data.evs)),
            }
            if name.startswith("mettleq"):
                diagnostics = estimator.last_mps_diagnostics[0]
                row.update(
                    {
                        "routing_swaps": diagnostics["routing_swaps"],
                        "maximum_bond_dimension": diagnostics[
                            "maximum_bond_dimension_reached"
                        ],
                        "routing_planner": diagnostics.get("routing_planner"),
                    }
                )
            rows.append(row)
    return {"threads": args.threads, "rows": rows}


def _run_parent(args: argparse.Namespace) -> dict:
    all_rows = []
    for threads in args.thread_counts:
        env = os.environ.copy()
        for variable in (
            "VECLIB_MAXIMUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            env[variable] = str(threads)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--threads",
            str(threads),
            "--repeats",
            str(args.repeats),
        ]
        for case in args.cases:
            command.extend(["--case", case])
        completed = subprocess.run(
            command,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        marker = next(
            line for line in completed.stdout.splitlines()
            if line.startswith("THREAD_BENCHMARK_JSON=")
        )
        all_rows.extend(json.loads(marker.split("=", 1)[1])["rows"])
    return {
        "thread_counts": args.thread_counts,
        "repeats": args.repeats,
        "cases": args.cases,
        "rows": all_rows,
        "environment_policy": {
            "mettleq": [
                "VECLIB_MAXIMUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            ],
            "qiskit_aer": "mps_omp_threads",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--thread-count",
        dest="thread_counts",
        type=int,
        action="append",
        default=None,
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--case", dest="cases", action="append", default=None)
    args = parser.parse_args()
    args.cases = args.cases or list(DEFAULT_CASES)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.worker:
        print("THREAD_BENCHMARK_JSON=" + json.dumps(_worker(args)))
        return 0
    args.thread_counts = args.thread_counts or [1, 2, 4, 6]
    if any(value < 1 for value in args.thread_counts):
        parser.error("thread counts must be positive")
    report = _run_parent(args)
    encoded = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n")
        print(f"Wrote {args.out}")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
