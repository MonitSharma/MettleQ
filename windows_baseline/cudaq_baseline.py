#!/usr/bin/env python3
"""CUDA Quantum baseline capture."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import cudaq


def _stats(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    stderr = stdev / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean_ms": mean,
        "stdev_ms": stdev,
        "stderr_ms": stderr,
        "ci95_ms": 1.96 * stderr,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def get_qft_kernel(n: int):
    @cudaq.kernel
    def qft():
        q = cudaq.qvector(n)
        for j in range(n):
            h(q[j])
            for k in range(j + 1, n):
                r1.ctrl(math.pi / (2 ** (k - j)), q[k], q[j])
    return qft


def get_qaoa_ring_kernel(n: int, layers: int):
    @cudaq.kernel
    def qaoa(gammas: list[float], betas: list[float]):
        q = cudaq.qvector(n)
        for layer in range(layers):
            g = gammas[layer]
            b = betas[layer]
            for i in range(n):
                r1.ctrl(g, q[i], q[(i + 1) % n])
            for wire in range(n):
                rx(2.0 * b, q[wire])
    return qaoa


def get_ghz_kernel(n: int):
    @cudaq.kernel
    def ghz():
        q = cudaq.qvector(n)
        h(q[0])
        for i in range(n - 1):
            x.ctrl(q[i], q[i + 1])
    return ghz


def get_grover_proxy_kernel(n: int):
    @cudaq.kernel
    def grover():
        q = cudaq.qvector(n)
        for i in range(n):
            h(q[i])
        for i in range(n):
            h(q[i])
            x(q[i])
        for i in range(n - 1):
            z.ctrl(q[i], q[i + 1])
        for i in range(n):
            x(q[i])
            h(q[i])
    return grover


def get_phase_estimation_kernel(n: int):
    @cudaq.kernel
    def qpe():
        q = cudaq.qvector(n)
        target = n - 1
        for p in range(n - 1):
            h(q[p])
        base = 0.4
        for p in range(n - 1):
            r1.ctrl(base * (2 ** p), q[p], q[target])
        # IQFT on 0 to n-2 (decomposed without using reversed() which is unsupported in AST)
        for j in range(n - 2, -1, -1):
            for k in range(n - 2, j, -1):
                r1.ctrl(-math.pi / (2 ** (k - j)), q[k], q[j])
            h(q[j])
    return qpe


def get_tfim_trotter_kernel(n: int):
    @cudaq.kernel
    def tfim(j_dt: float, h_dt: float, steps: int):
        q = cudaq.qvector(n)
        for step in range(steps):
            for i in range(n - 1):
                # RZZ(j_dt) decomposed as CNOT - RZ - CNOT
                x.ctrl(q[i], q[i+1])
                rz(j_dt, q[i+1])
                x.ctrl(q[i], q[i+1])
            for wire in range(n):
                rx(h_dt, q[wire])
    return tfim


def _build_kernel_and_args(benchmark: str, n: int, layers: int) -> tuple:
    if benchmark == "qft":
        return get_qft_kernel(n), []
    if benchmark == "qaoa_ring":
        gammas = [0.6 + 0.1 * l for l in range(layers)]
        betas = [0.4 + 0.05 * l for l in range(layers)]
        return get_qaoa_ring_kernel(n, layers), [gammas, betas]
    if benchmark == "ghz":
        return get_ghz_kernel(n), []
    if benchmark == "grover_proxy":
        return get_grover_proxy_kernel(n), []
    if benchmark == "phase_estimation":
        return get_phase_estimation_kernel(n), []
    if benchmark == "tfim_trotter":
        trotter_steps = 20
        time_total = 1.0
        J = 1.0
        h_param = 0.5
        dt = time_total / float(trotter_steps)
        j_dt = -2.0 * J * dt
        h_dt = 2.0 * h_param * dt
        return get_tfim_trotter_kernel(n), [j_dt, h_dt, trotter_steps]
    raise ValueError(f"unsupported benchmark: {benchmark}")


def run(args: argparse.Namespace) -> dict:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Set CUDA Quantum target (validate matching list)
    target_name = args.backend
    if target_name == "qpp":
        target_name = "qpp-cpu"

    cudaq.set_target(target_name)
    backend_label = f"cudaq_{args.backend}"

    qubits = [int(x) for x in args.qubits.split(",") if x.strip()]
    benchmarks = [x.strip() for x in args.benchmarks.split(",") if x.strip()]

    raw_rows: list[dict] = []
    summaries: list[dict] = []

    for benchmark in benchmarks:
        for n in qubits:
            # Skip large qubits for CPU (qpp-cpu) to prevent memory issues
            if target_name == "qpp-cpu" and n > 28:
                print(f"Skipping {benchmark} with {n} qubits on {target_name}")
                continue
            try:
                kernel, k_args = _build_kernel_and_args(benchmark, n, args.layers)
                measured: list[float] = []
                for run_index in range(args.warmups + args.repeats):
                    warmup = run_index < args.warmups
                    t0 = time.perf_counter()

                    # Force simulation by requesting the statevector
                    state = cudaq.get_state(kernel, *k_args)
                    checksum = len(state)

                    wall_ms = (time.perf_counter() - t0) * 1000.0
                    raw_rows.append({
                        "backend": backend_label,
                        "benchmark": benchmark,
                        "qubits": n,
                        "layers": args.layers if benchmark == "qaoa_ring" else "",
                        "run_index": run_index,
                        "warmup": warmup,
                        "wall_ms": wall_ms,
                        "state_size": checksum,
                    })
                    if not warmup:
                        measured.append(wall_ms)

                summary = {
                    "backend": backend_label,
                    "benchmark": benchmark,
                    "qubits": n,
                    "layers": args.layers if benchmark == "qaoa_ring" else "",
                    "warmups": args.warmups,
                    "repeats": args.repeats,
                }
                summary.update(_stats(measured))
                summaries.append(summary)
            except Exception as e:
                print(f"Error running {benchmark} with {n} qubits on {backend_label}: {e}")

    if not raw_rows:
        return {"status": "no_results"}

    raw_csv = outdir / f"{backend_label}_raw_runs.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(raw_rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_csv = outdir / f"{backend_label}_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(summaries[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summaries)

    manifest = {
        "status": "measured",
        "backend": backend_label,
        "python": sys.version,
        "platform": platform.platform(),
        "benchmarks": benchmarks,
        "qubits": qubits,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "layers": args.layers,
        "files": {"raw_runs": raw_csv.name, "summary": summary_csv.name},
    }
    manifest_json = outdir / f"{backend_label}_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Completed benchmark for {backend_label}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--backend", default="nvidia")
    parser.add_argument("--benchmarks", default="qft,qaoa_ring,ghz,grover_proxy,phase_estimation,tfim_trotter")
    parser.add_argument("--qubits", default="15,20,25")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--layers", type=int, default=6)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
