#!/usr/bin/env python3
"""Qiskit Aer CPU/GPU and Statevector/MPS baseline capture."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from importlib import metadata
from pathlib import Path


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


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


def _availability() -> dict:
    return {
        "qiskit": {
            "available": importlib.util.find_spec("qiskit") is not None,
            "version": _version("qiskit"),
        },
        "qiskit-aer": {
            "available": importlib.util.find_spec("qiskit_aer") is not None,
            "version": _version("qiskit-aer"),
        },
    }


def _write_unavailable(outdir: Path, availability: dict) -> None:
    payload = {
        "status": "unavailable",
        "reason": "qiskit and/or qiskit-aer is not installed in this Python environment",
        "availability": availability,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (outdir / "qiskit_aer_baseline_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _qft_circuit(QuantumCircuit, n: int):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(math.pi / (2 ** (k - j)), k, j)
    return qc


def _qaoa_ring_circuit(QuantumCircuit, n: int, layers: int):
    qc = QuantumCircuit(n)
    for layer in range(layers):
        gamma = 0.6 + 0.1 * layer
        beta = 0.4 + 0.05 * layer
        for i in range(n):
            qc.cp(gamma, i, (i + 1) % n)
        for q in range(n):
            qc.rx(2.0 * beta, q)
    return qc


def _ghz_circuit(QuantumCircuit, n: int):
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    return qc


def _grover_proxy_circuit(QuantumCircuit, n: int):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for q in range(n):
        qc.h(q)
        qc.x(q)
    for q in range(n - 1):
        qc.cz(q, q + 1)
    for q in range(n):
        qc.x(q)
        qc.h(q)
    return qc


def _phase_estimation_circuit(QuantumCircuit, n: int):
    qc = QuantumCircuit(n)
    target = n - 1
    for p in range(n - 1):
        qc.h(p)
    base = 0.4
    for p in range(n - 1):
        qc.cp(base * (2 ** p), p, target)
    for j in range(n - 2, -1, -1):
        for k in range(n - 2, j, -1):
            qc.cp(-math.pi / (2 ** (k - j)), k, j)
        qc.h(j)
    return qc


def _tfim_trotter_circuit(QuantumCircuit, n: int, trotter_steps: int = 20,
                          time_total: float = 1.0, J: float = 1.0, h: float = 0.5):
    qc = QuantumCircuit(n)
    dt = time_total / float(trotter_steps)
    for _ in range(trotter_steps):
        for i in range(n - 1):
            qc.rzz(-2.0 * J * dt, i, i + 1)
        for q in range(n):
            qc.rx(2.0 * h * dt, q)
    return qc


def _build_circuit(QuantumCircuit, benchmark: str, n: int, layers: int):
    if benchmark == "qft":
        return _qft_circuit(QuantumCircuit, n)
    if benchmark == "qaoa_ring":
        return _qaoa_ring_circuit(QuantumCircuit, n, layers)
    if benchmark == "ghz":
        return _ghz_circuit(QuantumCircuit, n)
    if benchmark == "grover_proxy":
        return _grover_proxy_circuit(QuantumCircuit, n)
    if benchmark == "phase_estimation":
        return _phase_estimation_circuit(QuantumCircuit, n)
    if benchmark == "tfim_trotter":
        return _tfim_trotter_circuit(QuantumCircuit, n)
    raise ValueError(f"unsupported benchmark: {benchmark}")


def run(args: argparse.Namespace) -> dict:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    availability = _availability()
    if not (availability["qiskit"]["available"] and availability["qiskit-aer"]["available"]):
        _write_unavailable(outdir, availability)
        return {"status": "unavailable", "availability": availability}

    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    qubits = [int(x) for x in args.qubits.split(",") if x.strip()]
    benchmarks = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    
    backend_label = f"qiskit_aer_{args.method}_{args.device.lower()}"
    simulator = AerSimulator(method=args.method, device=args.device)
    
    raw_rows: list[dict] = []
    summaries: list[dict] = []
    for benchmark in benchmarks:
        for n in qubits:
            # Skip large qubits for statevector if too memory intensive
            if args.method == "statevector" and n > 28:
                print(f"Skipping {benchmark} with {n} qubits on statevector (exceeds typical memory limits)")
                continue
            try:
                qc = _build_circuit(QuantumCircuit, benchmark, n, args.layers)
                if args.method == "statevector" or n <= 25:
                    qc.save_statevector()
                else:
                    qc.measure_all()
                circuit = transpile(qc, simulator, optimization_level=0)
                measured: list[float] = []
                for run_index in range(args.warmups + args.repeats):
                    warmup = run_index < args.warmups
                    t0 = time.perf_counter()
                    result = simulator.run(circuit, shots=1).result()
                    if args.method == "statevector" or n <= 25:
                        state = result.get_statevector(circuit)
                        checksum = len(state)
                    else:
                        counts = result.get_counts(circuit)
                        checksum = sum(counts.values())
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
        writer = csv.DictWriter(fh, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
        
    summary_csv = outdir / f"{backend_label}_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    manifest = {
        "status": "measured",
        "backend": backend_label,
        "availability": availability,
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
    parser.add_argument("--method", choices=["statevector", "matrix_product_state"], default="statevector")
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
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
