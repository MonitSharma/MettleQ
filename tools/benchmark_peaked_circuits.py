#!/usr/bin/env python3
"""Benchmark MettleQ on exact small peaked regressions and published P9.

The small mirrored family has a known output distribution and is suitable for
matched MettleQ/Aer checks. The published 56-qubit input is run separately in
a killable process because ordinary forward MPS is not the midpoint-MPO plus
unswapping algorithm used by the successful reference solver.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path


def _parse_csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _backend(implementation, dmax, eps):
    if implementation == "mettleq_mps_cpu":
        from mettleq.integrations.qiskit import MettleQBackend

        return MettleQBackend(
            method="matrix_product_state",
            device="cpu",
            mps_max_bond_dimension=dmax,
            mps_truncation_threshold=eps,
            mps_accuracy_policy="report",
        )
    if implementation == "qiskit_aer_mps_cpu":
        from qiskit_aer import AerSimulator

        return AerSimulator(
            method="matrix_product_state",
            device="CPU",
            matrix_product_state_max_bond_dimension=dmax,
            matrix_product_state_truncation_threshold=eps,
        )
    raise ValueError(f"Unknown implementation: {implementation}")


def _run_once(backend, circuit, shots, seed):
    started = time.perf_counter()
    result = backend.run(
        circuit, shots=shots, seed_simulator=seed
    ).result()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    counts = result.get_counts()
    top_bitstring = max(counts, key=counts.get)
    return elapsed_ms, counts, top_bitstring


def run_smoke(args):
    from mettleq.peaked import build_mirrored_peaked_circuit

    implementations = ["mettleq_mps_cpu", "qiskit_aer_mps_cpu"]
    rows = []
    case_index = 0
    for n_qubits in args.qubits:
        for topology in args.topologies:
            case_index += 1
            circuit = build_mirrored_peaked_circuit(
                n_qubits,
                depth=args.depth,
                topology=topology,
                seed=args.seed,
                noise_angle=args.noise_angle,
                measure=True,
            )
            peak = circuit.metadata["peak_bitstring"]
            expected_probability = circuit.metadata[
                "expected_peak_probability"
            ]
            backends = {
                name: _backend(name, args.dmax, args.eps)
                for name in implementations
            }
            for name in implementations:
                for warmup in range(args.warmups):
                    _run_once(
                        backends[name], circuit, args.shots,
                        args.seed + warmup,
                    )
            for repeat in range(args.repeats):
                order = (
                    implementations
                    if (case_index + repeat) % 2 == 0
                    else list(reversed(implementations))
                )
                for name in order:
                    elapsed_ms, counts, top = _run_once(
                        backends[name], circuit, args.shots,
                        args.seed + 100 + repeat,
                    )
                    observed = counts.get(peak, 0) / args.shots
                    diagnostics = None
                    if name == "mettleq_mps_cpu":
                        diagnostics = backends[name].last_mps_diagnostics[-1]
                    rows.append(
                        {
                            "case": f"{topology}_{n_qubits}q",
                            "n_qubits": n_qubits,
                            "topology": topology,
                            "depth": args.depth,
                            "implementation": name,
                            "repeat": repeat,
                            "time_ms": elapsed_ms,
                            "shots": args.shots,
                            "expected_peak": peak,
                            "expected_peak_probability": expected_probability,
                            "observed_peak_probability": observed,
                            "absolute_probability_error": abs(
                                observed - expected_probability
                            ),
                            "top_bitstring": top,
                            "recovered_expected_peak": top == peak,
                            "mps_diagnostics": diagnostics,
                        }
                    )
    return rows


def _published_worker(queue, options):
    try:
        from qiskit import QuantumCircuit, qasm2
        from mettleq.integrations.qiskit import MettleQBackend
        from mettleq.peaked import (
            PUBLISHED_P9_EXPECTED_BITSTRING,
            published_p9_qasm_path,
        )

        parse_started = time.perf_counter()
        source = qasm2.load(
            str(published_p9_qasm_path()),
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        )
        parse_ms = (time.perf_counter() - parse_started) * 1000.0
        source_operation_count = len(source.data)
        max_operations = options["max_operations"]
        if max_operations is not None:
            circuit = QuantumCircuit(source.num_qubits)
            for instruction in source.data[:max_operations]:
                circuit.append(
                    instruction.operation,
                    [source.find_bit(bit).index for bit in instruction.qubits],
                    [source.find_bit(bit).index for bit in instruction.clbits],
                )
        else:
            circuit = source.copy()
        operation_count = len(circuit.data)
        circuit.measure_all()
        backend = MettleQBackend(
            method="matrix_product_state",
            device="cpu",
            mps_max_bond_dimension=options["dmax"],
            mps_truncation_threshold=options["eps"],
            mps_accuracy_policy="report",
        )
        elapsed_ms, counts, top = _run_once(
            backend, circuit, options["shots"], options["seed"]
        )
        is_full = (
            max_operations is None
            or operation_count == source_operation_count
        )
        queue.put(
            {
                "status": "complete",
                "scope": "full_published_p9" if is_full else "published_p9_prefix",
                "algorithm": "mettleq_forward_mps",
                "reference_algorithm": "midpoint_mpo_with_greedy_unswapping",
                "algorithm_matched": False,
                "n_qubits": source.num_qubits,
                "source_operation_count": source_operation_count,
                "executed_operation_count": operation_count,
                "parse_ms": parse_ms,
                "time_ms": elapsed_ms,
                "shots": options["shots"],
                "top_bitstring": top,
                "expected_peak": (
                    PUBLISHED_P9_EXPECTED_BITSTRING if is_full else None
                ),
                "recovered_expected_peak": (
                    top == PUBLISHED_P9_EXPECTED_BITSTRING if is_full else None
                ),
                "expected_peak_count": (
                    counts.get(PUBLISHED_P9_EXPECTED_BITSTRING, 0)
                    if is_full else None
                ),
                "mps_diagnostics": backend.last_mps_diagnostics[-1],
            }
        )
    except BaseException as exc:  # process boundary must report failures
        queue.put(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )


def run_published_attempt(args, max_operations):
    context = mp.get_context("spawn")
    queue = context.Queue()
    options = {
        "max_operations": max_operations,
        "dmax": args.dmax,
        "eps": args.eps,
        "shots": args.published_shots,
        "seed": args.seed,
    }
    process = context.Process(
        target=_published_worker, args=(queue, options), daemon=False
    )
    started = time.perf_counter()
    process.start()
    process.join(args.published_timeout_s)
    wall_ms = (time.perf_counter() - started) * 1000.0
    if process.is_alive():
        process.terminate()
        process.join(10.0)
        return {
            "status": "timeout",
            "scope": (
                "full_published_p9"
                if max_operations is None
                else "published_p9_prefix"
            ),
            "algorithm": "mettleq_forward_mps",
            "reference_algorithm": "midpoint_mpo_with_greedy_unswapping",
            "algorithm_matched": False,
            "n_qubits": 56,
            "requested_operation_count": max_operations or 5807,
            "timeout_s": args.published_timeout_s,
            "wall_ms": wall_ms,
            "interpretation": (
                "A forward-MPS timeout is not a failure of the published "
                "midpoint-MPO method and is not a quantum-advantage result."
            ),
        }
    if not queue.empty():
        result = queue.get()
        result["wall_ms"] = wall_ms
        return result
    return {
        "status": "error",
        "scope": "full_published_p9" if max_operations is None else "published_p9_prefix",
        "exit_code": process.exitcode,
        "wall_ms": wall_ms,
        "error": "worker exited without returning a result",
    }


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row["case"], row["implementation"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (case, implementation), group in sorted(grouped.items()):
        summary.append(
            {
                "case": case,
                "n_qubits": group[0]["n_qubits"],
                "topology": group[0]["topology"],
                "implementation": implementation,
                "median_time_ms": statistics.median(
                    row["time_ms"] for row in group
                ),
                "median_peak_probability": statistics.median(
                    row["observed_peak_probability"] for row in group
                ),
                "max_absolute_probability_error": max(
                    row["absolute_probability_error"] for row in group
                ),
                "peak_recovery_rate": sum(
                    row["recovered_expected_peak"] for row in group
                ) / len(group),
            }
        )
    by_case = {}
    for row in summary:
        by_case.setdefault(row["case"], {})[row["implementation"]] = row
    for implementations in by_case.values():
        mettleq = implementations.get("mettleq_mps_cpu")
        aer = implementations.get("qiskit_aer_mps_cpu")
        if mettleq and aer:
            speedup = aer["median_time_ms"] / mettleq["median_time_ms"]
            mettleq["aer_over_mettleq_speedup"] = speedup
            aer["aer_over_mettleq_speedup"] = speedup
    return summary


def write_csv(path, rows):
    if not rows:
        return
    fields = [key for key in rows[0] if key != "mps_diagnostics"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def plot_summary(path, summary):
    import matplotlib.pyplot as plt
    import numpy as np

    cases = sorted({row["case"] for row in summary})
    implementations = ["mettleq_mps_cpu", "qiskit_aer_mps_cpu"]
    labels = ["MettleQ MPS CPU", "Qiskit Aer MPS CPU"]
    colors = ["#7557ff", "#34a0a4"]
    x = np.arange(len(cases))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(9, len(cases) * 0.75), 5.4))
    for index, (implementation, label, color) in enumerate(
        zip(implementations, labels, colors)
    ):
        lookup = {
            row["case"]: row["median_time_ms"]
            for row in summary
            if row["implementation"] == implementation
        }
        values = [lookup[case] for case in cases]
        axis.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=label,
            color=color,
        )
    axis.set_yscale("log")
    axis.set_ylabel("Median end-to-end time (ms, log scale)")
    axis.set_xticks(x, cases, rotation=35, ha="right")
    axis.set_title("MettleQ mirrored peaked-circuit regression")
    axis.grid(axis="y", alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qubits", type=lambda value: _parse_csv(value, int), default=[6, 8, 10])
    parser.add_argument("--topologies", type=lambda value: _parse_csv(value), default=["linear", "grid", "long_range", "all_to_all"])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--noise-angle", type=float, default=0.16)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=153)
    parser.add_argument("--dmax", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument(
        "--published-mode",
        choices=["none", "prefix", "full", "both"],
        default="none",
    )
    parser.add_argument("--published-prefix-operations", type=int, default=250)
    parser.add_argument("--published-shots", type=int, default=100)
    parser.add_argument("--published-timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    if args.repeats < 1 or args.warmups < 0 or args.shots < 1:
        parser.error("repeats/shots must be positive and warmups non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_smoke(args)
    summary = summarize(rows)
    published = []
    if args.published_mode in ("prefix", "both"):
        published.append(
            run_published_attempt(args, args.published_prefix_operations)
        )
    if args.published_mode in ("full", "both"):
        published.append(run_published_attempt(args, None))

    write_csv(args.output_dir / "peaked_raw.csv", rows)
    write_csv(args.output_dir / "peaked_summary.csv", summary)
    plot_summary(args.output_dir / "peaked_comparison.png", summary)
    manifest = {
        "benchmark": "mettleq_peaked_circuit_evidence",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "parameters": vars(args) | {"output_dir": str(args.output_dir)},
        "scope_note": (
            "The mirrored family is a correctness/performance regression, not "
            "a quantum-advantage claim. Published P9 uses a different reference "
            "algorithm than MettleQ forward MPS."
        ),
        "published_attempts": published,
        "summary": summary,
        "raw_rows": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "published": published}, indent=2, default=str))


if __name__ == "__main__":
    mp.freeze_support()
    main()
