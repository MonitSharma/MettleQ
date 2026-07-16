#!/usr/bin/env python3
"""Profile MettleQ two-site CPU/SVD work and the GPU-residency boundary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import mlx.core as mx
import numpy as np

import mettleq.mps_state as current_mps
from mettleq.gates import H, RY


def _median_ms(callable_, repeats: int) -> float:
    for _ in range(3):
        callable_()
    values = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        callable_()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return float(statistics.median(values))


def _load_baseline(ref: str):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:src/mettleq/mps_state.py"], text=True
    )
    module = types.ModuleType("mettleq._profile_baseline_mps_state")
    module.__package__ = "mettleq"
    sys.modules[module.__name__] = module
    exec(compile(source, f"{ref}:mps_state.py", "exec"), module.__dict__)
    return module


def _workload(module, n_qubits: int, depth: int):
    mx.set_default_device(mx.cpu)
    state = module.MPSState(
        n_qubits, module.MPSOptions(dmax=64, eps=0.0)
    )
    for qubit in range(n_qubits):
        state.apply_single(H(), qubit)
    started = time.perf_counter()
    for layer in range(depth):
        state.apply_zz_two_sweep(0.13 + 0.017 * layer)
        for qubit in range(n_qubits):
            state.apply_single(
                RY(0.07 * (qubit + 1) + 0.01 * layer), qubit
            )
    elapsed = time.perf_counter() - started
    return elapsed, state.truncation_diagnostics()


def _workload_profile(module, repeats: int, n_qubits: int, depth: int):
    runs = []
    diagnostic = None
    for _ in range(repeats):
        elapsed, diagnostic = _workload(module, n_qubits, depth)
        runs.append(elapsed)
    return {
        "runs_s": runs,
        "median_s": statistics.median(runs),
        "diagnostics": diagnostic,
    }


def _contraction_profile(bond: int, repeats: int, seed: int):
    rng = np.random.default_rng(seed + bond)
    left_np = (
        rng.standard_normal((bond, 2, bond))
        + 1j * rng.standard_normal((bond, 2, bond))
    ).astype(np.complex64)
    right_np = (
        rng.standard_normal((bond, 2, bond))
        + 1j * rng.standard_normal((bond, 2, bond))
    ).astype(np.complex64)
    gate_np = (
        rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    ).astype(np.complex64)

    def numpy_contraction():
        tensor = np.tensordot(left_np, right_np, axes=([2], [0]))
        merged = tensor.reshape(bond, 4, bond)
        return np.tensordot(
            gate_np, merged, axes=([1], [1])
        ).transpose(1, 0, 2).reshape(2 * bond, 2 * bond)

    cpu_ms = _median_ms(numpy_contraction, repeats)
    matrix = numpy_contraction()
    svd_ms = _median_ms(
        lambda: current_mps._safe_cpu_svd(matrix, "auto"), repeats
    )

    def mlx_times(device):
        mx.set_default_device(device)
        left = mx.array(left_np)
        right = mx.array(right_np)
        gate = mx.array(gate_np)

        def contract():
            tensor = mx.tensordot(left, right, axes=([2], [0]))
            merged = mx.reshape(tensor, (bond, 4, bond))
            transformed = mx.tensordot(gate, merged, axes=([1], [1]))
            matrix_out = mx.reshape(
                mx.transpose(transformed, (1, 0, 2)),
                (2 * bond, 2 * bond),
            )
            mx.eval(matrix_out)
            return matrix_out

        resident_ms = _median_ms(contract, repeats)

        def roundtrip():
            return np.asarray(contract(), dtype=np.complex64)

        roundtrip_ms = _median_ms(roundtrip, repeats)
        return resident_ms, roundtrip_ms

    mlx_cpu_ms, mlx_cpu_roundtrip_ms = mlx_times(mx.cpu)
    gpu_resident_ms, gpu_roundtrip_ms = mlx_times(mx.gpu)
    mx.set_default_device(mx.cpu)
    return {
        "bond": bond,
        "svd_matrix_shape": f"{2 * bond}x{2 * bond}",
        "matrix_elements": 4 * bond * bond,
        "numpy_cpu_contraction_ms": cpu_ms,
        "mlx_cpu_resident_ms": mlx_cpu_ms,
        "mlx_cpu_roundtrip_ms": mlx_cpu_roundtrip_ms,
        "mlx_gpu_resident_ms": gpu_resident_ms,
        "mlx_gpu_roundtrip_ms": gpu_roundtrip_ms,
        "cpu_svd_ms": svd_ms,
        "gpu_resident_speedup_over_numpy_cpu": (
            cpu_ms / gpu_resident_ms if gpu_resident_ms else None
        ),
        "gpu_roundtrip_speedup_over_numpy_cpu": (
            cpu_ms / gpu_roundtrip_ms if gpu_roundtrip_ms else None
        ),
        "svd_fraction_of_cpu_contraction_plus_svd": svd_ms / (cpu_ms + svd_ms),
    }


def _two_consecutive_crossover(rows, key, baseline_key, speedup: float):
    streak = 0
    first = None
    for row in sorted(rows, key=lambda item: item["bond"]):
        qualifies = row[key] * speedup <= row[baseline_key]
        if qualifies:
            streak += 1
            if first is None:
                first = row["bond"]
            if streak >= 2:
                return first
        else:
            streak = 0
            first = None
    return None


def _plot(path: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    bonds = [row["bond"] for row in rows]
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    for key, label, color in (
        ("numpy_cpu_contraction_ms", "NumPy CPU contraction", "#7557ff"),
        ("mlx_gpu_resident_ms", "MLX GPU resident contraction", "#34a0a4"),
        ("mlx_gpu_roundtrip_ms", "MLX GPU + CPU readback", "#e76f51"),
        ("cpu_svd_ms", "CPU SVD", "#264653"),
    ):
        axis.plot(bonds, [row[key] for row in rows], "o-", label=label, color=color)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(bonds, [str(bond) for bond in bonds])
    axis.set_xlabel("MPS bond dimension (square split is 2D × 2D)")
    axis.set_ylabel("Median time (ms, log scale)")
    axis.set_title("MettleQ two-site CPU/GPU residency profile")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--bonds", default="8,16,32,64,128")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--workload-repeats", type=int, default=5)
    parser.add_argument("--workload-qubits", type=int, default=18)
    parser.add_argument("--workload-depth", type=int, default=9)
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()
    bonds = [int(value) for value in args.bonds.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_baseline(args.baseline_ref)
    baseline_profile = _workload_profile(
        baseline,
        args.workload_repeats,
        args.workload_qubits,
        args.workload_depth,
    )
    current_profile = _workload_profile(
        current_mps,
        args.workload_repeats,
        args.workload_qubits,
        args.workload_depth,
    )
    rows = [
        _contraction_profile(bond, args.repeats, args.seed) for bond in bonds
    ]
    resident_crossover = _two_consecutive_crossover(
        rows,
        "mlx_gpu_resident_ms",
        "numpy_cpu_contraction_ms",
        1.10,
    )
    roundtrip_crossover = _two_consecutive_crossover(
        rows,
        "mlx_gpu_roundtrip_ms",
        "numpy_cpu_contraction_ms",
        1.10,
    )
    actual_max_elements = current_profile["diagnostics"].get(
        "two_site_max_matrix_elements", 0
    )
    resident_crossover_elements = (
        4 * resident_crossover * resident_crossover
        if resident_crossover is not None
        else None
    )
    actual_reaches_resident_crossover = bool(
        resident_crossover_elements is not None
        and actual_max_elements >= resident_crossover_elements
    )
    speedup = (
        baseline_profile["median_s"] / current_profile["median_s"]
        if current_profile["median_s"]
        else None
    )
    decision = {
        "native_gpu_mps_enabled": False,
        "gpu_resident_contraction_crossover_bond": resident_crossover,
        "gpu_roundtrip_crossover_bond": roundtrip_crossover,
        "profiled_workload_max_matrix_elements": actual_max_elements,
        "profiled_workload_reaches_resident_crossover": actual_reaches_resident_crossover,
        "reason": (
            "CPU SVD forces a GPU-to-CPU boundary after every two-site contraction; "
            "no two-consecutive-size roundtrip crossover was measured. Keep native "
            "GPU MPS disabled until SVD/truncation can remain GPU-resident."
            if roundtrip_crossover is None
            else "A roundtrip crossover was measured; validate it on end-to-end MPS circuits before enabling dispatch."
        ),
    }
    report = {
        "profile": "mettleq_mps_cpu_gpu_boundary",
        "baseline_ref": args.baseline_ref,
        "cpu_optimization": {
            "baseline": baseline_profile,
            "current": current_profile,
            "baseline_over_current_speedup": speedup,
        },
        "contractions": rows,
        "decision": decision,
    }
    (args.output_dir / "profile.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    with (args.output_dir / "contractions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(args.output_dir / "mps_cpu_gpu_profile.png", rows)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
