#!/usr/bin/env python3
"""Run the Windows-baseline circuits through MettleQ Metal at exact widths."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mettleq.benchmark_environment import capture_performance_environment
from mettleq.device import Device
from mettleq.execution import statevector_preflight
from tools.interleaved_campaign import mettleq_ops
from windows_baseline.qiskit_aer_baseline import _build_circuit


def _git(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _stats(values: list[float]) -> dict[str, float]:
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


def _sync(state) -> None:
    mx.eval(state)
    synchronize = getattr(getattr(mx, "metal", None), "synchronize", None)
    if callable(synchronize):
        synchronize()


def _release_metal_cache() -> None:
    """Release lazy-graph and allocator caches between measured cells."""
    gc.collect()
    mx.clear_cache()
    gc.collect()


def _memory_gate(n_qubits: int, minimum_headroom_gib: float) -> dict:
    preflight = statevector_preflight(n_qubits)
    memory = psutil.virtual_memory()
    minimum_headroom = int(minimum_headroom_gib * (1 << 30))
    minimum_peak = int(
        preflight.get("cost_model", {}).get(
            "minimum_input_plus_output_bytes", 0
        )
    )
    required_available = minimum_peak + minimum_headroom
    allowed = bool(
        preflight["allowed"] and memory.available >= required_available
    )
    return {
        "allowed": allowed,
        "statevector_preflight": preflight,
        "available_memory_bytes": int(memory.available),
        "minimum_headroom_bytes": minimum_headroom,
        "minimum_peak_bytes": minimum_peak,
        "required_available_bytes": required_available,
    }


def _reference_error(benchmark: str, n_qubits: int, candidate: np.ndarray) -> float:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    circuit = _build_circuit(QuantumCircuit, benchmark, n_qubits, 6)
    reference = np.asarray(Statevector.from_instruction(circuit).data)
    overlap = np.vdot(reference, candidate)
    phase = np.exp(-1j * np.angle(overlap)) if abs(overlap) else 1.0
    return float(np.max(np.abs(reference - candidate * phase), initial=0.0))


def run(args: argparse.Namespace) -> int:
    git_snapshot = {
        "commit": _git("rev-parse", "HEAD"),
        # Evidence directories are intentionally untracked until the campaign
        # completes; report whether tracked source/configuration was modified.
        "status_porcelain": _git(
            "status", "--porcelain", "--untracked-files=no"
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    widths = [int(item) for item in args.qubits.split(",") if item.strip()]
    workloads = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    os.environ["METTLEQ_METAL_KERNELS"] = "1"
    os.environ["METTLEQ_RX_RADIX16"] = "1"
    os.environ["METTLEQ_CHAINPHASE_RX_FUSION"] = "1"

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    gates: list[dict] = []
    for benchmark in workloads:
        for n_qubits in widths:
            memory_gate = _memory_gate(n_qubits, args.minimum_headroom_gib)
            gates.append({
                "benchmark": benchmark,
                "qubits": n_qubits,
                **memory_gate,
            })
            if not memory_gate["allowed"]:
                print(f"REFUSED {benchmark} {n_qubits}q: memory gate", flush=True)
                continue
            operations = mettleq_ops(benchmark, n_qubits)
            samples: list[float] = []
            last_state = None
            last_plan = None
            for run_index in range(args.warmups + args.repeats):
                warmup = run_index < args.warmups
                device = Device(n_qubits)
                started = time.perf_counter()
                device.execute(operations, report=True)
                _sync(device.sim.state)
                elapsed_ms = (time.perf_counter() - started) * 1_000.0
                last_state = device.sim.state
                last_plan = device.last_execution_plan
                row = {
                    "backend": "mettleq_metal",
                    "benchmark": benchmark,
                    "qubits": n_qubits,
                    "layers": 6 if benchmark == "qaoa_ring" else "",
                    "run_index": run_index,
                    "warmup": warmup,
                    "wall_ms": elapsed_ms,
                    "state_size": int(last_state.size),
                    "expected_metal_launches": int(
                        last_plan["expected_custom_kernel_launches"]
                    ),
                }
                raw_rows.append(row)
                if not warmup:
                    samples.append(elapsed_ms)
                if run_index + 1 < args.warmups + args.repeats:
                    del device
                    _release_metal_cache()
            assert last_state is not None and last_plan is not None
            norm = float(mx.sum(mx.abs(last_state) ** 2).item())
            max_error = ""
            if n_qubits <= args.validate_max_qubits:
                candidate = np.asarray(last_state)
                max_error = _reference_error(benchmark, n_qubits, candidate)
                if max_error > args.accuracy_atol:
                    raise AssertionError(
                        f"{benchmark} {n_qubits}q error {max_error} exceeds "
                        f"{args.accuracy_atol}"
                    )
            summary = {
                "backend": "mettleq_metal",
                "benchmark": benchmark,
                "qubits": n_qubits,
                "layers": 6 if benchmark == "qaoa_ring" else "",
                "warmups": args.warmups,
                "repeats": args.repeats,
                "result_contract": "synchronized_full_state_complex64",
                "expected_metal_launches": int(
                    last_plan["expected_custom_kernel_launches"]
                ),
                "state_norm": norm,
                "max_reference_error": max_error,
            }
            summary.update(_stats(samples))
            summary_rows.append(summary)
            print(
                f"DONE {benchmark} {n_qubits}q: "
                f"{summary['mean_ms']:.3f} ms, "
                f"launches={summary['expected_metal_launches']}, "
                f"norm={norm:.9f}",
                flush=True,
            )
            del device, last_state
            _release_metal_cache()

    if not raw_rows:
        raise RuntimeError("no benchmark cells completed")
    raw_path = output / "mettleq_metal_raw_runs.csv"
    summary_path = output / "mettleq_metal_summary.csv"
    for path, rows in ((raw_path, raw_rows), (summary_path, summary_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "status": "measured",
        "backend": "mettleq_metal",
        "result_contract": "synchronized_full_state_complex64",
        "benchmarks": workloads,
        "qubits": widths,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "accuracy_atol": args.accuracy_atol,
        "validated_through_qubits": args.validate_max_qubits,
        "git": git_snapshot,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("mettleq", "mlx", "numpy", "qiskit")
        },
        "environment": capture_performance_environment(),
        "memory_gates": gates,
        "optimizations": {
            "metal_kernels": True,
            "rx_radix16": True,
            "chain_phase_rx_fusion": True,
        },
        "files": {"raw_runs": raw_path.name, "summary": summary_path.name},
    }
    (output / "mettleq_metal_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qubits", default="15,20,24,26,28")
    parser.add_argument(
        "--benchmarks",
        default=(
            "qft,qaoa_ring,ghz,grover_proxy,phase_estimation,tfim_trotter"
        ),
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--validate-max-qubits", type=int, default=20)
    parser.add_argument("--accuracy-atol", type=float, default=5e-6)
    parser.add_argument("--minimum-headroom-gib", type=float, default=6.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
