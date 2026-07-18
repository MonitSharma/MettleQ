#!/usr/bin/env python3
"""Safely probe dense Aer CPU and MettleQ Apple-GPU capacity boundaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ENGINES = ("aer_cpu_statevector", "mettleq_gpu_statevector")


def _hardware_profile() -> dict[str, Any]:
    """Capture publishable machine facts while excluding serial identifiers."""
    memory = _memory()
    profile: dict[str, Any] = {
        "architecture": platform.machine(),
        "physical_memory_bytes": memory["total"],
    }
    try:
        raw = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        data = json.loads(raw)
        hardware = data.get("SPHardwareDataType", [{}])[0]
        display = data.get("SPDisplaysDataType", [{}])[0]
        profile.update(
            {
                "model_name": hardware.get("machine_name"),
                "model_identifier": hardware.get("machine_model"),
                "chip": hardware.get("chip_type"),
                "cpu_cores": hardware.get("number_processors"),
                "memory": hardware.get("physical_memory"),
                "gpu": display.get("sppci_model"),
                "gpu_cores": display.get("sppci_cores"),
                "metal_support": display.get("spdisplays_mtlgpufamilysupport"),
            }
        )
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        pass
    return {key: value for key, value in profile.items() if value is not None}


def _memory() -> dict[str, int]:
    import psutil

    value = psutil.virtual_memory()
    return {"total": int(value.total), "available": int(value.available)}


def _preflight(engine: str, width: int, args) -> dict[str, Any]:
    amplitudes = 1 << width
    if engine == "aer_cpu_statevector":
        state = 16 * amplitudes
        projected = int(1.5 * state)
        basis = "Aer complex128 state plus 50% execution/allocator guard"
    else:
        state = 8 * amplitudes
        projected = int(2.2 * state)
        basis = "MettleQ complex64 input/output states plus 10% guard"
    memory = _memory()
    fraction_limit = int(memory["total"] * args.maximum_memory_fraction)
    headroom = int(args.minimum_headroom_gib * 2**30)
    allowed = (
        projected <= fraction_limit
        and projected + headroom <= memory["available"]
    )
    return {
        "engine": engine,
        "width": width,
        "allowed": allowed,
        "state_bytes": state,
        "projected_peak_bytes": projected,
        "basis": basis,
        "physical_memory_bytes": memory["total"],
        "available_memory_bytes": memory["available"],
        "fraction_limit_bytes": fraction_limit,
        "minimum_headroom_bytes": headroom,
    }


def _circuit(width: int, depth: int):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(width, name=f"safe_dense_{width}q_d{depth}")
    for layer in range(depth):
        for wire in range(width):
            circuit.ry(0.011 * (wire + 1) * (layer + 1), wire)
            circuit.rz(-0.006 * (wire + 2) * (layer + 1), wire)
        for wire in range(layer % 2, width - 1, 2):
            circuit.cx(wire, wire + 1)
    return circuit


def _worker(args) -> int:
    output = Path(args.worker_output)
    circuit = _circuit(args.worker_width, args.depth)
    started = time.perf_counter()
    if args.worker_engine == "aer_cpu_statevector":
        from qiskit import transpile
        from qiskit.quantum_info import Pauli
        from qiskit_aer import AerSimulator

        backend = AerSimulator(
            method="statevector",
            device="CPU",
            precision="double",
            enable_truncation=False,
            max_memory_mb=int(args.worker_memory_limit_bytes / 2**20),
        )
        circuit.save_expectation_value(Pauli("Z"), [0], label="z0")
        compiled = transpile(circuit, backend, optimization_level=1)
        result = backend.run(compiled, shots=None).result()
        expectation = float(np.real(result.data(0)["z0"]))
        metadata = result.results[0].metadata
        evidence = {
            "parallel_threads": metadata.get("parallel_state_update"),
            "method": metadata.get("method"),
            "device": metadata.get("device"),
        }
        peak_bytes = None
    else:
        import mlx.core as mx
        from qiskit.quantum_info import SparsePauliOp
        from mettleq.integrations.qiskit import MettleQEstimatorV2

        mx.set_memory_limit(args.worker_memory_limit_bytes)
        mx.set_cache_limit(0)
        mx.reset_peak_memory()
        estimator = MettleQEstimatorV2(
            method="statevector",
            device="gpu",
            metal_checkpoint_budget_bytes=args.worker_memory_limit_bytes,
        )
        observable = SparsePauliOp("I" * (args.worker_width - 1) + "Z")
        result = estimator.run([(circuit, observable)]).result()[0]
        expectation = float(np.asarray(result.data.evs))
        peak_bytes = int(mx.get_peak_memory())
        selection = estimator.last_execution_selections[0]
        evidence = {
            "selection": selection,
            "metal_kernels_policy": os.environ.get(
                "METTLEQ_METAL_KERNELS", "auto_default"
            ),
        }
    row = {
        "status": "completed",
        "engine": args.worker_engine,
        "width": args.worker_width,
        "depth": args.depth,
        "execution_ms": (time.perf_counter() - started) * 1000.0,
        "expectation_z0": expectation,
        "engine_peak_bytes": peak_bytes,
        "evidence": evidence,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    return 0


def _run_monitored(command, *, timeout_s: float, critical_available_bytes: int):
    import psutil

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"},
    )
    child = psutil.Process(process.pid)
    started = time.monotonic()
    peak_rss = 0
    minimum_available = _memory()["available"]
    reason = None
    while process.poll() is None:
        try:
            peak_rss = max(peak_rss, int(child.memory_info().rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        available = _memory()["available"]
        minimum_available = min(minimum_available, available)
        if available < critical_available_bytes:
            reason = "system_headroom_guard_triggered"
            process.terminate()
            break
        if time.monotonic() - started > timeout_s:
            reason = "timeout"
            process.terminate()
            break
        time.sleep(0.25)
    try:
        stdout, _ = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, _ = process.communicate()
        reason = reason or "forced_kill_after_termination"
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "peak_rss_bytes": peak_rss,
        "minimum_system_available_bytes": minimum_available,
        "monitor_stop_reason": reason,
    }


def _plot(path: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9.5, 5.4))
    colors = {
        "aer_cpu_statevector": "#355c7d",
        "mettleq_gpu_statevector": "#6c5ce7",
    }
    for engine in ENGINES:
        completed = [
            row for row in rows
            if row["engine"] == engine and row["status"] == "completed"
        ]
        if completed:
            axis.plot(
                [row["width"] for row in completed],
                [row["execution_ms"] for row in completed],
                marker="o",
                linewidth=2,
                color=colors[engine],
                label=engine.replace("_", " "),
            )
        refused = [
            row for row in rows
            if row["engine"] == engine and row["status"] == "safety_refused"
        ]
        for row in refused:
            axis.scatter(
                row["width"], 1.0, marker="x", s=100, color=colors[engine]
            )
            axis.annotate(
                "safety refusal", (row["width"], 1.0),
                xytext=(0, 8), textcoords="offset points", ha="center",
            )
    axis.set_yscale("log")
    axis.set_xticks(sorted({row["width"] for row in rows}))
    axis.set_xlabel("Qubits")
    axis.set_ylabel("Analytic Z0 complete-call time (ms, log scale)")
    axis.set_title("Safe dense-state limit probe on Apple M3 Pro")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _write_readme(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Safe dense-state capacity probe",
        "",
        "This campaign runs each engine/width in a fresh monitored process and "
        "requests only analytic Z on qubit zero. It probes internal dense-state "
        "capacity without returning another giant host state.",
        "",
        f"Hardware: `{json.dumps(payload['hardware'], sort_keys=True)}`",
        "",
        "| Qubits | Aer CPU ms | MettleQ GPU ms | Aer / MettleQ | Absolute Z0 error |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["comparisons"]:
        lines.append(
            f"| {row['width']} | {row['aer_cpu_ms']:.3f} | "
            f"{row['mettleq_gpu_ms']:.3f} | "
            f"**{row['aer_over_mettleq']:.3f}x** | "
            f"`{row['expectation_absolute_error']:.2e}` |"
        )
    lines.extend([
        "",
        "A ratio above 1.0x favors MettleQ. Widths shown as safety refusals in "
        "the plot were never launched.",
        "",
        "The default policy caps each engine's projected peak at 45% of physical "
        "memory, requires 6 GiB available headroom before launch, terminates a "
        "worker if available memory falls below 5 GiB, disables the MLX free "
        "cache, and imposes a per-process timeout. Passing is a capacity result "
        "for this shallow structured circuit and scalar-output contract, not a "
        "guarantee for arbitrary depth or full-state return.",
        "",
        f"Protocol: `{json.dumps(payload['protocol'], sort_keys=True)}`",
        "",
    ])
    path.write_text("\n".join(lines))


def _campaign(args) -> int:
    outdir = Path(args.outdir)
    cells = outdir / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    rows = []
    worker_memory_limit = int(_memory()["total"] * args.maximum_memory_fraction)
    for width in args.widths:
        for engine in ENGINES:
            preflight = _preflight(engine, width, args)
            if not preflight["allowed"]:
                row = {
                    "status": "safety_refused",
                    "engine": engine,
                    "width": width,
                    "depth": args.depth,
                    "execution_ms": None,
                    "expectation_z0": None,
                    "preflight": preflight,
                }
                rows.append(row)
                print(f"{engine} {width}q: safety refused", flush=True)
                continue
            output = cells / f"{engine}_{width}q.json"
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--worker-engine", engine, "--worker-width", str(width),
                "--worker-output", str(output), "--depth", str(args.depth),
                "--worker-memory-limit-bytes", str(worker_memory_limit),
            ]
            monitor = _run_monitored(
                command,
                timeout_s=args.timeout_seconds,
                critical_available_bytes=int(args.critical_headroom_gib * 2**30),
            )
            if monitor["returncode"] == 0 and output.exists():
                row = json.loads(output.read_text())
            else:
                row = {
                    "status": "monitor_stopped" if monitor["monitor_stop_reason"] else "error",
                    "engine": engine,
                    "width": width,
                    "depth": args.depth,
                    "execution_ms": None,
                    "expectation_z0": None,
                    "error": monitor["stdout"][-2000:],
                }
            row["preflight"] = preflight
            row["monitor"] = monitor
            rows.append(row)
            print(
                f"{engine} {width}q: {row['status']}"
                + (f" {row['execution_ms']:.1f} ms" if row.get("execution_ms") else ""),
                flush=True,
            )

    comparisons = []
    for width in args.widths:
        selected = {
            row["engine"]: row for row in rows
            if row["width"] == width and row["status"] == "completed"
        }
        if len(selected) == 2:
            aer = selected["aer_cpu_statevector"]
            mettleq = selected["mettleq_gpu_statevector"]
            comparisons.append({
                "width": width,
                "aer_cpu_ms": aer["execution_ms"],
                "mettleq_gpu_ms": mettleq["execution_ms"],
                "aer_over_mettleq": aer["execution_ms"] / mettleq["execution_ms"],
                "expectation_absolute_error": abs(
                    aer["expectation_z0"] - mettleq["expectation_z0"]
                ),
            })
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware": _hardware_profile(),
        "protocol": {
            "widths": args.widths,
            "depth": args.depth,
            "output_contract": "analytic Z on qubit zero; no dense host return",
            "maximum_memory_fraction": args.maximum_memory_fraction,
            "minimum_headroom_gib": args.minimum_headroom_gib,
            "critical_runtime_headroom_gib": args.critical_headroom_gib,
            "timeout_seconds": args.timeout_seconds,
            "process_isolation": True,
        },
        "rows": rows,
        "comparisons": comparisons,
    }
    (outdir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _plot(outdir / "safe_dense_limit.png", rows)
    _write_readme(outdir / "README.md", payload)
    print(json.dumps(comparisons, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="bench/runs/safe-dense-limits")
    parser.add_argument("--widths", type=int, nargs="+", default=[27, 28, 29, 30])
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--maximum-memory-fraction", type=float, default=0.45)
    parser.add_argument("--minimum-headroom-gib", type=float, default=6.0)
    parser.add_argument("--critical-headroom-gib", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-engine", choices=ENGINES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-width", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--worker-memory-limit-bytes", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return _worker(args)
    return _campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
