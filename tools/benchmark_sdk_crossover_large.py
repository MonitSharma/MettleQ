#!/usr/bin/env python3
"""Benchmark large exact-state SDK CPU baselines against MettleQ's Apple GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "CROSSOVER_RESULT::"


def _system_memory() -> dict[str, int]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "total_bytes": int(memory.total),
            "available_bytes": int(memory.available),
        }
    except Exception:
        total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        return {"total_bytes": total, "available_bytes": total}


def _validation_memory_preflight(
    width: int,
    *,
    maximum_fraction: float,
    minimum_headroom_bytes: int,
) -> dict[str, Any]:
    """Conservatively bound a simultaneous reference/GPU full-state check."""
    amplitudes = 1 << int(width)
    # Aer double state (16 B), MettleQ input+output (16 B), plus 4 B/amplitude
    # for result wrappers, allocator fragmentation, and validation work.
    projected_peak = 36 * amplitudes
    memory = _system_memory()
    fraction_limit = int(memory["total_bytes"] * float(maximum_fraction))
    allowed = (
        projected_peak <= fraction_limit
        and projected_peak + int(minimum_headroom_bytes)
        <= memory["available_bytes"]
    )
    return {
        "allowed": allowed,
        "width": int(width),
        "projected_peak_bytes": projected_peak,
        "physical_memory_bytes": memory["total_bytes"],
        "available_memory_bytes": memory["available_bytes"],
        "maximum_fraction": float(maximum_fraction),
        "fraction_limit_bytes": fraction_limit,
        "minimum_headroom_bytes": int(minimum_headroom_bytes),
        "model": (
            "36 bytes per amplitude: Aer double + MettleQ two-state + guard"
        ),
    }


def _git(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str]:
    from importlib.metadata import version

    return {
        name: version(name)
        for name in (
            "mettleq",
            "mlx",
            "numpy",
            "pennylane",
            "pennylane-lightning",
            "qiskit",
            "qiskit-aer",
        )
    }


def _hardware_profile() -> dict[str, Any]:
    """Capture reproducible hardware facts without recording device IDs."""
    profile: dict[str, Any] = {
        "architecture": platform.machine(),
        "physical_memory_bytes": _system_memory()["total_bytes"],
    }
    try:
        raw = subprocess.check_output(
            [
                "system_profiler",
                "SPHardwareDataType",
                "SPDisplaysDataType",
                "-json",
            ],
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


def _qiskit_circuit(n_qubits: int, depth: int):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(n_qubits, name=f"crossover_{n_qubits}q_d{depth}")
    for layer in range(depth):
        for wire in range(n_qubits):
            circuit.ry(0.01 * (layer + 1) * (wire + 1), wire)
            circuit.rz(0.007 * (layer + 2) * (wire + 1), wire)
        for wire in range(layer % 2, n_qubits - 1, 2):
            circuit.cx(wire, wire + 1)
        if layer == 1:
            for wire in range(n_qubits // 4):
                circuit.cx(wire, n_qubits - 1 - wire)
    return circuit


def _pennylane_qnode(device, n_qubits: int, depth: int):
    import pennylane as qml

    @qml.qnode(device)
    def circuit():
        for layer in range(depth):
            for wire in range(n_qubits):
                qml.RY(0.01 * (layer + 1) * (wire + 1), wires=wire)
                qml.RZ(0.007 * (layer + 2) * (wire + 1), wires=wire)
            for wire in range(layer % 2, n_qubits - 1, 2):
                qml.CNOT(wires=[wire, wire + 1])
            if layer == 1:
                for wire in range(n_qubits // 4):
                    qml.CNOT(wires=[wire, n_qubits - 1 - wire])
        return qml.state()

    return circuit


def _benchmark(function, *, warmups: int, repeats: int):
    for _ in range(warmups):
        function()
    samples_ms = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        samples_ms.append((time.perf_counter() - started) * 1_000.0)
    if result is None:
        raise ValueError("repeats must be at least one")
    return result, float(np.median(samples_ms)), samples_ms


def _phase_aligned_error_chunked(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    chunk_amplitudes: int = 1 << 20,
) -> float:
    """Compare large states without allocating another full complex128 state."""
    reference = np.asarray(reference).reshape(-1)
    candidate = np.asarray(candidate).reshape(-1)
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"statevector shapes differ: {reference.shape} != {candidate.shape}"
        )
    overlap = np.vdot(reference, candidate)
    phase = np.exp(-1j * np.angle(overlap)) if abs(overlap) > 0.0 else 1.0
    maximum = 0.0
    for start in range(0, reference.size, chunk_amplitudes):
        stop = min(reference.size, start + chunk_amplitudes)
        delta = np.asarray(reference[start:stop], dtype=np.complex128) - (
            np.asarray(candidate[start:stop], dtype=np.complex128) * phase
        )
        maximum = max(maximum, float(np.max(np.abs(delta), initial=0.0)))
    return maximum


def _run_qiskit(
    n_qubits: int,
    depth: int,
    warmups: int,
    repeats: int,
    max_mps_width: int,
) -> dict[str, Any]:
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from mettleq.integrations.qiskit import MettleQBackend

    cpu_runs: dict[str, dict[str, Any]] = {}
    methods = ["statevector"]
    if n_qubits <= max_mps_width:
        methods.append("matrix_product_state")
    for method in methods:
        backend = AerSimulator(
            method=method,
            device="CPU",
            precision="double",
            enable_truncation=False,
        )
        circuit = _qiskit_circuit(n_qubits, depth)
        circuit.save_statevector()
        compiled = transpile(circuit, backend, optimization_level=1)

        def run_reference(b=backend, c=compiled):
            return np.asarray(b.run(c, shots=None).result().get_statevector(c))

        state, median_ms, samples_ms = _benchmark(
            run_reference, warmups=warmups, repeats=repeats
        )
        cpu_runs[method] = {
            "state": state,
            "median_ms": median_ms,
            "samples_ms": samples_ms,
        }

    fastest_method = min(cpu_runs, key=lambda key: cpu_runs[key]["median_ms"])
    reference = cpu_runs[fastest_method]["state"]
    method_agreement = None
    if len(cpu_runs) == 2:
        method_agreement = _phase_aligned_error_chunked(
            cpu_runs["statevector"]["state"],
            cpu_runs["matrix_product_state"]["state"],
        )

    backend = MettleQBackend(method="statevector", device="gpu")
    circuit = _qiskit_circuit(n_qubits, depth)
    compiled = transpile(circuit, backend, optimization_level=1)

    def run_mettleq():
        return np.asarray(
            backend.run(compiled, shots=1, return_statevector=True)
            .result()
            .data(0)["statevector"]
        )

    candidate, mettleq_ms, mettleq_samples_ms = _benchmark(
        run_mettleq, warmups=warmups, repeats=repeats
    )
    error = _phase_aligned_error_chunked(reference, candidate)
    result = {
        "cpu_methods": {
            method: {
                "median_ms": run["median_ms"],
                "samples_ms": run["samples_ms"],
            }
            for method, run in cpu_runs.items()
        },
        "cpu_fastest_method": fastest_method,
        "cpu_fastest_ms": cpu_runs[fastest_method]["median_ms"],
        "mettleq_gpu_ms": mettleq_ms,
        "mettleq_gpu_samples_ms": mettleq_samples_ms,
        "cpu_over_mettleq": cpu_runs[fastest_method]["median_ms"] / mettleq_ms,
        "max_amplitude_error": error,
        "aer_method_agreement_error": method_agreement,
    }
    del reference, candidate
    for run in cpu_runs.values():
        run.pop("state", None)
    return result


def _run_pennylane(
    n_qubits: int,
    depth: int,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    import pennylane as qml
    from mettleq.integrations.pennylane import MettleQDevice

    lightning = qml.device(
        "lightning.qubit",
        wires=n_qubits,
        c_dtype=np.complex128,
    )
    reference_qnode = _pennylane_qnode(lightning, n_qubits, depth)
    reference, reference_ms, reference_samples_ms = _benchmark(
        reference_qnode, warmups=warmups, repeats=repeats
    )

    mettleq_device = MettleQDevice(
        wires=n_qubits,
        method="statevector",
        device="gpu",
    )
    mettleq_qnode = _pennylane_qnode(mettleq_device, n_qubits, depth)
    candidate, mettleq_ms, mettleq_samples_ms = _benchmark(
        mettleq_qnode, warmups=warmups, repeats=repeats
    )
    error = _phase_aligned_error_chunked(reference, candidate)
    result = {
        "cpu_method": "lightning.qubit/complex128",
        "cpu_ms": reference_ms,
        "cpu_samples_ms": reference_samples_ms,
        "mettleq_gpu_ms": mettleq_ms,
        "mettleq_gpu_samples_ms": mettleq_samples_ms,
        "cpu_over_mettleq": reference_ms / mettleq_ms,
        "max_amplitude_error": error,
    }
    del reference, candidate
    return result


def _worker(args: argparse.Namespace) -> int:
    import mlx.core as mx

    memory = _system_memory()
    mx.set_memory_limit(
        int(memory["total_bytes"] * args.mlx_memory_limit_fraction)
    )
    mx.set_cache_limit(0)
    started = time.perf_counter()
    qiskit = _run_qiskit(
        args.worker_width,
        args.depth,
        args.warmups,
        args.repeats,
        args.max_qiskit_mps_width,
    )
    pennylane = _run_pennylane(
        args.worker_width,
        args.depth,
        args.warmups,
        args.repeats,
    )
    record = {
        "width": args.worker_width,
        "depth": args.depth,
        "state_mib_complex128": (2**args.worker_width * 16) / (1024**2),
        "qiskit": qiskit,
        "pennylane": pennylane,
        "worker_wall_s": time.perf_counter() - started,
    }
    maximum_error = max(
        qiskit["max_amplitude_error"], pennylane["max_amplitude_error"]
    )
    if maximum_error > args.accuracy_atol:
        raise AssertionError(
            f"{args.worker_width}q maximum state error {maximum_error} "
            f"exceeds {args.accuracy_atol}"
        )
    print(RESULT_PREFIX + json.dumps(record, sort_keys=True), flush=True)
    return 0


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "width",
        "depth",
        "state_mib_complex128",
        "qiskit_cpu_fastest_method",
        "qiskit_cpu_fastest_ms",
        "qiskit_mettleq_gpu_ms",
        "qiskit_cpu_over_mettleq",
        "qiskit_max_amplitude_error",
        "qiskit_aer_method_agreement_error",
        "pennylane_cpu_method",
        "pennylane_cpu_ms",
        "pennylane_mettleq_gpu_ms",
        "pennylane_cpu_over_mettleq",
        "pennylane_max_amplitude_error",
        "worker_wall_s",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            qiskit = record["qiskit"]
            pennylane = record["pennylane"]
            writer.writerow(
                {
                    "width": record["width"],
                    "depth": record["depth"],
                    "state_mib_complex128": record["state_mib_complex128"],
                    "qiskit_cpu_fastest_method": qiskit["cpu_fastest_method"],
                    "qiskit_cpu_fastest_ms": qiskit["cpu_fastest_ms"],
                    "qiskit_mettleq_gpu_ms": qiskit["mettleq_gpu_ms"],
                    "qiskit_cpu_over_mettleq": qiskit["cpu_over_mettleq"],
                    "qiskit_max_amplitude_error": qiskit["max_amplitude_error"],
                    "qiskit_aer_method_agreement_error": qiskit[
                        "aer_method_agreement_error"
                    ],
                    "pennylane_cpu_method": pennylane["cpu_method"],
                    "pennylane_cpu_ms": pennylane["cpu_ms"],
                    "pennylane_mettleq_gpu_ms": pennylane["mettleq_gpu_ms"],
                    "pennylane_cpu_over_mettleq": pennylane[
                        "cpu_over_mettleq"
                    ],
                    "pennylane_max_amplitude_error": pennylane[
                        "max_amplitude_error"
                    ],
                    "worker_wall_s": record["worker_wall_s"],
                }
            )


def _plot(path: Path, records: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    widths = np.asarray([record["width"] for record in records])
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.0), sharex="col")
    panels = [
        (
            "Qiskit",
            np.asarray([record["qiskit"]["cpu_fastest_ms"] for record in records]),
            np.asarray([record["qiskit"]["mettleq_gpu_ms"] for record in records]),
            np.asarray([record["qiskit"]["cpu_over_mettleq"] for record in records]),
            np.asarray([record["qiskit"]["max_amplitude_error"] for record in records]),
        ),
        (
            "PennyLane",
            np.asarray([record["pennylane"]["cpu_ms"] for record in records]),
            np.asarray([record["pennylane"]["mettleq_gpu_ms"] for record in records]),
            np.asarray([record["pennylane"]["cpu_over_mettleq"] for record in records]),
            np.asarray([record["pennylane"]["max_amplitude_error"] for record in records]),
        ),
    ]
    for column, (framework, cpu, gpu, speedup, error) in enumerate(panels):
        top = axes[0, column]
        top.plot(widths, cpu, "o-", color="#334e68", label="Fastest CPU reference")
        top.plot(widths, gpu, "o-", color="#7557ff", label="MettleQ Apple GPU")
        top.fill_between(
            widths,
            cpu,
            gpu,
            where=gpu < cpu,
            color="#49b675",
            alpha=0.16,
            label="MettleQ faster",
        )
        top.fill_between(
            widths,
            cpu,
            gpu,
            where=gpu >= cpu,
            color="#d65f5f",
            alpha=0.10,
            label="CPU reference faster",
        )
        top.set_yscale("log")
        top.set_ylabel("Median complete-call time (ms)")
        top.set_title(f"{framework}: CPU reference vs MettleQ GPU")
        top.grid(True, which="both", alpha=0.25)
        top.legend(fontsize=8)

        bottom = axes[1, column]
        colors = ["#49b675" if value >= 1.0 else "#d65f5f" for value in speedup]
        bottom.bar(widths, speedup, width=0.65, color=colors, alpha=0.82)
        bottom.axhline(1.0, color="#17213a", linestyle="--", linewidth=1)
        for annotation_index, (width, value, observed_error) in enumerate(
            zip(widths, speedup, error)
        ):
            bottom.annotate(
                f"{value:.2f}x\nerr {observed_error:.1e}",
                (width, value),
                xytext=(0, 5 + 8 * (annotation_index % 2)),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        bottom.set_xlabel("Qubits")
        bottom.set_ylabel("CPU / MettleQ GPU")
        bottom.set_title("Above 1.0× means MettleQ is faster")
        bottom.grid(True, axis="y", alpha=0.25)
        bottom.set_xticks(widths)
        bottom.set_ylim(0.0, max(1.12, max(speedup) * 1.18))

    figure.suptitle(
        "MettleQ large exact-state crossover on Apple M3 Pro\n"
        "same circuit and full-state contract; errors are global-phase aligned",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_readme(path: Path, payload: dict[str, Any]) -> None:
    qiskit_wins = [
        record for record in payload["records"]
        if record["qiskit"]["cpu_over_mettleq"] >= 1.0
    ]
    pennylane_wins = [
        record for record in payload["records"]
        if record["pennylane"]["cpu_over_mettleq"] >= 1.0
    ]
    lines = [
        "# Large SDK CPU/GPU crossover",
        "",
        "This experiment compares complete full-state calls on one Apple M3 Pro. "
        "Qiskit uses the fastest measured CPU method at each width from Aer "
        "statevector and, where scheduled, Aer MPS. PennyLane uses "
        "`lightning.qubit` with complex128. MettleQ is forced to its exact Apple "
        "GPU statevector path through each SDK adapter.",
        "",
        f"Hardware: `{json.dumps(payload['hardware'], sort_keys=True)}`",
        "",
        "Aer circuit truncation is disabled because its subset-observable "
        "pilot disagreed with two independent full-state references; disabling "
        "truncation restored agreement. Aer MPS "
        "is not scheduled above the configured cap after full-state materialization "
        "has removed its tensor-network advantage.",
        "",
        "| Qubits | Qiskit fastest CPU | CPU ms | MettleQ GPU ms | CPU / GPU | Max state error | Lightning CPU ms | MettleQ GPU ms | CPU / GPU | Max state error |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in payload["records"]:
        qiskit = record["qiskit"]
        pennylane = record["pennylane"]
        lines.append(
            f"| {record['width']} | Aer {qiskit['cpu_fastest_method']} | "
            f"{qiskit['cpu_fastest_ms']:.3f} | {qiskit['mettleq_gpu_ms']:.3f} | "
            f"**{qiskit['cpu_over_mettleq']:.3f}x** | "
            f"`{qiskit['max_amplitude_error']:.2e}` | "
            f"{pennylane['cpu_ms']:.3f} | {pennylane['mettleq_gpu_ms']:.3f} | "
            f"**{pennylane['cpu_over_mettleq']:.3f}x** | "
            f"`{pennylane['max_amplitude_error']:.2e}` |"
        )
    lines.extend(
        [
            "",
            "A ratio above 1.0× means MettleQ was faster. Every accepted row must "
            f"also satisfy maximum phase-aligned state error <= "
            f"`{payload['protocol']['accuracy_atol']}`.",
            "",
            "## Interpretation",
            "",
            f"On this workload MettleQ first crosses Qiskit Aer at "
            f"{qiskit_wins[0]['width'] if qiskit_wins else 'no measured'} "
            f"qubits and `lightning.qubit` at "
            f"{pennylane_wins[0]['width'] if pennylane_wins else 'no measured'} "
            "qubits. Aer MPS "
            "is slower here because the contract requires materializing the "
            "complete dense statevector; bounded observables or samples are the "
            "appropriate contract for demonstrating an MPS advantage.",
            "",
            "The Qiskit path applies dependency-preserving scheduling before "
            "Metal fusion, so frontend ASAP serialization does not hide legal "
            "single-qubit and affine layers. Operations sharing a wire retain "
            "their original order. Generic single-qubit layers use radix-16 "
            "passes that apply four different 2x2 matrices per state traversal.",
            "",
            "Safety refusals are results, not missing data: widths whose modeled "
            "simultaneous validation footprint violates the configured RAM "
            "fraction or OS-headroom reserve are never launched.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=src caffeinate -i .venv/bin/python \\",
            "  tools/benchmark_sdk_crossover_large.py \\",
            "  --widths 16,18,20,22,24,26,27,28,29,30 --depth 3 \\",
            "  --warmups 1 --repeats 3 \\",
            "  --max-qiskit-mps-width 22 --accuracy-atol 5e-6 \\",
            "  --output-dir bench/runs/sdk-crossover-large",
            "```",
            "",
            "The worker-per-width design releases large statevectors before "
            "advancing. `caffeinate` exits when the benchmark process finishes.",
            "",
            f"Environment: `{json.dumps(payload['environment'], sort_keys=True)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _main(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    widths = sorted(set(int(value) for value in args.widths.split(",")))
    records = []
    safety_refusals = []
    started = time.perf_counter()
    for index, width in enumerate(widths, start=1):
        safety = _validation_memory_preflight(
            width,
            maximum_fraction=args.max_validation_memory_fraction,
            minimum_headroom_bytes=int(
                args.minimum_system_headroom_gib * 2**30
            ),
        )
        if not safety["allowed"]:
            safety_refusals.append(safety)
            print(
                f"[{index}/{len(widths)}] {width} qubits refused by memory "
                f"safety (projected "
                f"{safety['projected_peak_bytes'] / 2**30:.1f} GiB)",
                flush=True,
            )
            continue
        print(f"[{index}/{len(widths)}] {width} qubits", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-width",
            str(width),
            "--depth",
            str(args.depth),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--max-qiskit-mps-width",
            str(args.max_qiskit_mps_width),
            "--accuracy-atol",
            str(args.accuracy_atol),
            "--mlx-memory-limit-fraction",
            str(args.mlx_memory_limit_fraction),
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.worker_timeout_s,
        )
        if completed.returncode:
            print(completed.stdout, file=sys.stderr)
            raise RuntimeError(f"{width}q worker exited {completed.returncode}")
        result_lines = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith(RESULT_PREFIX)
        ]
        if len(result_lines) != 1:
            raise RuntimeError(
                f"{width}q worker emitted {len(result_lines)} result records"
            )
        record = json.loads(result_lines[0][len(RESULT_PREFIX) :])
        records.append(record)
        print(
            f"  Qiskit {record['qiskit']['cpu_over_mettleq']:.3f}x; "
            f"PennyLane {record['pennylane']['cpu_over_mettleq']:.3f}x; "
            f"max error {max(record['qiskit']['max_amplitude_error'], record['pennylane']['max_amplitude_error']):.2e}",
            flush=True,
        )
        partial = {
            "schema_version": 1,
            "status": "running",
            "records": records,
        }
        (output_dir / "partial_results.json").write_text(
            json.dumps(partial, indent=2, sort_keys=True) + "\n"
        )

    payload = {
        "schema_version": 1,
        "benchmark": "mettleq_large_sdk_cpu_gpu_crossover",
        "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware": _hardware_profile(),
        "environment": _versions(),
        "protocol": {
            "widths": widths,
            "depth": args.depth,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "accuracy_atol": args.accuracy_atol,
            "max_qiskit_mps_width": args.max_qiskit_mps_width,
            "qiskit_aer_enable_truncation": False,
            "memory_safety": {
                "maximum_validation_fraction": (
                    args.max_validation_memory_fraction
                ),
                "minimum_system_headroom_gib": (
                    args.minimum_system_headroom_gib
                ),
                "mlx_memory_limit_fraction": args.mlx_memory_limit_fraction,
            },
            "output_contract": "complete analytic statevector",
            "timing_scope": "compiled complete SDK execution and full-state return; transpilation excluded",
        },
        "elapsed_s": time.perf_counter() - started,
        "records": records,
        "safety_refusals": safety_refusals,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(output_dir / "results.csv", records)
    _plot(output_dir / "sdk_cpu_gpu_crossover.png", records)
    _write_readme(output_dir / "README.md", payload)
    (output_dir / "partial_results.json").unlink(missing_ok=True)
    print(json.dumps({"output_dir": str(output_dir), "elapsed_s": payload["elapsed_s"]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="bench/runs/sdk-crossover-large")
    parser.add_argument("--widths", default="16,18,20,22,24,26")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-qiskit-mps-width", type=int, default=22)
    parser.add_argument("--accuracy-atol", type=float, default=5e-6)
    parser.add_argument("--worker-timeout-s", type=int, default=1800)
    parser.add_argument(
        "--max-validation-memory-fraction", type=float, default=0.50
    )
    parser.add_argument(
        "--minimum-system-headroom-gib", type=float, default=6.0
    )
    parser.add_argument(
        "--mlx-memory-limit-fraction", type=float, default=0.50
    )
    parser.add_argument("--worker-width", type=int)
    args = parser.parse_args()
    if args.worker_width is not None:
        return _worker(args)
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
