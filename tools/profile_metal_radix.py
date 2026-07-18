#!/usr/bin/env python3
"""Calibrate radix-8 vs radix-16 and optionally write an Xcode GPU trace."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import mlx.core as mx

from mettleq.benchmark_environment import capture_performance_environment
from mettleq.device import Device


def _operations(qubits: int, depth: int):
    operations = []
    for layer in range(depth):
        for wire in range(qubits):
            operations.append({"name": "RY", "wires": [wire],
                               "parameters": [0.013 * (wire + 1) * (layer + 1)]})
            operations.append({"name": "RZ", "wires": [wire],
                               "parameters": [-0.009 * (wire + 1)]})
    return operations


def _measure(qubits, depth, radix, repeats, trace):
    os.environ["METTLEQ_METAL_KERNELS"] = "1"
    os.environ["METTLEQ_SINGLE_QUBIT_RADIX"] = str(radix)
    operations = _operations(qubits, depth)
    warmup = Device(qubits)
    warmup.execute(operations)
    warmup.synchronize()
    if trace:
        trace.parent.mkdir(parents=True, exist_ok=True)
        mx.metal.start_capture(str(trace))
    samples = []
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            device = Device(qubits)
            device.execute(operations)
            device.synchronize()
            samples.append(time.perf_counter() - started)
    finally:
        if trace:
            mx.metal.stop_capture()
    return {"radix": radix, "samples_s": samples,
            "median_s": statistics.median(samples), "minimum_s": min(samples)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=22)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("bench/runs/radix-profile.json"))
    parser.add_argument("--capture-dir", type=Path)
    args = parser.parse_args()
    if not 3 <= args.qubits <= 28:
        parser.error("qubits must be between 3 and the safety cap of 28")
    records = []
    for radix in (8, 16):
        trace = (args.capture_dir / f"radix-{radix}.gputrace"
                 if args.capture_dir else None)
        records.append(_measure(args.qubits, args.depth, radix, args.repeats, trace))
    fastest = min(records, key=lambda row: row["median_s"])["radix"]
    payload = {
        "schema_version": 1,
        "environment": capture_performance_environment(),
        "contract": {"qubits": args.qubits, "depth": args.depth,
                     "repeats": args.repeats, "precision": "complex64"},
        "variants": records,
        "recommended_radix": fastest,
        "counter_note": (
            "Open any .gputrace in Xcode and inspect occupancy, register pressure, "
            "and achieved bandwidth. A trace is hardware-specific and is not committed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
