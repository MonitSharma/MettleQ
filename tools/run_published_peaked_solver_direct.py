#!/usr/bin/env python3
"""Run the published P9 solver core under MettleQ's matched result contract.

This imports ``p9solver.pipeline`` from the published repository environment,
not MettleQ's vendored copy. It deliberately bypasses the published CLI's
per-cycle JSON checkpoint rewrites so the comparison measures the simulation
core under the same in-memory telemetry and seeded-sampling contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import qiskit
from qiskit import qasm2
import quimb
from p9solver.pipeline import mpo_compress_unswap, mpo_to_mps

import mettleq.midpoint_mpo as midpoint_mpo
from mettleq.midpoint_mpo import (
    MidpointMPOOptions,
    MidpointMPOSimulator,
)


PUBLISHED_COMMIT = "3bcdc1e5bfd6abb9425f71bd43e560d2b27f45c1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qasm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--expected-bitstring", required=True)
    parser.add_argument("--max-bond", type=int, default=512)
    parser.add_argument("--cutoff", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sabre-trials", type=int, default=90)
    parser.add_argument("--post-sabre-trials", type=int, default=50)
    parser.add_argument("--no-progress-limit", type=int, default=20)
    args = parser.parse_args()

    # Select the published repository's core while retaining the exact same
    # MettleQ-side consolidation, telemetry, sampling seed, and result schema.
    midpoint_mpo._require_tensor_network = lambda: (
        qiskit,
        quimb,
        mpo_compress_unswap,
        mpo_to_mps,
    )
    circuit = qasm2.load(
        str(args.qasm), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
    )
    options = MidpointMPOOptions(
        max_bond=args.max_bond,
        cutoff=args.cutoff,
        seed=args.seed,
        sabre_trials=args.sabre_trials,
        post_sabre_trials=args.post_sabre_trials,
        abort_after_no_progress_unswap_cycles=args.no_progress_limit,
    )

    def progress(row):
        if row.get("stage") == "cycle_progress":
            print(
                f"[cycle {row.get('unswap_cycle')}] "
                f"{row.get('gates_consumed')}/{row.get('total_work_gates')} "
                f"after {float(row.get('time', 0.0)):.0f}s",
                flush=True,
            )

    result = MidpointMPOSimulator(options).run(
        circuit,
        shots=args.shots,
        expected_bitstring=args.expected_bitstring,
        on_progress=progress,
    )
    result.diagnostics.update(
        {
            "implementation": "published_p9solver.pipeline_direct",
            "published_solver_commit": PUBLISHED_COMMIT,
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
    )
    midpoint_mpo._write_result(args.output_dir, result)
    print(json.dumps(result.to_summary(include_counts=False), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
