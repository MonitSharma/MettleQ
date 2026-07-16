#!/usr/bin/env python3
"""Run the order-balanced midpoint-MPO priority campaign on one Mac."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from qiskit import qasm2
import qiskit

from mettleq.midpoint_mpo import (
    IsolatedMidpointMPOSimulator,
    MidpointMPOOptions,
    PUBLISHED_P9_EXPECTED_BITSTRING,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QASM = ROOT / "src/mettleq/datasets/peaked_circuit_P9_Hqap_56x1917.qasm"
PUBLISHED_COMMIT = "3bcdc1e5bfd6abb9425f71bd43e560d2b27f45c1"
MAIN_ARMS = ("mettleq_d512", "mettleq_d768", "published_d512")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str):
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json(path: Path):
    return json.loads(path.read_text())


def _worker_environment(worker_python: Path) -> dict:
    code = (
        "import json,platform,qiskit,quimb,numpy,scipy;"
        "print(json.dumps({'python':platform.python_version(),"
        "'qiskit':qiskit.__version__,'quimb':quimb.__version__,"
        "'numpy':numpy.__version__,'scipy':scipy.__version__}))"
    )
    return json.loads(subprocess.check_output([str(worker_python), "-c", code]))


def _main_schedule(repeats: int) -> list[dict]:
    rows = []
    for repeat in range(repeats):
        offset = repeat % len(MAIN_ARMS)
        order = MAIN_ARMS[offset:] + MAIN_ARMS[:offset]
        for position, arm in enumerate(order):
            rows.append(
                {
                    "phase": "main",
                    "repeat": repeat,
                    "position": position,
                    "arm": arm,
                }
            )
    return rows


def _cutoff_schedule(repeats: int) -> list[dict]:
    rows = []
    arms = ("mettleq_d512_c5e-4", "mettleq_d512_c7e-4")
    for repeat in range(repeats):
        order = arms if repeat % 2 == 0 else tuple(reversed(arms))
        for position, arm in enumerate(order):
            rows.append(
                {
                    "phase": "cutoff",
                    "repeat": repeat,
                    "position": position,
                    "arm": arm,
                }
            )
    return rows


def _arm_contract(arm: str) -> tuple[str, int, float]:
    if arm == "mettleq_d512":
        return "mettleq_isolated", 512, 6e-4
    if arm == "mettleq_d768":
        return "mettleq_isolated", 768, 6e-4
    if arm == "published_d512":
        return "published_core", 512, 6e-4
    if arm == "mettleq_d512_c5e-4":
        return "mettleq_isolated", 512, 5e-4
    if arm == "mettleq_d512_c7e-4":
        return "mettleq_isolated", 512, 7e-4
    raise ValueError(f"unknown arm: {arm}")


def _run_mettleq(
    *,
    circuit,
    run_dir: Path,
    worker_python: Path,
    max_bond: int,
    cutoff: float,
    shots: int,
    seed: int,
    timeout_seconds: float,
) -> dict:
    simulator = IsolatedMidpointMPOSimulator(
        MidpointMPOOptions(
            max_bond=max_bond,
            cutoff=cutoff,
            seed=seed,
            sabre_trials=90,
            post_sabre_trials=50,
            abort_after_no_progress_unswap_cycles=20,
            parallel_rewire=False,
        ),
        worker_python=worker_python,
        timeout_seconds=timeout_seconds,
    )
    started = time.perf_counter()
    result = simulator.run(
        circuit,
        shots=shots,
        expected_bitstring=PUBLISHED_P9_EXPECTED_BITSTRING,
        output_dir=run_dir,
    )
    wall_time_s = time.perf_counter() - started
    return {
        "campaign_wall_time_s": wall_time_s,
        "caller_qiskit_version": qiskit.__version__,
        "worker_wall_time_s": result.diagnostics["worker_wall_time_s"],
    }


def _run_published(
    *,
    qasm: Path,
    run_dir: Path,
    worker_python: Path,
    published_repo: Path,
    max_bond: int,
    cutoff: float,
    shots: int,
    seed: int,
    timeout_seconds: float,
) -> dict:
    command = [
        str(worker_python),
        str(ROOT / "tools/run_published_peaked_solver_direct.py"),
        "--qasm",
        str(qasm),
        "--output-dir",
        str(run_dir),
        "--shots",
        str(shots),
        "--expected-bitstring",
        PUBLISHED_P9_EXPECTED_BITSTRING,
        "--max-bond",
        str(max_bond),
        "--cutoff",
        str(cutoff),
        "--seed",
        str(seed),
        "--sabre-trials",
        "90",
        "--post-sabre-trials",
        "50",
        "--no-progress-limit",
        "20",
    ]
    source_root = ROOT / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(source_root),
            str(published_repo),
            env.get("PYTHONPATH"),
        )
        if value
    )
    started = time.perf_counter()
    with (run_dir / "worker.log").open("w") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=timeout_seconds,
            check=False,
        )
    wall_time_s = time.perf_counter() - started
    if completed.returncode != 0:
        tail = "\n".join(
            (run_dir / "worker.log").read_text(errors="replace").splitlines()[-40:]
        )
        raise RuntimeError(f"published core exited {completed.returncode}:\n{tail}")
    return {"campaign_wall_time_s": wall_time_s}


def _record_from_summary(plan: dict, run_dir: Path, extra: dict) -> dict:
    summary = _json(run_dir / "summary.json")
    implementation, max_bond, cutoff = _arm_contract(plan["arm"])
    diagnostics = summary["diagnostics"]
    return {
        **plan,
        "status": "complete",
        "implementation": implementation,
        "max_bond": max_bond,
        "cutoff": cutoff,
        "shots": summary["shots"],
        "predicted_bitstring": summary["predicted_bitstring"],
        "expected_peak_count": summary["expected_peak_count"],
        "expected_peak_fraction": summary["expected_peak_fraction"],
        "matches_expected_bitstring": summary["matches_expected_bitstring"],
        "compression_time_s": summary["compression_time_s"],
        "materialize_time_s": summary["materialize_time_s"],
        "sampling_time_s": summary["sampling_time_s"],
        "algorithm_time_s": summary["total_time_s"],
        "peak_max_bond": diagnostics.get("peak_max_bond"),
        "peak_total_elements": diagnostics.get("peak_total_elements"),
        "work_gates_consumed": diagnostics.get("work_gates_consumed"),
        "total_work_gates": diagnostics.get("total_work_gates"),
        "termination_reason": diagnostics.get("termination_reason"),
        "run_dir": str(run_dir),
        **extra,
    }


def _summarize(records: list[dict], plan: list[dict]) -> dict:
    complete = [record for record in records if record["status"] == "complete"]
    by_arm = {}
    for arm in sorted({record["arm"] for record in complete}):
        group = [record for record in complete if record["arm"] == arm]
        by_arm[arm] = {
            "runs": len(group),
            "positions": [record["position"] for record in group],
            "algorithm_times_s": [record["algorithm_time_s"] for record in group],
            "wall_times_s": [record["campaign_wall_time_s"] for record in group],
            "median_algorithm_time_s": statistics.median(
                record["algorithm_time_s"] for record in group
            ),
            "median_wall_time_s": statistics.median(
                record["campaign_wall_time_s"] for record in group
            ),
            "expected_peak_counts": [
                record["expected_peak_count"] for record in group
            ],
            "expected_peak_fractions": [
                record["expected_peak_fraction"] for record in group
            ],
            "all_recovered_expected_peak": all(
                record["matches_expected_bitstring"] for record in group
            ),
        }

    paired = []
    for repeat in sorted(
        {record["repeat"] for record in complete if record["phase"] == "main"}
    ):
        group = {
            record["arm"]: record
            for record in complete
            if record["phase"] == "main" and record["repeat"] == repeat
        }
        if set(group) != set(MAIN_ARMS):
            continue
        paired.append(
            {
                "repeat": repeat,
                "mettleq_d512_over_published_algorithm_ratio": (
                    group["mettleq_d512"]["algorithm_time_s"]
                    / group["published_d512"]["algorithm_time_s"]
                ),
                "mettleq_d512_over_published_wall_ratio": (
                    group["mettleq_d512"]["campaign_wall_time_s"]
                    / group["published_d512"]["campaign_wall_time_s"]
                ),
                "mettleq_d512_over_d768_algorithm_ratio": (
                    group["mettleq_d512"]["algorithm_time_s"]
                    / group["mettleq_d768"]["algorithm_time_s"]
                ),
            }
        )

    cutoff_records = [
        record
        for record in complete
        if record["implementation"] == "mettleq_isolated"
        and record["max_bond"] == 512
        and record["cutoff"] in {5e-4, 6e-4, 7e-4}
    ]
    cutoff_groups = {}
    for cutoff in (5e-4, 6e-4, 7e-4):
        group = [record for record in cutoff_records if record["cutoff"] == cutoff]
        cutoff_groups[str(cutoff)] = {
            "runs": len(group),
            "expected_peak_fractions": [
                record["expected_peak_fraction"] for record in group
            ],
            "median_expected_peak_fraction": (
                statistics.median(record["expected_peak_fraction"] for record in group)
                if group
                else None
            ),
            "median_algorithm_time_s": (
                statistics.median(record["algorithm_time_s"] for record in group)
                if group
                else None
            ),
            "all_recovered_expected_peak": bool(group)
            and all(record["matches_expected_bitstring"] for record in group),
        }
    cutoff_medians = [
        group["median_expected_peak_fraction"]
        for group in cutoff_groups.values()
        if group["median_expected_peak_fraction"] is not None
    ]
    cutoff_complete = all(group["runs"] > 0 for group in cutoff_groups.values())
    cutoff_recovered = all(
        group["all_recovered_expected_peak"] for group in cutoff_groups.values()
    )
    cutoff_spread = (
        max(cutoff_medians) - min(cutoff_medians)
        if len(cutoff_medians) == 3
        else None
    )
    cutoff_converged = bool(
        cutoff_complete
        and cutoff_recovered
        and cutoff_spread is not None
        and cutoff_spread <= 0.03
    )
    return {
        "planned_runs": len(plan),
        "completed_runs": len(complete),
        "by_arm": by_arm,
        "paired_main_repeats": paired,
        "paired_main_median_ratios": {
            key: statistics.median(row[key] for row in paired) if paired else None
            for key in (
                "mettleq_d512_over_published_algorithm_ratio",
                "mettleq_d512_over_published_wall_ratio",
                "mettleq_d512_over_d768_algorithm_ratio",
            )
        },
        "cutoff_convergence": {
            "classification": "converged" if cutoff_converged else "not_converged",
            "converged": cutoff_converged,
            "fixed_max_bond": 512,
            "peak_fraction_atol": 0.03,
            "median_expected_peak_fraction_spread": cutoff_spread,
            "groups": cutoff_groups,
        },
    }


def _write_records(path: Path, records: list[dict]) -> None:
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qasm", type=Path, default=DEFAULT_QASM)
    parser.add_argument(
        "--worker-python", type=Path, default=ROOT / ".venv-mpo/bin/python"
    )
    parser.add_argument(
        "--published-repo",
        type=Path,
        default=Path("/tmp/peaked-mpo-solver-reference-20260716"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cutoff-repeats", type=int, default=2)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--phase", choices=("all", "main", "cutoff"), default="all"
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    if args.repeats < 3:
        parser.error("--repeats must be at least 3 for three-position balance")
    if args.cutoff_repeats < 2:
        parser.error("--cutoff-repeats must be at least 2 for reversed order")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_python = Path(os.path.abspath(os.fspath(args.worker_python.expanduser())))
    published_repo = args.published_repo.resolve()
    if not worker_python.exists():
        parser.error(f"worker interpreter not found: {worker_python}")
    if not (published_repo / "src/p9solver").exists():
        parser.error(f"published solver checkout not found: {published_repo}")
    published_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=published_repo, text=True
    ).strip()
    if published_head != PUBLISHED_COMMIT:
        parser.error(
            f"published solver must be at {PUBLISHED_COMMIT}; found {published_head}"
        )

    source_circuit = qasm2.load(
        str(args.qasm.resolve()),
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )
    input_dir = output_dir / "input"
    input_dir.mkdir(exist_ok=True)
    exported_qasm = input_dir / "p9-qiskit2-export.qasm"
    qasm2.dump(source_circuit, str(exported_qasm))
    plan = []
    if args.phase in {"all", "main"}:
        plan.extend(_main_schedule(args.repeats))
    if args.phase in {"all", "cutoff"}:
        plan.extend(_cutoff_schedule(args.cutoff_repeats))
    manifest = {
        "benchmark": "mettleq_midpoint_mpo_priority_phase12",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "caller_environment": {
            "python": platform.python_version(),
            "qiskit": qiskit.__version__,
        },
        "worker_environment": _worker_environment(worker_python),
        "worker_python": str(worker_python),
        "published_solver_commit": published_head,
        "source_qasm": str(args.qasm.resolve()),
        "source_qasm_sha256": _sha256(args.qasm.resolve()),
        "exported_qasm": str(exported_qasm),
        "exported_qasm_sha256": _sha256(exported_qasm),
        "contract": {
            "shots": args.shots,
            "routing_seed": args.seed,
            "sampling_seed": args.seed,
            "sabre_trials": 90,
            "post_sabre_trials": 50,
            "parallel_rewire": False,
            "expected_bitstring": PUBLISHED_P9_EXPECTED_BITSTRING,
        },
        "ordering": (
            "three-arm Latin rotation for main repeats; reversed two-arm "
            "ordering for cutoff endpoints"
        ),
        "plan": plan,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )

    records = []
    for index, item in enumerate(plan, start=1):
        phase_dir = output_dir / item["phase"] / f"repeat-{item['repeat']}"
        run_dir = phase_dir / item["arm"]
        run_dir.mkdir(parents=True, exist_ok=True)
        record_path = run_dir / "campaign_record.json"
        if args.resume and record_path.exists() and (run_dir / "summary.json").exists():
            record = _json(record_path)
            records.append(record)
            print(
                f"[{index}/{len(plan)}] reuse {item['phase']} r{item['repeat']} "
                f"p{item['position']} {item['arm']}",
                flush=True,
            )
            continue
        implementation, max_bond, cutoff = _arm_contract(item["arm"])
        print(
            f"[{index}/{len(plan)}] start {item['phase']} r{item['repeat']} "
            f"p{item['position']} {item['arm']} D={max_bond} cutoff={cutoff:g}",
            flush=True,
        )
        try:
            if implementation == "mettleq_isolated":
                extra = _run_mettleq(
                    circuit=source_circuit,
                    run_dir=run_dir,
                    worker_python=worker_python,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    shots=args.shots,
                    seed=args.seed,
                    timeout_seconds=args.timeout_seconds,
                )
                if _sha256(run_dir / "input.qasm") != _sha256(exported_qasm):
                    raise RuntimeError("isolated worker QASM differs from campaign QASM")
            else:
                extra = _run_published(
                    qasm=exported_qasm,
                    run_dir=run_dir,
                    worker_python=worker_python,
                    published_repo=published_repo,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    shots=args.shots,
                    seed=args.seed,
                    timeout_seconds=args.timeout_seconds,
                )
            record = _record_from_summary(item, run_dir, extra)
        except Exception as error:
            record = {
                **item,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "run_dir": str(run_dir),
            }
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            _write_records(output_dir / "records.csv", records)
            (output_dir / "summary.partial.json").write_text(
                json.dumps(_summarize(records, plan), indent=2, default=str) + "\n"
            )
            raise
        record_path.write_text(json.dumps(record, indent=2, default=str) + "\n")
        records.append(record)
        _write_records(output_dir / "records.csv", records)
        partial = _summarize(records, plan)
        (output_dir / "summary.partial.json").write_text(
            json.dumps(partial, indent=2, default=str) + "\n"
        )
        print(
            f"[{index}/{len(plan)}] done {item['arm']} "
            f"algorithm={record['algorithm_time_s']:.2f}s "
            f"wall={record['campaign_wall_time_s']:.2f}s "
            f"peak={record['expected_peak_count']}/{record['shots']}",
            flush=True,
        )

    summary = _summarize(records, plan)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0 if summary["completed_runs"] == summary["planned_runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
