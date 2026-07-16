#!/usr/bin/env python3
"""Run MettleQ midpoint-MPO convergence and a matched published-solver arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


EXPECTED_P9 = "01101110111001100000100000001010011100101101010111110111"


def _configs(value: str):
    result = []
    for item in value.split(","):
        bond, cutoff = item.strip().split(":", 1)
        result.append((int(bond), float(cutoff)))
    return result


def _label(max_bond: int, cutoff: float) -> str:
    cutoff_label = f"{cutoff:.0e}".replace("-0", "-").replace("+0", "+")
    return f"D{max_bond}_cutoff{cutoff_label}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str):
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run(command: list[str], log_path: Path, *, cwd: Path) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            check=False,
        )
    return {
        "exit_code": completed.returncode,
        "wall_time_s": time.perf_counter() - started,
        "log": str(log_path),
    }


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _convergence(points: list[dict], tolerance: float) -> dict:
    complete = [point for point in points if point.get("status") == "complete"]
    fractions = [
        point["expected_peak_fraction"]
        for point in complete
        if point.get("expected_peak_fraction") is not None
    ]
    predictions = [point.get("predicted_bitstring") for point in complete]
    spread = max(fractions) - min(fractions) if len(fractions) >= 2 else None
    same_prediction = len(set(predictions)) == 1 if predictions else False
    recovered = bool(complete) and all(
        point.get("matches_expected_bitstring") is True for point in complete
    )
    axis_reports = []

    def add_axis_reports(axis: str, varied_key: str, fixed_key: str) -> None:
        groups = {}
        for point in complete:
            groups.setdefault(point[fixed_key], []).append(point)
        for fixed_value, group in groups.items():
            varied_values = sorted({point[varied_key] for point in group})
            if len(varied_values) < 2:
                continue
            group_fractions = [
                point["expected_peak_fraction"]
                for point in group
                if point.get("expected_peak_fraction") is not None
            ]
            group_spread = (
                max(group_fractions) - min(group_fractions)
                if len(group_fractions) >= 2
                else None
            )
            group_predictions = {
                point.get("predicted_bitstring") for point in group
            }
            group_recovered = all(
                point.get("matches_expected_bitstring") is True for point in group
            )
            group_converged = bool(
                len(group_predictions) == 1
                and group_recovered
                and group_spread is not None
                and group_spread <= tolerance
            )
            axis_reports.append(
                {
                    "axis": axis,
                    "fixed_parameter": fixed_key,
                    "fixed_value": fixed_value,
                    "varied_parameter": varied_key,
                    "varied_values": varied_values,
                    "expected_peak_fraction_spread": group_spread,
                    "converged": group_converged,
                }
            )

    add_axis_reports("bond", "max_bond", "cutoff")
    add_axis_reports("cutoff", "cutoff", "max_bond")
    converged = bool(axis_reports) and all(
        report["converged"] for report in axis_reports
    )
    return {
        "classification": "converged" if converged else "not_converged",
        "converged": converged,
        "peak_fraction_atol": tolerance,
        "expected_peak_fraction_spread": spread,
        "same_predicted_bitstring": same_prediction,
        "expected_peak_recovered_all_points": recovered,
        "complete_points": len(complete),
        "qualified_comparison_count": len(axis_reports),
        "convergence_axes": sorted({report["axis"] for report in axis_reports}),
        "axis_reports": axis_reports,
        "points": points,
    }


def _plot(path: Path, convergence: dict, matched: dict | None) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    points = [
        point for point in convergence["points"]
        if point.get("status") == "complete"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    labels = [point["label"] for point in points]
    fractions = [point["expected_peak_fraction"] for point in points]
    runtimes = [point["total_time_s"] for point in points]
    x = np.arange(len(points))
    axes[0].plot(x, fractions, "o-", color="#7557ff", linewidth=2)
    axes[0].axhline(0.1, color="#34a0a4", linestyle="--", label="expected ~10% peak")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Observed expected-peak fraction")
    axes[0].set_title("MettleQ cutoff / bond convergence")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    if matched:
        names = ["MettleQ", "Published solver"]
        values = [matched["mettleq_total_time_s"], matched["published_total_time_s"]]
        colors = ["#7557ff", "#34a0a4"]
        axes[1].bar(names, values, color=colors)
        axes[1].set_title("Same Mac, matched P9 contract")
    else:
        axes[1].bar(labels, runtimes, color="#7557ff")
        axes[1].set_xticks(x, labels, rotation=25, ha="right")
        axes[1].set_title("MettleQ end-to-end runtime")
    axes[1].set_ylabel("Wall time (seconds)")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("MettleQ midpoint-MPO/TNO + greedy unswapping evidence")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    default_qasm = (
        root / "src/mettleq/datasets/peaked_circuit_P9_Hqap_56x1917.qasm"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qasm", type=Path, default=default_qasm)
    parser.add_argument(
        "--configs",
        type=_configs,
        default=[(256, 1e-3), (512, 1e-3), (512, 6e-4)],
        help="comma-separated max_bond:cutoff pairs",
    )
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sabre-trials", type=int, default=90)
    parser.add_argument("--post-sabre-trials", type=int, default=50)
    parser.add_argument("--no-progress-limit", type=int, default=20)
    parser.add_argument("--peak-fraction-atol", type=float, default=0.05)
    parser.add_argument(
        "--reference-repo",
        type=Path,
        default=Path("/tmp/peaked-mpo-solver-reference-20260716"),
    )
    parser.add_argument(
        "--reference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qasm = args.qasm.resolve()
    contract = {
        "qasm": str(qasm),
        "qasm_sha256": _sha256(qasm),
        "shots": args.shots,
        "routing_seed": args.seed,
        "sampling_seed": args.seed,
        "sabre_trials": args.sabre_trials,
        "post_sabre_trials": args.post_sabre_trials,
        "parallel_rewire": False,
        "expected_bitstring": EXPECTED_P9,
    }
    points = []
    for max_bond, cutoff in args.configs:
        label = _label(max_bond, cutoff)
        run_dir = args.output_dir / "mettleq" / label
        summary_path = run_dir / "summary.json"
        execution = None
        if not (args.skip_existing and summary_path.exists()):
            command = [
                sys.executable,
                "-m",
                "mettleq.midpoint_mpo",
                "--qasm",
                str(qasm),
                "--output-dir",
                str(run_dir),
                "--shots",
                str(args.shots),
                "--expected-bitstring",
                EXPECTED_P9,
                "--max-bond",
                str(max_bond),
                "--cutoff",
                str(cutoff),
                "--seed",
                str(args.seed),
                "--sabre-trials",
                str(args.sabre_trials),
                "--post-sabre-trials",
                str(args.post_sabre_trials),
                "--no-progress-limit",
                str(args.no_progress_limit),
            ]
            execution = _run(command, run_dir / "run.log", cwd=root)
        summary = _load(summary_path)
        if summary:
            points.append(
                {
                    "status": "complete",
                    "label": label,
                    "max_bond": max_bond,
                    "cutoff": cutoff,
                    "predicted_bitstring": summary.get("predicted_bitstring"),
                    "expected_peak_count": summary.get("expected_peak_count"),
                    "expected_peak_fraction": summary.get("expected_peak_fraction"),
                    "matches_expected_bitstring": summary.get("matches_expected_bitstring"),
                    "total_time_s": summary.get("total_time_s"),
                    "summary": str(summary_path),
                    "execution": execution,
                }
            )
        else:
            points.append(
                {
                    "status": "failed",
                    "label": label,
                    "max_bond": max_bond,
                    "cutoff": cutoff,
                    "execution": execution,
                }
            )
        partial = {
            "contract": contract,
            "convergence": _convergence(points, args.peak_fraction_atol),
        }
        (args.output_dir / "manifest.partial.json").write_text(
            json.dumps(partial, indent=2, default=str) + "\n"
        )

    convergence = _convergence(points, args.peak_fraction_atol)
    matched = None
    reference_record = None
    baseline = next(
        (
            point
            for point in points
            if point["status"] == "complete"
            and point["max_bond"] == 512
            and abs(point["cutoff"] - 6e-4) < 1e-15
        ),
        None,
    )
    if args.reference and baseline is not None:
        reference_repo = args.reference_repo.resolve()
        reference_python = reference_repo / ".venv/bin/python"
        sync_record = None
        if not reference_python.exists():
            sync_record = _run(
                ["uv", "sync", "--locked"],
                args.output_dir / "published_solver" / "sync.log",
                cwd=reference_repo,
            )
        reference_root = args.output_dir / "published_solver"
        reference_summary = reference_root / "matched" / "summary.json"
        execution = None
        if not (args.skip_existing and reference_summary.exists()):
            command = [
                str(reference_python),
                str(root / "tools/run_published_peaked_solver.py"),
                "--sampling-seed",
                str(args.seed),
                "--qasm",
                str(qasm),
                "--outdir",
                str(reference_root),
                "--tag",
                "matched",
                "--samples",
                str(args.shots),
                "--expected-bitstring",
                EXPECTED_P9,
                "--max-bond",
                "512",
                "--cutoff",
                "0.0006",
                "--seed",
                str(args.seed),
                "--sabre-trials",
                str(args.sabre_trials),
                "--post-sabre-trials",
                str(args.post_sabre_trials),
                "--abort-after-no-progress-unswap-cycles",
                str(args.no_progress_limit),
                "--no-parallel-rewire",
                "--no-plots",
            ]
            execution = _run(
                command,
                args.output_dir / "published_solver" / "harness.log",
                cwd=reference_repo,
            )
        summary = _load(reference_summary)
        reference_record = {
            "sync": sync_record,
            "execution": execution,
            "summary": str(reference_summary),
            "data": summary,
        }
        if summary:
            published_total = float(summary.get("compress_time_s", 0.0)) + float(
                summary.get("sample_total_time_s", 0.0)
            )
            matched = {
                "contract": {
                    **contract,
                    "max_bond": 512,
                    "cutoff": 6e-4,
                },
                "sampling_seed_harness": (
                    "published CLI omits quimb's seed argument; harness supplies "
                    "the contract seed without changing compression"
                ),
                "mettleq_total_time_s": baseline["total_time_s"],
                "published_total_time_s": published_total,
                "published_over_mettleq_ratio": (
                    published_total / baseline["total_time_s"]
                    if baseline["total_time_s"]
                    else None
                ),
                "mettleq_predicted_bitstring": baseline["predicted_bitstring"],
                "published_predicted_bitstring": summary.get("predicted_bitstring"),
                "mettleq_expected_peak_count": baseline["expected_peak_count"],
                "published_expected_peak_count": (
                    summary.get("sample_peak_count")
                    if summary.get("matches_expected_bitstring")
                    else None
                ),
                "both_recovered_expected_peak": bool(
                    baseline["matches_expected_bitstring"]
                    and summary.get("matches_expected_bitstring")
                ),
            }

    manifest = {
        "benchmark": "mettleq_midpoint_mpo_phase",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "contract": contract,
        "convergence": convergence,
        "matched_comparison": matched,
        "published_solver": reference_record,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    _plot(args.output_dir / "midpoint_mpo_evidence.png", convergence, matched)
    print(json.dumps(manifest, indent=2, default=str))
    return 0 if all(point["status"] == "complete" for point in points) else 1


if __name__ == "__main__":
    raise SystemExit(main())
