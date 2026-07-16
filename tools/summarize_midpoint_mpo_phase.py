#!/usr/bin/env python3
"""Freeze completed midpoint-MPO phase evidence into a compact bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time


EXPECTED = "01101110111001100000100000001010011100101101010111110111"


def _load(path: Path):
    return json.loads(path.read_text())


def _sha256(path: Path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _last_progress(path: Path):
    text = path.read_text(errors="replace")
    matches = re.findall(
        r"\[cycle\s+(\d+)\]\s+(\d+)/1885(?:\s+work gates)?\s+after\s+(\d+)s",
        text,
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    cycle, gates, seconds = matches[-1]
    return {
        "last_cycle": int(cycle),
        "last_work_gates": int(gates),
        "last_elapsed_s": int(seconds),
    }


def _copy_run(source: Path, target: Path, names):
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in names:
        src = source / name
        if src.exists():
            shutil.copy2(src, target / name)
            copied.append(name)
    return copied


def _plot(path: Path, rows, profile):
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    complete = [row for row in rows if row["status"] == "complete"]
    names = [row["label"] for row in complete]
    runtimes = [row["total_time_s"] for row in complete]
    peaks = [row["expected_peak_fraction"] for row in complete]
    colors = ["#7557ff", "#9b7cff", "#34a0a4"]
    axes[0].bar(names, runtimes, color=colors[: len(names)])
    axes[0].set_ylabel("End-to-end time (s)")
    axes[0].set_title("Same-Mac P9 runtime")
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(names, peaks, color=colors[: len(names)])
    axes[1].axhline(0.1, color="#264653", linestyle="--", label="expected ~10%")
    axes[1].set_ylim(0.0, max(0.125, max(peaks) * 1.15))
    axes[1].set_ylabel("Expected-peak fraction")
    axes[1].set_title("Seeded 1,000-shot peak evidence")
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    contractions = profile["contractions"]
    bonds = [row["bond"] for row in contractions]
    axes[2].plot(
        bonds,
        [row["numpy_cpu_contraction_ms"] for row in contractions],
        "o-",
        label="NumPy CPU contraction",
        color="#7557ff",
    )
    axes[2].plot(
        bonds,
        [row["mlx_gpu_resident_ms"] for row in contractions],
        "o-",
        label="MLX GPU resident",
        color="#34a0a4",
    )
    axes[2].plot(
        bonds,
        [row["cpu_svd_ms"] for row in contractions],
        "o-",
        label="CPU SVD",
        color="#e76f51",
    )
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].set_xticks(bonds, [str(bond) for bond in bonds])
    axes[2].set_xlabel("Bond dimension")
    axes[2].set_ylabel("Median time (ms, log)")
    axes[2].set_title("GPU MPS phase gate")
    axes[2].grid(alpha=0.25, which="both")
    axes[2].legend(fontsize=8)
    figure.suptitle("MettleQ midpoint-MPO/TNO + unswapping — M3 Pro evidence")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mettleq_512_dir = args.run_dir / "mettleq/D512_cutoff6e-4"
    mettleq_768_dir = args.run_dir / "mettleq/D768_cutoff6e-4"
    published_dir = args.run_dir / "published_solver/direct"
    m512 = _load(mettleq_512_dir / "summary.json")
    m768 = _load(mettleq_768_dir / "summary.json")
    published = _load(published_dir / "summary.json")
    profile = _load(args.profile_dir / "profile.json")

    rows = []
    for label, implementation, summary, source in (
        ("MettleQ D=512", "mettleq_midpoint_mpo", m512, mettleq_512_dir),
        ("MettleQ D=768", "mettleq_midpoint_mpo", m768, mettleq_768_dir),
        ("Published core D=512", "published_p9solver_core", published, published_dir),
    ):
        rows.append(
            {
                "status": "complete",
                "label": label,
                "implementation": implementation,
                "max_bond": summary["diagnostics"]["options"]["max_bond"],
                "cutoff": summary["diagnostics"]["options"]["cutoff"],
                "shots": summary["shots"],
                "expected_peak_count": summary["expected_peak_count"],
                "expected_peak_fraction": summary["expected_peak_fraction"],
                "matches_expected_bitstring": summary["matches_expected_bitstring"],
                "compression_time_s": summary["compression_time_s"],
                "sampling_time_s": summary["sampling_time_s"],
                "total_time_s": summary["total_time_s"],
                "peak_max_bond": summary["diagnostics"]["peak_max_bond"],
                "peak_total_elements": summary["diagnostics"]["peak_total_elements"],
                "source": str(source),
            }
        )

    qasm = Path("src/mettleq/datasets/peaked_circuit_P9_Hqap_56x1917.qasm")
    cutoff_failure = _last_progress(args.run_dir / "mettleq/D512_cutoff1e-3/run.log")
    modern_256 = _last_progress(
        Path("bench/runs/midpoint-mpo-phase11/mettleq/D256_cutoff1e-3/run.log")
    )
    modern_512 = _last_progress(
        Path("bench/runs/midpoint-mpo-phase11/mettleq/D512_cutoff1e-3/run.log")
    )
    cli_partial = _load(args.run_dir / "published_solver/matched/summary.json")
    peak_spread = abs(
        m768["expected_peak_fraction"] - m512["expected_peak_fraction"]
    )
    bond_converged = bool(
        m512["matches_expected_bitstring"]
        and m768["matches_expected_bitstring"]
        and peak_spread <= 0.03
    )
    matched_ratio = m512["total_time_s"] / published["total_time_s"]
    bond_speedup = m512["total_time_s"] / m768["total_time_s"]
    manifest = {
        "benchmark": "mettleq_midpoint_mpo_phase11",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "implementation_commit": "72582c7",
        "current_git_commit": _git("rev-parse", "HEAD"),
        "current_git_dirty": bool(_git("status", "--porcelain")),
        "platform": platform.platform(),
        "qasm_sha256": _sha256(qasm),
        "expected_bitstring": EXPECTED,
        "sampling_contract": {
            "shots": 1000,
            "routing_seed": 123,
            "sampling_seed": 123,
            "sabre_trials": 90,
            "post_sabre_trials": 50,
            "parallel_rewire": False,
            "cutoff": 0.0006,
        },
        "pinned_environment": {
            "python": "3.10.16",
            "qiskit": "1.4.5",
            "quimb": "1.11.2",
            "numpy": "2.2.6",
            "scipy": "1.15.3",
            "published_solver_commit": "3bcdc1e5bfd6abb9425f71bd43e560d2b27f45c1",
        },
        "completed_runs": rows,
        "bond_convergence": {
            "classification": "converged" if bond_converged else "not_converged",
            "converged": bond_converged,
            "peak_fraction_atol": 0.03,
            "expected_peak_fraction_spread": peak_spread,
            "d512_fraction": m512["expected_peak_fraction"],
            "d768_fraction": m768["expected_peak_fraction"],
            "d512_over_d768_runtime_speedup": bond_speedup,
            "d768_time_reduction_fraction": 1.0 - 1.0 / bond_speedup,
        },
        "cutoff_convergence": {
            "classification": "not_converged_operationally",
            "successful_cutoff": 0.0006,
            "failed_cutoff": 0.001,
            "failed_cutoff_progress": cutoff_failure,
            "reason": "1e-3 consumed only 197/1885 gates after 501 s and was stopped as an impractical trajectory; no peak claim is made for it",
        },
        "matched_published_comparison": {
            "classification": "performance_parity_single_ordered_pair",
            "mettleq_total_time_s": m512["total_time_s"],
            "published_core_total_time_s": published["total_time_s"],
            "mettleq_over_published_runtime_ratio": matched_ratio,
            "mettleq_time_delta_fraction": matched_ratio - 1.0,
            "both_expected_peak_count": [
                m512["expected_peak_count"],
                published["expected_peak_count"],
            ],
            "both_recovered_expected_peak": bool(
                m512["matches_expected_bitstring"]
                and published["matches_expected_bitstring"]
            ),
            "interpretation": "same algorithm and identical contract; a single ordered pair does not support a speedup claim",
        },
        "compatibility_failures": {
            "qiskit2_quimb114_d256_cutoff1e3": modern_256,
            "qiskit2_quimb114_d512_cutoff1e3": modern_512,
            "published_cli_external_interruption": {
                "run_status": cli_partial.get("run_status"),
                "last_work_consumed": cli_partial.get("last_work_consumed"),
                "compress_time_s": cli_partial.get("compress_time_s"),
            },
        },
        "cpu_gpu_profile": profile,
    }

    copied = {}
    copied["mettleq_d512"] = _copy_run(
        mettleq_512_dir,
        args.output_dir / "mettleq_d512",
        ["summary.json", "stats.json", "samples.tsv", "run.log"],
    )
    copied["mettleq_d768"] = _copy_run(
        mettleq_768_dir,
        args.output_dir / "mettleq_d768",
        ["summary.json", "stats.json", "samples.tsv", "run.log"],
    )
    copied["published_core"] = _copy_run(
        published_dir,
        args.output_dir / "published_core",
        ["summary.json", "stats.json", "samples.tsv", "run.log"],
    )
    copied["cutoff_failure"] = _copy_run(
        args.run_dir / "mettleq/D512_cutoff1e-3",
        args.output_dir / "cutoff_failure",
        ["run.log"],
    )
    copied["published_cli_partial"] = _copy_run(
        args.run_dir / "published_solver/matched",
        args.output_dir / "published_cli_partial",
        ["summary.json", "run.log"],
    )
    copied["profile"] = _copy_run(
        args.profile_dir,
        args.output_dir / "mps_profile",
        ["profile.json", "contractions.csv", "mps_cpu_gpu_profile.png"],
    )
    manifest["copied_files"] = copied
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    with (args.output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(args.output_dir / "midpoint_mpo_phase.png", rows, profile)
    readme = f"""# MettleQ midpoint-MPO phase evidence

Frozen on an Apple M3 Pro using implementation commit `72582c7`.

- MettleQ D=512: {m512['total_time_s']:.2f} s, expected peak {m512['expected_peak_count']}/1000.
- MettleQ D=768: {m768['total_time_s']:.2f} s, expected peak {m768['expected_peak_count']}/1000.
- Published core D=512: {published['total_time_s']:.2f} s, expected peak {published['expected_peak_count']}/1000.
- Bond convergence: **{manifest['bond_convergence']['classification']}**, peak-fraction spread {peak_spread:.3f}.
- Cutoff 1e-3: operational non-convergence ({cutoff_failure['last_work_gates']}/1885 gates after {cutoff_failure['last_elapsed_s']} s); no peak claim.
- Native GPU MPS: remains disabled; no resident or round-trip crossover was measured through D=128.

See `manifest.json` for contracts, environments, raw-run paths, failure arms, and interpretation limits.
"""
    (args.output_dir / "README.md").write_text(readme)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
