#!/usr/bin/env python3
"""Freeze the repeated midpoint-MPO priority campaign into reviewable evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path):
    return json.loads(path.read_text())


def _git(*arguments: str):
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_run(source: Path, target: Path) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in (
        "campaign_record.json",
        "summary.json",
        "samples.tsv",
        "worker.log",
    ):
        path = source / name
        if path.exists():
            shutil.copy2(path, target / name)
            copied.append(name)
    stats = source / "stats.json"
    if stats.exists():
        with (
            stats.open("rb") as input_handle,
            (target / "stats.json.gz").open("wb") as raw_output,
            gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_output,
                mtime=0,
            ) as output_handle,
        ):
            shutil.copyfileobj(input_handle, output_handle)
        copied.append("stats.json.gz")
    return copied


def _copy_failures(source: Path, target: Path) -> dict[str, list[str]]:
    copied = {}
    if not source.exists():
        return copied
    for failure in sorted(path for path in source.iterdir() if path.is_dir()):
        destination = target / failure.name
        destination.mkdir(parents=True, exist_ok=True)
        names = []
        for path in sorted(failure.iterdir()):
            if not path.is_file():
                continue
            shutil.copy2(path, destination / path.name)
            names.append(path.name)
        copied[failure.name] = names
    return copied


def _safe_svd_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        if (
            record["status"] != "complete"
            or record["implementation"] != "mettleq_isolated"
        ):
            continue
        summary = _json(Path(record["run_dir"]) / "summary.json")
        telemetry = summary["diagnostics"].get("quimb_safe_svd", {})
        rows.append(
            {
                "phase": record["phase"],
                "repeat": record["repeat"],
                "position": record["position"],
                "arm": record["arm"],
                **telemetry,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, records: list[dict], summary: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {
        "mettleq_d512": "#7557ff",
        "mettleq_d768": "#9b7cff",
        "published_d512": "#34a0a4",
        "mettleq_d512_c5e-4": "#4958b8",
        "mettleq_d512_c7e-4": "#a768d4",
    }
    labels = {
        "mettleq_d512": "MettleQ D512",
        "mettleq_d768": "MettleQ D768",
        "published_d512": "Published core D512",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    main = [
        row
        for row in records
        if row["phase"] == "main" and row["status"] == "complete"
    ]
    arms = list(labels)
    x = np.arange(len(arms))
    for index, arm in enumerate(arms):
        group = [row for row in main if row["arm"] == arm]
        times = [row["algorithm_time_s"] for row in group]
        axes[0, 0].bar(
            index,
            statistics.median(times),
            color=colors[arm],
            alpha=0.82,
        )
        for offset, row in zip((-0.10, 0.10), group):
            axes[0, 0].scatter(
                index + offset,
                row["algorithm_time_s"],
                color="#17213a",
                s=30,
                zorder=3,
            )
            axes[0, 0].annotate(
                f"r{row['repeat']} p{row['position']}",
                (index + offset, row["algorithm_time_s"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    axes[0, 0].set_xticks(x, [labels[arm] for arm in arms], rotation=15)
    axes[0, 0].set_ylabel("Algorithm time (s)")
    axes[0, 0].set_title("Repeated same-Mac runtime (bars are medians)")
    axes[0, 0].grid(axis="y", alpha=0.25)

    for index, arm in enumerate(arms):
        group = [row for row in main if row["arm"] == arm]
        for offset, row in zip((-0.08, 0.08), group):
            axes[0, 1].scatter(
                index + offset,
                row["expected_peak_fraction"],
                color=colors[arm],
                s=55,
            )
    axes[0, 1].axhline(
        0.1, color="#17213a", linestyle="--", linewidth=1, label="Expected ~10%"
    )
    axes[0, 1].set_xticks(x, [labels[arm] for arm in arms], rotation=15)
    axes[0, 1].set_ylabel("Expected-peak fraction")
    axes[0, 1].set_title("Seeded 1,000-shot peak recovery")
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend()

    cutoff_rows = [
        row
        for row in records
        if row["status"] == "complete"
        and row["implementation"] == "mettleq_isolated"
        and row["max_bond"] == 512
    ]
    cutoff_values = (5e-4, 6e-4, 7e-4)
    cutoff_labels = ("5e-4", "6e-4", "7e-4")
    for index, cutoff in enumerate(cutoff_values):
        group = [row for row in cutoff_rows if row["cutoff"] == cutoff]
        for offset, row in zip(np.linspace(-0.08, 0.08, len(group)), group):
            axes[1, 0].scatter(
                index + offset,
                row["expected_peak_fraction"],
                color="#7557ff",
                s=50,
            )
    axes[1, 0].axhline(0.1, color="#17213a", linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(range(3), cutoff_labels)
    axes[1, 0].set_xlabel("Cutoff at fixed D=512")
    axes[1, 0].set_ylabel("Expected-peak fraction")
    classification = summary["cutoff_convergence"]["classification"]
    classification_label = classification.replace("_", " ")
    spread = summary["cutoff_convergence"]["median_expected_peak_fraction_spread"]
    spread_label = "n/a" if spread is None else f"{spread:.3f}"
    axes[1, 0].set_title(
        f"Cutoff convergence: {classification_label} (spread {spread_label})"
    )
    strict_failures = summary["cutoff_convergence"]["groups"]["0.0005"][
        "failures"
    ]
    if strict_failures:
        axes[1, 0].annotate(
            f"{strict_failures}/2 operational\nfailures at 176 gates",
            (0, 0.026),
            ha="center",
            va="center",
            fontsize=8,
            color="#a63d40",
        )
    axes[1, 0].grid(axis="y", alpha=0.25)

    for index, cutoff in enumerate(cutoff_values):
        group = [row for row in cutoff_rows if row["cutoff"] == cutoff]
        times = [row["algorithm_time_s"] for row in group]
        if times:
            axes[1, 1].bar(
                index,
                statistics.median(times),
                color="#7557ff",
                alpha=0.82,
            )
        for offset, row in zip(np.linspace(-0.08, 0.08, len(group)), group):
            axes[1, 1].scatter(
                index + offset,
                row["algorithm_time_s"],
                color="#17213a",
                s=28,
                zorder=3,
            )
    axes[1, 1].set_xticks(range(3), cutoff_labels)
    axes[1, 1].set_xlabel("Cutoff at fixed D=512")
    axes[1, 1].set_ylabel("Algorithm time (s)")
    axes[1, 1].set_title("Cutoff cost (bars are medians)")
    axes[1, 1].grid(axis="y", alpha=0.25)
    if strict_failures:
        axes[1, 1].annotate(
            f"{strict_failures}/2 operational\nfailures; no timing",
            (0, 100),
            ha="center",
            va="center",
            fontsize=8,
            color="#a63d40",
        )

    figure.suptitle(
        "MettleQ midpoint-MPO/TNO + unswapping — repeated M3 Pro evidence",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--failures-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_git_commit = _git("rev-parse", "HEAD")
    source_git_dirty = bool(_git("status", "--porcelain"))
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_manifest = _json(run_dir / "manifest.json")
    campaign_summary = _json(run_dir / "summary.json")
    with (run_dir / "records.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    numeric_ints = {
        "repeat",
        "position",
        "max_bond",
        "shots",
        "expected_peak_count",
        "peak_max_bond",
        "peak_total_elements",
        "work_gates_consumed",
        "total_work_gates",
    }
    numeric_floats = {
        "cutoff",
        "expected_peak_fraction",
        "compression_time_s",
        "materialize_time_s",
        "sampling_time_s",
        "algorithm_time_s",
        "campaign_wall_time_s",
        "worker_wall_time_s",
    }
    for row in records:
        for key in numeric_ints:
            if row.get(key):
                value = float(row[key])
                if not value.is_integer():
                    raise ValueError(
                        f"expected integral CSV value for {key}: {row[key]!r}"
                    )
                row[key] = int(value)
        for key in numeric_floats:
            if row.get(key):
                row[key] = float(row[key])
        row["matches_expected_bitstring"] = (
            row["matches_expected_bitstring"].lower() == "true"
        )
    if campaign_summary["attempted_runs"] != campaign_summary["planned_runs"]:
        raise RuntimeError("refusing to freeze a partially attempted campaign")
    if not campaign_summary["main_complete"]:
        raise RuntimeError("refusing to freeze incomplete main comparisons")
    invalid_failures = [
        row
        for row in records
        if row["status"] != "complete" and row["phase"] != "cutoff"
    ]
    if invalid_failures:
        raise RuntimeError("refusing to freeze failed main comparison records")

    copied_runs = {}
    for row in records:
        relative = Path(row["phase"]) / f"repeat-{row['repeat']}" / row["arm"]
        copied_runs[str(relative)] = _copy_run(
            Path(row["run_dir"]), output_dir / "runs" / relative
        )
    shutil.copy2(run_dir / "records.csv", output_dir / "records.csv")
    shutil.copy2(run_dir / "summary.json", output_dir / "campaign_summary.json")
    shutil.copy2(run_dir / "manifest.json", output_dir / "campaign_manifest.json")
    shutil.copy2(
        run_dir / "input/p9-qiskit2-export.qasm",
        output_dir / "p9-qiskit2-export.qasm",
    )

    failures = (
        _copy_failures(
            args.failures_dir.resolve(),
            output_dir / "recovery-evidence",
        )
        if args.failures_dir
        else {}
    )
    safe_svd = _safe_svd_rows(records)
    _write_csv(output_dir / "safe_svd_telemetry.csv", safe_svd)
    _plot(output_dir / "midpoint_mpo_repeated_evidence.png", records, campaign_summary)

    main_ratios = campaign_summary["paired_main_median_ratios"]
    cutoff = campaign_summary["cutoff_convergence"]
    by_arm = campaign_summary["by_arm"]
    fallback_totals = {
        key: sum(int(row.get(key, 0)) for row in safe_svd)
        for key in (
            "calls",
            "native_in_process_calls",
            "native_service_starts",
            "native_service_calls",
            "native_service_successes",
            "native_service_failures",
            "isolated_scipy_gesvd_calls",
            "isolated_scipy_gesvd_successes",
            "isolated_scipy_gesvd_failures",
            "rescaled_calls",
            "numpy_complex128_failures",
            "scipy_gesdd_failures",
            "eigh_fallbacks",
        )
    }
    evidence_manifest = {
        "benchmark": "mettleq_midpoint_mpo_priority_phase12_frozen",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": source_git_commit,
        "git_dirty": source_git_dirty,
        "platform": platform.platform(),
        "source_campaign": str(run_dir),
        "campaign_manifest_sha256": _sha256(run_dir / "manifest.json"),
        "campaign_summary_sha256": _sha256(run_dir / "summary.json"),
        "exported_qasm_sha256": _sha256(run_dir / "input/p9-qiskit2-export.qasm"),
        "copied_runs": copied_runs,
        "copied_recovery_evidence": failures,
        "safe_svd_totals": fallback_totals,
        "interpretation": {
            "runtime": (
                "paired, forward/reverse order-balanced medians on one Mac; "
                "ratios are evidence for this P9 contract, not universal speedups"
            ),
            "peak": (
                "seeded finite-shot recovery of the published expected bitstring; "
                "not a proof of full-state fidelity"
            ),
            "cutoff": cutoff,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(evidence_manifest, indent=2, default=str) + "\n"
    )

    def arm_line(arm: str) -> str:
        group = by_arm[arm]
        fractions = ", ".join(
            f"{value:.3f}" for value in group["expected_peak_fractions"]
        )
        return (
            f"| {arm} | {group['runs']} | "
            f"{group['median_algorithm_time_s']:.2f} | {fractions} | "
            f"{group['all_recovered_expected_peak']} |"
        )

    def cutoff_line(cutoff_key: str, label: str) -> str:
        group = cutoff["groups"][cutoff_key]
        fractions = ", ".join(
            f"{value:.3f}" for value in group["expected_peak_fractions"]
        ) or "no accepted result"
        median_time = group["median_algorithm_time_s"]
        time_label = "n/a" if median_time is None else f"{median_time:.2f}"
        return (
            f"| {label} | {group['attempts']} | {group['runs']} | "
            f"{group['failures']} | {time_label} | {fractions} | "
            f"{group['all_recovered_expected_peak']} |"
        )

    spread = cutoff["median_expected_peak_fraction_spread"]
    spread_text = "not measurable" if spread is None else f"{spread:.3f}"
    cutoff_failures = sum(
        group["failures"] for group in cutoff["groups"].values()
    )
    readme = f"""# MettleQ repeated midpoint-MPO evidence

This bundle freezes the order-balanced Apple M3 Pro campaign for the isolated
midpoint-MPO/TNO + unswapping worker. The normal caller used Qiskit
{campaign_manifest['caller_environment']['qiskit']}; the pinned worker used Qiskit
{campaign_manifest['worker_environment']['qiskit']} and Quimb
{campaign_manifest['worker_environment']['quimb']}.

| Arm | Repeats | Median algorithm time (s) | Expected-peak fractions | Peak recovered in every run |
|---|---:|---:|---|---|
{arm_line('mettleq_d512')}
{arm_line('mettleq_d768')}
{arm_line('published_d512')}

| Fixed-D512 cutoff | Attempts | Completed | Operational failures | Median algorithm time (s) | Expected-peak fractions | Peak recovered in every completed run |
|---|---:|---:|---:|---:|---|---|
{cutoff_line('0.0005', '5e-4')}
{cutoff_line('0.0006', '6e-4')}
{cutoff_line('0.0007', '7e-4')}

- Paired median MettleQ D512 / published-core D512 runtime ratio:
  **{main_ratios['mettleq_d512_over_published_algorithm_ratio']:.3f}x**.
- Paired median MettleQ D512 / D768 runtime ratio:
  **{main_ratios['mettleq_d512_over_d768_algorithm_ratio']:.3f}x**.
- Fixed-D512 cutoff classification: **{cutoff['classification']}**; median
  expected-peak fraction spread across 5e-4, 6e-4, and 7e-4 is
  **{spread_text}**. The endpoint schedule contains **{cutoff_failures}**
  recorded operational failure(s); failed arms contribute no peak estimate.
- The looser 7e-4 endpoint was **{by_arm['mettleq_d512']['median_algorithm_time_s'] / by_arm['mettleq_d512_c7e-4']['median_algorithm_time_s']:.3f}x**
  faster than 6e-4, but its expected-peak fraction fell reproducibly from
  0.100 to 0.024. It is therefore a failed accuracy endpoint, not a speedup.
- Safe-SVD telemetry across MettleQ arms: {fallback_totals['calls']} calls,
  {fallback_totals['native_service_calls']} routed to the persistent killable
  native service, {fallback_totals['native_service_failures']} service
  failures, and {fallback_totals['isolated_scipy_gesvd_successes']} successful
  fresh-process Quimb-compatible fallbacks; {fallback_totals['eigh_fallbacks']}
  calls reached the final Hermitian-eigensolver fallback.

`recovery-evidence/` preserves the unsafe Quimb `gesvd` process failure and the
superseded all-scaled-SVD experiment. Raw contraction statistics are losslessly
compressed as `stats.json.gz`; summaries, samples, logs, ordering, hashes, and
environment versions remain directly inspectable.

These timings characterize one P9 instance and one Mac. Finite-shot peak recovery
does not establish general circuit fidelity, and no general speedup claim is made.
"""
    (output_dir / "README.md").write_text(readme)
    print(json.dumps(evidence_manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
