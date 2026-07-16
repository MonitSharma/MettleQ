#!/usr/bin/env python3
"""Combine MPS limit campaigns into publication-ready evidence charts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FAMILIES = (
    "ghz_chain",
    "line_brickwork",
    "ring_brickwork",
    "grid_2d",
    "rainbow",
    "random_long_range",
    "all_to_all",
)


def _optional_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            for source_row in csv.DictReader(handle):
                row = dict(source_row)
                row["qubits"] = int(row["qubits"])
                row["depth"] = int(row["depth"])
                row["dmax"] = int(row["dmax"])
                for field in (
                    "execution_ms",
                    "exact_absolute_error",
                    "state_norm",
                    "maximum_bond_dimension_reached",
                    "local_discarded_weight_sum",
                ):
                    row[field] = _optional_float(row[field])
                row["truncated"] = (
                    None
                    if row["truncated"] == ""
                    else row["truncated"] == "True"
                )
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def _case_label(row: dict) -> str:
    family = row["family"].replace("_", " ")
    return f"{family} {row['qubits']}q d{row['depth']}"


def _plot_landscape(
    rows: list[dict], output: Path, timeout_seconds: float
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = dict(zip(FAMILIES, plt.cm.tab10.colors))
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    maximum_runtime = max(
        row["execution_ms"]
        for row in rows
        if row["execution_ms"] is not None
    )
    timeout_ms = timeout_seconds * 1_000.0
    failure_level = max(timeout_ms, maximum_runtime * 1.25)

    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        completed = [row for row in selected if row["status"] == "completed"]
        failed = [row for row in selected if row["status"] != "completed"]
        if completed:
            axes[0].scatter(
                [row["qubits"] for row in completed],
                [row["execution_ms"] for row in completed],
                color=colors[family],
                s=48,
                label=family.replace("_", " "),
                zorder=3,
            )
            untruncated = [row for row in completed if not row["truncated"]]
            truncated = [row for row in completed if row["truncated"]]
            if untruncated:
                axes[1].scatter(
                    [row["qubits"] for row in untruncated],
                    [row["maximum_bond_dimension_reached"] for row in untruncated],
                    color=colors[family],
                    s=48,
                    zorder=3,
                )
            if truncated:
                axes[1].scatter(
                    [row["qubits"] for row in truncated],
                    [row["maximum_bond_dimension_reached"] for row in truncated],
                    facecolors="none",
                    edgecolors=colors[family],
                    linewidths=1.6,
                    s=55,
                    zorder=3,
                )
        for row in failed:
            marker = "^" if row["status"] == "timeout" else "X"
            axes[0].scatter(
                row["qubits"],
                failure_level,
                color=colors[family],
                marker=marker,
                s=70,
                zorder=4,
            )

    axes[0].axhline(timeout_ms, color="#777777", linestyle="--", linewidth=1)
    axes[0].text(
        0.98,
        0.03,
        f"{timeout_seconds:g} s per-case ceiling; X/▲ are failure markers, not runtimes",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Qubits (log scale)")
    axes[0].set_ylabel("Qiskit Estimator execution (ms, log scale)")
    axes[0].set_title("Completion envelope at Dmax=64")

    axes[1].axhline(64, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log", base=2)
    axes[1].set_xlabel("Qubits (log scale)")
    axes[1].set_ylabel("Maximum bond dimension (log₂ scale)")
    axes[1].set_title("Entanglement pressure and truncation")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.22)

    family_handles, family_labels = axes[0].get_legend_handles_labels()
    status_handles = [
        Line2D([], [], color="black", marker="o", linestyle="None", label="no truncation"),
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            markerfacecolor="none",
            linestyle="None",
            label="local truncation observed",
        ),
        Line2D([], [], color="black", marker="X", linestyle="None", label="error"),
        Line2D([], [], color="black", marker="^", linestyle="None", label="timeout"),
    ]
    figure.legend(
        family_handles + status_handles,
        family_labels + [handle.get_label() for handle in status_handles],
        loc="outside lower center",
        ncol=4,
        fontsize=8,
    )
    figure.suptitle("MettleQ MPS limits depend on entanglement, not qubits alone")
    figure.tight_layout(rect=(0, 0.18, 1, 1))
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_convergence(rows: list[dict], output: Path, exact_atol: float) -> None:
    import matplotlib.pyplot as plt

    keys = sorted(
        {(row["family"], row["qubits"], row["depth"]) for row in rows},
        key=lambda key: (FAMILIES.index(key[0]), key[1], key[2]),
    )
    colors = dict(zip(keys, plt.cm.turbo([index / max(1, len(keys) - 1) for index in range(len(keys))])))
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    for key in keys:
        selected = sorted(
            [
                row
                for row in rows
                if (row["family"], row["qubits"], row["depth"]) == key
                and row["status"] == "completed"
            ],
            key=lambda row: row["dmax"],
        )
        if not selected:
            continue
        label = _case_label(selected[0])
        dmax = [row["dmax"] for row in selected]
        axes[0].plot(
            dmax,
            [row["execution_ms"] for row in selected],
            marker="o",
            color=colors[key],
            label=label,
        )
        axes[1].plot(
            dmax,
            [max(abs(row["state_norm"] - 1.0), 1e-9) for row in selected],
            marker="o",
            color=colors[key],
        )
        exact = [row for row in selected if row["exact_absolute_error"] is not None]
        if exact:
            axes[2].plot(
                [row["dmax"] for row in exact],
                [max(row["exact_absolute_error"], 1e-9) for row in exact],
                marker="o",
                color=colors[key],
            )

    axes[0].set_ylabel("Execution (ms, log scale)")
    axes[1].set_ylabel("|state norm − 1| (log scale)")
    axes[2].set_ylabel("Exact-reference |Δ⟨Z₀⟩| (log scale)")
    axes[2].axhline(exact_atol, color="#777777", linestyle="--", linewidth=1)
    axes[2].text(33, exact_atol * 1.25, f"acceptance {exact_atol:g}", fontsize=8)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks([32, 64, 128], labels=["32", "64", "128"])
        axis.set_xlabel("Maximum bond dimension")
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_title("Cost")
    axes[1].set_title("Norm stability")
    axes[2].set_title("Small-circuit validation")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7.5)
    figure.suptitle("Dmax convergence is required for truncated MPS results")
    figure.tight_layout(rect=(0, 0.2, 1, 1))
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _write_rows(rows: list[dict], output: Path) -> None:
    fieldnames = list(rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _repeated_dmax_cases(rows: list[dict]) -> list[dict]:
    dmax_by_case: dict[tuple[str, int, int], set[int]] = {}
    for row in rows:
        key = (row["family"], row["qubits"], row["depth"])
        dmax_by_case.setdefault(key, set()).add(row["dmax"])
    return [
        row
        for row in rows
        if len(
            dmax_by_case[(row["family"], row["qubits"], row["depth"])]
        )
        >= 2
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--convergence", type=Path, action="append", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--exact-atol", type=float, default=5e-5)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    campaign_rows = _read_rows(args.campaign)
    convergence_rows = _repeated_dmax_cases(_read_rows(args.convergence))
    if not campaign_rows or not convergence_rows:
        parser.error("campaign and convergence inputs must contain rows")
    _plot_landscape(
        campaign_rows,
        args.outdir / "mps_limit_landscape.png",
        args.timeout_seconds,
    )
    _plot_convergence(
        convergence_rows,
        args.outdir / "mps_dmax_convergence.png",
        args.exact_atol,
    )
    _write_rows(campaign_rows, args.outdir / "mps_limit_combined.csv")
    _write_rows(convergence_rows, args.outdir / "mps_convergence_combined.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
