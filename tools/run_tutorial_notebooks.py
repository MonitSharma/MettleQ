#!/usr/bin/env python3
"""Execute every MettleQ tutorial notebook and collect verified results."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
RESULT_PREFIX = "TUTORIAL_RESULT::"


def _git(*arguments: str):
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str]:
    import mettleq
    import pennylane
    import qiskit
    import qiskit_aer

    return {
        "mettleq": mettleq.__version__,
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "pennylane": pennylane.__version__,
        "numpy": __import__("numpy").__version__,
        "mlx": version("mlx"),
    }


def _extract_result(notebook, path: Path) -> dict:
    matches = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                texts = [output.get("text", "")]
            elif output.get("output_type") in {"execute_result", "display_data"}:
                texts = [str(output.get("data", {}).get("text/plain", ""))]
            else:
                texts = []
            for text in texts:
                for line in text.splitlines():
                    if line.startswith(RESULT_PREFIX):
                        matches.append(json.loads(line[len(RESULT_PREFIX) :]))
    if len(matches) != 1:
        raise RuntimeError(
            f"{path.relative_to(ROOT)} emitted {len(matches)} result records; expected 1"
        )
    return matches[0]


def _execute(path: Path, *, timeout: int, write_back: bool) -> dict:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
        record_timing=True,
    )
    client.execute()
    if write_back:
        nbformat.write(notebook, path)
    return _extract_result(notebook, path)


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Executed tutorial results",
        "",
        (
            f"All **{payload['passed_notebooks']}/{payload['total_notebooks']}** "
            "notebooks passed their declared comparison on this machine."
        ),
        "",
        "`Reference / MettleQ` above 1.0 means MettleQ was faster for that notebook's complete call; below 1.0 means the SDK reference was faster.",
        "",
        "| Notebook | Check | Reference (ms) | MettleQ (ms) | Reference / MettleQ | Method/device | Exact? |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for record in payload["results"]:
        method_device = "/".join(
            value
            for value in (
                record.get("selected_method"),
                record.get("selected_device"),
            )
            if value
        ) or "n/a"
        exact = record.get("exact_match")
        exact_label = "n/a" if exact is None else str(bool(exact))
        ratio = record.get("reference_over_mettleq")
        ratio_label = "n/a" if ratio is None else f"{ratio:.3f}x"
        lines.append(
            f"| `{record['notebook']}` | {record['check']} | "
            f"{record['reference_median_ms']:.3f} | "
            f"{record['mettleq_median_ms']:.3f} | {ratio_label} | "
            f"{method_device} | {exact_label} |"
        )
    lines.extend(
        [
            "",
            "Finite-shot rows are expected to show `Exact? False` when independent RNG algorithms produce different count dictionaries; their declared statistical check is the pass criterion.",
            "",
            f"Environment: `{json.dumps(payload['environment'], sort_keys=True)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--pattern", default="*.ipynb")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="verify notebooks without modifying outputs or aggregate result files",
    )
    args = parser.parse_args()

    # A generic ipykernel kernelspec launches `python`; put this interpreter's
    # directory first so notebooks execute in the same verified environment.
    os.environ["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).resolve().parent), os.environ.get("PATH", "")]
    )
    paths = sorted(TUTORIALS.glob(f"qiskit/{args.pattern}")) + sorted(
        TUTORIALS.glob(f"pennylane/{args.pattern}")
    )
    if not paths:
        parser.error("no tutorial notebooks matched")

    results = []
    source_git_commit = _git("rev-parse", "HEAD")
    source_git_dirty = bool(_git("status", "--porcelain"))
    started = time.perf_counter()
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] execute {path.relative_to(ROOT)}", flush=True)
        result = _execute(
            path, timeout=args.timeout, write_back=not args.no_write
        )
        if not result.get("passed"):
            raise RuntimeError(f"tutorial reported failure: {path.relative_to(ROOT)}")
        results.append(result)
        print(
            f"[{index}/{len(paths)}] passed reference={result['reference_median_ms']:.3f}ms "
            f"mettleq={result['mettleq_median_ms']:.3f}ms",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": source_git_commit,
        "git_dirty": source_git_dirty,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "environment": _versions(),
        "elapsed_s": time.perf_counter() - started,
        "total_notebooks": len(paths),
        "passed_notebooks": sum(bool(record["passed"]) for record in results),
        "all_passed": all(bool(record["passed"]) for record in results),
        "results": results,
    }
    if not args.no_write:
        (TUTORIALS / "results.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        _write_markdown(TUTORIALS / "results.md", payload)
    print(json.dumps({key: payload[key] for key in ("total_notebooks", "passed_notebooks", "elapsed_s")}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
