"""Small, inspectable helpers shared by the executed tutorial notebooks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import platform
import statistics
import time
from typing import Any

import numpy as np


RESULT_PREFIX = "TUTORIAL_RESULT::"


def _synchronize(value: Any) -> None:
    """Synchronize lazy array results without depending directly on MLX."""
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    if isinstance(value, (tuple, list)):
        for item in value:
            _synchronize(item)
    if isinstance(value, Mapping):
        for item in value.values():
            _synchronize(item)


def benchmark(
    function: Callable[[], Any],
    *,
    warmups: int = 1,
    repeats: int = 3,
) -> tuple[Any, float, list[float]]:
    """Return the last result, median milliseconds, and individual timings."""
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be non-negative and repeats positive")
    for _ in range(warmups):
        _synchronize(function())
    result = None
    timings_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        _synchronize(result)
        timings_ms.append((time.perf_counter() - start) * 1_000.0)
    return result, statistics.median(timings_ms), timings_ms


def phase_aligned_statevector_error(reference: Any, candidate: Any) -> float:
    """Maximum amplitude error after removing one unobservable global phase."""
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"statevector shapes differ: {reference.shape} != {candidate.shape}"
        )
    overlap = np.vdot(reference, candidate)
    if abs(overlap) > 0.0:
        candidate = candidate * np.exp(-1j * np.angle(overlap))
    return float(np.max(np.abs(reference - candidate), initial=0.0))


def max_abs_error(reference: Any, candidate: Any) -> float:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"result shapes differ: {reference.shape} != {candidate.shape}"
        )
    return float(np.max(np.abs(reference - candidate), initial=0.0))


def counts_to_probabilities(
    counts: Mapping[str, int], *, support: set[str] | None = None
) -> dict[str, float]:
    total = int(sum(counts.values()))
    if total <= 0:
        raise AssertionError("counts must contain at least one shot")
    keys = set(counts) if support is None else set(support) | set(counts)
    return {key: int(counts.get(key, 0)) / total for key in sorted(keys)}


def total_variation_distance(
    first: Mapping[str, int] | Mapping[str, float],
    second: Mapping[str, int] | Mapping[str, float],
) -> float:
    def normalized(values):
        total = float(sum(values.values()))
        if total <= 0.0:
            raise AssertionError("distribution has zero mass")
        return {key: float(value) / total for key, value in values.items()}

    first_probabilities = normalized(first)
    second_probabilities = normalized(second)
    support = set(first_probabilities) | set(second_probabilities)
    return 0.5 * sum(
        abs(first_probabilities.get(key, 0.0) - second_probabilities.get(key, 0.0))
        for key in support
    )


def print_scaling_table(rows: list[Mapping[str, Any]]) -> None:
    """Print a compact per-width table without requiring pandas."""
    print("qubits | reference ms | MettleQ ms | ref/MettleQ | path | max error")
    print("------ | ------------ | ---------- | ----------- | ---- | ---------")
    for row in rows:
        ratio = float(row["reference_ms"]) / float(row["mettleq_ms"])
        path = f"{row.get('method', '?')}/{row.get('device', '?')}"
        print(
            f"{int(row['width']):>6} | {float(row['reference_ms']):>12.3f} | "
            f"{float(row['mettleq_ms']):>10.3f} | {ratio:>10.3f}x | "
            f"{path:<15} | {float(row['error']):.2e}"
        )


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def emit_result(
    *,
    notebook: str,
    framework: str,
    reference_ms: float,
    mettleq_ms: float,
    check: str,
    passed: bool,
    metrics: Mapping[str, Any],
    exact_match: bool | None = None,
    selected_method: str | None = None,
    selected_device: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Print and return the stable record consumed by the suite runner."""
    record = {
        "schema_version": 1,
        "notebook": notebook,
        "framework": framework,
        "passed": bool(passed),
        "check": check,
        "exact_match": exact_match,
        "reference_median_ms": float(reference_ms),
        "mettleq_median_ms": float(mettleq_ms),
        "reference_over_mettleq": (
            float(reference_ms) / float(mettleq_ms)
            if float(mettleq_ms) > 0.0
            else None
        ),
        "selected_method": selected_method,
        "selected_device": selected_device,
        "metrics": json_value(metrics),
        "notes": notes,
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    ratio = record["reference_over_mettleq"]
    print("\nComparison summary")
    print("------------------")
    print(f"Correctness contract: {'PASS' if passed else 'FAIL'} — {check}")
    print(f"SDK reference median: {float(reference_ms):.3f} ms")
    print(f"MettleQ median:       {float(mettleq_ms):.3f} ms")
    if ratio is not None and ratio >= 1.0:
        print(f"Timing interpretation: MettleQ was {ratio:.3f}x faster in this run.")
    elif ratio is not None and ratio > 0.0:
        print(
            "Timing interpretation: the SDK reference was "
            f"{(1.0 / ratio):.3f}x faster in this run."
        )
    if selected_method or selected_device:
        print(f"MettleQ selected: {selected_method or '?'} / {selected_device or '?'}")
    if exact_match is not None:
        print(
            "Byte-for-byte result equality: "
            f"{'yes' if exact_match else 'no (see the declared tolerance/statistical check)'}"
        )
    if notes:
        print(f"Note: {notes}")
    print("\nMachine-readable record (used by the suite runner):")
    print(RESULT_PREFIX + json.dumps(record, sort_keys=True))
    if not passed:
        raise AssertionError(f"{notebook} comparison failed: {metrics}")
    return record


def qiskit_selection(backend) -> tuple[str | None, str | None]:
    selections = getattr(backend, "last_execution_selections", None) or []
    if not selections:
        return None, None
    selection = selections[-1]
    return selection.get("selected_method"), selection.get("selected_device")


def pennylane_selection(device) -> tuple[str | None, str | None]:
    selection = getattr(device, "last_execution_selection", None) or {}
    return selection.get("selected_method"), selection.get("selected_device")
