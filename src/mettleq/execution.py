"""Inspectable runtime policy and execution-plan reporting.

The custom shader path is intentionally explicit: requesting Metal is not the
same thing as proving that the current process can use it.  This module keeps
the policy, capability probes, memory estimates, and dispatch evidence in one
place so SDK adapters and benchmark tools can report the same facts.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from importlib import import_module, metadata as importlib_metadata
import math
import os
import platform
from typing import Any, Dict, Iterable, List, Optional

import mlx.core as mx


STATE_DTYPE = "complex64"
STATE_BYTES_PER_AMPLITUDE = 8
METAL_INDEX_BITS = 32
# n_state itself is passed as uint32, so 2**32 cannot be represented.
METAL_INDEX_QUBIT_LIMIT = 31
METAL_CHECKPOINT_BUDGET_ENV = "METTLEQ_METAL_CHECKPOINT_BUDGET_MB"
STATEVECTOR_UNSAFE_OVERRIDE_ENV = "METTLEQ_ALLOW_UNSAFE_STATEVECTOR"

_TRUE_VALUES = {"1", "true", "on", "yes", "enabled"}
_FALSE_VALUES = {"0", "false", "off", "no", "disabled", ""}


class StatevectorMemoryError(MemoryError):
    """Raised before allocation when reported device limits are insufficient."""

    def __init__(self, message: str, report: Dict[str, Any]):
        super().__init__(message)
        self.report = report


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "unavailable"


def _device_info() -> Dict[str, Any]:
    fn = getattr(mx, "device_info", None)
    if not callable(fn):
        metal = getattr(mx, "metal", None)
        fn = getattr(metal, "device_info", None) if metal is not None else None
    if not callable(fn):
        return {}
    try:
        return dict(fn())
    except Exception:
        return {}


def _metal_available() -> bool:
    try:
        metal = getattr(mx, "metal", None)
        fn = getattr(metal, "is_available", None) if metal is not None else None
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def _default_device() -> str:
    try:
        return str(mx.default_device())
    except Exception:
        return "unavailable"


@lru_cache(maxsize=1)
def _static_metal_capabilities() -> Dict[str, Any]:
    """Probe process-stable Metal facts once.

    Package metadata lookup dominates the former per-dispatch capability cost.
    Platform identity, Metal availability, the custom-kernel API, and physical
    device limits cannot meaningfully change inside one Python process.  The
    selected MLX device is deliberately excluded because callers may change it
    at runtime with ``mx.set_default_device``.
    """
    system = platform.system()
    machine = platform.machine().lower()
    info = _device_info()
    max_buffer = int(info.get("max_buffer_length", 0) or 0)
    working_set = int(info.get("max_recommended_working_set_size", 0) or 0)
    buffer_qubits = _max_qubits_for_bytes(max_buffer, STATE_BYTES_PER_AMPLITUDE)
    working_set_qubits = _max_qubits_for_bytes(
        working_set, 2 * STATE_BYTES_PER_AMPLITUDE
    )
    limits = [METAL_INDEX_QUBIT_LIMIT]
    if buffer_qubits is not None:
        limits.append(buffer_qubits)
    if working_set_qubits is not None:
        limits.append(working_set_qubits)
    return {
        "system": system,
        "machine": machine,
        "apple_silicon": (
            system == "Darwin" and machine in {"arm64", "aarch64"}
        ),
        "metal_available": _metal_available(),
        "custom_kernel_api": callable(
            getattr(getattr(mx, "fast", None), "metal_kernel", None)
        ),
        "mlx_version": _package_version("mlx"),
        "device_name": info.get("device_name", "unavailable"),
        "device_architecture": info.get("architecture", "unavailable"),
        "max_buffer_length": max_buffer or None,
        "max_recommended_working_set_size": working_set or None,
        "buffer_qubit_limit": buffer_qubits,
        "two_state_working_set_qubit_limit": working_set_qubits,
        "effective_qubit_limit": min(limits),
    }


def clear_metal_capability_cache() -> None:
    """Clear cached static probes for tests or deliberate runtime re-probing."""
    _static_metal_capabilities.cache_clear()


def _metal_policy() -> tuple[Optional[str], str, bool, bool]:
    raw = os.environ.get("METTLEQ_METAL_KERNELS")
    normalized = "" if raw is None else raw.strip().lower()
    if raw is None:
        return raw, "auto_default", True, True
    if normalized == "auto":
        return raw, "auto", True, True
    if normalized in _TRUE_VALUES:
        return raw, "enabled", True, True
    if normalized in _FALSE_VALUES:
        return raw, "disabled", False, True
    return raw, "invalid", False, False


def metal_runtime_enabled(
    n_qubits: Optional[int] = None,
    *,
    dtype: str = STATE_DTYPE,
    backend: str = "sv",
) -> bool:
    """Fast boolean selector using cached static and live dynamic checks."""
    _, _, requested, valid_policy = _metal_policy()
    if not requested or not valid_policy:
        return False
    static = _static_metal_capabilities()
    if not (
        backend == "sv"
        and static["apple_silicon"]
        and static["metal_available"]
        and "gpu" in _default_device().lower()
        and static["custom_kernel_api"]
        and dtype == STATE_DTYPE
        and os.environ.get("METTLEQ_DENSE_ONLY", "0") != "1"
    ):
        return False
    if n_qubits is None:
        return True
    n = int(n_qubits)
    return (
        0 <= n <= METAL_INDEX_QUBIT_LIMIT
        and n <= static["effective_qubit_limit"]
    )


def _max_qubits_for_bytes(limit: int, bytes_per_state: int) -> Optional[int]:
    if limit <= 0 or bytes_per_state <= 0:
        return None
    amplitudes = limit // bytes_per_state
    if amplitudes <= 0:
        return 0
    return int(math.floor(math.log2(amplitudes)))


def state_memory_estimate(n_qubits: int, dtype: str = STATE_DTYPE) -> Dict[str, Any]:
    """Return transparent state and temporary-memory expectations.

    Custom kernels are out-of-place.  Two state vectors are therefore the
    minimum useful temporary expectation; MLX may retain additional lazy
    intermediates depending on the graph and cache state.
    """
    n = int(n_qubits)
    if n < 0:
        raise ValueError("n_qubits must be non-negative")
    bytes_per_amplitude = STATE_BYTES_PER_AMPLITUDE if dtype == STATE_DTYPE else None
    state_bytes = ((1 << n) * bytes_per_amplitude
                   if bytes_per_amplitude is not None else None)
    temporary_bytes = state_bytes if state_bytes is not None else None
    expected_peak = (2 * state_bytes) if state_bytes is not None else None
    return {
        "dtype": dtype,
        "amplitudes": 1 << n,
        "bytes_per_amplitude": bytes_per_amplitude,
        "state_bytes": state_bytes,
        "state_mib": (state_bytes / (1024.0 * 1024.0)
                      if state_bytes is not None else None),
        "temporary_memory_expectation_bytes": temporary_bytes,
        "minimum_input_plus_output_bytes": expected_peak,
        "expectation_basis": (
            "out-of-place custom kernels require one input and one output "
            "state; lazy graphs and lookup tables can add intermediates"
        ),
    }


def statevector_unsafe_override_policy(
    allow_unsafe: Optional[bool] = None,
) -> Dict[str, Any]:
    """Resolve the deliberate escape hatch for statevector preflight refusals.

    An explicit constructor argument takes precedence over the environment.
    Invalid environment values fail closed so a typo cannot silently permit a
    statevector allocation that the device limits reject.
    """
    raw = os.environ.get(STATEVECTOR_UNSAFE_OVERRIDE_ENV)
    if allow_unsafe is not None:
        if not isinstance(allow_unsafe, bool):
            raise TypeError("allow_unsafe_statevector must be a boolean or None")
        return {
            "enabled": allow_unsafe,
            "source": "constructor_argument",
            "environment_variable": STATEVECTOR_UNSAFE_OVERRIDE_ENV,
            "environment_value": raw,
        }

    normalized = "" if raw is None else raw.strip().lower()
    if raw is None:
        return {
            "enabled": False,
            "source": "default_disabled",
            "environment_variable": STATEVECTOR_UNSAFE_OVERRIDE_ENV,
            "environment_value": None,
        }
    if normalized in _TRUE_VALUES:
        enabled = True
    elif normalized in _FALSE_VALUES:
        enabled = False
    else:
        raise ValueError(
            f"{STATEVECTOR_UNSAFE_OVERRIDE_ENV} must be a boolean value"
        )
    return {
        "enabled": enabled,
        "source": "environment",
        "environment_variable": STATEVECTOR_UNSAFE_OVERRIDE_ENV,
        "environment_value": raw,
    }


def statevector_preflight(
    n_qubits: int,
    *,
    dtype: str = STATE_DTYPE,
    allow_unsafe: Optional[bool] = None,
) -> Dict[str, Any]:
    """Estimate a statevector allocation before constructing the MLX array.

    The check is intentionally one-sided. It refuses allocations whose single
    state buffer or two-state out-of-place lower bound already exceeds a
    reported device limit. Passing means only that those lower bounds fit; MLX
    lazy graphs, lookup tables, caches, and other processes can consume more.
    """
    n = int(n_qubits)
    if n < 0:
        raise ValueError("n_qubits must be non-negative")
    if dtype != STATE_DTYPE:
        raise ValueError(
            f"statevector preflight currently supports only {STATE_DTYPE}"
        )
    memory = state_memory_estimate(n, dtype=dtype)
    override = statevector_unsafe_override_policy(allow_unsafe)
    static = _static_metal_capabilities()
    max_buffer = static.get("max_buffer_length")
    working_set = static.get("max_recommended_working_set_size")
    state_bytes = memory.get("state_bytes")
    minimum_peak = memory.get("minimum_input_plus_output_bytes")

    buffer_check = (
        state_bytes <= max_buffer
        if state_bytes is not None and max_buffer is not None else None
    )
    working_set_check = (
        minimum_peak <= working_set
        if minimum_peak is not None and working_set is not None else None
    )
    failures: List[str] = []
    if buffer_check is False:
        failures.append("single_state_exceeds_max_buffer_length")
    if working_set_check is False:
        failures.append("two_state_lower_bound_exceeds_recommended_working_set")

    refused_without_override = bool(failures)
    overridden = refused_without_override and bool(override["enabled"])
    allowed = not refused_without_override or overridden
    limits_available = buffer_check is not None and working_set_check is not None
    if overridden:
        decision = "allowed_with_unsafe_override"
        reason = (
            "reported device limits reject the allocation, but the explicit "
            "unsafe override permits it"
        )
    elif refused_without_override:
        decision = "refused"
        reason = "reported device limits are below the statevector lower bound"
    elif limits_available:
        decision = "allowed_within_reported_lower_bounds"
        reason = "statevector lower bounds fit within the reported device limits"
    else:
        decision = "allowed_with_unverified_device_limits"
        reason = "one or more device memory limits are unavailable"

    headroom = (
        working_set - minimum_peak
        if working_set is not None and minimum_peak is not None else None
    )
    return {
        "schema_version": 1,
        "allowed": allowed,
        "decision": decision,
        "reason": reason,
        "overridden": overridden,
        "override": override,
        "failure_reasons": failures,
        "qubits": n,
        "dtype": dtype,
        "device": {
            "name": static.get("device_name"),
            "architecture": static.get("device_architecture"),
            "max_buffer_length": max_buffer,
            "max_recommended_working_set_size": working_set,
        },
        "checks": {
            "single_state_within_max_buffer_length": buffer_check,
            "two_state_lower_bound_within_recommended_working_set": (
                working_set_check
            ),
        },
        "cost_model": {
            **memory,
            "recommended_working_set_headroom_bytes": headroom,
            "scope": "allocation preflight lower bound",
            "not_included": [
                "additional lazy-graph intermediates",
                "lookup and index buffers",
                "allocator cache",
                "memory used by other processes",
            ],
        },
        "passing_guarantee": (
            "Passing does not guarantee that an arbitrary circuit fits; it "
            "only establishes that the reported single-buffer and two-state "
            "lower bounds do not already rule it out."
        ),
    }


def require_statevector_preflight(
    n_qubits: int,
    *,
    dtype: str = STATE_DTYPE,
    allow_unsafe: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return the preflight report or raise before statevector allocation."""
    report = statevector_preflight(
        n_qubits, dtype=dtype, allow_unsafe=allow_unsafe
    )
    if report["allowed"]:
        return report
    cost = report["cost_model"]
    device = report["device"]
    gib = 1024.0 ** 3
    state_gib = float(cost["state_bytes"]) / gib
    lower_gib = float(cost["minimum_input_plus_output_bytes"]) / gib
    def _limit_text(value: Optional[int]) -> str:
        return f"{float(value) / gib:.2f} GiB" if value is not None else "unknown"

    message = (
        f"Refusing {n_qubits}-qubit {dtype} statevector before allocation: "
        f"one state is {state_gib:.2f} GiB and the out-of-place lower bound "
        f"is {lower_gib:.2f} GiB; device max buffer is "
        f"{_limit_text(device['max_buffer_length'])} and recommended working "
        f"set is {_limit_text(device['max_recommended_working_set_size'])}. "
        "Use the MPS backend, reduce qubits, or set "
        f"{STATEVECTOR_UNSAFE_OVERRIDE_ENV}=1 to deliberately bypass this "
        "preflight."
    )
    raise StatevectorMemoryError(message, report)


def metal_checkpoint_policy(
    budget_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve the opt-in lazy-graph checkpoint budget.

    An explicit byte budget is intended for SDKs and tests. When it is not
    supplied, ``METTLEQ_METAL_CHECKPOINT_BUDGET_MB`` accepts a positive MiB value.
    Unset, empty, ``0``, and normal false-like values preserve the existing
    fully lazy behavior. Invalid or negative values are rejected instead of
    silently changing execution semantics.
    """
    if budget_bytes is not None:
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int):
            raise TypeError("metal checkpoint budget must be an integer byte count")
        if budget_bytes < 0:
            raise ValueError("metal checkpoint budget must be non-negative")
        return {
            "configured": budget_bytes > 0,
            "budget_bytes": budget_bytes or None,
            "source": "device_argument",
            "environment_variable": METAL_CHECKPOINT_BUDGET_ENV,
            "environment_value": os.environ.get(METAL_CHECKPOINT_BUDGET_ENV),
        }

    raw = os.environ.get(METAL_CHECKPOINT_BUDGET_ENV)
    normalized = "" if raw is None else raw.strip().lower()
    if raw is None:
        return {
            "configured": False,
            "budget_bytes": None,
            "source": "default_disabled",
            "environment_variable": METAL_CHECKPOINT_BUDGET_ENV,
            "environment_value": None,
        }
    if normalized in _FALSE_VALUES:
        return {
            "configured": False,
            "budget_bytes": None,
            "source": "environment",
            "environment_variable": METAL_CHECKPOINT_BUDGET_ENV,
            "environment_value": raw,
        }
    try:
        mib = float(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{METAL_CHECKPOINT_BUDGET_ENV} must be a non-negative MiB value"
        ) from exc
    if not math.isfinite(mib) or mib < 0:
        raise ValueError(
            f"{METAL_CHECKPOINT_BUDGET_ENV} must be a finite non-negative MiB value"
        )
    resolved = int(mib * 1024 * 1024)
    if mib > 0 and resolved <= 0:
        raise ValueError(
            f"{METAL_CHECKPOINT_BUDGET_ENV} is smaller than one byte"
        )
    return {
        "configured": resolved > 0,
        "budget_bytes": resolved or None,
        "source": "environment",
        "environment_variable": METAL_CHECKPOINT_BUDGET_ENV,
        "environment_value": raw,
    }


def metal_runtime_status(
    n_qubits: Optional[int] = None,
    *,
    dtype: str = STATE_DTYPE,
    backend: str = "sv",
) -> Dict[str, Any]:
    """Explain whether the custom Metal path is requested and usable.

    ``METTLEQ_METAL_KERNELS`` accepts explicit on/off values plus ``auto``.
    Unset uses capability-probed automatic selection. ``0`` remains a stable
    pure-MLX ablation/compatibility path. Explicit on is still safely refused
    when the platform or kernel constraints are unsupported.
    """
    raw, policy, requested, valid_policy = _metal_policy()
    static = _static_metal_capabilities()
    cache_info = _static_metal_capabilities.cache_info()
    default_device = _default_device()
    gpu_selected = "gpu" in default_device.lower()

    checks: Dict[str, bool] = {
        "valid_policy": valid_policy,
        "statevector_backend": backend == "sv",
        "apple_silicon": static["apple_silicon"],
        "metal_available": static["metal_available"],
        "gpu_device_selected": gpu_selected,
        "custom_kernel_api": static["custom_kernel_api"],
        "dtype_complex64": dtype == STATE_DTYPE,
        "dense_ablation_disabled": os.environ.get("METTLEQ_DENSE_ONLY", "0") != "1",
    }
    if n_qubits is not None:
        checks["non_negative_qubits"] = int(n_qubits) >= 0
        checks["index_width"] = 0 <= int(n_qubits) <= METAL_INDEX_QUBIT_LIMIT
        checks["device_memory_limit"] = (
            0 <= int(n_qubits) <= static["effective_qubit_limit"]
        )

    capable = all(checks.values())
    enabled = requested and capable
    failed = [name for name, ok in checks.items() if not ok]
    if enabled:
        reason = "custom Metal kernels selected"
    elif not valid_policy:
        reason = (
            "invalid METTLEQ_METAL_KERNELS value; use 0, 1, off, on, or auto"
        )
    elif not requested:
        reason = "custom Metal kernels are disabled by policy"
    else:
        reason = "custom Metal kernels unavailable: " + ", ".join(failed)

    return {
        "environment_value": raw,
        "policy": policy,
        "requested": requested,
        "enabled": enabled,
        "reason": reason,
        "checks": checks,
        "failed_checks": failed,
        "platform": {
            "system": static["system"],
            "machine": static["machine"],
        },
        "mlx": {
            "version": static["mlx_version"],
            "declared_requirement": "mlx>=0.6.0",
            "compatibility_policy": (
                "capability-probed; mx.fast.metal_kernel is required and no "
                "untested upper version bound is assumed"
            ),
            "default_device": default_device,
            "metal_available": static["metal_available"],
        },
        "device": {
            "name": static["device_name"],
            "architecture": static["device_architecture"],
            "max_buffer_length": static["max_buffer_length"],
            "max_recommended_working_set_size": static[
                "max_recommended_working_set_size"
            ],
        },
        "capability_cache": {
            "scope": "process",
            "static_facts": [
                "platform",
                "metal_available",
                "custom_kernel_api",
                "mlx_version",
                "device_limits",
            ],
            "dynamic_facts": [
                "environment_policy",
                "selected_device",
                "backend",
                "dtype",
                "dense_ablation",
                "requested_qubits",
            ],
            "info": {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "current_size": cache_info.currsize,
            },
        },
        "kernel_constraints": {
            "backend": "statevector only",
            "dtype": STATE_DTYPE,
            "index_bits": METAL_INDEX_BITS,
            "index_qubit_limit": METAL_INDEX_QUBIT_LIMIT,
            "buffer_qubit_limit": static["buffer_qubit_limit"],
            "two_state_working_set_qubit_limit": static[
                "two_state_working_set_qubit_limit"
            ],
            "effective_qubit_limit": static["effective_qubit_limit"],
            "requested_qubits": n_qubits,
        },
    }


def metal_memory_snapshot() -> Dict[str, Optional[int]]:
    metal = getattr(mx, "metal", None)
    out: Dict[str, Optional[int]] = {}
    for label, name in (
        ("active_bytes", "get_active_memory"),
        ("peak_bytes", "get_peak_memory"),
        ("cache_bytes", "get_cache_memory"),
    ):
        fn = getattr(mx, name, None)
        if not callable(fn):
            fn = getattr(metal, name, None) if metal is not None else None
        try:
            out[label] = int(fn()) if callable(fn) else None
        except Exception:
            out[label] = None
    return out


_CACHE_COMPONENTS = {
    "mettleq_zz_chain_layer": ("mettleq.shaders.zz", "_zz_chain_kernel"),
    "mettleq_zz_generic_layer": ("mettleq.shaders.zz", "_zz_generic_kernel"),
    "mettleq_zz_weighted_layer": ("mettleq.shaders.zz", "_zz_weighted_kernel"),
    "mettleq_qft_stage_gen": ("mettleq.shaders.qft", "_qft_stage_gen_kernel"),
    "mettleq_qft_radix4": ("mettleq.shaders.qft", "_qft_radix4_kernel"),
    "mettleq_qft_stage_pair": ("mettleq.shaders.qft", "_qft_kernel"),
    "mettleq_rx_pair": ("mettleq.shaders.single_qubit", "_rx_pair_kernel"),
    "mettleq_rx_single": ("mettleq.shaders.single_qubit", "_rx_single_kernel"),
    "mettleq_u2_pair": ("mettleq.shaders.single_qubit", "_u2_pair_kernel"),
    "mettleq_u2_single": ("mettleq.shaders.single_qubit", "_u2_single_kernel"),
    "mettleq_u2_list_pair": ("mettleq.shaders.single_qubit", "_u2l_pair_kernel"),
    "mettleq_walsh4": ("mettleq.shaders.single_qubit", "_walsh4_kernel"),
    "mettleq_phase_popcount": ("mettleq.shaders.single_qubit", "_phase_popcount_kernel"),
    "mettleq_diag_chain_layer": ("mettleq.shaders.diag", "_diag_chain_kernel"),
    "mettleq_diag_pair_layer": ("mettleq.shaders.diag", "_diag_pair_kernel"),
    "mettleq_diag_weighted_layer": ("mettleq.shaders.diag", "_diag_weighted_kernel"),
    "mettleq_xor_flip": ("mettleq.shaders.xor_affine", "_xor_flip_kernel"),
    "mettleq_xor_affine_gather": ("mettleq.shaders.xor_affine", "_xor_gather_kernel"),
}


def kernel_cache_snapshot() -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for kernel, (module_name, attribute) in _CACHE_COMPONENTS.items():
        try:
            module = import_module(module_name)
            out[kernel] = getattr(module, attribute, None) is not None
        except Exception:
            out[kernel] = False
    return out


def _h_launches(n: int) -> int:
    remainder = n % 4
    return n // 4 + remainder // 2 + remainder % 2


def _is_identity_rows(rows: Iterable[int], n: int) -> bool:
    return list(rows) == [1 << (n - 1 - q) for q in range(n)]


def _custom_dispatch_spec(op: Dict[str, Any], n: int) -> Optional[Dict[str, Any]]:
    name = str(op.get("name", "")).upper()
    kernels: List[str]
    launches: int
    family: str
    if name == "_ZZLAYER":
        full_chain = sorted(op.get("bonds", [])) == [(i, i + 1) for i in range(n - 1)]
        kernels = ["mettleq_zz_chain_layer" if full_chain else "mettleq_zz_generic_layer"]
        launches, family = 1, "zz_layer"
    elif name == "_ZZWEIGHTED":
        kernels, launches, family = ["mettleq_zz_weighted_layer"], 1, "zz_weighted"
    elif name == "_DIAGLAYER":
        bonds = [tuple(sorted(bond)) for bond in op.get("bonds", [])]
        chain = (len(bonds) == len(set(bonds)) and all(
            b - a == 1 or (a == 0 and b == n - 1) for a, b in bonds
        ))
        kernels = ["mettleq_diag_chain_layer" if chain else "mettleq_diag_pair_layer"]
        launches, family = 1, "diagonal_pair_layer"
    elif name == "_DIAGWEIGHTED":
        kernels, launches, family = ["mettleq_diag_weighted_layer"], 1, "diagonal_weighted"
    elif name == "_QFTSTAGE":
        kernels, launches, family = ["mettleq_qft_stage_gen"], 1, "qft_stage"
    elif name == "_RXLAYER":
        kernels = ["mettleq_rx_pair"] + (["mettleq_rx_single"] if n % 2 else [])
        launches, family = n // 2 + n % 2, "uniform_rx_layer"
    elif name == "_U2LAYER":
        tail = n % 4
        kernels = ["mettleq_u2_list_quad"]
        if tail >= 2:
            kernels.append("mettleq_u2_pair")
        if tail % 2:
            kernels.append("mettleq_u2_single")
        launches, family = n // 4 + tail // 2 + tail % 2, "uniform_single_qubit_layer"
    elif name == "_U2LISTLAYER":
        active = set(op.get("active") or range(n))
        launches = sum(1 for q in range(0, n - 3, 4)
                       if any(q + offset in active for offset in range(4)))
        tail_start = n - (n % 4)
        if n % 4 >= 2 and (tail_start in active or tail_start + 1 in active):
            launches += 1
        if n % 2 and n - 1 in active:
            launches += 1
        kernels = ["mettleq_u2_list_quad"]
        if n % 4 >= 2:
            kernels.append("mettleq_u2_list_pair")
        if n % 2 and n - 1 in active:
            kernels.append("mettleq_u2_single")
        family = "per_qubit_single_qubit_layer"
    elif name == "_XORLAYER":
        identity = _is_identity_rows(op.get("rows", []), n)
        kernels = ["mettleq_xor_flip" if identity else "mettleq_xor_affine_gather"]
        launches, family = 1, "xor_affine_permutation"
    elif name in {"_XXLAYER", "_YYLAYER"}:
        h_launches = _h_launches(n)
        kernels = ["mettleq_walsh4", "mettleq_u2_pair", "mettleq_u2_single",
                   "mettleq_zz_chain_layer"]
        kernels = [k for k in kernels if not (
            (k == "mettleq_walsh4" and n < 4)
            or (k == "mettleq_u2_pair" and n % 4 < 2)
            or (k == "mettleq_u2_single" and n % 2 == 0)
        )]
        launches = 2 * h_launches + 1
        if name == "_YYLAYER":
            kernels.append("mettleq_phase_popcount")
            launches += 2
        family = "xx_layer" if name == "_XXLAYER" else "yy_layer"
    else:
        return None
    return {
        "optimized_op": name,
        "kernel_family": family,
        "concrete_kernels": kernels,
        "expected_metal_launches": launches,
        "supports_intra_layer_streaming": (
            launches > 1
            and name in {
                "_RXLAYER",
                "_U2LAYER",
                "_U2LISTLAYER",
                "_XXLAYER",
                "_YYLAYER",
            }
        ),
    }


def build_execution_plan(
    *,
    n_qubits: int,
    backend: str,
    operations: List[Dict[str, Any]],
    optimized_operations: List[Dict[str, Any]],
    checkpoint_policy_data: Optional[Dict[str, Any]] = None,
    statevector_preflight_data: Optional[Dict[str, Any]] = None,
    initial_pending_custom_passes: int = 0,
    initial_pending_custom_io_bytes: int = 0,
) -> Dict[str, Any]:
    status = metal_runtime_status(n_qubits, backend=backend)
    cache = kernel_cache_snapshot()
    matched: List[Dict[str, Any]] = []
    for optimized_index, op in enumerate(optimized_operations):
        spec = _custom_dispatch_spec(op, n_qubits)
        if spec is not None:
            spec["optimized_operation_index"] = optimized_index
            spec["cache_before"] = {
                kernel: cache.get(kernel, False)
                for kernel in spec["concrete_kernels"]
            }
            matched.append(spec)

    selected = matched if status["enabled"] else []
    pattern_counts = Counter(item["kernel_family"] for item in matched)
    remaining = Counter(
        str(op.get("name", "unknown")).upper()
        for op in optimized_operations
        if _custom_dispatch_spec(op, n_qubits) is None
    )
    fallback_reasons: List[str] = []
    if not status["enabled"]:
        fallback_reasons.append(status["reason"])
    if remaining:
        fallback_reasons.append(
            "operations without a custom structured match use the pure MLX "
            "statevector path"
        )

    selected_kernels = sorted({
        kernel for item in selected for kernel in item["concrete_kernels"]
    })
    cache_values = [cache.get(kernel, False) for kernel in selected_kernels]
    if not selected_kernels:
        compilation_status = "not_selected"
    elif all(cache_values):
        compilation_status = "process_wrappers_warm_compilation_cache_opaque"
    else:
        compilation_status = "cold_process_wrapper_lazy_compilation_pending"

    memory = state_memory_estimate(n_qubits)
    state_bytes = int(memory.get("state_bytes") or 0)
    policy = dict(checkpoint_policy_data or metal_checkpoint_policy())
    checkpoint_enabled = bool(
        policy.get("configured")
        and status["enabled"]
        and backend == "sv"
        and state_bytes > 0
    )
    budget = int(policy.get("budget_bytes") or 0)
    pass_io_bytes = 2 * state_bytes
    max_pending_passes = (
        max(1, budget // pass_io_bytes)
        if checkpoint_enabled and pass_io_bytes > 0 else None
    )
    pending_passes = int(initial_pending_custom_passes)
    pending_io_bytes = int(initial_pending_custom_io_bytes)
    predicted_checkpoints: List[Dict[str, Any]] = []
    for spec in selected:
        layer_passes = int(spec["expected_metal_launches"])
        layer_io_bytes = 2 * state_bytes * layer_passes
        spec["estimated_input_output_bytes"] = layer_io_bytes
        if (checkpoint_enabled and pending_io_bytes > 0
                and pending_io_bytes + layer_io_bytes > budget):
            predicted_checkpoints.append({
                "boundary": "before_optimized_operation",
                "optimized_operation_index": spec["optimized_operation_index"],
                "reason": "next_fused_layer_would_exceed_budget",
                "estimated_passes_evaluated": pending_passes,
                "estimated_input_output_bytes_evaluated": pending_io_bytes,
            })
            pending_passes = 0
            pending_io_bytes = 0
        remaining_layer_passes = layer_passes
        if (checkpoint_enabled
                and spec["supports_intra_layer_streaming"]
                and max_pending_passes is not None
                and layer_passes > max_pending_passes):
            launch_index = 0
            while remaining_layer_passes > max_pending_passes:
                launch_index += max_pending_passes
                predicted_checkpoints.append({
                    "boundary": "within_optimized_operation",
                    "optimized_operation_index": spec[
                        "optimized_operation_index"
                    ],
                    "within_layer_launch_index": launch_index,
                    "within_layer_launch_count": layer_passes,
                    "reason": "multi_launch_layer_streaming_budget",
                    "estimated_passes_evaluated": max_pending_passes,
                    "estimated_input_output_bytes_evaluated": (
                        max_pending_passes * pass_io_bytes
                    ),
                })
                remaining_layer_passes -= max_pending_passes
        pending_passes += remaining_layer_passes
        pending_io_bytes += remaining_layer_passes * pass_io_bytes
        if checkpoint_enabled and pending_io_bytes > budget:
            reason = (
                "single_custom_launch_exceeds_budget"
                if remaining_layer_passes == 1 and pass_io_bytes > budget
                else "single_fused_layer_exceeds_budget"
            )
            predicted_checkpoints.append({
                "boundary": "after_optimized_operation",
                "optimized_operation_index": spec["optimized_operation_index"],
                "reason": reason,
                "estimated_passes_evaluated": pending_passes,
                "estimated_input_output_bytes_evaluated": pending_io_bytes,
            })
            pending_passes = 0
            pending_io_bytes = 0

    if checkpoint_enabled:
        checkpoint_reason = "configured budget and Metal statevector path selected"
    elif not policy.get("configured"):
        checkpoint_reason = "checkpoint budget is not configured"
    else:
        checkpoint_reason = (
            "checkpoint budget configured but custom Metal is not selected: "
            + status["reason"]
        )

    predicted_custom_io_bytes = sum(
        int(item.get("estimated_input_output_bytes") or 0)
        for item in selected
    )

    if backend == "sv":
        preflight = dict(
            statevector_preflight_data or statevector_preflight(n_qubits)
        )
    else:
        preflight = {
            "schema_version": 1,
            "applicable": False,
            "reason": "statevector allocation preflight does not apply to MPS",
        }

    return {
        "schema_version": 4,
        "backend": backend,
        "selected_device": status["mlx"]["default_device"],
        "dtype": STATE_DTYPE,
        "qubits": n_qubits,
        "input_operation_count": len(operations),
        "optimized_operation_count": len(optimized_operations),
        "optimization_passes": [
            {
                "name": "structured_runtime_fusion",
                "enabled": os.environ.get("METTLEQ_DENSE_ONLY", "0") != "1",
                "matched_patterns": dict(sorted(pattern_counts.items())),
            },
            {
                "name": "custom_metal_selection",
                "enabled": status["enabled"],
                "policy": status["policy"],
            },
        ],
        "matched_structured_patterns": dict(sorted(pattern_counts.items())),
        "selected_custom_kernels": selected,
        "selected_concrete_kernels": selected_kernels,
        "expected_custom_kernel_launches": sum(
            item["expected_metal_launches"] for item in selected
        ),
        "pure_mlx_fallback_operations": dict(sorted(remaining.items())),
        "fallback_reasons": fallback_reasons,
        "compilation_cache": {
            "status": compilation_status,
            "process_wrapper_cache_before": cache,
            "note": (
                "MLX does not expose per-kernel compiler-cache hit state; "
                "cold/warm timing is required to quantify compilation overhead"
            ),
        },
        "capabilities": status,
        "memory": memory,
        "statevector_preflight": preflight,
        "execution_cost_model": {
            "scope": "selected custom Metal launches",
            "state_bytes": state_bytes,
            "out_of_place_bytes_per_custom_launch": pass_io_bytes,
            "predicted_custom_launches": sum(
                item["expected_metal_launches"] for item in selected
            ),
            "predicted_custom_input_output_bytes": predicted_custom_io_bytes,
            "pure_mlx_operations_modeled": False,
            "note": (
                "Traffic is a deterministic launch-accounting estimate, not "
                "an allocator peak or a memory-bandwidth measurement."
            ),
        },
        "checkpointing": {
            **policy,
            "enabled": checkpoint_enabled,
            "reason": checkpoint_reason,
            "accounting_basis": (
                "two statevector byte-transfers (input plus output) per "
                "expected custom Metal launch"
            ),
            "state_bytes": state_bytes,
            "estimated_input_output_bytes_per_pass": 2 * state_bytes,
            "max_pending_custom_passes_per_streamed_chunk": (
                max_pending_passes
            ),
            "safe_boundaries": (
                "between custom Metal launches inside supported multi-launch "
                "layers; otherwise between optimized fused operations; never "
                "inside one custom kernel launch"
            ),
            "initial_pending_custom_passes": int(initial_pending_custom_passes),
            "initial_pending_custom_io_bytes": int(
                initial_pending_custom_io_bytes
            ),
            "predicted_checkpoints": predicted_checkpoints,
            "predicted_checkpoint_count": len(predicted_checkpoints),
            "predicted_pending_custom_passes_after_graph_build": pending_passes,
            "predicted_pending_custom_io_bytes_after_graph_build": (
                pending_io_bytes
            ),
            "actual_checkpoints": [],
            "actual_checkpoint_count": 0,
            "actual_evaluated_custom_passes": 0,
            "actual_evaluated_custom_io_bytes": 0,
            "pending_custom_passes_after_graph_build": int(
                initial_pending_custom_passes
            ),
            "pending_custom_io_bytes_after_graph_build": int(
                initial_pending_custom_io_bytes
            ),
        },
        "observed_custom_dispatches": [],
        "execution_status": "planned",
        "synchronized": False,
        "metal_memory_before": metal_memory_snapshot(),
    }


def record_custom_dispatch(
    plan: Optional[Dict[str, Any]],
    op: Dict[str, Any],
    *,
    n_qubits: Optional[int] = None,
    optimized_operation_index: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Return a dispatch estimate and append it to an enabled report."""
    if n_qubits is None:
        if plan is None:
            return None
        n_qubits = int(plan["qubits"])
    spec = _custom_dispatch_spec(op, int(n_qubits))
    if spec is not None and optimized_operation_index is not None:
        spec["optimized_operation_index"] = optimized_operation_index
    if (spec is not None and plan is not None
            and plan.get("capabilities", {}).get("enabled", False)):
        plan["observed_custom_dispatches"].append(spec)
    return spec


def record_evaluation_checkpoint(
    plan: Optional[Dict[str, Any]], event: Dict[str, Any]
) -> None:
    if plan is None:
        return
    checkpointing = plan.get("checkpointing")
    if not checkpointing or not checkpointing.get("enabled", False):
        return
    checkpointing["actual_checkpoints"].append(event)
    checkpointing["actual_checkpoint_count"] += 1
    checkpointing["actual_evaluated_custom_passes"] += int(
        event["estimated_passes_evaluated"]
    )
    checkpointing["actual_evaluated_custom_io_bytes"] += int(
        event["estimated_input_output_bytes_evaluated"]
    )


def mark_graph_built(
    plan: Optional[Dict[str, Any]],
    *,
    pending_custom_passes: int = 0,
    pending_custom_io_bytes: int = 0,
) -> None:
    if plan is not None:
        checkpoints = plan.get("checkpointing", {}).get(
            "actual_checkpoint_count", 0
        )
        plan["execution_status"] = (
            "checkpointed_lazy_graph_built" if checkpoints
            else "lazy_graph_built"
        )
        checkpointing = plan.get("checkpointing")
        if checkpointing is not None:
            checkpointing["pending_custom_passes_after_graph_build"] = int(
                pending_custom_passes
            )
            checkpointing["pending_custom_io_bytes_after_graph_build"] = int(
                pending_custom_io_bytes
            )
        plan["metal_memory_after_graph_build"] = metal_memory_snapshot()


def mark_synchronized(plan: Optional[Dict[str, Any]]) -> None:
    if plan is None:
        return
    plan["execution_status"] = "evaluated"
    plan["synchronized"] = True
    checkpointing = plan.get("checkpointing")
    if checkpointing is not None:
        checkpointing["pending_custom_passes_after_synchronize"] = 0
        checkpointing["pending_custom_io_bytes_after_synchronize"] = 0
    plan["metal_memory_after_evaluation"] = metal_memory_snapshot()
    after = kernel_cache_snapshot()
    plan["compilation_cache"]["process_wrapper_cache_after"] = after
    if plan.get("selected_concrete_kernels"):
        plan["compilation_cache"]["status"] = (
            "compiled_or_loaded_and_synchronously_executed"
        )


__all__ = [
    "METAL_INDEX_BITS",
    "METAL_INDEX_QUBIT_LIMIT",
    "METAL_CHECKPOINT_BUDGET_ENV",
    "STATE_DTYPE",
    "build_execution_plan",
    "clear_metal_capability_cache",
    "kernel_cache_snapshot",
    "mark_graph_built",
    "mark_synchronized",
    "metal_checkpoint_policy",
    "metal_memory_snapshot",
    "metal_runtime_enabled",
    "metal_runtime_status",
    "record_custom_dispatch",
    "record_evaluation_checkpoint",
    "state_memory_estimate",
]
