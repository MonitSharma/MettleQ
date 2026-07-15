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

_TRUE_VALUES = {"1", "true", "on", "yes", "enabled"}
_FALSE_VALUES = {"0", "false", "off", "no", "disabled", ""}


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
    raw = os.environ.get("MLXQ_METAL_KERNELS")
    normalized = "" if raw is None else raw.strip().lower()
    if raw is None:
        return raw, "off_by_default", False, True
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
        and os.environ.get("MLXQ_DENSE_ONLY", "0") != "1"
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


def metal_runtime_status(
    n_qubits: Optional[int] = None,
    *,
    dtype: str = STATE_DTYPE,
    backend: str = "sv",
) -> Dict[str, Any]:
    """Explain whether the custom Metal path is requested and usable.

    ``MLXQ_METAL_KERNELS`` accepts explicit on/off values plus ``auto``.
    Unset remains off for compatibility.  ``auto`` enables custom kernels only
    when every hard capability check passes.  Explicit on is still safely
    refused when the platform or kernel constraints are unsupported.
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
        "dense_ablation_disabled": os.environ.get("MLXQ_DENSE_ONLY", "0") != "1",
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
            "invalid MLXQ_METAL_KERNELS value; use 0, 1, off, on, or auto"
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
    "mlxq_zz_chain_layer": ("mlxq.shaders.zz", "_zz_chain_kernel"),
    "mlxq_zz_generic_layer": ("mlxq.shaders.zz", "_zz_generic_kernel"),
    "mlxq_zz_weighted_layer": ("mlxq.shaders.zz", "_zz_weighted_kernel"),
    "mlxq_qft_stage_gen": ("mlxq.shaders.qft", "_qft_stage_gen_kernel"),
    "mlxq_qft_radix4": ("mlxq.shaders.qft", "_qft_radix4_kernel"),
    "mlxq_qft_stage_pair": ("mlxq.shaders.qft", "_qft_kernel"),
    "mlxq_rx_pair": ("mlxq.shaders.single_qubit", "_rx_pair_kernel"),
    "mlxq_rx_single": ("mlxq.shaders.single_qubit", "_rx_single_kernel"),
    "mlxq_u2_pair": ("mlxq.shaders.single_qubit", "_u2_pair_kernel"),
    "mlxq_u2_single": ("mlxq.shaders.single_qubit", "_u2_single_kernel"),
    "mlxq_u2_list_pair": ("mlxq.shaders.single_qubit", "_u2l_pair_kernel"),
    "mlxq_walsh4": ("mlxq.shaders.single_qubit", "_walsh4_kernel"),
    "mlxq_phase_popcount": ("mlxq.shaders.single_qubit", "_phase_popcount_kernel"),
    "mlxq_diag_chain_layer": ("mlxq.shaders.diag", "_diag_chain_kernel"),
    "mlxq_diag_pair_layer": ("mlxq.shaders.diag", "_diag_pair_kernel"),
    "mlxq_diag_weighted_layer": ("mlxq.shaders.diag", "_diag_weighted_kernel"),
    "mlxq_xor_flip": ("mlxq.shaders.xor_affine", "_xor_flip_kernel"),
    "mlxq_xor_affine_gather": ("mlxq.shaders.xor_affine", "_xor_gather_kernel"),
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
        kernels = ["mlxq_zz_chain_layer" if full_chain else "mlxq_zz_generic_layer"]
        launches, family = 1, "zz_layer"
    elif name == "_ZZWEIGHTED":
        kernels, launches, family = ["mlxq_zz_weighted_layer"], 1, "zz_weighted"
    elif name == "_DIAGLAYER":
        bonds = [tuple(sorted(bond)) for bond in op.get("bonds", [])]
        chain = (len(bonds) == len(set(bonds)) and all(
            b - a == 1 or (a == 0 and b == n - 1) for a, b in bonds
        ))
        kernels = ["mlxq_diag_chain_layer" if chain else "mlxq_diag_pair_layer"]
        launches, family = 1, "diagonal_pair_layer"
    elif name == "_DIAGWEIGHTED":
        kernels, launches, family = ["mlxq_diag_weighted_layer"], 1, "diagonal_weighted"
    elif name == "_QFTSTAGE":
        kernels, launches, family = ["mlxq_qft_stage_gen"], 1, "qft_stage"
    elif name == "_RXLAYER":
        kernels = ["mlxq_rx_pair"] + (["mlxq_rx_single"] if n % 2 else [])
        launches, family = n // 2 + n % 2, "uniform_rx_layer"
    elif name == "_U2LAYER":
        kernels = ["mlxq_u2_pair"] + (["mlxq_u2_single"] if n % 2 else [])
        launches, family = n // 2 + n % 2, "uniform_single_qubit_layer"
    elif name == "_U2LISTLAYER":
        active = set(op.get("active") or range(n))
        launches = sum(
            1 for q in range(0, n - 1, 2) if q in active or q + 1 in active
        )
        if n % 2 and n - 1 in active:
            launches += 1
        kernels = ["mlxq_u2_list_pair"]
        if n % 2 and n - 1 in active:
            kernels.append("mlxq_u2_single")
        family = "per_qubit_single_qubit_layer"
    elif name == "_XORLAYER":
        identity = _is_identity_rows(op.get("rows", []), n)
        kernels = ["mlxq_xor_flip" if identity else "mlxq_xor_affine_gather"]
        launches, family = 1, "xor_affine_permutation"
    elif name in {"_XXLAYER", "_YYLAYER"}:
        h_launches = _h_launches(n)
        kernels = ["mlxq_walsh4", "mlxq_u2_pair", "mlxq_u2_single",
                   "mlxq_zz_chain_layer"]
        kernels = [k for k in kernels if not (
            (k == "mlxq_walsh4" and n < 4)
            or (k == "mlxq_u2_pair" and n % 4 < 2)
            or (k == "mlxq_u2_single" and n % 2 == 0)
        )]
        launches = 2 * h_launches + 1
        if name == "_YYLAYER":
            kernels.append("mlxq_phase_popcount")
            launches += 2
        family = "xx_layer" if name == "_XXLAYER" else "yy_layer"
    else:
        return None
    return {
        "optimized_op": name,
        "kernel_family": family,
        "concrete_kernels": kernels,
        "expected_metal_launches": launches,
    }


def build_execution_plan(
    *,
    n_qubits: int,
    backend: str,
    operations: List[Dict[str, Any]],
    optimized_operations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    status = metal_runtime_status(n_qubits, backend=backend)
    cache = kernel_cache_snapshot()
    matched: List[Dict[str, Any]] = []
    for op in optimized_operations:
        spec = _custom_dispatch_spec(op, n_qubits)
        if spec is not None:
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

    return {
        "schema_version": 1,
        "backend": backend,
        "selected_device": status["mlx"]["default_device"],
        "dtype": STATE_DTYPE,
        "qubits": n_qubits,
        "input_operation_count": len(operations),
        "optimized_operation_count": len(optimized_operations),
        "optimization_passes": [
            {
                "name": "structured_runtime_fusion",
                "enabled": os.environ.get("MLXQ_DENSE_ONLY", "0") != "1",
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
        "memory": state_memory_estimate(n_qubits),
        "observed_custom_dispatches": [],
        "execution_status": "planned",
        "synchronized": False,
        "metal_memory_before": metal_memory_snapshot(),
    }


def record_custom_dispatch(plan: Optional[Dict[str, Any]], op: Dict[str, Any]) -> None:
    if plan is None or not plan.get("capabilities", {}).get("enabled", False):
        return
    spec = _custom_dispatch_spec(op, int(plan["qubits"]))
    if spec is not None:
        plan["observed_custom_dispatches"].append(spec)


def mark_graph_built(plan: Optional[Dict[str, Any]]) -> None:
    if plan is not None:
        plan["execution_status"] = "lazy_graph_built"
        plan["metal_memory_after_graph_build"] = metal_memory_snapshot()


def mark_synchronized(plan: Optional[Dict[str, Any]]) -> None:
    if plan is None:
        return
    plan["execution_status"] = "evaluated"
    plan["synchronized"] = True
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
    "STATE_DTYPE",
    "build_execution_plan",
    "clear_metal_capability_cache",
    "kernel_cache_snapshot",
    "mark_graph_built",
    "mark_synchronized",
    "metal_memory_snapshot",
    "metal_runtime_enabled",
    "metal_runtime_status",
    "record_custom_dispatch",
    "state_memory_estimate",
]
