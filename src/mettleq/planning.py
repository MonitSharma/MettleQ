"""Inspectable execution-method and Apple device selection for SDK callers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import mlx.core as mx

from .execution import statevector_preflight


METHOD_ALIASES = {
    "auto": "automatic",
    "automatic": "automatic",
    "sv": "statevector",
    "statevector": "statevector",
    "mps": "matrix_product_state",
    "matrix_product_state": "matrix_product_state",
}
DEVICE_ALIASES = {
    "auto": "auto",
    "cpu": "cpu",
    "gpu": "gpu",
}

# Conservative defaults. Matched complete-call Qiskit/PennyLane measurements
# on the reference M3 Pro keep 14-qubit work on CPU and cross to GPU at 16
# qubits. The calibration benchmark can override this per backend/device
# instance without changing circuit code.
DEFAULT_STATEVECTOR_GPU_MIN_QUBITS = 16
# MLX 0.32 performs MPS SVD on CPU. On the reference M3 Pro, the explicit GPU
# tensor path did not beat the all-CPU path through 32 qubits, so automatic
# MPS remains on CPU unless a caller supplies a measured crossover.
DEFAULT_MPS_GPU_MIN_QUBITS: Optional[int] = None
DEFAULT_AUTOMATIC_MPS_MIN_QUBITS = 24


@dataclass(frozen=True)
class ExecutionSelection:
    requested_method: str
    selected_method: str
    requested_device: str
    selected_device: str
    allow_approximation: bool
    reason: str
    statevector_preflight: Optional[dict]
    mps_compatible: bool
    mps_compatibility_reasons: tuple[str, ...]
    cpu_gpu_policy: str = (
        "one numerical device is selected per execution; CPU and GPU times "
        "are benchmarked independently and are not combined into a speed claim"
    )
    mps_svd_device: Optional[str] = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["mps_compatibility_reasons"] = list(
            self.mps_compatibility_reasons
        )
        return result


def normalize_method(method: str) -> str:
    try:
        return METHOD_ALIASES[str(method).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "method must be one of: automatic, statevector, "
            "matrix_product_state"
        ) from exc


def normalize_device(device: str) -> str:
    try:
        return DEVICE_ALIASES[str(device).strip().lower()]
    except KeyError as exc:
        raise ValueError("device must be one of: auto, cpu, gpu") from exc


def _mps_compatibility(
    n_qubits: int,
    operations: Sequence[dict],
) -> tuple[bool, tuple[str, ...], int]:
    reasons = []
    entanglers = 0
    for operation in operations:
        wires = list(operation.get("wires", []))
        if len(wires) >= 3:
            reasons.append("contains_three_or_more_qubit_operation")
        elif len(wires) == 2:
            entanglers += 1
            if abs(int(wires[0]) - int(wires[1])) != 1:
                reasons.append("contains_non_adjacent_two_qubit_operation")
    # Automatic MPS selection is intentionally restricted to shallow local
    # circuits. Explicit MPS remains available for deeper/non-local circuits,
    # with truncation telemetry making approximation visible.
    shallow_limit = max(1, 4 * max(1, n_qubits - 1))
    if entanglers > shallow_limit:
        reasons.append("entangling_operation_count_exceeds_auto_mps_limit")
    unique_reasons = tuple(sorted(set(reasons)))
    return not unique_reasons, unique_reasons, entanglers


def _gpu_available() -> bool:
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


def select_execution(
    n_qubits: int,
    operations: Sequence[dict],
    *,
    method: str = "automatic",
    device: str = "auto",
    allow_approximation: bool = False,
    allow_unsafe_statevector: Optional[bool] = None,
    statevector_gpu_min_qubits: int = DEFAULT_STATEVECTOR_GPU_MIN_QUBITS,
    mps_gpu_min_qubits: Optional[int] = DEFAULT_MPS_GPU_MIN_QUBITS,
    automatic_mps_min_qubits: int = DEFAULT_AUTOMATIC_MPS_MIN_QUBITS,
) -> ExecutionSelection:
    """Select one method and one numerical device with an auditable reason."""
    n_qubits = int(n_qubits)
    requested_method = normalize_method(method)
    requested_device = normalize_device(device)
    if not isinstance(allow_approximation, bool):
        raise TypeError("allow_approximation must be a boolean")
    if statevector_gpu_min_qubits < 1 or (
        mps_gpu_min_qubits is not None and mps_gpu_min_qubits < 1
    ):
        raise ValueError("GPU crossover thresholds must be positive")

    mps_compatible, mps_reasons, _ = _mps_compatibility(
        n_qubits, operations
    )
    preflight = None
    if requested_method in ("automatic", "statevector"):
        preflight = statevector_preflight(
            n_qubits, allow_unsafe=allow_unsafe_statevector
        )

    if requested_method == "statevector":
        selected_method = "statevector"
        reason = "explicit_statevector_request"
    elif requested_method == "matrix_product_state":
        selected_method = "matrix_product_state"
        reason = "explicit_matrix_product_state_request"
    elif (
        allow_approximation
        and n_qubits >= int(automatic_mps_min_qubits)
        and mps_compatible
    ):
        selected_method = "matrix_product_state"
        reason = "automatic_shallow_local_mps_with_approximation_allowed"
    elif preflight is not None and preflight["allowed"]:
        selected_method = "statevector"
        reason = "automatic_exact_statevector_preflight_passed"
    elif allow_approximation and mps_compatible:
        selected_method = "matrix_product_state"
        reason = "automatic_statevector_refused_mps_fallback_allowed"
    else:
        # Preserve the exact request. StateVectorSimulator will raise the
        # detailed StatevectorMemoryError before allocation.
        selected_method = "statevector"
        reason = "automatic_exact_request_has_no_allowed_approximate_fallback"

    if requested_device != "auto":
        if requested_device == "gpu" and not _gpu_available():
            raise RuntimeError("Apple GPU execution was requested but Metal is unavailable")
        selected_device = requested_device
        device_reason = f"explicit_{requested_device}_request"
    elif not _gpu_available():
        selected_device = "cpu"
        device_reason = "automatic_gpu_unavailable"
    elif (
        selected_method == "matrix_product_state"
        and mps_gpu_min_qubits is None
    ):
        selected_device = "cpu"
        device_reason = "automatic_mps_gpu_crossover_not_observed"
    else:
        crossover = (
            statevector_gpu_min_qubits
            if selected_method == "statevector"
            else mps_gpu_min_qubits
        )
        if n_qubits < int(crossover):
            selected_device = "cpu"
            device_reason = (
                f"automatic_below_measured_{selected_method}_gpu_crossover"
            )
        else:
            selected_device = "gpu"
            device_reason = (
                f"automatic_at_or_above_measured_{selected_method}_gpu_crossover"
            )

    return ExecutionSelection(
        requested_method=requested_method,
        selected_method=selected_method,
        requested_device=requested_device,
        selected_device=selected_device,
        allow_approximation=allow_approximation,
        reason=f"{reason}; {device_reason}",
        statevector_preflight=preflight,
        mps_compatible=mps_compatible,
        mps_compatibility_reasons=mps_reasons,
        mps_svd_device=(
            "cpu" if selected_method == "matrix_product_state" else None
        ),
    )
