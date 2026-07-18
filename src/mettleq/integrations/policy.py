"""Circuit-aware SDK engine recommendations for Apple Silicon.

The policy is deliberately inspectable.  It never invents a speedup: every
decision includes the calibrated crossover, fusion estimate, topology, output
contract, precision, and dense-memory preflight that led to the recommendation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from ..execution import statevector_preflight


_SINGLE = {
    "H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "SX", "RX", "RY",
    "RZ", "U1", "U2", "U3",
}
_AFFINE = {"X", "CNOT", "CX", "SWAP"}
_DIAGONAL = {"Z", "S", "SDG", "T", "TDG", "RZ", "U1", "CZ", "CPHASE"}
_PAIR_PHASE = {"XXPHASE", "YYPHASE", "ZZPHASE"}
_CONTROLLED = {"CH", "CRX", "CRY", "CRZ"}
_SUPPORTED = _SINGLE | _AFFINE | _DIAGONAL | _PAIR_PHASE | _CONTROLLED | {
    "ISWAP", "TOFFOLI", "FREDKIN",
}


@dataclass(frozen=True)
class CircuitProfile:
    qubits: int
    operations: int
    single_qubit_operations: int
    two_qubit_operations: int
    nonlocal_two_qubit_operations: int
    maximum_two_qubit_span: int
    affine_operations: int
    diagonal_operations: int
    pair_phase_operations: int
    controlled_single_operations: int
    unsupported_operations: int
    estimated_native_operations: int
    estimated_fusion_coverage: float
    estimated_metal_launches: int
    estimated_operations_per_launch: float
    topology_class: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SDKEngineDecision:
    sdk: str
    engine: str
    reason: str
    precision: str
    output_contract: str
    calibrated_gpu_crossover_qubits: int
    adjusted_gpu_crossover_qubits: int
    statevector_preflight: dict
    profile: CircuitProfile
    fallback_available: bool

    def to_dict(self) -> dict:
        result = asdict(self)
        result["profile"] = self.profile.to_dict()
        return result


def profile_operations(n_qubits: int, operations: Sequence[dict]) -> CircuitProfile:
    """Summarize gate mix and topology without allocating a statevector."""
    n_qubits = int(n_qubits)
    names = [str(operation.get("name", "")).upper() for operation in operations]
    singles = sum(len(operation.get("wires", [])) == 1 for operation in operations)
    pairs = [
        tuple(int(wire) for wire in operation.get("wires", []))
        for operation in operations
        if len(operation.get("wires", [])) == 2
    ]
    spans = [abs(first - second) for first, second in pairs]
    nonlocal_pairs = sum(span != 1 for span in spans)
    native = sum(name in _SUPPORTED for name in names)
    operation_count = len(operations)
    coverage = native / operation_count if operation_count else 1.0
    # Conservative structural launch estimate. It intentionally counts only
    # patterns the runtime fuser can prove without changing gate order.
    launches = 0
    index = 0
    while index < operation_count:
        name = names[index]
        if name in _SINGLE:
            end = index + 1
            while end < operation_count and names[end] in _SINGLE:
                end += 1
            active_wires = len({int(op["wires"][0]) for op in operations[index:end]
                                if len(op.get("wires", [])) == 1})
            launches += max(1, (active_wires + 3) // 4)
        elif name in _AFFINE:
            end = index + 1
            while end < operation_count and names[end] in _AFFINE:
                end += 1
            launches += 1
        elif name in _CONTROLLED:
            end = index + 1
            while end < operation_count and names[end] in _CONTROLLED:
                end += 1
            launches += (end - index + 1) // 2
        elif name in _PAIR_PHASE:
            end = index + 1
            while end < operation_count and names[end] == name:
                end += 1
            launches += 1
        else:
            end = index + 1
            launches += 1
        index = end
    if not pairs:
        topology = "single_qubit_only"
    elif nonlocal_pairs == 0:
        topology = "nearest_neighbor"
    elif nonlocal_pairs <= max(1, len(pairs) // 5):
        topology = "mostly_local"
    else:
        topology = "long_range"
    return CircuitProfile(
        qubits=n_qubits,
        operations=operation_count,
        single_qubit_operations=int(singles),
        two_qubit_operations=len(pairs),
        nonlocal_two_qubit_operations=int(nonlocal_pairs),
        maximum_two_qubit_span=max(spans, default=0),
        affine_operations=sum(name in _AFFINE for name in names),
        diagonal_operations=sum(name in _DIAGONAL for name in names),
        pair_phase_operations=sum(name in _PAIR_PHASE for name in names),
        controlled_single_operations=sum(name in _CONTROLLED for name in names),
        unsupported_operations=sum(name not in _SUPPORTED for name in names),
        estimated_native_operations=native,
        estimated_fusion_coverage=float(coverage),
        estimated_metal_launches=int(launches),
        estimated_operations_per_launch=(
            float(operation_count / launches) if launches else 0.0
        ),
        topology_class=topology,
    )


def recommend_sdk_engine(
    sdk: str,
    n_qubits: int,
    operations: Sequence[dict],
    *,
    precision: str = "single",
    output_contract: str = "samples",
    method: str = "statevector",
    allow_approximation: bool = False,
    fallback_available: bool = True,
    allow_unsafe_statevector: Optional[bool] = None,
    qiskit_gpu_crossover_qubits: int = 20,
    pennylane_gpu_crossover_qubits: int = 16,
) -> SDKEngineDecision:
    """Recommend Aer/Lightning CPU or MettleQ using calibrated evidence.

    ``output_contract`` is one of ``statevector``, ``expectation``,
    ``probabilities``, or ``samples``. Double precision always selects the SDK
    CPU reference because MLX has no native GPU complex128 execution path.
    """
    sdk = str(sdk).strip().lower()
    if sdk not in {"qiskit", "pennylane"}:
        raise ValueError("sdk must be 'qiskit' or 'pennylane'")
    precision = str(precision).strip().lower()
    if precision not in {"single", "double"}:
        raise ValueError("precision must be 'single' or 'double'")
    output_contract = str(output_contract).strip().lower()
    if output_contract not in {
        "statevector", "expectation", "probabilities", "samples"
    }:
        raise ValueError("unsupported output contract")
    method = str(method).strip().lower()
    profile = profile_operations(n_qubits, operations)
    preflight = statevector_preflight(
        int(n_qubits), allow_unsafe=allow_unsafe_statevector
    )
    base = (
        int(qiskit_gpu_crossover_qubits)
        if sdk == "qiskit"
        else int(pennylane_gpu_crossover_qubits)
    )
    adjusted = base
    reasons = []
    if profile.estimated_fusion_coverage < 0.5:
        adjusted += 2
        reasons.append("low_native_fusion_coverage")
    elif profile.estimated_fusion_coverage >= 0.9 and profile.operations >= 2 * n_qubits:
        adjusted = max(1, adjusted - 1)
        reasons.append("high_native_fusion_coverage")
    if output_contract == "statevector":
        adjusted += 0
    elif output_contract in {"expectation", "probabilities"}:
        adjusted = max(1, adjusted - 1)
        reasons.append("gpu_native_reduction_contract")
    if profile.unsupported_operations:
        adjusted += 2
        reasons.append("unsupported_or_generic_operations_present")
    if profile.estimated_operations_per_launch < 1.5 and profile.operations:
        adjusted += 1
        reasons.append("high_expected_metal_launch_density")

    cpu_engine = (
        "qiskit_aer_cpu_statevector"
        if sdk == "qiskit"
        else "pennylane_lightning_cpu"
    )
    if method in {"matrix_product_state", "mps"}:
        engine = "qiskit_aer_cpu_mps" if sdk == "qiskit" else "mettleq_cpu_mps"
        reason = "MPS GPU crossover is not established; use the strongest CPU path"
    elif precision == "double":
        engine = cpu_engine if fallback_available else "mettleq_cpu_statevector"
        reason = "double precision requires a CPU complex128 path"
    elif not preflight["allowed"]:
        if allow_approximation and profile.topology_class in {
            "single_qubit_only", "nearest_neighbor", "mostly_local"
        }:
            engine = "qiskit_aer_cpu_mps" if sdk == "qiskit" else "mettleq_cpu_mps"
            reason = "dense preflight refused; bounded local MPS fallback was explicitly allowed"
        else:
            engine = "refused"
            reason = "dense preflight refused and no trustworthy approximate fallback was allowed"
    elif int(n_qubits) >= adjusted:
        engine = "mettleq_gpu_statevector"
        reason = "circuit reached the adjusted Apple-GPU crossover"
    else:
        engine = cpu_engine if fallback_available else "mettleq_cpu_statevector"
        reason = "circuit remains below the adjusted Apple-GPU crossover"
    if reasons:
        reason += "; " + ", ".join(reasons)
    return SDKEngineDecision(
        sdk=sdk,
        engine=engine,
        reason=reason,
        precision=precision,
        output_contract=output_contract,
        calibrated_gpu_crossover_qubits=base,
        adjusted_gpu_crossover_qubits=adjusted,
        statevector_preflight=preflight,
        profile=profile,
        fallback_available=bool(fallback_available),
    )
