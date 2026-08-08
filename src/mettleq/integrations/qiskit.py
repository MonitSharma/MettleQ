"""Qiskit ``BackendV2`` adapter for MettleQ's Apple GPU engine."""

from __future__ import annotations

import time
import uuid
from collections import Counter
from collections.abc import Sequence
from typing import Optional

import numpy as np

try:
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library.standard_gates import (
        get_standard_gate_name_mapping,
    )
    from qiskit.exceptions import QiskitError
    from qiskit.primitives import (
        BackendSamplerV2,
        BaseEstimatorV2,
        PrimitiveJob,
    )
    from qiskit.primitives.containers import (
        DataBin,
        EstimatorPub,
        PrimitiveResult,
        PubResult,
    )
    from qiskit.providers import BackendV2, JobStatus, JobV1, Options
    from qiskit.result import Result
    from qiskit.transpiler import Target
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "The MettleQ Qiskit backend requires the optional Qiskit dependency. "
        "Install it with `python -m pip install -e '.[qiskit]'`."
    ) from exc

from ._common import (
    apply_global_phase,
    core_operation,
    execute_operations,
    pauli_product_expectation,
    pauli_product_expectations,
    sample_bits,
    statevector_numpy,
)
from ..mps_accuracy import build_convergence_report
from .. import __version__
from ..planning import (
    DEFAULT_AUTOMATIC_MPS_MIN_QUBITS,
    DEFAULT_STATEVECTOR_GPU_MIN_QUBITS,
    normalize_device,
    normalize_method,
)
from .policy import recommend_sdk_engine


_QISKIT_TO_CORE = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sdg": "SDG",
    "t": "T",
    "tdg": "TDG",
    "sx": "SX",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "p": "U1",
    "u1": "U1",
    "u2": "U2",
    "u": "U3",
    "u3": "U3",
    "cx": "CNOT",
    "ch": "CH",
    "cz": "CZ",
    "cp": "CPHASE",
    "crx": "CRX",
    "cry": "CRY",
    "crz": "CRZ",
    "swap": "SWAP",
    "iswap": "ISWAP",
    "ccx": "TOFFOLI",
    "cswap": "FREDKIN",
    "rxx": "XXPHASE",
    "ryy": "YYPHASE",
    "rzz": "ZZPHASE",
}

_TARGET_GATES = tuple(
    name for name in _QISKIT_TO_CORE if name not in ("ccx", "cswap")
) + ("id", "measure")


class _CompletedJob(JobV1):
    """Synchronous Qiskit job returned after MLX execution completes."""

    def __init__(self, backend: BackendV2, job_id: str, result: Result):
        super().__init__(backend, job_id)
        self._result = result

    def submit(self):
        return None

    def result(self, timeout=None):
        return self._result

    def status(self):
        return JobStatus.DONE


class AdaptiveQiskitBackend(BackendV2):
    """Delegate each Qiskit circuit to Aer CPU or MettleQ Apple GPU.

    The selection is based on the translated gate stream, requested precision,
    output contract, topology, fusion coverage, and dense-memory preflight.
    Aer remains optional; if it is unavailable the policy falls back to the
    MettleQ CPU path below crossover rather than failing package import.
    """

    version = 2

    def __init__(
        self,
        provider=None,
        *,
        precision: str = "single",
        output_contract: str = "samples",
        qiskit_gpu_crossover_qubits: int = 20,
        mettleq_options: Optional[dict] = None,
        **fields,
    ) -> None:
        super().__init__(
            provider=provider,
            name="mettleq_adaptive",
            description="Circuit-aware Aer CPU / MettleQ Apple-GPU dispatcher",
            backend_version=__version__,
            **fields,
        )
        precision = str(precision).strip().lower()
        if precision not in {"single", "double"}:
            raise ValueError("precision must be 'single' or 'double'")
        self.precision = precision
        self.output_contract = str(output_contract).strip().lower()
        self.qiskit_gpu_crossover_qubits = int(qiskit_gpu_crossover_qubits)
        self._mettleq = MettleQBackend(**dict(mettleq_options or {}))
        self._target = self._mettleq.target
        self.last_adaptive_decisions = []

    @classmethod
    def _default_options(cls):
        return Options(
            shots=1024,
            memory=False,
            seed_simulator=None,
            return_statevector=False,
            execution_report=False,
            precision=None,
            output_contract=None,
        )

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @staticmethod
    def _aer_available() -> bool:
        try:
            import qiskit_aer  # noqa: F401
            return True
        except ImportError:
            return False

    def _decision(self, circuit, options, *, fallback_available):
        operations, _, _ = self._mettleq._translate_circuit(circuit)
        contract = options.get("output_contract") or (
            "statevector" if options.get("return_statevector")
            else self.output_contract
        )
        precision = options.get("precision") or self.precision
        return recommend_sdk_engine(
            "qiskit",
            circuit.num_qubits,
            operations,
            precision=precision,
            output_contract=contract,
            method="statevector",
            fallback_available=fallback_available,
            qiskit_gpu_crossover_qubits=self.qiskit_gpu_crossover_qubits,
        )

    @staticmethod
    def _result_from_jobs(backend, jobs):
        results = []
        success = True
        total_time = 0.0
        for job in jobs:
            payload = job.result().to_dict()
            results.extend(payload.get("results", []))
            success = success and bool(payload.get("success", True))
            total_time += float(payload.get("time_taken") or 0.0)
        result = Result.from_dict(
            {
                "backend_name": backend.name,
                "backend_version": backend.backend_version,
                "qobj_id": None,
                "job_id": str(uuid.uuid4()),
                "success": success,
                "results": results,
                "status": "COMPLETED" if success else "ERROR",
                "time_taken": total_time,
            }
        )
        return _CompletedJob(backend, str(uuid.uuid4()), result)

    def _run_aer(self, circuits, decision, options):
        from qiskit_aer import AerSimulator

        method = (
            "matrix_product_state"
            if decision.engine == "qiskit_aer_cpu_mps"
            else "statevector"
        )
        backend = AerSimulator(
            method=method,
            device="CPU",
            precision="double" if decision.precision == "double" else "single",
            enable_truncation=False,
        )
        candidates = []
        for circuit in circuits:
            candidate = circuit.copy()
            if options.get("return_statevector"):
                candidate.save_statevector()
            candidates.append(candidate)
        kwargs = {
            "shots": int(options["shots"]),
            "memory": bool(options["memory"]),
        }
        if options.get("seed_simulator") is not None:
            kwargs["seed_simulator"] = options["seed_simulator"]
        return backend.run(candidates, **kwargs)

    def run(self, run_input, **run_options):
        circuits = (
            [run_input]
            if isinstance(run_input, QuantumCircuit)
            else list(run_input)
            if isinstance(run_input, Sequence)
            else None
        )
        if not circuits or not all(isinstance(c, QuantumCircuit) for c in circuits):
            raise QiskitError(
                "AdaptiveQiskitBackend.run expects a circuit or non-empty circuit sequence"
            )
        unknown = set(run_options) - set(self.options)
        if unknown:
            raise QiskitError(
                "Unsupported adaptive option(s): " + ", ".join(sorted(unknown))
            )
        options = {name: getattr(self.options, name) for name in self.options}
        options.update(run_options)
        fallback_available = self._aer_available()
        decisions = [
            self._decision(circuit, options, fallback_available=fallback_available)
            for circuit in circuits
        ]
        self.last_adaptive_decisions = [decision.to_dict() for decision in decisions]
        groups = {}
        for index, decision in enumerate(decisions):
            if decision.engine == "refused":
                raise QiskitError(decision.reason)
            groups.setdefault(decision.engine, []).append(index)
        experiment_slots = [None] * len(circuits)
        success = True
        elapsed = 0.0
        for engine, indices in groups.items():
            selected = [circuits[index] for index in indices]
            decision = decisions[indices[0]]
            if engine.startswith("qiskit_aer_cpu"):
                job = self._run_aer(selected, decision, options)
            else:
                mettleq_options = {
                    "shots": options["shots"],
                    "memory": options["memory"],
                    "seed_simulator": options["seed_simulator"],
                    "return_statevector": options["return_statevector"],
                    "execution_report": options["execution_report"],
                    "method": "statevector",
                    "device": "gpu" if "gpu" in decision.engine else "cpu",
                }
                job = self._mettleq.run(selected, **mettleq_options)
            payload = job.result().to_dict()
            success = success and bool(payload.get("success", True))
            elapsed += float(payload.get("time_taken") or 0.0)
            for index, experiment in zip(indices, payload.get("results", [])):
                experiment_slots[index] = experiment
        job_id = str(uuid.uuid4())
        result = Result.from_dict({
            "backend_name": self.name,
            "backend_version": self.backend_version,
            "qobj_id": None,
            "job_id": job_id,
            "success": success,
            "results": experiment_slots,
            "status": "COMPLETED" if success else "ERROR",
            "time_taken": elapsed,
        })
        return _CompletedJob(self, job_id, result)


class MettleQBackend(BackendV2):
    """Run final-measurement Qiskit circuits on MettleQ and Apple GPUs.

    The target deliberately has ``num_qubits=None``.  Qiskit's transpiler then
    preserves each input circuit's width instead of padding it to a simulator
    maximum that would allocate an unnecessarily large statevector.
    """

    version = 2

    def __init__(
        self,
        provider=None,
        target: Optional[Target] = None,
        *,
        method: str = "automatic",
        device: str = "auto",
        allow_approximation: bool = False,
        mps_max_bond_dimension: int = 64,
        mps_truncation_threshold: float = 1e-10,
        mps_svd_driver: str = "auto",
        mps_routing_strategy: str = "lookahead",
        mps_routing_lookahead: int = 8,
        mps_accuracy_policy: str = "report",
        mps_max_relative_discarded_weight: Optional[float] = 1e-6,
        mps_max_norm_error: Optional[float] = 1e-5,
        mps_convergence_bond_dimensions: Optional[Sequence[int]] = None,
        mps_convergence_atol: float = 5e-5,
        statevector_gpu_min_qubits: int = DEFAULT_STATEVECTOR_GPU_MIN_QUBITS,
        automatic_mps_min_qubits: int = DEFAULT_AUTOMATIC_MPS_MIN_QUBITS,
        metal_checkpoint_budget_bytes: Optional[int] = None,
        allow_unsafe_statevector: Optional[bool] = None,
        **fields,
    ) -> None:
        super().__init__(
            provider=provider,
            name="mettleq",
            description=(
                "MettleQ Apple Silicon statevector and matrix-product-state "
                "simulator"
            ),
            backend_version=__version__,
            **fields,
        )
        self._target = target
        method = normalize_method(method)
        device = normalize_device(device)
        convergence_dimensions = tuple(
            sorted(
                {
                    int(dimension)
                    for dimension in (mps_convergence_bond_dimensions or ())
                }
            )
        )
        if convergence_dimensions and (
            len(convergence_dimensions) < 2
            or convergence_dimensions[0] < 1
        ):
            raise ValueError(
                "MPS convergence needs at least two positive bond dimensions"
            )
        if mps_convergence_atol <= 0.0:
            raise ValueError("MPS convergence tolerance must be positive")
        self.set_options(
            method=method,
            device=device,
            allow_approximation=bool(allow_approximation),
            mps_max_bond_dimension=int(mps_max_bond_dimension),
            mps_truncation_threshold=float(mps_truncation_threshold),
            mps_svd_driver=str(mps_svd_driver),
            mps_routing_strategy=str(mps_routing_strategy),
            mps_routing_lookahead=int(mps_routing_lookahead),
            mps_accuracy_policy=str(mps_accuracy_policy),
            mps_max_relative_discarded_weight=(
                None
                if mps_max_relative_discarded_weight is None
                else float(mps_max_relative_discarded_weight)
            ),
            mps_max_norm_error=(
                None
                if mps_max_norm_error is None
                else float(mps_max_norm_error)
            ),
            mps_convergence_bond_dimensions=convergence_dimensions,
            mps_convergence_atol=float(mps_convergence_atol),
            statevector_gpu_min_qubits=int(statevector_gpu_min_qubits),
            automatic_mps_min_qubits=int(automatic_mps_min_qubits),
        )
        self._metal_checkpoint_budget_bytes = metal_checkpoint_budget_bytes
        self._allow_unsafe_statevector = allow_unsafe_statevector
        self.last_execution_plans = []
        self.last_statevector_preflights = []
        self.last_execution_selections = []
        self.last_mps_diagnostics = []
        self.last_mps_accuracy_reports = []
        self.last_mps_convergence_reports = []

    @classmethod
    def _default_options(cls):
        return Options(
            shots=1024,
            memory=False,
            seed_simulator=None,
            return_statevector=False,
            execution_report=False,
            method="automatic",
            device="auto",
            allow_approximation=False,
            mps_max_bond_dimension=64,
            mps_truncation_threshold=1e-10,
            mps_svd_driver="auto",
            mps_routing_strategy="lookahead",
            mps_routing_lookahead=8,
            mps_accuracy_policy="report",
            mps_max_relative_discarded_weight=1e-6,
            mps_max_norm_error=1e-5,
            mps_convergence_bond_dimensions=(),
            mps_convergence_atol=5e-5,
            statevector_gpu_min_qubits=DEFAULT_STATEVECTOR_GPU_MIN_QUBITS,
            automatic_mps_min_qubits=DEFAULT_AUTOMATIC_MPS_MIN_QUBITS,
        )

    @property
    def target(self) -> Target:
        if self._target is None:
            target = Target(
                description="MettleQ dynamic-width simulator target",
                num_qubits=None,
            )
            standard_gates = get_standard_gate_name_mapping()
            for name in _TARGET_GATES:
                target.add_instruction(
                    standard_gates[name], properties=None, name=name
                )
            self._target = target
        return self._target

    @property
    def max_circuits(self):
        return None

    def run(self, run_input, **run_options) -> _CompletedJob:
        circuits = (
            [run_input]
            if isinstance(run_input, QuantumCircuit)
            else list(run_input)
            if isinstance(run_input, Sequence)
            else None
        )
        if circuits is None or not circuits or not all(
            isinstance(circuit, QuantumCircuit) for circuit in circuits
        ):
            raise QiskitError(
                "MettleQBackend.run expects a QuantumCircuit or a non-empty "
                "sequence of QuantumCircuit objects"
            )

        unknown_options = set(run_options) - set(self.options)
        if unknown_options:
            names = ", ".join(sorted(unknown_options))
            raise QiskitError(f"Unsupported MettleQ run option(s): {names}")
        options = {name: getattr(self.options, name) for name in self.options}
        options.update(run_options)
        try:
            options["method"] = normalize_method(options["method"])
            options["device"] = normalize_device(options["device"])
        except ValueError as exc:
            raise QiskitError(str(exc)) from exc
        shots = options["shots"]
        if shots is None or int(shots) <= 0:
            raise QiskitError("shots must be a positive integer")
        shots = int(shots)
        seed = options["seed_simulator"]
        rng = np.random.default_rng(seed)
        job_id = str(uuid.uuid4())
        experiment_results = []
        self.last_execution_plans = []
        self.last_statevector_preflights = []
        self.last_execution_selections = []
        self.last_mps_diagnostics = []
        self.last_mps_accuracy_reports = []
        self.last_mps_convergence_reports = []
        execution_cache = {}

        start = time.perf_counter()
        for circuit in circuits:
            experiment_results.append(
                self._run_circuit(
                    circuit,
                    shots=shots,
                    rng=rng,
                    options=options,
                    execution_cache=execution_cache,
                )
            )
        elapsed = time.perf_counter() - start
        result = Result.from_dict(
            {
                "backend_name": self.name,
                "backend_version": self.backend_version,
                "qobj_id": None,
                "job_id": job_id,
                "success": True,
                "results": experiment_results,
                "status": "COMPLETED",
                "time_taken": elapsed,
            }
        )
        return _CompletedJob(self, job_id, result)

    @staticmethod
    def _translate_circuit(circuit: QuantumCircuit):
        """Translate a bound circuit and preserve final measurement mapping."""
        if circuit.num_qubits <= 0:
            raise QiskitError("MettleQ requires at least one circuit qubit")
        operations = []
        measured_clbits: dict[int, int] = {}
        saw_measurement = False

        for instruction in circuit.data:
            operation = instruction.operation
            name = operation.name
            if getattr(operation, "condition", None) is not None:
                raise QiskitError(
                    "Classically conditioned operations are not supported yet"
                )
            if name == "barrier":
                continue
            if name == "measure":
                saw_measurement = True
                qubit = circuit.find_bit(instruction.qubits[0]).index
                clbit = circuit.find_bit(instruction.clbits[0]).index
                measured_clbits[clbit] = qubit
                continue
            if saw_measurement:
                raise QiskitError(
                    "MettleQ's Qiskit adapter currently supports final "
                    "measurements only"
                )
            if name == "id":
                continue
            if name not in _QISKIT_TO_CORE:
                raise QiskitError(
                    f"Unsupported instruction {name!r}; transpile the circuit "
                    "against MettleQBackend before running it"
                )
            try:
                parameters = [float(value) for value in operation.params]
            except (TypeError, ValueError) as exc:
                raise QiskitError(
                    f"Instruction {name!r} contains unbound parameters"
                ) from exc
            if name in ("rxx", "ryy", "rzz"):
                # Qiskit defines exp(-i theta P⊗P / 2); MettleQ's native
                # pair-phase primitive defines exp(-i theta P⊗P).
                parameters[0] /= 2.0
            qiskit_wires = [
                circuit.find_bit(qubit).index for qubit in instruction.qubits
            ]
            core_wires = [
                circuit.num_qubits - 1 - qubit for qubit in qiskit_wires
            ]
            try:
                operations.append(
                    core_operation(
                        _QISKIT_TO_CORE[name], core_wires, parameters
                    )
                )
            except ValueError as exc:
                raise QiskitError(str(exc)) from exc
        try:
            global_phase = float(circuit.global_phase)
        except (TypeError, ValueError) as exc:
            raise QiskitError("Circuit global phase is unbound") from exc
        return operations, measured_clbits, global_phase

    def _execute_bound_circuit(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int,
        options: dict,
        execution_cache: Optional[dict] = None,
    ):
        operations, measured_clbits, global_phase = self._translate_circuit(
            circuit
        )
        if (
            options["method"] == "matrix_product_state"
            and any(len(operation["wires"]) >= 3 for operation in operations)
        ):
            raise QiskitError(
                "Three-qubit operations must be transpiled into MettleQ's "
                "one- and two-qubit target before MPS execution"
            )
        device = execute_operations(
            circuit.num_qubits,
            operations,
            shots=shots,
            report=bool(options["execution_report"]),
            metal_checkpoint_budget_bytes=self._metal_checkpoint_budget_bytes,
            allow_unsafe_statevector=self._allow_unsafe_statevector,
            method=options["method"],
            execution_device=options["device"],
            allow_approximation=bool(options["allow_approximation"]),
            mps_max_bond_dimension=int(options["mps_max_bond_dimension"]),
            mps_truncation_threshold=float(
                options["mps_truncation_threshold"]
            ),
            mps_svd_driver=str(options["mps_svd_driver"]),
            mps_routing_strategy=str(options["mps_routing_strategy"]),
            mps_routing_lookahead=int(options["mps_routing_lookahead"]),
            mps_accuracy_policy=str(options["mps_accuracy_policy"]),
            mps_max_relative_discarded_weight=(
                None
                if options["mps_max_relative_discarded_weight"] is None
                else float(options["mps_max_relative_discarded_weight"])
            ),
            mps_max_norm_error=(
                None
                if options["mps_max_norm_error"] is None
                else float(options["mps_max_norm_error"])
            ),
            statevector_gpu_min_qubits=int(
                options["statevector_gpu_min_qubits"]
            ),
            automatic_mps_min_qubits=int(
                options["automatic_mps_min_qubits"]
            ),
            execution_cache=execution_cache,
        )
        apply_global_phase(device, global_phase)
        return device, measured_clbits, global_phase

    def _run_circuit(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int,
        rng: np.random.Generator,
        options: dict,
        execution_cache: Optional[dict] = None,
    ) -> dict:
        device, measured_clbits, global_phase = self._execute_bound_circuit(
            circuit,
            shots=shots,
            options=options,
            execution_cache=execution_cache,
        )
        self.last_execution_plans.append(device.last_execution_plan)
        self.last_statevector_preflights.append(device.statevector_preflight)
        self.last_execution_selections.append(device.execution_selection)
        mps_diagnostics = (
            device.sim.truncation_diagnostics()
            if device.backend == "mps"
            else None
        )
        self.last_mps_diagnostics.append(mps_diagnostics)
        mps_accuracy = getattr(device, "mps_accuracy_report", None)
        self.last_mps_accuracy_reports.append(mps_accuracy)

        data = {}
        memory = []
        if measured_clbits:
            clbits = sorted(measured_clbits)
            internal_wires = [
                circuit.num_qubits - 1 - measured_clbits[clbit]
                for clbit in clbits
            ]
            samples = sample_bits(device, shots, internal_wires, rng)
            for row in samples:
                value = 0
                for bit, clbit in zip(row, clbits):
                    value |= int(bit) << clbit
                memory.append(hex(value))
            data["counts"] = dict(Counter(memory))
        else:
            data["counts"] = {}
        if options["memory"]:
            data["memory"] = memory
        if options["return_statevector"]:
            data["statevector"] = statevector_numpy(device)
        if options["execution_report"]:
            data["mettleq_execution_plan"] = device.last_execution_plan
            data["mettleq_statevector_preflight"] = (
                device.statevector_preflight
            )
            data["mettleq_execution_selection"] = (
                device.execution_selection
            )
            data["mettleq_mps_diagnostics"] = mps_diagnostics
            data["mettleq_mps_accuracy"] = mps_accuracy

        header = {
            "name": circuit.name,
            "n_qubits": circuit.num_qubits,
            "qreg_sizes": [
                [register.name, register.size] for register in circuit.qregs
            ],
            "creg_sizes": [
                [register.name, register.size] for register in circuit.cregs
            ],
            "qubit_labels": [
                [register.name, index]
                for register in circuit.qregs
                for index in range(register.size)
            ],
            "clbit_labels": [
                [register.name, index]
                for register in circuit.cregs
                for index in range(register.size)
            ],
            "memory_slots": circuit.num_clbits,
            "global_phase": global_phase,
            "metadata": circuit.metadata or {},
        }
        return {
            "name": circuit.name,
            "seed_simulator": options["seed_simulator"],
            "shots": shots,
            "data": data,
            "status": "DONE",
            "success": True,
            "header": header,
        }


class MettleQMidpointMPOBackend(BackendV2):
    """First-class Qiskit backend for midpoint-MPO/TNO plus unswapping.

    The pinned worker remains a numerical isolation detail; callers use the
    ordinary ``BackendV2.run(...).result().get_counts()`` contract.
    """

    version = 2

    def __init__(
        self,
        provider=None,
        *,
        mpo_options=None,
        worker_python=None,
        timeout_seconds=None,
        simulator=None,
        **fields,
    ) -> None:
        super().__init__(
            provider=provider,
            name="mettleq_midpoint_mpo",
            description="MettleQ midpoint-MPO/TNO and greedy-unswapping backend",
            backend_version=__version__,
            **fields,
        )
        from ..midpoint_mpo import IsolatedMidpointMPOSimulator, MidpointMPOOptions

        if mpo_options is None:
            mpo_options = MidpointMPOOptions()
        elif isinstance(mpo_options, dict):
            mpo_options = MidpointMPOOptions(**mpo_options)
        self._simulator = simulator or IsolatedMidpointMPOSimulator(
            mpo_options,
            worker_python=worker_python,
            timeout_seconds=timeout_seconds,
        )
        self._target = MettleQBackend().target
        self.last_mpo_results = []

    @classmethod
    def _default_options(cls):
        return Options(shots=1024, memory=False, expected_bitstring=None)

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @staticmethod
    def _unitary_and_measurements(circuit):
        measured = {}
        saw_measurement = False
        for instruction in circuit.data:
            name = instruction.operation.name
            if name == "measure":
                saw_measurement = True
                qubit = circuit.find_bit(instruction.qubits[0]).index
                clbit = circuit.find_bit(instruction.clbits[0]).index
                measured[clbit] = qubit
            elif saw_measurement and name != "barrier":
                raise QiskitError("midpoint-MPO supports final measurements only")
        return circuit.remove_final_measurements(inplace=False), measured

    def run(self, run_input, **run_options):
        circuits = ([run_input] if isinstance(run_input, QuantumCircuit)
                    else list(run_input) if isinstance(run_input, Sequence) else None)
        if not circuits or not all(isinstance(c, QuantumCircuit) for c in circuits):
            raise QiskitError("MettleQMidpointMPOBackend.run expects Qiskit circuits")
        unknown = set(run_options) - set(self.options)
        if unknown:
            raise QiskitError("Unsupported midpoint-MPO option(s): "
                              + ", ".join(sorted(unknown)))
        options = {name: getattr(self.options, name) for name in self.options}
        options.update(run_options)
        shots = int(options["shots"])
        if shots < 1:
            raise QiskitError("shots must be positive")

        started = time.perf_counter()
        experiments = []
        self.last_mpo_results = []
        for circuit in circuits:
            unitary, measured = self._unitary_and_measurements(circuit)
            mpo_result = self._simulator.run(
                unitary,
                shots=shots,
                expected_bitstring=options["expected_bitstring"],
            )
            self.last_mpo_results.append(mpo_result)
            memory = []
            counts = Counter()
            if measured:
                for sample in mpo_result.samples:
                    value = 0
                    for clbit, qubit in measured.items():
                        value |= int(sample[qubit]) << clbit
                    encoded = hex(value)
                    counts[encoded] += 1
                    memory.append(encoded)
            data = {
                "counts": dict(counts),
                "mettleq_midpoint_mpo": mpo_result.to_summary(include_counts=False),
            }
            if options["memory"]:
                data["memory"] = memory
            experiments.append({
                "name": circuit.name,
                "shots": shots,
                "data": data,
                "status": "DONE",
                "success": True,
                "header": {
                    "name": circuit.name,
                    "n_qubits": circuit.num_qubits,
                    "memory_slots": circuit.num_clbits,
                    "metadata": circuit.metadata or {},
                },
            })
        job_id = str(uuid.uuid4())
        result = Result.from_dict({
            "backend_name": self.name,
            "backend_version": self.backend_version,
            "qobj_id": None,
            "job_id": job_id,
            "success": True,
            "results": experiments,
            "status": "COMPLETED",
            "time_taken": time.perf_counter() - started,
        })
        return _CompletedJob(self, job_id, result)


class MettleQSamplerV2(BackendSamplerV2):
    """Qiskit SamplerV2 bound to a MettleQ backend by default.

    Qiskit's standard PUB batching and BitArray packing are retained, while
    the backend supplies MLX-device sampling without a probability-vector host
    readback.
    """

    def __init__(
        self,
        *,
        backend: Optional[MettleQBackend] = None,
        options: Optional[dict] = None,
        **backend_options,
    ) -> None:
        if backend is not None and backend_options:
            raise ValueError(
                "backend_options cannot be combined with an existing backend"
            )
        backend = backend or MettleQBackend(**backend_options)
        super().__init__(backend=backend, options=options)


class MettleQEstimatorV2(BaseEstimatorV2):
    """Qiskit EstimatorV2 with exact statevector or bounded MPS execution."""

    def __init__(
        self,
        *,
        backend: Optional[MettleQBackend] = None,
        default_precision: float = 0.0,
        **backend_options,
    ) -> None:
        if backend is not None and backend_options:
            raise ValueError(
                "backend_options cannot be combined with an existing backend"
            )
        if default_precision < 0.0:
            raise ValueError("default_precision must be non-negative")
        self._backend = backend or MettleQBackend(**backend_options)
        self._default_precision = float(default_precision)
        self.last_execution_selections = []
        self.last_mps_diagnostics = []
        self.last_mps_accuracy_reports = []
        self.last_mps_convergence_reports = []

    @property
    def backend(self) -> MettleQBackend:
        return self._backend

    def run(self, pubs, *, precision: Optional[float] = None):
        precision = self._default_precision if precision is None else precision
        if precision < 0.0:
            raise ValueError("precision must be non-negative")
        coerced = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        job = PrimitiveJob(self._run, coerced)
        job._submit()
        return job

    def _run(self, pubs):
        self.last_execution_selections = []
        self.last_mps_diagnostics = []
        self.last_mps_accuracy_reports = []
        self.last_mps_convergence_reports = []
        return PrimitiveResult(
            [self._run_pub(pub) for pub in pubs],
            metadata={"version": 2, "backend": "mettleq"},
        )

    def _evaluate_bound_pub(self, circuits, observables, options):
        evs = np.zeros(circuits.shape, dtype=np.float64)
        selections = []
        diagnostics = []
        accuracy_reports = []
        # Group broadcast observables by bound circuit. This executes each
        # parameter point once while retaining at most one simulator state at
        # a time, rather than keeping one exponential state per parameter set.
        circuit_groups = {}
        for index in np.ndindex(*circuits.shape):
            circuit = circuits[index]
            key = id(circuit)
            circuit_groups.setdefault(key, (circuit, []))[1].append(index)

        execution_cache = {}
        for circuit, indices in circuit_groups.values():
            device, measurements, _ = self._backend._execute_bound_circuit(
                circuit,
                shots=1,
                options=options,
                execution_cache=execution_cache,
            )
            if measurements:
                raise QiskitError(
                    "Estimator circuits must not contain measurements"
                )
            selections.append(device.execution_selection)
            diagnostics.append(
                device.sim.truncation_diagnostics()
                if device.backend == "mps"
                else None
            )
            accuracy_reports.append(
                getattr(device, "mps_accuracy_report", None)
            )

            requests = []
            for index in indices:
                observable = observables[index]
                for pauli, coefficient in observable.items():
                    label = (
                        pauli.to_label()
                        if hasattr(pauli, "to_label")
                        else str(pauli)
                    )
                    if len(label) != circuit.num_qubits:
                        raise QiskitError(
                            "Observable width does not match circuit width"
                        )
                    word = {
                        internal_wire: symbol
                        for internal_wire, symbol in enumerate(label)
                        if symbol != "I"
                    }
                    requests.append((index, complex(coefficient), word))
            reduced = pauli_product_expectations(
                device, [word for _, _, word in requests]
            )
            accumulated = {index: 0.0 + 0.0j for index in indices}
            for (index, coefficient, _), term in zip(requests, reduced):
                accumulated[index] += coefficient * term
            for index in indices:
                value = accumulated[index]
                if abs(value.imag) > 5e-5 * max(1.0, abs(value.real)):
                    raise QiskitError("Estimator observable is not Hermitian")
                evs[index] = value.real
        return evs, selections, diagnostics, accuracy_reports

    def _run_pub(self, pub):
        bound_circuits = pub.parameter_values.bind_all(pub.circuit)
        circuits, observables = np.broadcast_arrays(
            bound_circuits, pub.observables
        )
        options = {
            name: getattr(self._backend.options, name)
            for name in self._backend.options
        }
        options["execution_report"] = False
        evs, selections, diagnostics, accuracy_reports = (
            self._evaluate_bound_pub(circuits, observables, options)
        )
        stds = np.zeros(circuits.shape, dtype=np.float64)

        convergence = None
        dimensions = tuple(options["mps_convergence_bond_dimensions"])
        selected_mps = any(
            selection["selected_method"] == "matrix_product_state"
            for selection in selections
        )
        if dimensions and selected_mps:
            base_dmax = int(options["mps_max_bond_dimension"])
            dimensions = tuple(sorted(set(dimensions + (base_dmax,))))
            runs = []
            for dmax in dimensions:
                if dmax == base_dmax:
                    run_evs = evs
                    run_diagnostics = diagnostics
                    run_accuracy = accuracy_reports
                else:
                    convergence_options = dict(options)
                    convergence_options["mps_max_bond_dimension"] = dmax
                    convergence_options["mps_accuracy_policy"] = "report"
                    (
                        run_evs,
                        _,
                        run_diagnostics,
                        run_accuracy,
                    ) = self._evaluate_bound_pub(
                        circuits, observables, convergence_options
                    )
                runs.append(
                    {
                        "dmax": dmax,
                        "value": run_evs,
                        "diagnostics": run_diagnostics,
                        "accuracy": run_accuracy,
                    }
                )
            convergence = build_convergence_report(
                runs, atol=float(options["mps_convergence_atol"])
            )

        data = DataBin(evs=evs, stds=stds, shape=evs.shape)
        self.last_execution_selections.extend(selections)
        self.last_mps_diagnostics.extend(diagnostics)
        self.last_mps_accuracy_reports.extend(accuracy_reports)
        self.last_mps_convergence_reports.append(convergence)
        return PubResult(
            data,
            metadata={
                "target_precision": pub.precision,
                "achieved_precision": 0.0,
                "circuit_metadata": pub.circuit.metadata,
                "mettleq_execution_selections": selections,
                "mettleq_mps_diagnostics": diagnostics,
                "mettleq_mps_accuracy": accuracy_reports,
                "mettleq_mps_convergence": convergence,
            },
        )


# Source-compatible aliases for applications written before the MettleQ 0.2
# rename. New code should use the MettleQ names above.
QupertinoBackend = MettleQBackend
QupertinoSamplerV2 = MettleQSamplerV2
QupertinoEstimatorV2 = MettleQEstimatorV2
