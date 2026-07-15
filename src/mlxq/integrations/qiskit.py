"""Qiskit ``BackendV2`` adapter for Qupertino's Apple GPU engine."""

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
        "The Qupertino Qiskit backend requires the optional Qiskit dependency. "
        "Install it with `python -m pip install -e '.[qiskit]'`."
    ) from exc

from ._common import (
    apply_global_phase,
    core_operation,
    execute_operations,
    pauli_product_expectation,
    sample_bits,
    statevector_numpy,
)
from .. import __version__
from ..planning import (
    DEFAULT_AUTOMATIC_MPS_MIN_QUBITS,
    DEFAULT_MPS_GPU_MIN_QUBITS,
    DEFAULT_STATEVECTOR_GPU_MIN_QUBITS,
    normalize_device,
    normalize_method,
)


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


class QupertinoBackend(BackendV2):
    """Run final-measurement Qiskit circuits on Qupertino and Apple GPUs.

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
        statevector_gpu_min_qubits: int = DEFAULT_STATEVECTOR_GPU_MIN_QUBITS,
        mps_gpu_min_qubits: Optional[int] = DEFAULT_MPS_GPU_MIN_QUBITS,
        automatic_mps_min_qubits: int = DEFAULT_AUTOMATIC_MPS_MIN_QUBITS,
        metal_checkpoint_budget_bytes: Optional[int] = None,
        allow_unsafe_statevector: Optional[bool] = None,
        **fields,
    ) -> None:
        super().__init__(
            provider=provider,
            name="qupertino",
            description=(
                "Qupertino Apple Silicon statevector and matrix-product-state "
                "simulator"
            ),
            backend_version=__version__,
            **fields,
        )
        self._target = target
        method = normalize_method(method)
        device = normalize_device(device)
        self.set_options(
            method=method,
            device=device,
            allow_approximation=bool(allow_approximation),
            mps_max_bond_dimension=int(mps_max_bond_dimension),
            mps_truncation_threshold=float(mps_truncation_threshold),
            statevector_gpu_min_qubits=int(statevector_gpu_min_qubits),
            mps_gpu_min_qubits=(
                None
                if mps_gpu_min_qubits is None
                else int(mps_gpu_min_qubits)
            ),
            automatic_mps_min_qubits=int(automatic_mps_min_qubits),
        )
        self._metal_checkpoint_budget_bytes = metal_checkpoint_budget_bytes
        self._allow_unsafe_statevector = allow_unsafe_statevector
        self.last_execution_plans = []
        self.last_statevector_preflights = []
        self.last_execution_selections = []
        self.last_mps_diagnostics = []

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
            statevector_gpu_min_qubits=DEFAULT_STATEVECTOR_GPU_MIN_QUBITS,
            mps_gpu_min_qubits=DEFAULT_MPS_GPU_MIN_QUBITS,
            automatic_mps_min_qubits=DEFAULT_AUTOMATIC_MPS_MIN_QUBITS,
        )

    @property
    def target(self) -> Target:
        if self._target is None:
            target = Target(
                description="Qupertino dynamic-width simulator target",
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
                "QupertinoBackend.run expects a QuantumCircuit or a non-empty "
                "sequence of QuantumCircuit objects"
            )

        unknown_options = set(run_options) - set(self.options)
        if unknown_options:
            names = ", ".join(sorted(unknown_options))
            raise QiskitError(f"Unsupported Qupertino run option(s): {names}")
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
            raise QiskitError("Qupertino requires at least one circuit qubit")
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
                    "Qupertino's Qiskit adapter currently supports final "
                    "measurements only"
                )
            if name == "id":
                continue
            if name not in _QISKIT_TO_CORE:
                raise QiskitError(
                    f"Unsupported instruction {name!r}; transpile the circuit "
                    "against QupertinoBackend before running it"
                )
            try:
                parameters = [float(value) for value in operation.params]
            except (TypeError, ValueError) as exc:
                raise QiskitError(
                    f"Instruction {name!r} contains unbound parameters"
                ) from exc
            if name in ("rxx", "ryy", "rzz"):
                # Qiskit defines exp(-i theta P⊗P / 2); Qupertino's native
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
                "Three-qubit operations must be transpiled into Qupertino's "
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
            statevector_gpu_min_qubits=int(
                options["statevector_gpu_min_qubits"]
            ),
            mps_gpu_min_qubits=(
                None
                if options["mps_gpu_min_qubits"] is None
                else int(options["mps_gpu_min_qubits"])
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
            data["qupertino_execution_plan"] = device.last_execution_plan
            data["qupertino_statevector_preflight"] = (
                device.statevector_preflight
            )
            data["qupertino_execution_selection"] = (
                device.execution_selection
            )
            data["qupertino_mps_diagnostics"] = mps_diagnostics

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


class QupertinoSamplerV2(BackendSamplerV2):
    """Qiskit SamplerV2 bound to a Qupertino backend by default.

    Qiskit's standard PUB batching and BitArray packing are retained, while
    the backend supplies MLX-device sampling without a probability-vector host
    readback.
    """

    def __init__(
        self,
        *,
        backend: Optional[QupertinoBackend] = None,
        options: Optional[dict] = None,
        **backend_options,
    ) -> None:
        if backend is not None and backend_options:
            raise ValueError(
                "backend_options cannot be combined with an existing backend"
            )
        backend = backend or QupertinoBackend(**backend_options)
        super().__init__(backend=backend, options=options)


class QupertinoEstimatorV2(BaseEstimatorV2):
    """Exact Qiskit EstimatorV2 using device-resident Pauli expectations."""

    def __init__(
        self,
        *,
        backend: Optional[QupertinoBackend] = None,
        default_precision: float = 0.0,
        **backend_options,
    ) -> None:
        if backend is not None and backend_options:
            raise ValueError(
                "backend_options cannot be combined with an existing backend"
            )
        if default_precision < 0.0:
            raise ValueError("default_precision must be non-negative")
        self._backend = backend or QupertinoBackend(**backend_options)
        self._default_precision = float(default_precision)
        self.last_execution_selections = []
        self.last_mps_diagnostics = []

    @property
    def backend(self) -> QupertinoBackend:
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
        return PrimitiveResult(
            [self._run_pub(pub) for pub in pubs],
            metadata={"version": 2, "backend": "qupertino"},
        )

    def _run_pub(self, pub):
        bound_circuits = pub.parameter_values.bind_all(pub.circuit)
        circuits, observables = np.broadcast_arrays(
            bound_circuits, pub.observables
        )
        evs = np.zeros(circuits.shape, dtype=np.float64)
        stds = np.zeros(circuits.shape, dtype=np.float64)
        options = {
            name: getattr(self._backend.options, name)
            for name in self._backend.options
        }
        options["execution_report"] = False
        selections = []
        diagnostics = []
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

            for index in indices:
                value = 0.0 + 0.0j
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
                    value += complex(
                        coefficient
                    ) * pauli_product_expectation(device, word)
                if abs(value.imag) > 5e-5 * max(1.0, abs(value.real)):
                    raise QiskitError("Estimator observable is not Hermitian")
                evs[index] = value.real

        data = DataBin(evs=evs, stds=stds, shape=evs.shape)
        self.last_execution_selections.extend(selections)
        self.last_mps_diagnostics.extend(diagnostics)
        return PubResult(
            data,
            metadata={
                "target_precision": pub.precision,
                "achieved_precision": 0.0,
                "circuit_metadata": pub.circuit.metadata,
                "qupertino_execution_selections": selections,
                "qupertino_mps_diagnostics": diagnostics,
            },
        )
