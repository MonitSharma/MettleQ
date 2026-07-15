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
    sample_bits,
    statevector_numpy,
)
from .. import __version__


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

_TARGET_GATES = tuple(_QISKIT_TO_CORE) + ("id", "measure")


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
        metal_checkpoint_budget_bytes: Optional[int] = None,
        allow_unsafe_statevector: Optional[bool] = None,
        **fields,
    ) -> None:
        super().__init__(
            provider=provider,
            name="qupertino",
            description="Qupertino Apple GPU statevector simulator",
            backend_version=__version__,
            **fields,
        )
        self._target = target
        self._metal_checkpoint_budget_bytes = metal_checkpoint_budget_bytes
        self._allow_unsafe_statevector = allow_unsafe_statevector
        self.last_execution_plans = []
        self.last_statevector_preflights = []

    @classmethod
    def _default_options(cls):
        return Options(
            shots=1024,
            memory=False,
            seed_simulator=None,
            return_statevector=False,
            execution_report=False,
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

        start = time.perf_counter()
        for circuit in circuits:
            experiment_results.append(
                self._run_circuit(circuit, shots=shots, rng=rng, options=options)
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

    def _run_circuit(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int,
        rng: np.random.Generator,
        options: dict,
    ) -> dict:
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

        device = execute_operations(
            circuit.num_qubits,
            operations,
            shots=shots,
            report=bool(options["execution_report"]),
            metal_checkpoint_budget_bytes=self._metal_checkpoint_budget_bytes,
            allow_unsafe_statevector=self._allow_unsafe_statevector,
        )
        try:
            global_phase = float(circuit.global_phase)
        except (TypeError, ValueError) as exc:
            raise QiskitError("Circuit global phase is unbound") from exc
        apply_global_phase(device, global_phase)
        self.last_execution_plans.append(device.last_execution_plan)
        self.last_statevector_preflights.append(device.statevector_preflight)

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
