"""Native PennyLane device backed by Qupertino and Apple GPUs."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from os import path
from typing import Optional

import numpy as np

try:
    import pennylane as qml
    from pennylane.devices import Device as PennyLaneDevice
    from pennylane.devices import ExecutionConfig
    from pennylane.devices.modifiers import simulator_tracking, single_tape_support
    from pennylane.devices.preprocess import (
        decompose,
        validate_device_wires,
        validate_measurements,
        validate_observables,
    )
    from pennylane.exceptions import DeviceError
    from pennylane.measurements import (
        CountsMP,
        ExpectationMP,
        ProbabilityMP,
        SampleMP,
        StateMP,
        VarianceMP,
    )
    from pennylane.transforms import broadcast_expand, convert_to_numpy_parameters
    from pennylane.transforms.core import CompilePipeline
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "The Qupertino PennyLane device requires the optional PennyLane "
        'dependency. Install it with `python -m pip install -e ".[pennylane]"`.'
    ) from exc

from ._common import (
    apply_global_phase,
    bit_counts,
    core_operation,
    execute_operations,
    local_expectation,
    marginal_probabilities,
    observable_samples,
    pauli_product_expectation,
    sample_bits,
    statevector_numpy,
)
from ..planning import (
    DEFAULT_AUTOMATIC_MPS_MIN_QUBITS,
    DEFAULT_MPS_GPU_MIN_QUBITS,
    DEFAULT_STATEVECTOR_GPU_MIN_QUBITS,
    normalize_device,
    normalize_method,
)


_PENNYLANE_TO_CORE = {
    "Hadamard": "H",
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "S": "S",
    "T": "T",
    "SX": "SX",
    "RX": "RX",
    "RY": "RY",
    "RZ": "RZ",
    "PhaseShift": "U1",
    "CNOT": "CNOT",
    "CH": "CH",
    "CZ": "CZ",
    "ControlledPhaseShift": "CPHASE",
    "CRX": "CRX",
    "CRY": "CRY",
    "CRZ": "CRZ",
    "SWAP": "SWAP",
    "ISWAP": "ISWAP",
    "Toffoli": "TOFFOLI",
    "CSWAP": "FREDKIN",
    "IsingXX": "XXPHASE",
    "IsingYY": "YYPHASE",
    "IsingZZ": "ZZPHASE",
}

_ANALYTIC_MEASUREMENTS = (StateMP, ProbabilityMP, ExpectationMP, VarianceMP)
_SHOT_MEASUREMENTS = (ProbabilityMP, SampleMP, CountsMP, ExpectationMP, VarianceMP)


@simulator_tracking
@single_tape_support
class QupertinoDevice(PennyLaneDevice):
    """PennyLane device using Qupertino's Apple-native simulation methods."""

    config_filepath = path.join(
        path.dirname(__file__), "pennylane_capabilities.toml"
    )

    def __init__(
        self,
        wires,
        shots=None,
        *,
        seed=None,
        method: str = "automatic",
        device: str = "auto",
        allow_approximation: bool = False,
        mps_max_bond_dimension: int = 64,
        mps_truncation_threshold: float = 1e-10,
        statevector_gpu_min_qubits: int = DEFAULT_STATEVECTOR_GPU_MIN_QUBITS,
        mps_gpu_min_qubits: Optional[int] = DEFAULT_MPS_GPU_MIN_QUBITS,
        automatic_mps_min_qubits: int = DEFAULT_AUTOMATIC_MPS_MIN_QUBITS,
        execution_report: bool = False,
        metal_checkpoint_budget_bytes: Optional[int] = None,
        allow_unsafe_statevector: Optional[bool] = None,
    ) -> None:
        if wires is None:
            raise DeviceError("QupertinoDevice requires an explicit wire set")
        super().__init__(wires=wires, shots=shots)
        if self.wires is None or len(self.wires) <= 0:
            raise DeviceError("QupertinoDevice requires at least one wire")
        self._rng = np.random.default_rng(seed)
        self._method = normalize_method(method)
        self._device = normalize_device(device)
        self._allow_approximation = bool(allow_approximation)
        self._mps_max_bond_dimension = int(mps_max_bond_dimension)
        self._mps_truncation_threshold = float(mps_truncation_threshold)
        self._statevector_gpu_min_qubits = int(statevector_gpu_min_qubits)
        self._mps_gpu_min_qubits = (
            None
            if mps_gpu_min_qubits is None
            else int(mps_gpu_min_qubits)
        )
        self._automatic_mps_min_qubits = int(automatic_mps_min_qubits)
        self._execution_report = bool(execution_report)
        self._metal_checkpoint_budget_bytes = metal_checkpoint_budget_bytes
        self._allow_unsafe_statevector = allow_unsafe_statevector
        self.last_execution_plan = None
        self.statevector_preflight = None
        self.last_execution_selection = None
        self.last_mps_diagnostics = None

    @property
    def name(self) -> str:
        return "qupertino"

    def setup_execution_config(
        self,
        config: Optional[ExecutionConfig] = None,
        circuit=None,
    ) -> ExecutionConfig:
        config = ExecutionConfig() if config is None else config
        options = {
            "method": self._method,
            "device": self._device,
            "allow_approximation": self._allow_approximation,
            "mps_max_bond_dimension": self._mps_max_bond_dimension,
            "mps_truncation_threshold": self._mps_truncation_threshold,
            "statevector_gpu_min_qubits": self._statevector_gpu_min_qubits,
            "mps_gpu_min_qubits": self._mps_gpu_min_qubits,
            "automatic_mps_min_qubits": self._automatic_mps_min_qubits,
            **config.device_options,
        }
        return replace(
            config,
            device_options=options,
            convert_to_numpy=True,
            use_device_gradient=False,
            use_device_jacobian_product=False,
            grad_on_execution=False,
        )

    def _supports_operation(self, operation, *, method=None) -> bool:
        method = self._method if method is None else method
        if (
            method == "matrix_product_state"
            and operation.name in {"Toffoli", "CSWAP"}
        ):
            return False
        return operation.name in _PENNYLANE_TO_CORE or operation.name in {
            "Identity",
            "GlobalPhase",
        }

    def preprocess_transforms(
        self, execution_config: Optional[ExecutionConfig] = None
    ) -> CompilePipeline:
        config = self.setup_execution_config(execution_config)
        method = normalize_method(config.device_options["method"])
        program = CompilePipeline()
        program.add_transform(convert_to_numpy_parameters)
        program.add_transform(
            decompose,
            stopping_condition=lambda operation: self._supports_operation(
                operation, method=method
            ),
            device_wires=self.wires,
            name=self.name,
        )
        program.add_transform(broadcast_expand)
        program.add_transform(validate_device_wires, self.wires, name=self.name)
        program.add_transform(
            validate_measurements,
            analytic_measurements=lambda measurement: isinstance(
                measurement, _ANALYTIC_MEASUREMENTS
            ),
            sample_measurements=lambda measurement: isinstance(
                measurement, _SHOT_MEASUREMENTS
            ),
            name=self.name,
        )
        program.add_transform(
            validate_observables,
            stopping_condition=lambda observable: bool(observable.has_matrix),
            name=self.name,
        )
        return program

    def execute(self, circuits, execution_config=None):
        config = self.setup_execution_config(execution_config)
        execution_cache = {}
        return tuple(
            self._execute_circuit(
                circuit,
                config.device_options,
                execution_cache=execution_cache,
            )
            for circuit in circuits
        )

    def _execute_circuit(
        self, circuit, execution_options, *, execution_cache=None
    ):
        operations = []
        global_phase = 0.0
        for operation in circuit.operations:
            if operation.name == "Identity":
                continue
            if operation.name == "GlobalPhase":
                # PennyLane's GlobalPhase(phi) is exp(-i phi) I.
                global_phase -= self._parameter(operation.parameters[0], operation.name)
                continue
            if operation.name not in _PENNYLANE_TO_CORE:
                raise DeviceError(
                    f"Operation {operation.name!r} reached Qupertino without "
                    "a supported decomposition"
                )
            parameters = [
                self._parameter(value, operation.name)
                for value in operation.parameters
            ]
            if operation.name in ("IsingXX", "IsingYY", "IsingZZ"):
                # PennyLane uses exp(-i theta P⊗P / 2).
                parameters[0] /= 2.0
            wires = [self.wires.index(wire) for wire in operation.wires]
            try:
                operations.append(
                    core_operation(
                        _PENNYLANE_TO_CORE[operation.name], wires, parameters
                    )
                )
            except ValueError as exc:
                raise DeviceError(str(exc)) from exc

        effective_shots = circuit.shots
        if not effective_shots and self.shots:
            effective_shots = self.shots
        default_shots = effective_shots.total_shots or 1000
        device = execute_operations(
            len(self.wires),
            operations,
            shots=default_shots,
            report=self._execution_report,
            metal_checkpoint_budget_bytes=self._metal_checkpoint_budget_bytes,
            allow_unsafe_statevector=self._allow_unsafe_statevector,
            method=execution_options["method"],
            execution_device=execution_options["device"],
            allow_approximation=bool(
                execution_options["allow_approximation"]
            ),
            mps_max_bond_dimension=int(
                execution_options["mps_max_bond_dimension"]
            ),
            mps_truncation_threshold=float(
                execution_options["mps_truncation_threshold"]
            ),
            statevector_gpu_min_qubits=int(
                execution_options["statevector_gpu_min_qubits"]
            ),
            mps_gpu_min_qubits=(
                None
                if execution_options["mps_gpu_min_qubits"] is None
                else int(execution_options["mps_gpu_min_qubits"])
            ),
            automatic_mps_min_qubits=int(
                execution_options["automatic_mps_min_qubits"]
            ),
            execution_cache=execution_cache,
        )
        apply_global_phase(device, global_phase)
        self.last_execution_plan = device.last_execution_plan
        self.statevector_preflight = device.statevector_preflight
        self.last_execution_selection = device.execution_selection
        self.last_mps_diagnostics = (
            device.sim.truncation_diagnostics()
            if device.backend == "mps"
            else None
        )

        if effective_shots.has_partitioned_shots:
            return tuple(
                self._measure_all(device, circuit.measurements, int(shots))
                for shots in effective_shots
            )
        shots = effective_shots.total_shots
        return self._measure_all(device, circuit.measurements, shots)

    @staticmethod
    def _parameter(value, operation_name: str) -> float:
        array = np.asarray(value)
        if array.ndim != 0:
            raise DeviceError(
                f"Operation {operation_name!r} still has broadcast parameters"
            )
        try:
            return float(array)
        except (TypeError, ValueError) as exc:
            raise DeviceError(
                f"Operation {operation_name!r} has a non-numeric parameter"
            ) from exc

    def _measure_all(self, device, measurements, shots):
        # Every returned measurement is an evaluation boundary.  Synchronize
        # once so execution-plan evidence reflects work that actually ran.
        device.synchronize()
        values = tuple(
            self._measure(device, measurement, shots)
            for measurement in measurements
        )
        return values[0] if len(values) == 1 else values

    def _measure(self, device, measurement, shots):
        if isinstance(measurement, StateMP):
            if shots is not None:
                raise DeviceError("State measurements require analytic execution")
            return statevector_numpy(device)

        if measurement.obs is None:
            wires = (
                [self.wires.index(wire) for wire in measurement.wires]
                if len(measurement.wires)
                else list(range(len(self.wires)))
            )
            if isinstance(measurement, ProbabilityMP):
                if shots is None:
                    return marginal_probabilities(device, wires)
                samples = sample_bits(device, shots, wires, self._rng)
                dimension = 1 << len(wires)
                indices = samples.dot(
                    1 << np.arange(len(wires) - 1, -1, -1)
                )
                return np.bincount(indices, minlength=dimension) / float(shots)
            if isinstance(measurement, SampleMP):
                if shots is None:
                    raise DeviceError("Sample measurements require finite shots")
                return sample_bits(device, shots, wires, self._rng)
            if isinstance(measurement, CountsMP):
                if shots is None:
                    raise DeviceError("Counts measurements require finite shots")
                samples = sample_bits(device, shots, wires, self._rng)
                counts = bit_counts(samples)
                if measurement.all_outcomes:
                    for value in range(1 << len(wires)):
                        counts.setdefault(format(value, f"0{len(wires)}b"), 0)
                return counts

        if isinstance(measurement, (ExpectationMP, VarianceMP)):
            if shots is None:
                expectation, second_moment = self._observable_moments(
                    device, measurement.obs
                )
                if isinstance(measurement, ExpectationMP):
                    return self._real_scalar(expectation, "expectation")
                variance = second_moment - expectation * expectation
                return self._real_scalar(variance, "variance")
            samples = self._sample_observable(device, measurement.obs, shots)
            if isinstance(measurement, ExpectationMP):
                return self._real_scalar(np.mean(samples), "expectation")
            return self._real_scalar(np.var(samples), "variance")

        if isinstance(measurement, (SampleMP, CountsMP)):
            if shots is None:
                raise DeviceError("Observable sampling requires finite shots")
            samples = self._sample_observable(device, measurement.obs, shots)
            if isinstance(measurement, SampleMP):
                return samples
            counts = dict(Counter(np.real_if_close(value).item() for value in samples))
            if measurement.all_outcomes:
                for value in np.unique(np.real_if_close(measurement.eigvals())):
                    counts.setdefault(value.item(), 0)
            return counts

        raise DeviceError(
            f"Measurement {type(measurement).__name__!r} is not supported"
        )

    def _observable_moments(self, device, observable):
        sentence = self._pauli_sentence(observable)
        if sentence is not None:
            expectation = self._sentence_expectation(device, sentence)
            second_moment = self._sentence_expectation(
                device, sentence @ sentence
            )
            return expectation, second_moment

        wires = [self.wires.index(wire) for wire in observable.wires]
        dense_limit = 4 if device.backend == "mps" else 8
        if len(wires) > dense_limit:
            raise DeviceError(
                f"Non-Pauli observables spanning more than {dense_limit} "
                "wires would require an unsafe dense observable matrix"
            )
        matrix = np.asarray(qml.matrix(observable, wire_order=observable.wires))
        expectation = local_expectation(device, wires, matrix)
        second_moment = local_expectation(device, wires, matrix @ matrix)
        return expectation, second_moment

    def _sample_observable(self, device, observable, shots):
        sentence = self._pauli_sentence(observable)
        if sentence is not None and len(sentence) == 1:
            word, coefficient = next(iter(sentence.items()))
            if not word:
                return np.full(int(shots), np.real_if_close(coefficient))
            expectation = pauli_product_expectation(
                device,
                {
                    self.wires.index(wire): pauli
                    for wire, pauli in word.items()
                },
            )
            expectation = float(np.clip(expectation.real, -1.0, 1.0))
            signs = self._rng.choice(
                np.array([-1.0, 1.0]),
                size=int(shots),
                p=((1.0 - expectation) / 2.0, (1.0 + expectation) / 2.0),
            )
            return np.real_if_close(complex(coefficient) * signs)

        wires = [self.wires.index(wire) for wire in observable.wires]
        dense_limit = 4 if device.backend == "mps" else 8
        if len(wires) > dense_limit:
            raise DeviceError(
                "Finite-shot sampling of a non-Pauli observable spanning more "
                f"than {dense_limit} wires is not supported"
            )
        matrix = np.asarray(qml.matrix(observable, wire_order=observable.wires))
        return observable_samples(
            device, wires, matrix, int(shots), self._rng
        )

    def _sentence_expectation(self, device, sentence) -> complex:
        value = 0.0 + 0.0j
        for word, coefficient in sentence.items():
            value += complex(coefficient) * pauli_product_expectation(
                device,
                {
                    self.wires.index(wire): pauli
                    for wire, pauli in word.items()
                },
            )
        return value

    @staticmethod
    def _pauli_sentence(observable):
        try:
            return qml.pauli.pauli_sentence(observable)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _real_scalar(value, label: str) -> float:
        value = complex(value)
        tolerance = 5e-5 * max(1.0, abs(value.real))
        if abs(value.imag) > tolerance:
            raise DeviceError(
                f"Hermitian {label} developed an imaginary component "
                f"{value.imag:.3e}, above the {tolerance:.3e} tolerance"
            )
        return float(value.real)
