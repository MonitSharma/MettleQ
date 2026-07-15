"""Native PennyLane device backed by Qupertino and Apple GPUs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Optional

import mlx.core as mx
import numpy as np

try:
    import pennylane as qml
    from pennylane.devices import Device as PennyLaneDevice
    from pennylane.devices import ExecutionConfig
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
    _apply_local_matrix,
    bit_counts,
    core_operation,
    execute_operations,
    local_expectation,
    marginal_probabilities,
    observable_samples,
    sample_bits,
    statevector_numpy,
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

_PAULI_MATRICES = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex64),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex64),
}

_ANALYTIC_MEASUREMENTS = (StateMP, ProbabilityMP, ExpectationMP, VarianceMP)
_SHOT_MEASUREMENTS = (ProbabilityMP, SampleMP, CountsMP, ExpectationMP, VarianceMP)


def _supports_operation(operation) -> bool:
    return operation.name in _PENNYLANE_TO_CORE or operation.name in {
        "Identity",
        "GlobalPhase",
    }


class QupertinoDevice(PennyLaneDevice):
    """PennyLane device using Qupertino's validated statevector engine."""

    def __init__(
        self,
        wires,
        shots=None,
        *,
        seed=None,
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
        self._execution_report = bool(execution_report)
        self._metal_checkpoint_budget_bytes = metal_checkpoint_budget_bytes
        self._allow_unsafe_statevector = allow_unsafe_statevector
        self.last_execution_plan = None
        self.statevector_preflight = None

    @property
    def name(self) -> str:
        return "qupertino"

    def preprocess_transforms(
        self, execution_config: Optional[ExecutionConfig] = None
    ) -> CompilePipeline:
        program = CompilePipeline()
        program.add_transform(convert_to_numpy_parameters)
        program.add_transform(
            decompose,
            stopping_condition=_supports_operation,
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
        single_circuit = not isinstance(circuits, Sequence)
        batch = [circuits] if single_circuit else list(circuits)
        results = tuple(self._execute_circuit(circuit) for circuit in batch)
        return results[0] if single_circuit else results

    def _execute_circuit(self, circuit):
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
        )
        if global_phase:
            device.sim.state = device.sim.state * complex(
                np.cos(global_phase), np.sin(global_phase)
            )
        self.last_execution_plan = device.last_execution_plan
        self.statevector_preflight = device.statevector_preflight

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
            acted = mx.zeros_like(device.sim.state)
            for word, coefficient in sentence.items():
                term = device.sim.state
                for wire, pauli in word.items():
                    term = _apply_local_matrix(
                        term,
                        len(self.wires),
                        [self.wires.index(wire)],
                        _PAULI_MATRICES[pauli],
                    )
                acted = acted + complex(coefficient) * term
            expectation = mx.sum(mx.conj(device.sim.state) * acted)
            second_moment = mx.sum(mx.conj(acted) * acted)
            mx.eval(expectation, second_moment)
            return complex(expectation.item()), complex(second_moment.item())

        wires = [self.wires.index(wire) for wire in observable.wires]
        if len(wires) > 8:
            raise DeviceError(
                "Non-Pauli observables spanning more than 8 wires would "
                "require an unsafe dense observable matrix"
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
            acted = device.sim.state
            for wire, pauli in word.items():
                acted = _apply_local_matrix(
                    acted,
                    len(self.wires),
                    [self.wires.index(wire)],
                    _PAULI_MATRICES[pauli],
                )
            expectation = mx.sum(mx.conj(device.sim.state) * acted)
            mx.eval(expectation)
            expectation = float(np.clip(expectation.item().real, -1.0, 1.0))
            signs = self._rng.choice(
                np.array([-1.0, 1.0]),
                size=int(shots),
                p=((1.0 - expectation) / 2.0, (1.0 + expectation) / 2.0),
            )
            return np.real_if_close(complex(coefficient) * signs)

        wires = [self.wires.index(wire) for wire in observable.wires]
        if len(wires) > 8:
            raise DeviceError(
                "Finite-shot sampling of a non-Pauli observable spanning more "
                "than 8 wires is not supported"
            )
        matrix = np.asarray(qml.matrix(observable, wire_order=observable.wires))
        return observable_samples(
            device, wires, matrix, int(shots), self._rng
        )

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
