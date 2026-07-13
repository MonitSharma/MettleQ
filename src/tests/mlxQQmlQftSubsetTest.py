import math

import mlx.core as mx
import numpy as np

from mlxq.qml import (
    Device,
    H,
    IQFT,
    QFT,
    RX,
    RY,
    RZ,
    probs,
    qnode,
    recipe_qft,
    sample,
    state,
)
from mlxq.sim import StateVectorSimulator, iqft, qft


def _numpy_state(state_array) -> np.ndarray:
    mx.eval(state_array)
    return np.asarray(state_array.tolist(), dtype=np.complex128)


def _random_state(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    return vector / np.linalg.norm(vector)


def _bit_reverse(value: int, width: int) -> int:
    result = 0
    for bit in range(width):
        result |= ((value >> bit) & 1) << (width - 1 - bit)
    return result


def _qft_gate_without_final_swaps(n: int) -> np.ndarray:
    size = 1 << n
    indices = np.arange(size)
    fourier = np.exp(2j * np.pi * np.outer(indices, indices) / size) / math.sqrt(size)
    bit_reversed_rows = [_bit_reverse(index, n) for index in range(size)]
    return fourier[bit_reversed_rows, :]


def _apply_reference_qft(
    vector: np.ndarray,
    n: int,
    wires: tuple[int, ...],
    *,
    inverse: bool = False,
) -> np.ndarray:
    gate = _qft_gate_without_final_swaps(len(wires))
    if inverse:
        gate = gate.conj().T
    remaining = [wire for wire in range(n) if wire not in wires]
    permutation = remaining + list(wires)
    tensor = vector.reshape([2] * n).transpose(permutation)
    matrix = tensor.reshape(1 << (n - len(wires)), 1 << len(wires))
    updated = matrix @ gate.T
    updated_tensor = updated.reshape([2] * n)
    return updated_tensor.transpose(np.argsort(permutation)).reshape(1 << n)


def test_recipe_qft_three_qubits_executes():
    _, circuit = recipe_qft(3)
    actual = _numpy_state(circuit())
    expected = np.full(8, 1 / math.sqrt(8), dtype=np.complex128)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0.0)


def test_qft_matches_independent_full_and_ordered_subset_reference():
    n = 4
    initial = _random_state(n, seed=208)
    for wires in ((0, 1, 2, 3), (2, 0), (3, 1, 0)):
        sim = StateVectorSimulator(n)
        sim.state = mx.array(initial.astype(np.complex64))
        qft(sim, list(wires))
        expected = _apply_reference_qft(initial, n, wires)
        np.testing.assert_allclose(_numpy_state(sim.state), expected, atol=2e-5, rtol=0.0)


def test_qft_then_iqft_recovers_random_states_on_ordered_subsets():
    n = 4
    for seed, wires in ((13, (0, 1, 2, 3)), (71, (2, 0)), (911, (3, 1, 0))):
        initial = _random_state(n, seed)
        sim = StateVectorSimulator(n)
        sim.state = mx.array(initial.astype(np.complex64))
        qft(sim, list(wires))
        iqft(sim, list(wires))
        np.testing.assert_allclose(_numpy_state(sim.state), initial, atol=2e-5, rtol=0.0)


def test_qml_subset_qft_and_iqft_match_the_ordered_reference():
    dev = Device(wires=3)

    def prepare():
        RX(0.37, wires=0)
        RY(-0.52, wires=1)
        RZ(0.91, wires=2)
        H(2)

    @qnode(dev)
    def baseline():
        prepare()
        return state()

    @qnode(dev)
    def forward():
        prepare()
        QFT((2, 0))
        return state()

    @qnode(dev)
    def inverse():
        prepare()
        IQFT((2, 0))
        return state()

    initial = _numpy_state(baseline())
    expected_forward = _apply_reference_qft(initial, 3, (2, 0))
    expected_inverse = _apply_reference_qft(initial, 3, (2, 0), inverse=True)
    np.testing.assert_allclose(_numpy_state(forward()), expected_forward, atol=2e-5, rtol=0.0)
    np.testing.assert_allclose(_numpy_state(inverse()), expected_inverse, atol=2e-5, rtol=0.0)


def test_qml_subset_probabilities_and_samples_preserve_wire_order_and_shape():
    dev = Device(wires=3)

    @qnode(dev)
    def probability_circuit():
        RX(math.pi, wires=0)
        H(1)
        return probs(wires=(2, 0))

    @qnode(dev)
    def sample_circuit():
        RX(math.pi, wires=0)
        H(1)
        return sample(wires=(2, 0), shots=32)

    np.testing.assert_allclose(
        probability_circuit(), [0.0, 1.0, 0.0, 0.0], atol=2e-6, rtol=0.0
    )
    samples = np.asarray(sample_circuit())
    assert samples.shape == (32, 2)
    assert np.all(samples == np.asarray([0, 1]))
