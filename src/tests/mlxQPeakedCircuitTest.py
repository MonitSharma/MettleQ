import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from mettleq.integrations.qiskit import MettleQBackend
from mettleq.peaked import (
    PUBLISHED_P9_EXPECTED_BITSTRING,
    PUBLISHED_P9_SHA256,
    build_mirrored_peaked_circuit,
    published_p9_manifest,
)


def test_published_p9_fixture_has_expected_integrity_and_shape():
    manifest = published_p9_manifest()
    assert manifest["sha256"] == PUBLISHED_P9_SHA256
    assert manifest["integrity_ok"] is True
    assert manifest["num_qubits"] == 56
    assert manifest["u_gates"] == 3890
    assert manifest["rzz_gates"] == 1917
    assert manifest["operation_count"] == 5807
    assert len(PUBLISHED_P9_EXPECTED_BITSTRING) == 56


@pytest.mark.parametrize("topology", ["linear", "grid", "long_range", "all_to_all"])
def test_mirrored_peaked_family_has_analytic_unique_mode(topology):
    circuit = build_mirrored_peaked_circuit(
        6, depth=2, topology=topology, seed=153, measure=False
    )
    peak = circuit.metadata["peak_bitstring"]
    expected_probability = circuit.metadata["expected_peak_probability"]
    probabilities = Statevector.from_instruction(circuit).probabilities_dict()
    assert max(probabilities, key=probabilities.get) == peak
    assert probabilities[peak] == pytest.approx(expected_probability, abs=2e-12)


def test_mettleq_mps_recovers_mirrored_peaked_mode_and_probability():
    circuit = build_mirrored_peaked_circuit(
        8, depth=3, topology="long_range", seed=153, measure=True
    )
    peak = circuit.metadata["peak_bitstring"]
    expected_probability = circuit.metadata["expected_peak_probability"]
    backend = MettleQBackend(
        method="matrix_product_state",
        device="cpu",
        mps_max_bond_dimension=32,
        mps_truncation_threshold=1e-12,
    )
    shots = 4096
    counts = backend.run(
        circuit, shots=shots, seed_simulator=153
    ).result().get_counts()
    observed_probability = counts.get(peak, 0) / shots
    assert max(counts, key=counts.get) == peak
    assert observed_probability == pytest.approx(expected_probability, abs=0.025)
    assert backend.last_mps_diagnostics[0]["state_norm"] == pytest.approx(
        1.0, abs=2e-5
    )
