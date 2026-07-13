import math

import mlx.core as mx
import numpy as np
import pytest

import mlxq.qasm as qasm
from mlxq.information import ptrace
from mlxq.qml import recipe_qpe_energy_single_qubit


QASMParseError = getattr(qasm, "QASMParseError", ValueError)


def _write_qasm(tmp_path, source: str):
    path = tmp_path / "circuit.qasm"
    path.write_text(source, encoding="utf-8")
    return path


def test_qasm_bound_parameter_arithmetic_is_evaluated(tmp_path):
    path = _write_qasm(
        tmp_path,
        """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
gate twist(theta, phi) a {
  rz(-(theta/2) + pi/8 + phi*2) a;
}
twist(pi/2, 0.125) q[0];
""",
    )
    n, operations = qasm.parse_qasm_file(str(path))
    assert n == 1
    assert operations == [
        {
            "name": "RZ",
            "wires": [0],
            "parameters": [pytest.approx(-math.pi / 8 + 0.25, abs=1e-12)],
        }
    ]


def test_qasm_rejects_unknown_or_unsupported_expressions_with_line_numbers(tmp_path):
    for expression in ("secret", "pi**2"):
        path = _write_qasm(
            tmp_path,
            f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rx({expression}) q[0];
""",
        )
        with pytest.raises(QASMParseError, match=r"circuit\.qasm:4:.*expression"):
            qasm.parse_qasm_file(str(path))


def test_qasm_rejects_reset_and_classical_control_instead_of_skipping(tmp_path):
    sources = (
        (
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
reset q[0];
""",
            4,
            "reset",
        ),
        (
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
if(c==1) x q[0];
""",
            5,
            "classical control",
        ),
    )
    for source, line, feature in sources:
        path = _write_qasm(tmp_path, source)
        with pytest.raises(
            QASMParseError, match=rf"circuit\.qasm:{line}:.*{feature}.*not supported"
        ):
            qasm.parse_qasm_file(str(path))


def test_qasm_allows_terminal_measurements_but_rejects_later_quantum_work(tmp_path):
    terminal = _write_qasm(
        tmp_path,
        """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
h q[0];
measure q[0] -> c[0];
""",
    )
    assert qasm.parse_qasm_file(str(terminal))[1] == [
        {"name": "H", "wires": [0], "parameters": []}
    ]

    nonterminal = _write_qasm(
        tmp_path,
        """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
h q[0];
measure q[0] -> c[0];
z q[0];
""",
    )
    with pytest.raises(QASMParseError, match=r"circuit\.qasm:7:.*terminal"):
        qasm.parse_qasm_file(str(nonterminal))


def test_qasm_rejects_unknown_includes_statements_and_gate_arity(tmp_path):
    cases = (
        (
            """OPENQASM 2.0;
include "unknown.inc";
qreg q[1];
""",
            2,
            "include",
        ),
        (
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
nonsense;
""",
            4,
            "statement",
        ),
        (
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
gate pair a,b {
  cx a,b;
}
pair q[0];
""",
            7,
            "expects 2 wires",
        ),
    )
    for source, line, message in cases:
        path = _write_qasm(tmp_path, source)
        with pytest.raises(QASMParseError, match=rf"circuit\.qasm:{line}:.*{message}"):
            qasm.parse_qasm_file(str(path))


def test_partial_trace_supports_multiple_arbitrary_subsystems():
    rng = np.random.default_rng(411)
    vector = rng.normal(size=12) + 1j * rng.normal(size=12)
    vector /= np.linalg.norm(vector)
    density = np.outer(vector, vector.conj())

    tensor = density.reshape(2, 3, 2, 2, 3, 2)
    reference = np.trace(np.trace(tensor, axis1=2, axis2=5), axis1=0, axis2=2)
    actual = ptrace(mx.array(density.astype(np.complex64)), traced_out=[0, 2], dims=[2, 3, 2])
    np.testing.assert_allclose(
        np.asarray(actual.tolist(), dtype=np.complex128), reference, atol=2e-6, rtol=0.0
    )


def test_unimplemented_qml_qpe_recipe_fails_explicitly():
    with pytest.raises(NotImplementedError, match="controlled-unitary QPE"):
        recipe_qpe_energy_single_qubit(0.0, 0.0, 0.0, 0.7, times=[0.0, 1.0])
