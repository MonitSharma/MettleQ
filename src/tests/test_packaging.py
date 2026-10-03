"""Portable public API and distribution contract checks (no MLX required)."""
from importlib import metadata, resources
import hashlib
import subprocess
import sys

import pytest

from mettleq.draw import circuit_ascii
from mettleq.paths import qasm_local_path
from mettleq.qasm import QASMParseError, parse_qasm_file


def test_bundled_qasm_parses_without_source_datasets():
    assert parse_qasm_file(qasm_local_path("bell.qasm")) is not None
    data = resources.files("mettleq").joinpath("datasets/peaked_circuit_P9_Hqap_56x1917.qasm")
    assert hashlib.sha256(data.read_bytes()).hexdigest() == (
        "cff3496c45d9133c1f1693f1d3b0cf1fc2da338f13cd7b339db330a4762d0f35"
    )


def test_qasm_rejects_unsupported_dynamic_execution(tmp_path):
    source = tmp_path / "dynamic.qasm"
    source.write_text('OPENQASM 2.0;\nqreg q[1];\nreset q[0];')
    with pytest.raises(QASMParseError, match="reset"):
        parse_qasm_file(str(source))


def test_portable_import_does_not_initialize_mlx():
    # A fresh interpreter detects accidental eager MLX imports even on a Mac.
    subprocess.run([sys.executable, "-c", "import sys; import mettleq; "
                    "import mettleq.qasm; import mettleq.draw; import mlxq; "
                    "assert 'mlx.core' not in sys.modules"], check=True)


def test_public_entry_points_and_license_metadata():
    info = metadata.metadata("mettleq")
    assert info["License-Expression"] == "MIT AND Apache-2.0"
    assert {"mettleq", "mettleq.adaptive", "mettleq.compat"} <= {
        ep.name for ep in metadata.entry_points(group="pennylane.plugins")
    }
    assert "q0" in circuit_ascii(1, [])
