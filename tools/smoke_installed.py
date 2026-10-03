"""Exercise an installed distribution from outside the checkout."""
from __future__ import annotations

import argparse
from importlib import metadata, resources
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metal", action="store_true")
    parser.add_argument("--require-native", action="store_true")
    args = parser.parse_args()

    import mettleq
    from mettleq.draw import circuit_ascii
    from mettleq.paths import qasm_local_path
    from mettleq.qasm import parse_qasm_file

    checkout = Path(__file__).resolve().parents[1]
    location = Path(mettleq.__file__).resolve()
    assert not location.is_relative_to(checkout / "src"), location
    assert mettleq.__version__ == metadata.version("mettleq")
    assert "q0" in circuit_ascii(1, [])
    parsed = parse_qasm_file(qasm_local_path("bell.qasm"))
    assert parsed is not None
    data = resources.files("mettleq")
    assert data.joinpath("py.typed").is_file()
    assert data.joinpath("integrations/pennylane_capabilities.toml").is_file()
    eps = metadata.entry_points(group="pennylane.plugins")
    assert {"mettleq", "mettleq.adaptive", "mettleq.compat"} <= {ep.name for ep in eps}
    subprocess.run([sys.executable, "-m", "mettleq.midpoint_mpo", "--help"],
                   check=True, capture_output=True)
    if args.require_native:
        from mettleq import _mps_native
        assert _mps_native.__file__
    if args.metal:
        import mlx.core as mx
        import numpy as np
        from qiskit import QuantumCircuit
        import pennylane as qml
        from mettleq.integrations.qiskit import MettleQBackend

        assert mx.metal.is_available(), "Apple Metal is required for release validation"
        with mx.stream(mx.gpu):
            probe = mx.array([1.0]) + 1
            mx.eval(probe)
            assert float(probe.item()) == 2.0
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        result = MettleQBackend(method="statevector").run(
            circuit, shots=100, seed_simulator=7).result()
        counts = result.get_counts()
        assert set(counts) == {"00", "11"}, counts
        dev = qml.device("mettleq", wires=2, method="mps")

        @qml.qnode(dev)
        def bell():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        np.testing.assert_allclose(bell(), [0.5, 0, 0, 0.5], atol=2e-6)
    print(json.dumps({"version": mettleq.__version__, "installed_at": str(location),
                      "metal_checked": args.metal, "native_required": args.require_native}))


if __name__ == "__main__":
    main()
