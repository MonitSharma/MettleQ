<div align="center">
  <h1>MettleQ</h1>
  <p><strong>Local quantum-circuit simulation for Apple Silicon — with Qiskit and PennyLane adapters.</strong></p>
  <p>
    <img src="https://img.shields.io/badge/platform-Apple%20Silicon-111111" alt="Apple Silicon"/>
    <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB" alt="Python 3.11+"/>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2EA44F" alt="MIT license"/></a>
  </p>
</div>

MettleQ is a local quantum-circuit simulation engine for Apple Silicon. It
provides exact statevector and bounded matrix-product-state (MPS) methods,
MLX and hand-written Metal execution, statevector memory preflight, and
inspectable correctness evidence — exposed through a native Qiskit `BackendV2`
and a registered PennyLane device so you can drop it into an existing workflow.

> MettleQ is alpha research software. Validate important results against an
> independent backend and keep the execution report.

## Install

Requirements: Apple Silicon, macOS, Python 3.11+.

```bash
# from PyPI (once published)
python -m pip install 'mettleq[sdk]'

# or from source
python -m pip install 'mettleq[sdk] @ git+https://github.com/MonitSharma/MettleQ.git'
```

The `sdk` extra pulls in Qiskit and PennyLane. For local development, install
editable from a clone:

```bash
git clone https://github.com/MonitSharma/MettleQ.git
cd MettleQ
python -m pip install -e '.[sdk,tests]'
```

## Quickstart

### Qiskit

```python
from qiskit import QuantumCircuit
from mettleq.integrations.qiskit import MettleQBackend

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

backend = MettleQBackend(method="statevector")   # or method="mps"
result = backend.run(qc, shots=1000).result()
print(result.get_counts())                        # {'00': ~500, '11': ~500}
```

`AdaptiveQiskitBackend` additionally auto-selects between Aer/Lightning CPU and
the MettleQ Apple-GPU path based on circuit shape, precision, and memory.

### PennyLane

```python
import pennylane as qml

dev = qml.device("mettleq", wires=4, method="mps")   # registered entry point

@qml.qnode(dev)
def ghz():
    qml.Hadamard(0)
    for i in range(3):
        qml.CNOT(wires=[i, i + 1])
    return qml.probs(wires=range(4))

print(ghz())    # peaks at |0000> and |1111>
```

Use `qml.device("mettleq.adaptive", ...)` for the auto-selecting device.

## Simulation methods

| Method | Device | Notes |
| --- | --- | --- |
| Exact statevector (`statevector`) | Apple GPU / CPU | Hand-written Metal kernels for structured layers; memory-preflighted |
| Bounded MPS (`mps`) | CPU | Approximate with explicit `Dmax`, truncation, and convergence evidence |
| Midpoint-MPO | CPU (isolated env) | Opt-in Qiskit method for peaked/low-entanglement circuits |

CPU and GPU are independent alternatives — MettleQ never sums their timings.

## Development

```bash
python -m pip install -e '.[sdk,tests]'
python -m pytest src/tests -q
```

## Research, benchmarks, and full history

This `main` branch is the clean, installable package. The full research
monorepo — reproducible benchmarks, frozen evidence, comparison plots, tutorial
notebooks, the MettleQ Studio desktop app, the technical report, and the
Windows/WSL baselines — lives on the [`development`](https://github.com/MonitSharma/MettleQ/tree/development)
branch.

## Attribution and license

MettleQ is an independently maintained, MIT-licensed fork of
[BoltzmannEntropy/Qupertino](https://github.com/BoltzmannEntropy/Qupertino) by
Shlomo Kashani. The midpoint-MPO solver is derived from
[alexgalda-m/peaked-mpo-solver](https://github.com/alexgalda-m/peaked-mpo-solver)
(Apache-2.0). Original authorship, licensing, and citation details are retained
in [`CITATION.cff`](CITATION.cff) and the vendored source notices.

Released under the [MIT License](LICENSE).
