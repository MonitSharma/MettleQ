# MettleQ

MettleQ is a local quantum-circuit simulation engine for Apple Silicon. It
provides Qiskit and PennyLane adapters, exact statevector and bounded MPS
methods, MLX/custom Metal execution, memory preflight, and inspectable
correctness evidence.

> MettleQ is alpha research software. Validate important results against an
> independent backend and retain the execution report.

## Install

Requirements: Apple Silicon, macOS 14 or newer, and Python 3.11 or newer.

The MLX simulation kernels support macOS arm64 (Apple Silicon). Linux,
Windows, and Intel Macs can install the package for MLX-independent drawing,
QASM parsing, and documentation tooling, but simulation requires an Apple
Silicon Mac.

```bash
python -m pip install 'mettleq[sdk]'
```

The `sdk` extra installs Qiskit Aer and PennyLane Lightning so adaptive mode can
use strong CPU fallbacks below the calibrated Apple-GPU crossover.

For the optional midpoint-MPO command, install its dependencies explicitly:

```bash
python -m pip install 'mettleq[mpo]'
```

The command runs in the current environment. The isolated worker API also
accepts `METTLEQ_MPO_PYTHON` when a separately pinned worker is required.

Until the first PyPI release is published, install from the public repository:

```bash
python -m pip install \
  'mettleq[sdk] @ git+https://github.com/MonitSharma/MettleQ.git'
```

## Qiskit

```python
from qiskit import QuantumCircuit, transpile
from mettleq.integrations.qiskit import AdaptiveQiskitBackend

backend = AdaptiveQiskitBackend(precision="single")
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

compiled = transpile(circuit, backend)
result = backend.run(compiled, shots=1000, seed_simulator=7).result()
print(result.get_counts())
```

## PennyLane

```python
import pennylane as qml

device = qml.device("mettleq.adaptive", wires=2, precision="single")

@qml.qnode(device)
def bell_circuit():
    qml.Hadamard(0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.Z(0) @ qml.Z(1))

print(bell_circuit())
```

## Choosing a path

- Adaptive mode uses Aer or Lightning CPU for small and double-precision work.
- The exact MettleQ statevector path targets the Apple GPU when circuit shape,
  output contract, memory, and calibrated crossover justify its overhead.
- MettleQ MPS is currently a CPU-first bounded approximation with explicit
  convergence and truncation evidence; MPS is CPU-native.
- Midpoint-MPO is a separate opt-in Qiskit method with an isolated dependency
  environment and its own approximation contract.

See the [complete README](https://github.com/MonitSharma/MettleQ#readme),
[tutorial notebooks](https://github.com/MonitSharma/MettleQ/tree/main/tutorials),
[examples](https://github.com/MonitSharma/MettleQ/tree/main/examples), and
[technical report](https://github.com/MonitSharma/MettleQ/blob/main/output/pdf/MettleQ_technical_report.pdf).

MettleQ is an independently maintained MIT-licensed fork of
[BoltzmannEntropy/Qupertino](https://github.com/BoltzmannEntropy/Qupertino).
Original authorship and project lineage are retained in the repository.
