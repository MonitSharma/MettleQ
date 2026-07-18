# Large SDK CPU/GPU crossover

This experiment compares complete full-state calls on one Apple M3 Pro. Qiskit uses the fastest measured CPU method at each width from Aer statevector and, where scheduled, Aer MPS. PennyLane uses `lightning.qubit` with complex128. MettleQ is forced to its exact Apple GPU statevector path through each SDK adapter.

Aer circuit truncation is disabled because its subset-observable pilot disagreed with two independent full-state references; disabling truncation restored agreement. Aer MPS is not scheduled above the configured cap after full-state materialization has removed its tensor-network advantage.

| Qubits | Qiskit fastest CPU | CPU ms | MettleQ GPU ms | CPU / GPU | Max state error | Lightning CPU ms | MettleQ GPU ms | CPU / GPU | Max state error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | Aer statevector | 5.043 | 26.711 | **0.189x** | `4.39e-07` | 16.501 | 10.534 | **1.567x** | `3.71e-07` |
| 18 | Aer statevector | 12.655 | 32.702 | **0.387x** | `4.52e-07` | 46.998 | 17.172 | **2.737x** | `4.41e-07` |
| 20 | Aer statevector | 41.616 | 39.917 | **1.043x** | `1.25e-07` | 181.961 | 21.189 | **8.588x** | `2.86e-07` |
| 22 | Aer statevector | 276.039 | 88.542 | **3.118x** | `2.29e-07` | 729.526 | 63.499 | **11.489x** | `5.56e-07` |
| 24 | Aer statevector | 1050.644 | 305.850 | **3.435x** | `2.45e-07` | 2962.134 | 226.452 | **13.081x** | `1.93e-07` |
| 26 | Aer statevector | 3470.464 | 1051.053 | **3.302x** | `2.45e-07` | 9460.133 | 892.971 | **10.594x** | `1.66e-07` |
| 27 | Aer statevector | 5252.406 | 1799.607 | **2.919x** | `1.49e-07` | 23075.791 | 1764.872 | **13.075x** | `1.42e-07` |
| 28 | Aer statevector | 7447.867 | 4246.323 | **1.754x** | `1.53e-07` | 40934.126 | 3856.384 | **10.615x** | `1.10e-07` |
| 29 | Aer statevector | 19614.343 | 8453.778 | **2.320x** | `1.14e-07` | 88010.174 | 7622.193 | **11.547x** | `9.28e-08` |

A ratio above 1.0× means MettleQ was faster. Every accepted row must also satisfy maximum phase-aligned state error <= `5e-06`.

## Interpretation

On this workload MettleQ first crosses Qiskit Aer at 20 qubits and `lightning.qubit` at 16 qubits. Aer MPS is slower here because the contract requires materializing the complete dense statevector; bounded observables or samples are the appropriate contract for demonstrating an MPS advantage.

The Qiskit path applies dependency-preserving scheduling before Metal fusion, so frontend ASAP serialization does not hide legal single-qubit and affine layers. Operations sharing a wire retain their original order.

Safety refusals are results, not missing data: widths whose modeled simultaneous validation footprint violates the configured RAM fraction or OS-headroom reserve are never launched.

## Reproduce

```bash
PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/benchmark_sdk_crossover_large.py \
  --widths 16,18,20,22,24,26,27,28,29,30 --depth 3 \
  --warmups 1 --repeats 3 \
  --max-qiskit-mps-width 22 --accuracy-atol 5e-6 \
  --output-dir bench/runs/sdk-crossover-large
```

The worker-per-width design releases large statevectors before advancing. `caffeinate` exits when the benchmark process finishes.

Environment: `{"mettleq": "0.2.0", "mlx": "0.32.0", "numpy": "2.4.6", "pennylane": "0.45.1", "pennylane-lightning": "0.45.0", "qiskit": "2.5.0", "qiskit-aer": "0.17.2"}`
