# Large SDK CPU/GPU crossover

This experiment compares complete full-state calls on one Apple M3 Pro. Qiskit uses the fastest measured CPU method at each width from Aer statevector and, where scheduled, Aer MPS. PennyLane uses `lightning.qubit` with complex128. MettleQ is forced to its exact Apple GPU statevector path through each SDK adapter.

Aer circuit truncation is disabled because its subset-observable pilot disagreed with two independent full-state references; disabling truncation restored agreement. Aer MPS is not scheduled above the configured cap after full-state materialization has removed its tensor-network advantage.

| Qubits | Qiskit fastest CPU | CPU ms | MettleQ GPU ms | CPU / GPU | Max state error | Lightning CPU ms | MettleQ GPU ms | CPU / GPU | Max state error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | Aer statevector | 4.096 | 9.505 | **0.431x** | `3.12e-07` | 15.595 | 20.326 | **0.767x** | `4.22e-07` |
| 18 | Aer statevector | 19.036 | 19.547 | **0.974x** | `2.13e-07` | 44.696 | 28.552 | **1.565x** | `2.90e-07` |
| 20 | Aer statevector | 54.272 | 71.865 | **0.755x** | `2.97e-07` | 171.267 | 78.677 | **2.177x** | `3.12e-07` |
| 22 | Aer statevector | 193.835 | 306.035 | **0.633x** | `3.36e-07` | 862.820 | 454.955 | **1.896x** | `9.99e-08` |
| 24 | Aer statevector | 1202.066 | 1444.005 | **0.832x** | `1.85e-07` | 2923.091 | 1520.666 | **1.922x** | `9.05e-08` |
| 26 | Aer statevector | 2278.732 | 5931.567 | **0.384x** | `1.94e-07` | 12164.674 | 7168.681 | **1.697x** | `1.86e-07` |

A ratio above 1.0× means MettleQ was faster. Every accepted row must also satisfy maximum phase-aligned state error <= `5e-06`.

## Interpretation

On this workload MettleQ crosses `lightning.qubit` at 18 qubits and stays faster through 26 qubits. It does not cross Qiskit Aer: Aer statevector remains faster at every measured width. Aer MPS is slower here because the contract requires materializing the complete dense statevector; bounded observables or samples are the appropriate contract for demonstrating an MPS advantage.

The benchmark intentionally reports this negative Qiskit result. It identifies Qiskit adapter/result conversion, GPU memory traffic, and dense-kernel performance as optimization targets before an automatic Qiskit GPU crossover can be claimed.

## Reproduce

```bash
PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/benchmark_sdk_crossover_large.py \
  --widths 16,18,20,22,24,26 --depth 3 --warmups 1 --repeats 3 \
  --max-qiskit-mps-width 22 --accuracy-atol 5e-6 \
  --output-dir bench/runs/sdk-crossover-large
```

The worker-per-width design releases large statevectors before advancing. `caffeinate` exits when the benchmark process finishes.

Environment: `{"mettleq": "0.2.0", "mlx": "0.32.0", "numpy": "2.4.6", "pennylane": "0.45.1", "pennylane-lightning": "0.45.0", "qiskit": "2.5.0", "qiskit-aer": "0.17.2"}`
