# Large SDK CPU/GPU crossover

This experiment compares complete full-state calls on one Apple M3 Pro. Qiskit uses the fastest measured CPU method at each width from Aer statevector and, where scheduled, Aer MPS. PennyLane uses `lightning.qubit` with complex128. MettleQ is forced to its exact Apple GPU statevector path through each SDK adapter.

Hardware: `{"architecture": "arm64", "chip": "Apple M3 Pro", "cpu_cores": "proc 12:6:6:0", "gpu": "Apple M3 Pro", "gpu_cores": "18", "memory": "36 GB", "metal_support": "spdisplays_metal4", "model_identifier": "Mac15,7", "model_name": "MacBook Pro", "physical_memory_bytes": 38654705664}`

Aer circuit truncation is disabled because its subset-observable pilot disagreed with two independent full-state references; disabling truncation restored agreement. Aer MPS is not scheduled above the configured cap after full-state materialization has removed its tensor-network advantage.

| Qubits | Qiskit fastest CPU | CPU ms | MettleQ GPU ms | CPU / GPU | Max state error | Lightning CPU ms | MettleQ GPU ms | CPU / GPU | Max state error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | Aer statevector | 3.760 | 14.728 | **0.255x** | `3.75e-07` | 9.003 | 6.233 | **1.445x** | `3.49e-07` |
| 18 | Aer statevector | 12.608 | 20.119 | **0.627x** | `3.98e-07` | 46.538 | 12.543 | **3.710x** | `3.96e-07` |
| 20 | Aer statevector | 47.574 | 35.859 | **1.327x** | `4.02e-07` | 118.461 | 10.675 | **11.097x** | `3.03e-07` |
| 22 | Aer statevector | 138.076 | 43.700 | **3.160x** | `1.47e-07` | 470.725 | 28.653 | **16.429x** | `3.35e-07` |
| 24 | Aer statevector | 687.554 | 150.980 | **4.554x** | `1.99e-07` | 2027.497 | 108.346 | **18.713x** | `1.63e-07` |
| 26 | Aer statevector | 2257.433 | 530.395 | **4.256x** | `2.60e-07` | 8360.268 | 434.765 | **19.229x** | `7.52e-08` |
| 27 | Aer statevector | 3954.558 | 1110.417 | **3.561x** | `2.21e-07` | 17369.838 | 955.118 | **18.186x** | `1.41e-07` |
| 28 | Aer statevector | 5833.566 | 2059.440 | **2.833x** | `8.59e-08` | 35927.705 | 1696.528 | **21.177x** | `1.22e-07` |
| 29 | Aer statevector | 14252.833 | 5203.575 | **2.739x** | `1.33e-07` | 74375.386 | 4646.928 | **16.005x** | `1.20e-07` |

A ratio above 1.0× means MettleQ was faster. Every accepted row must also satisfy maximum phase-aligned state error <= `5e-06`.

## Interpretation

On this workload MettleQ first crosses Qiskit Aer at 20 qubits and `lightning.qubit` at 16 qubits. Aer MPS is slower here because the contract requires materializing the complete dense statevector; bounded observables or samples are the appropriate contract for demonstrating an MPS advantage.

The Qiskit path applies dependency-preserving scheduling before Metal fusion, so frontend ASAP serialization does not hide legal single-qubit and affine layers. Operations sharing a wire retain their original order. Generic single-qubit layers use radix-16 passes that apply four different 2x2 matrices per state traversal.

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
