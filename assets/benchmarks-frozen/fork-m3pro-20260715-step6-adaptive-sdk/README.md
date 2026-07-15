# Step 6 adaptive SDK evidence — Apple M3 Pro

This bundle validates engine commit
`3f40a474b337cb094e4c40c57fa9e1ba2002f890` on an Apple M3 Pro with
Python 3.13.2 and MLX 0.32.0. Every timing family used one warmup and seven
rotating repeats. CPU and GPU arms ran independently; their times are never
added together.

## Execution-policy calibration

The policy workload is two steps of `H + nearest-neighbor ZZ + RX`, followed by
a local Pauli-Z expectation. It measures allocation, execution,
synchronization, and measurement.

| Statevector policy at 20 qubits | CPU | GPU | CPU / GPU | Sustained crossover |
| --- | ---: | ---: | ---: | ---: |
| Pure MLX (safe default) | 230.50 ms | 41.67 ms | 5.53× | 14 qubits |
| Compatible Metal enabled | 231.56 ms | 12.42 ms | 18.64× | 6 qubits |

MPS used `Dmax=32`, `eps=1e-10`, and had no truncation events. CPU remained
faster at every measured size from 4 through 32 qubits. At 32 qubits, pure-MLX
MPS took 10.69 ms on CPU and 28.41 ms on the explicit GPU tensor path. The GPU
arm reports GPU tensors and CPU SVD, which is why no automatic MPS GPU
crossover is recorded.

The maximum CPU/GPU expectation delta across either policy run was
`2.776e-7`.

## Qiskit and PennyLane method matrix

All rows use the same 20-qubit, two-step circuit and analytic local Pauli-Z
expectation. Qiskit compares with `StatevectorEstimator`; PennyLane compares
with `default.qubit`.

| SDK | Implementation | Median | Reference / implementation | Absolute error |
| --- | --- | ---: | ---: | ---: |
| Qiskit | Reference | 661.79 ms | 1.00× | — |
| Qiskit | Qupertino statevector CPU | 238.69 ms | 2.77× | `4.42e-11` |
| Qiskit | Qupertino statevector GPU | 18.11 ms | 36.54× | `4.42e-10` |
| Qiskit | **Qupertino MPS CPU** | **7.80 ms** | **84.81×** | `6.22e-8` |
| Qiskit | Qupertino MPS GPU tensors | 24.67 ms | 26.83× | `1.12e-7` |
| PennyLane | Reference | 610.25 ms | 1.00× | — |
| PennyLane | Qupertino statevector CPU | 236.91 ms | 2.58× | `2.79e-9` |
| PennyLane | Qupertino statevector GPU | 16.28 ms | 37.49× | `1.12e-8` |
| PennyLane | **Qupertino MPS CPU** | **10.79 ms** | **56.56×** | `1.95e-7` |
| PennyLane | Qupertino MPS GPU tensors | 27.75 ms | 21.99× | `2.26e-7` |

MPS had zero truncation events and zero recorded discarded weight for this
shallow local circuit. These MPS timings are not a general dense/random-circuit
claim; bond growth and truncation determine performance and accuracy.

## Previous-protocol refresh

The older four-step SDK protocol was rerun unchanged. Qiskit full-state
readback took 11.01 ms versus Aer CPU at 77.61 ms (7.05×). PennyLane local
expectation took 21.44 ms versus `default.qubit` at 1,075.41 ms (50.15×).
Qupertino's absolute times were 8.98% and 17.12% slower than the preceding
Step 5 session, respectively; the larger PennyLane speedup comes from reference
variation, not a claim that the engine became faster.

## Contents

- `policy_pure_mlx/`: raw/default CPU-GPU policy data, summary, chart, manifest
- `policy_metal/`: custom-Metal policy data with the same protocol
- `sdk_method_matrix/`: Qiskit/PennyLane method/device raw rows, diagnostics,
  selections, summary, chart, and manifest
- `sdk_adapter_refresh/`: unchanged Step 5 protocol rerun
- `evidence_summary.json`: reviewed headline values
- `test_summary.json`: full repository test result
