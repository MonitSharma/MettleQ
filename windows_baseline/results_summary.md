# Windows / WSL Baseline Performance Report

This report aggregates simulation results for Qiskit Aer, PennyLane, and CUDA Quantum backends run on Windows WSL (Ubuntu) with an NVIDIA RTX 3070 8GB GPU.

## Workload: ghz

| Backend / Qubits | 15q | 20q | 24q | 26q | 28q | 30q | 40q |
|---||---:||---:||---:||---:||---:||---:||---:|
| cudaq_nvidia | 64.80 ms | 68.81 ms | 70.08 ms | 91.78 ms | 173.24 ms | - | - |
| cudaq_qpp | 78.56 ms | 434.22 ms | 9.244 s | 39.813 s | 186.627 s | - | - |
| default.qubit | 2.18 ms | 76.41 ms | 1.745 s | - | - | - | - |
| lightning.gpu | 3.35 ms | 7.74 ms | 70.15 ms | 312.64 ms | 1.142 s | - | - |
| lightning.qubit | 2.92 ms | 15.52 ms | 361.55 ms | 1.354 s | 6.390 s | - | - |
| qiskit_aer_matrix_product_state_cpu | 8.57 ms | 286.92 ms | 6.290 s | 4.65 ms | 0.92 ms | 1.30 ms | 1.69 ms |
| qiskit_aer_statevector_cpu | 52.25 ms | 29.53 ms | 231.42 ms | 803.08 ms | 3.332 s | - | - |

## Workload: grover_proxy

| Backend / Qubits | 15q | 20q | 24q | 26q | 28q | 30q | 40q |
|---||---:||---:||---:||---:||---:||---:||---:|
| cudaq_nvidia | 55.78 ms | 58.12 ms | 63.17 ms | 83.31 ms | 165.94 ms | - | - |
| cudaq_qpp | 99.30 ms | 2.272 s | 51.459 s | 237.494 s | 1089.298 s | - | - |
| default.qubit | 9.56 ms | 590.88 ms | 13.395 s | - | - | - | - |
| lightning.gpu | 7.75 ms | 21.17 ms | 275.62 ms | 1.018 s | 4.350 s | - | - |
| lightning.qubit | 3.74 ms | 74.34 ms | 2.380 s | 9.978 s | 45.136 s | - | - |
| qiskit_aer_matrix_product_state_cpu | 9.16 ms | 285.56 ms | 6.142 s | 2.44 ms | 2.32 ms | 3.71 ms | 3.32 ms |
| qiskit_aer_statevector_cpu | 21.64 ms | 52.65 ms | 223.16 ms | 706.20 ms | 3.002 s | - | - |

## Workload: phase_estimation

| Backend / Qubits | 15q | 20q | 24q | 26q | 28q | 30q | 40q |
|---||---:||---:||---:||---:||---:||---:||---:|
| cudaq_nvidia | 52.56 ms | 60.23 ms | 84.99 ms | 140.96 ms | 363.80 ms | - | - |
| cudaq_qpp | 116.49 ms | 4.427 s | 116.986 s | 569.963 s | 2846.848 s | - | - |
| default.qubit | 26.65 ms | 1.392 s | 42.245 s | - | - | - | - |
| lightning.gpu | 10.46 ms | 30.89 ms | 360.20 ms | 1.481 s | 6.882 s | - | - |
| lightning.qubit | 7.42 ms | 114.98 ms | 3.773 s | 16.568 s | 70.062 s | - | - |
| qiskit_aer_matrix_product_state_cpu | 12.28 ms | 295.16 ms | 6.168 s | 23.36 ms | 21.40 ms | 24.71 ms | 49.24 ms |
| qiskit_aer_statevector_cpu | 13.07 ms | 42.18 ms | 1.092 s | 4.286 s | 16.895 s | - | - |

## Workload: qaoa_ring

| Backend / Qubits | 15q | 20q | 24q | 26q | 28q | 30q | 40q |
|---||---:||---:||---:||---:||---:||---:||---:|
| cudaq_nvidia | 58.06 ms | 70.47 ms | 85.68 ms | 155.19 ms | 463.80 ms | - | - |
| cudaq_qpp | 135.17 ms | 4.520 s | 103.615 s | 470.291 s | 2196.922 s | - | - |
| default.qubit | 30.72 ms | 1.246 s | 35.289 s | - | - | - | - |
| lightning.gpu | 14.07 ms | 35.84 ms | 413.74 ms | 1.612 s | 7.254 s | - | - |
| lightning.qubit | 9.47 ms | 139.84 ms | 3.998 s | 17.306 s | 72.095 s | - | - |
| qiskit_aer_matrix_product_state_cpu | 219.32 ms | 1.619 s | 14.291 s | 1.760 s | 1.986 s | 2.263 s | 3.687 s |
| qiskit_aer_statevector_cpu | 33.37 ms | 367.63 ms | 1.259 s | 4.233 s | 17.439 s | - | - |

## Workload: qft

| Backend / Qubits | 15q | 20q | 24q | 26q | 28q | 30q | 40q |
|---||---:||---:||---:||---:||---:||---:||---:|
| cudaq_nvidia | 61.52 ms | 63.02 ms | 82.93 ms | 150.64 ms | 340.38 ms | - | - |
| cudaq_qpp | 377.57 ms | 4.233 s | 118.124 s | 570.398 s | 2697.125 s | - | - |
| default.qubit | 25.80 ms | 960.51 ms | 35.429 s | - | - | - | - |
| lightning.gpu | 11.56 ms | 33.25 ms | 323.40 ms | 1.529 s | 6.635 s | - | - |
| lightning.qubit | 6.12 ms | 90.70 ms | 3.040 s | 14.185 s | 63.080 s | - | - |
| qiskit_aer_matrix_product_state_cpu | 14.29 ms | 338.63 ms | 6.529 s | 24.27 ms | 28.03 ms | 29.64 ms | 52.75 ms |
| qiskit_aer_statevector_cpu | 24.12 ms | 70.79 ms | 868.87 ms | 3.381 s | 13.861 s | - | - |

## Workload: tfim_trotter

| Backend / Qubits | 15q | 20q | 24q | 26q | 28q | 30q | 40q |
|---||---:||---:||---:||---:||---:||---:||---:|
| cudaq_nvidia | 95.49 ms | 118.24 ms | 204.35 ms | 463.70 ms | 1.566 s | - | - |
| cudaq_qpp | 629.03 ms | 27.743 s | 673.310 s | 3060.610 s | 14210.794 s | - | - |
| default.qubit | 68.49 ms | 4.054 s | 116.363 s | - | - | - | - |
| lightning.gpu | 31.21 ms | 119.68 ms | 1.513 s | 6.301 s | 27.164 s | - | - |
| lightning.qubit | 14.95 ms | 501.42 ms | 14.957 s | 62.993 s | 276.929 s | - | - |
| qiskit_aer_matrix_product_state_cpu | 16.55 ms | 331.22 ms | 6.342 s | 18.56 ms | 25.67 ms | 21.46 ms | 27.77 ms |
| qiskit_aer_statevector_cpu | 26.53 ms | 310.13 ms | 3.614 s | 13.031 s | 54.528 s | - | - |

