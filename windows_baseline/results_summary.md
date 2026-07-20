# Windows / WSL Baseline Performance Report

This report aggregates simulation results for Qiskit Aer, PennyLane, and CUDA Quantum backends run on Windows WSL (Ubuntu) with an NVIDIA RTX 3070 8GB GPU.

## Workload: ghz

| Backend / Qubits | 15q | 20q | 24q | 26q | 28q | 30q | 40q |
|---| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cudaq_nvidia | 63.84 ms | 64.86 ms | 70.40 ms | 87.03 ms | 164.80 ms | - | - |
| cudaq_qpp | 85.17 ms | 388.91 ms | 8.832 s | - | - | - | - |
| default.qubit | 2.49 ms | 66.43 ms | 1.646 s | - | - | - | - |
| lightning.gpu | 6.07 ms | 10.59 ms | 92.13 ms | 305.40 ms | 1.357 s | - | - |
| lightning.qubit | 2.88 ms | 15.60 ms | 328.18 ms | 1.318 s | 5.614 s | - | - |
| qiskit_aer_matrix_product_state_cpu | 8.44 ms | 278.52 ms | 6.185 s | 8.84 ms | 3.43 ms | 0.94 ms | 3.17 ms |
| qiskit_aer_matrix_product_state_gpu | 8.72 ms | 285.56 ms | 6.148 s | 2.32 ms | 1.18 ms | 1.41 ms | 1.88 ms |
| qiskit_aer_statevector_cpu | 43.01 ms | 22.37 ms | 212.79 ms | 728.25 ms | 3.111 s | - | - |
| qiskit_aer_statevector_gpu | 3.51 ms | 34.72 ms | 196.82 ms | 968.08 ms | 4.140 s | - | - |

## Workload: grover_proxy

| Backend / Qubits | 15q | 20q | 24q | 26q | 28q | 30q | 40q |
|---| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cudaq_nvidia | 63.47 ms | 64.75 ms | 65.41 ms | 84.84 ms | 164.14 ms | - | - |
| cudaq_qpp | 260.98 ms | 2.293 s | 51.169 s | - | - | - | - |
| default.qubit | 8.23 ms | 436.40 ms | 13.237 s | - | - | - | - |
| lightning.gpu | 8.47 ms | 22.28 ms | 285.54 ms | 1.023 s | 4.541 s | - | - |
| lightning.qubit | 3.85 ms | 85.86 ms | 2.240 s | 9.738 s | 41.909 s | - | - |
| qiskit_aer_matrix_product_state_cpu | 9.22 ms | 283.94 ms | 6.043 s | 4.10 ms | 2.67 ms | 2.52 ms | 3.36 ms |
| qiskit_aer_matrix_product_state_gpu | 9.10 ms | 273.21 ms | 6.028 s | 5.87 ms | 3.21 ms | 2.99 ms | 6.37 ms |
| qiskit_aer_statevector_cpu | 3.05 ms | 12.29 ms | 178.33 ms | 700.79 ms | 2.999 s | - | - |
| qiskit_aer_statevector_gpu | 5.76 ms | 30.29 ms | 245.47 ms | 837.16 ms | 3.845 s | - | - |

## Workload: phase_estimation

| Backend / Qubits | 15q | 20q | 24q | 26q | 28q | 30q | 40q |
|---| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cudaq_nvidia | 58.42 ms | 67.22 ms | 81.54 ms | 147.11 ms | 359.29 ms | - | - |
| cudaq_qpp | 111.03 ms | 4.029 s | 117.566 s | - | - | - | - |
| default.qubit | 27.93 ms | 1.231 s | 39.729 s | - | - | - | - |
| lightning.gpu | 10.35 ms | 31.36 ms | 361.63 ms | 1.437 s | 7.687 s | - | - |
| lightning.qubit | 6.58 ms | 104.16 ms | 3.435 s | 15.180 s | 68.617 s | - | - |
| qiskit_aer_matrix_product_state_cpu | 12.13 ms | 289.11 ms | 5.994 s | 18.10 ms | 24.33 ms | 29.35 ms | 48.95 ms |
| qiskit_aer_matrix_product_state_gpu | 12.85 ms | 284.97 ms | 6.230 s | 23.71 ms | 27.50 ms | 33.40 ms | 52.01 ms |
| qiskit_aer_statevector_cpu | 7.62 ms | 46.54 ms | 1.014 s | 4.110 s | 16.735 s | - | - |
| qiskit_aer_statevector_gpu | 9.78 ms | 46.97 ms | 344.36 ms | 1.333 s | 6.155 s | - | - |

## Workload: qaoa_ring

| Backend / Qubits | 15q | 20q | 24q | 26q | 28q | 30q | 40q |
|---| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cudaq_nvidia | 59.13 ms | 63.52 ms | 84.92 ms | 160.26 ms | 464.77 ms | - | - |
| cudaq_qpp | 161.01 ms | 4.211 s | 103.435 s | - | - | - | - |
| default.qubit | 28.65 ms | 1.181 s | 32.663 s | - | - | - | - |
| lightning.gpu | 13.57 ms | 34.85 ms | 374.35 ms | 1.552 s | 7.170 s | - | - |
| lightning.qubit | 9.31 ms | 137.98 ms | 3.853 s | 16.134 s | 69.362 s | - | - |
| qiskit_aer_matrix_product_state_cpu | 210.87 ms | 1.592 s | 13.995 s | 1.686 s | 1.948 s | 2.301 s | 3.585 s |
| qiskit_aer_matrix_product_state_gpu | 210.69 ms | 1.520 s | 14.014 s | 1.683 s | 1.960 s | 2.224 s | 3.700 s |
| qiskit_aer_statevector_cpu | 8.53 ms | 56.88 ms | 1.085 s | 4.159 s | 16.994 s | - | - |
| qiskit_aer_statevector_gpu | 10.52 ms | 83.76 ms | 557.39 ms | 2.332 s | 11.150 s | - | - |

## Workload: qft

| Backend / Qubits | 15q | 20q | 24q | 26q | 28q | 30q | 40q |
|---| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cudaq_nvidia | 53.69 ms | 63.65 ms | 86.77 ms | 144.68 ms | 341.45 ms | - | - |
| cudaq_qpp | 194.10 ms | 4.249 s | 108.406 s | - | - | - | - |
| default.qubit | 26.24 ms | 1.347 s | 36.712 s | - | - | - | - |
| lightning.gpu | 12.95 ms | 33.84 ms | 316.40 ms | 1.446 s | 6.184 s | - | - |
| lightning.qubit | 7.55 ms | 88.93 ms | 3.055 s | 13.653 s | 62.266 s | - | - |
| qiskit_aer_matrix_product_state_cpu | 13.73 ms | 348.31 ms | 6.460 s | 17.89 ms | 21.57 ms | 24.02 ms | 48.73 ms |
| qiskit_aer_matrix_product_state_gpu | 15.32 ms | 343.87 ms | 6.402 s | 18.42 ms | 22.09 ms | 27.33 ms | 54.83 ms |
| qiskit_aer_statevector_cpu | 7.67 ms | 40.21 ms | 805.13 ms | 3.184 s | 13.268 s | - | - |
| qiskit_aer_statevector_gpu | 6.76 ms | 27.69 ms | 290.34 ms | 1.198 s | 6.251 s | - | - |

## Workload: tfim_trotter

| Backend / Qubits | 15q | 20q | 24q | 26q | 28q | 30q | 40q |
|---| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cudaq_nvidia | 97.58 ms | 115.76 ms | 207.23 ms | 471.45 ms | 1.582 s | - | - |
| cudaq_qpp | 1.179 s | 27.586 s | 681.807 s | - | - | - | - |
| default.qubit | 67.31 ms | 3.570 s | 110.186 s | - | - | - | - |
| lightning.gpu | 33.12 ms | 125.56 ms | 1.487 s | 6.278 s | 26.346 s | - | - |
| lightning.qubit | 16.41 ms | 522.26 ms | 15.064 s | 63.526 s | 271.655 s | - | - |
| qiskit_aer_matrix_product_state_cpu | 16.71 ms | 305.95 ms | 6.318 s | 17.79 ms | 18.80 ms | 22.96 ms | 27.96 ms |
| qiskit_aer_matrix_product_state_gpu | 17.41 ms | 326.97 ms | 6.991 s | 25.90 ms | 25.19 ms | 25.15 ms | 34.53 ms |
| qiskit_aer_statevector_cpu | 21.68 ms | 255.59 ms | 3.486 s | 12.872 s | 53.621 s | - | - |
| qiskit_aer_statevector_gpu | 26.05 ms | 102.53 ms | 1.495 s | 6.559 s | 28.747 s | - | - |

