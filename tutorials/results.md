# Executed tutorial results

All **29/29** notebooks passed their declared comparison on this machine.

`Reference / MettleQ` above 1.0 means MettleQ was faster for that notebook's complete call; below 1.0 means the SDK reference was faster.

| Notebook | Check | Reference (ms) | MettleQ (ms) | Reference / MettleQ | Method/device | Exact? |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `qiskit/01_backend_quickstart.ipynb` | phase-aligned statevector atol=2e-6 | 0.065 | 0.318 | 0.204x | statevector/cpu | False |
| `qiskit/02_statevectors_and_gates.ipynb` | phase-aligned statevector atol=2e-6 | 0.159 | 0.930 | 0.171x | statevector/cpu | False |
| `qiskit/03_sampler_and_counts.ipynb` | finite-shot total-variation distance <= 0.05 | 6.213 | 8.729 | 0.712x | statevector/cpu | False |
| `qiskit/04_estimator_chsh.ipynb` | EstimatorV2 correlation curve atol=2e-6 | 6.451 | 29.114 | 0.222x | statevector/cpu | False |
| `qiskit/05_deutsch_jozsa.ipynb` | probability vector atol=2e-6 and exact oracle class | 0.626 | 1.297 | 0.482x | statevector/cpu | True |
| `qiskit/06_bernstein_vazirani.ipynb` | probability vector atol=2e-6 and exact hidden string | 0.561 | 1.072 | 0.524x | statevector/cpu | True |
| `qiskit/07_grover_search.ipynb` | probability vector atol=2e-6 and exact marked item | 0.190 | 0.502 | 0.377x | statevector/cpu | True |
| `qiskit/08_qft_and_phase_estimation.ipynb` | phase-register probabilities atol=2e-6 and identical mode | 1.538 | 0.650 | 2.367x | statevector/cpu | True |
| `qiskit/09_shor_order_finding.ipynb` | compiled phase distribution atol=2e-6 and factors 3,5 | 0.496 | 0.918 | 0.541x | statevector/cpu | True |
| `qiskit/10_vqe.ipynb` | VQE energy trace atol=3e-6 | 16.987 | 120.086 | 0.141x | statevector/cpu | True |
| `qiskit/11_qaoa_maxcut.ipynb` | QAOA cost landscape atol=3e-6 | 22.666 | 162.675 | 0.139x | statevector/cpu | True |
| `qiskit/12_hamiltonian_simulation.ipynb` | observable trajectory atol=3e-6 | 9.321 | 46.800 | 0.199x | statevector/cpu | False |
| `qiskit/13_quantum_kernel.ipynb` | fidelity kernel matrix atol=4e-6 | 0.871 | 3.073 | 0.283x | statevector/cpu | False |
| `qiskit/14_mps_topology_and_convergence.ipynb` | phase-aligned MPS state atol=8e-5 and convergence report | 0.869 | 12.102 | 0.072x | matrix_product_state/cpu | False |
| `qiskit/15_apple_gpu_scaling.ipynb` | per-width statevector atol=3e-6, policy-selected GPU, and 20q speedup >=1.5x | 10.678 | 5.704 | 1.872x | statevector/gpu | False |
| `qiskit/16_peaked_circuit_smoke.ipynb` | expected mode, peak probability atol=0.025, and TVD<=0.06 | 9.365 | 36.335 | 0.258x | matrix_product_state/cpu | False |
| `pennylane/01_qnodes_and_measurements.ipynb` | all analytic measurements atol=2e-6 | 0.753 | 4.091 | 0.184x | statevector/cpu | False |
| `pennylane/02_finite_shots.ipynb` | finite-shot total-variation distance <= 0.05 | 6.041 | 6.429 | 0.940x | statevector/cpu | False |
| `pennylane/03_parameter_shift_gradients.ipynb` | value and parameter-shift gradient atol=3e-6 | 2.828 | 8.030 | 0.352x | statevector/cpu | False |
| `pennylane/04_variational_optimization.ipynb` | optimization trace atol=4e-5 | 46.591 | 99.578 | 0.468x | statevector/cpu | False |
| `pennylane/05_vqe.ipynb` | VQE energy trace atol=5e-5 | 64.782 | 321.869 | 0.201x | statevector/cpu | False |
| `pennylane/06_qaoa_maxcut.ipynb` | QAOA landscape atol=4e-6 | 31.589 | 105.114 | 0.301x | statevector/cpu | True |
| `pennylane/07_qft_and_qpe.ipynb` | phase probabilities atol=3e-6 and identical mode | 1.271 | 1.407 | 0.903x | statevector/cpu | True |
| `pennylane/08_variational_classifier.ipynb` | training trace and predictions atol=8e-5 | 133.507 | 291.579 | 0.458x | statevector/cpu | False |
| `pennylane/09_quantum_kernel.ipynb` | kernel matrix atol=4e-6 | 2.400 | 3.230 | 0.743x | statevector/cpu | False |
| `pennylane/10_teleportation_deferred.ipynb` | receiver observables atol=3e-6 | 7.967 | 26.814 | 0.297x | statevector/cpu | False |
| `pennylane/11_hamiltonian_simulation.ipynb` | observable trajectory atol=3e-6 | 18.792 | 45.988 | 0.409x | statevector/cpu | False |
| `pennylane/12_mps_convergence.ipynb` | MPS state atol=8e-5 and Dmax convergence atol=5e-4 | 4.558 | 47.129 | 0.097x | matrix_product_state/cpu | False |
| `pennylane/13_apple_gpu_scaling.ipynb` | per-width statevector atol=3e-6, policy-selected GPU, and 20q speedup >=1.5x | 14.106 | 6.760 | 2.087x | statevector/gpu | False |

## Apple Silicon crossover by width

The SDK references already use the Apple CPU. MettleQ's additional opportunity is its MLX/Metal GPU path. Ratios above 1.0 mean MettleQ was faster for the complete statevector call; ratios below 1.0 mean the reference was faster.

### Qiskit

| Qubits | Reference (ms) | MettleQ (ms) | Reference / MettleQ | MettleQ path | Max state error |
| ---: | ---: | ---: | ---: | --- | ---: |
| 12 | 2.386 | 2.865 | **0.833x** | statevector/cpu | `2.26e-07` |
| 14 | 5.038 | 5.772 | **0.873x** | statevector/cpu | `2.27e-07` |
| 16 | 10.678 | 3.460 | **3.086x** | statevector/gpu | `8.76e-08` |
| 18 | 63.265 | 5.704 | **11.091x** | statevector/gpu | `1.21e-07` |
| 20 | 441.510 | 5.888 | **74.984x** | statevector/gpu | `8.31e-08` |

### PennyLane

| Qubits | Reference (ms) | MettleQ (ms) | Reference / MettleQ | MettleQ path | Max state error |
| ---: | ---: | ---: | ---: | --- | ---: |
| 12 | 3.561 | 4.415 | **0.807x** | statevector/cpu | `1.31e-07` |
| 14 | 6.699 | 7.052 | **0.950x** | statevector/cpu | `1.67e-07` |
| 16 | 14.106 | 5.738 | **2.458x** | statevector/gpu | `6.96e-08` |
| 18 | 49.707 | 6.760 | **7.353x** | statevector/gpu | `1.18e-07` |
| 20 | 380.684 | 7.529 | **50.565x** | statevector/gpu | `1.61e-07` |


Finite-shot rows are expected to show `Exact? False` when independent RNG algorithms produce different count dictionaries; their declared statistical check is the pass criterion.

Environment: `{"mettleq": "0.2.0", "mlx": "0.32.0", "numpy": "2.4.6", "pennylane": "0.45.1", "qiskit": "2.5.0", "qiskit_aer": "0.17.2"}`
