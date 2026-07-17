# Executed tutorial results

All **29/29** notebooks passed their declared comparison on this machine.

`Reference / MettleQ` above 1.0 means MettleQ was faster for that notebook's complete call; below 1.0 means the SDK reference was faster.

| Notebook | Check | Reference (ms) | MettleQ (ms) | Reference / MettleQ | Method/device | Exact? |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `qiskit/01_backend_quickstart.ipynb` | phase-aligned statevector atol=2e-6 | 0.063 | 0.274 | 0.231x | statevector/cpu | False |
| `qiskit/02_statevectors_and_gates.ipynb` | phase-aligned statevector atol=2e-6 | 0.145 | 0.889 | 0.163x | statevector/cpu | False |
| `qiskit/03_sampler_and_counts.ipynb` | finite-shot total-variation distance <= 0.05 | 4.467 | 6.794 | 0.657x | statevector/cpu | False |
| `qiskit/04_estimator_chsh.ipynb` | EstimatorV2 correlation curve atol=2e-6 | 3.274 | 24.589 | 0.133x | statevector/cpu | False |
| `qiskit/05_deutsch_jozsa.ipynb` | probability vector atol=2e-6 and exact oracle class | 0.411 | 0.841 | 0.489x | statevector/cpu | True |
| `qiskit/06_bernstein_vazirani.ipynb` | probability vector atol=2e-6 and exact hidden string | 0.329 | 0.722 | 0.455x | statevector/cpu | True |
| `qiskit/07_grover_search.ipynb` | probability vector atol=2e-6 and exact marked item | 0.186 | 0.394 | 0.472x | statevector/cpu | True |
| `qiskit/08_qft_and_phase_estimation.ipynb` | phase-register probabilities atol=2e-6 and identical mode | 0.364 | 0.525 | 0.694x | statevector/cpu | True |
| `qiskit/09_shor_order_finding.ipynb` | compiled phase distribution atol=2e-6 and factors 3,5 | 0.407 | 0.613 | 0.665x | statevector/cpu | True |
| `qiskit/10_vqe.ipynb` | VQE energy trace atol=3e-6 | 7.833 | 81.276 | 0.096x | statevector/cpu | True |
| `qiskit/11_qaoa_maxcut.ipynb` | QAOA cost landscape atol=3e-6 | 13.176 | 139.270 | 0.095x | statevector/cpu | True |
| `qiskit/12_hamiltonian_simulation.ipynb` | observable trajectory atol=3e-6 | 7.673 | 40.468 | 0.190x | statevector/cpu | False |
| `qiskit/13_quantum_kernel.ipynb` | fidelity kernel matrix atol=4e-6 | 0.559 | 2.548 | 0.219x | statevector/cpu | False |
| `qiskit/14_mps_topology_and_convergence.ipynb` | phase-aligned MPS state atol=8e-5 and convergence report | 0.781 | 10.053 | 0.078x | matrix_product_state/cpu | False |
| `qiskit/15_apple_gpu_scaling.ipynb` | per-width statevector atol=3e-6 and policy-selected GPU | 1.417 | 2.105 | 0.673x | statevector/gpu | False |
| `qiskit/16_peaked_circuit_smoke.ipynb` | expected mode, peak probability atol=0.025, and TVD<=0.06 | 5.589 | 29.768 | 0.188x | matrix_product_state/cpu | False |
| `pennylane/01_qnodes_and_measurements.ipynb` | all analytic measurements atol=2e-6 | 0.635 | 2.557 | 0.248x | statevector/cpu | False |
| `pennylane/02_finite_shots.ipynb` | finite-shot total-variation distance <= 0.05 | 3.690 | 3.867 | 0.954x | statevector/cpu | False |
| `pennylane/03_parameter_shift_gradients.ipynb` | value and parameter-shift gradient atol=3e-6 | 1.469 | 4.006 | 0.367x | statevector/cpu | False |
| `pennylane/04_variational_optimization.ipynb` | optimization trace atol=4e-5 | 24.828 | 50.577 | 0.491x | statevector/cpu | False |
| `pennylane/05_vqe.ipynb` | VQE energy trace atol=5e-5 | 28.813 | 176.084 | 0.164x | statevector/cpu | False |
| `pennylane/06_qaoa_maxcut.ipynb` | QAOA landscape atol=4e-6 | 19.336 | 77.591 | 0.249x | statevector/cpu | True |
| `pennylane/07_qft_and_qpe.ipynb` | phase probabilities atol=3e-6 and identical mode | 0.634 | 1.597 | 0.397x | statevector/cpu | True |
| `pennylane/08_variational_classifier.ipynb` | training trace and predictions atol=8e-5 | 69.275 | 166.633 | 0.416x | statevector/cpu | False |
| `pennylane/09_quantum_kernel.ipynb` | kernel matrix atol=4e-6 | 1.210 | 1.905 | 0.635x | statevector/cpu | False |
| `pennylane/10_teleportation_deferred.ipynb` | receiver observables atol=3e-6 | 4.786 | 13.508 | 0.354x | statevector/cpu | False |
| `pennylane/11_hamiltonian_simulation.ipynb` | observable trajectory atol=3e-6 | 11.421 | 21.755 | 0.525x | statevector/cpu | False |
| `pennylane/12_mps_convergence.ipynb` | MPS state atol=8e-5 and Dmax convergence atol=5e-4 | 2.239 | 27.275 | 0.082x | matrix_product_state/cpu | False |
| `pennylane/13_apple_gpu_scaling.ipynb` | per-width statevector atol=3e-6 and policy-selected GPU | 1.815 | 2.904 | 0.625x | statevector/gpu | False |

Finite-shot rows are expected to show `Exact? False` when independent RNG algorithms produce different count dictionaries; their declared statistical check is the pass criterion.

Environment: `{"mettleq": "0.2.0", "mlx": "0.32.0", "numpy": "2.4.6", "pennylane": "0.45.1", "qiskit": "2.5.0", "qiskit_aer": "0.17.2"}`
