# Tutorial correctness audit

**Result: PASS — 29 notebooks validated.**

`exact_match` means bit-for-bit equality, which is stricter than numerical correctness. MettleQ's Apple GPU statevector commonly uses complex64 while the CPU references use complex128, so harmless last-bit differences are expected. Finite-shot results use independent random samples and therefore must be compared statistically. MPS is approximate and additionally needs bond-dimension convergence evidence.

- Bit-for-bit matches: 9
- Tolerance/statistically validated matches: 20
- Largest primary-error/tolerance fraction: 0.407

| Notebook | Validation class | Exact | Primary error | Tolerance | Fraction used |
| --- | --- | ---: | ---: | ---: | ---: |
| `qiskit/01_backend_quickstart.ipynb` | numerically equivalent within an explicit tolerance | no | 1.21e-08 | 2.00e-06 | 0.006 |
| `qiskit/02_statevectors_and_gates.ipynb` | numerically equivalent within an explicit tolerance | no | 4.35e-08 | 2.00e-06 | 0.022 |
| `qiskit/03_sampler_and_counts.ipynb` | statistical agreement (independent random samples) | no | 2.20e-03 | 5.00e-02 | 0.044 |
| `qiskit/04_estimator_chsh.ipynb` | numerically equivalent within an explicit tolerance | no | 1.19e-07 | 2.00e-06 | 0.060 |
| `qiskit/05_deutsch_jozsa.ipynb` | bit-for-bit match | yes | 2.03e-07 | 2.00e-06 | 0.101 |
| `qiskit/06_bernstein_vazirani.ipynb` | bit-for-bit match | yes | 2.03e-07 | 2.00e-06 | 0.101 |
| `qiskit/07_grover_search.ipynb` | bit-for-bit match | yes | 2.38e-07 | 2.00e-06 | 0.119 |
| `qiskit/08_qft_and_phase_estimation.ipynb` | bit-for-bit match | yes | 2.38e-07 | 2.00e-06 | 0.119 |
| `qiskit/09_shor_order_finding.ipynb` | bit-for-bit match | yes | 2.38e-07 | 2.00e-06 | 0.119 |
| `qiskit/10_vqe.ipynb` | bit-for-bit match | yes | 1.35e-07 | 3.00e-06 | 0.045 |
| `qiskit/11_qaoa_maxcut.ipynb` | bit-for-bit match | yes | 9.80e-07 | 3.00e-06 | 0.327 |
| `qiskit/12_hamiltonian_simulation.ipynb` | numerically equivalent within an explicit tolerance | no | 1.22e-06 | 3.00e-06 | 0.407 |
| `qiskit/13_quantum_kernel.ipynb` | numerically equivalent within an explicit tolerance | no | 3.86e-07 | 4.00e-06 | 0.097 |
| `qiskit/14_mps_topology_and_convergence.ipynb` | approximate MPS, independently checked with convergence evidence | no | 6.87e-07 | 8.00e-05 | 0.009 |
| `qiskit/15_apple_gpu_scaling.ipynb` | numerically equivalent within an explicit tolerance | no | 2.27e-07 | 3.00e-06 | 0.076 |
| `qiskit/16_peaked_circuit_smoke.ipynb` | statistical agreement (independent random samples) | no | 4.88e-03 | 2.50e-02 | 0.195 |
| `pennylane/01_qnodes_and_measurements.ipynb` | numerically equivalent within an explicit tolerance | no | 1.80e-08 | 2.00e-06 | 0.009 |
| `pennylane/02_finite_shots.ipynb` | statistical agreement (independent random samples) | no | 1.71e-03 | 5.00e-02 | 0.034 |
| `pennylane/03_parameter_shift_gradients.ipynb` | numerically equivalent within an explicit tolerance | no | 7.08e-09 | 3.00e-06 | 0.002 |
| `pennylane/04_variational_optimization.ipynb` | numerically equivalent within an explicit tolerance | no | 9.73e-08 | 4.00e-05 | 0.002 |
| `pennylane/05_vqe.ipynb` | numerically equivalent within an explicit tolerance | no | 1.38e-07 | 5.00e-05 | 0.003 |
| `pennylane/06_qaoa_maxcut.ipynb` | bit-for-bit match | yes | 1.01e-06 | 4.00e-06 | 0.253 |
| `pennylane/07_qft_and_qpe.ipynb` | bit-for-bit match | yes | 3.54e-08 | 3.00e-06 | 0.012 |
| `pennylane/08_variational_classifier.ipynb` | numerically equivalent within an explicit tolerance | no | 1.11e-07 | 8.00e-05 | 0.001 |
| `pennylane/09_quantum_kernel.ipynb` | numerically equivalent within an explicit tolerance | no | 2.38e-07 | 4.00e-06 | 0.060 |
| `pennylane/10_teleportation_deferred.ipynb` | numerically equivalent within an explicit tolerance | no | 1.82e-07 | 3.00e-06 | 0.061 |
| `pennylane/11_hamiltonian_simulation.ipynb` | numerically equivalent within an explicit tolerance | no | 8.91e-07 | 3.00e-06 | 0.297 |
| `pennylane/12_mps_convergence.ipynb` | approximate MPS, independently checked with convergence evidence | no | 5.48e-07 | 8.00e-05 | 0.007 |
| `pennylane/13_apple_gpu_scaling.ipynb` | numerically equivalent within an explicit tolerance | no | 1.67e-07 | 3.00e-06 | 0.056 |

A passing tolerance is evidence for the tested circuits and methods, not a proof that every possible circuit is correct. The project test suite, full-state comparisons, MPS convergence runs, and benchmark accuracy gates provide complementary coverage.
