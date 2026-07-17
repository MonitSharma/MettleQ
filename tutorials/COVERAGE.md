# Tutorial coverage audit

This audit prevents “all tutorials” from becoming an inaccurate compatibility
claim. It maps the current official SDK learning catalogs to what a local ideal
statevector/MPS backend can execute. “Core represented” means the notebook
implements a small original version of the quantum circuit and result contract;
it does not reproduce the official text, cloud workflow, dataset, or hardware
experiment.

Catalogs checked on 2026-07-17:

- [IBM Quantum tutorials](https://quantum.cloud.ibm.com/docs/en/tutorials/index)
- [Qiskit exact local simulation](https://quantum.cloud.ibm.com/docs/en/guides/simulate-with-qiskit-sdk-primitives)
- [PennyLane demonstrations](https://www.pennylane.ai/demonstrations)
- [PennyLane circuits and QNodes](https://docs.pennylane.ai/en/stable/introduction/circuits.html)

## IBM/Qiskit catalog

| Official tutorial or capability | Coverage | MettleQ notebook / reason |
| --- | --- | --- |
| CHSH inequality | Direct local counterpart | `qiskit/04_estimator_chsh.ipynb` |
| Sample-based quantum diagonalization (chemistry) | Core represented | `qiskit/10_vqe.ipynb`; full chemistry mapping is a separate classical dependency |
| Sample-based Krylov diagonalization (fermionic lattice) | Core represented | `qiskit/10_vqe.ipynb`, `qiskit/12_hamiltonian_simulation.ipynb` |
| Quantum approximate optimization algorithm | Direct local counterpart | `qiskit/11_qaoa_maxcut.ipynb` |
| Advanced QAOA and Pauli correlation encoding | Core represented | `qiskit/11_qaoa_maxcut.ipynb`; addon-specific mapping is not bundled |
| Implicit solvent with Qiskit Serverless | Not a local backend feature | Requires Serverless/cloud services and chemistry dependencies |
| Krylov lattice Hamiltonians | Core represented | `qiskit/12_hamiltonian_simulation.ipynb` |
| Nishimori phase transition | Core represented | `qiskit/12_hamiltonian_simulation.ipynb`; full hardware-scale study is not reproduced |
| Heisenberg-chain VQE | Core represented | `qiskit/10_vqe.ipynb` |
| Quantum kernel training and projected kernels | Direct local counterpart | `qiskit/13_quantum_kernel.ipynb` |
| Shor's algorithm | Small compiled core represented | `qiskit/09_shor_order_finding.ipynb`; no scalable modular-arithmetic claim |
| Grover's algorithm | Direct local counterpart | `qiskit/07_grover_search.ipynb` |
| Dynamic-circuit Bell-pair benchmark | Unsupported semantics | Native mid-circuit measurement/control is rejected |
| Fractional gates | Output circuits only | Compatible decomposed unitary circuits can run; the hardware feature is not emulated |
| AI-powered transpiler | External service | A transpiled supported circuit can run, but MettleQ does not provide that service |
| SABRE transpilation optimization | Backend target represented | All Qiskit notebooks transpile to `MettleQBackend`; MPS routing is covered by `qiskit/14_mps_topology_and_convergence.ipynb` |
| Hamiltonian-simulation compilation | Core represented | `qiskit/12_hamiltonian_simulation.ipynb` |
| Long-range entanglement / kicked Ising with dynamic circuits | Static unitary core only | `qiskit/12_hamiltonian_simulation.ipynb`; dynamic control is unsupported |
| Qiskit Functions tutorials | External managed services | Function-specific suppression, mitigation, PDE, chemistry, and optimization services are not local simulator features |
| Qiskit addon tutorials (MPF, AQC, OBP, circuit cutting) | Post-transform circuits only | Supported unitary outputs may run; addon algorithms are not reimplemented |
| Readout and utility-scale error mitigation | Unsupported model | MettleQ currently simulates ideal unitary statevector/MPS execution, not device noise |
| Repetition/spacetime error detection | Unsupported hardware contract | Requires noisy shots, syndrome workflows, and/or dynamic circuits |
| StatevectorEstimator / StatevectorSampler exact local guide | Direct counterpart | `qiskit/01` through `04`, with MettleQ BackendV2/SamplerV2/EstimatorV2 |
| Deutsch–Jozsa and Bernstein–Vazirani learning algorithms | Direct local counterparts | `qiskit/05_deutsch_jozsa.ipynb`, `qiskit/06_bernstein_vazirani.ipynb` |
| QFT and phase estimation learning material | Direct local counterpart | `qiskit/08_qft_and_phase_estimation.ipynb` |
| Matrix-product-state simulation | MettleQ-specific extension | `qiskit/14_mps_topology_and_convergence.ipynb`, `qiskit/16_peaked_circuit_smoke.ipynb` |
| Apple GPU crossover | MettleQ-specific extension | `qiskit/15_apple_gpu_scaling.ipynb` |

## PennyLane catalog categories

The PennyLane demonstrations page is a growing research catalog rather than a
fixed compatibility suite. The mapping below covers the device contracts and
algorithm families relevant to MettleQ.

| Catalog area | Coverage | MettleQ notebooks / boundary |
| --- | --- | --- |
| Getting started: devices, QNodes, measurements | Direct counterparts | `pennylane/01`, `02` |
| Gradients and hybrid optimization | Direct counterparts | `pennylane/03`, `04` using parameter shift |
| VQE / quantum chemistry | Core represented | `pennylane/05_vqe.ipynb`; molecular integral generation and datasets are external |
| QAOA / optimization | Direct local counterpart | `pennylane/06_qaoa_maxcut.ipynb` |
| QFT / QPE algorithms | Direct local counterpart | `pennylane/07_qft_and_qpe.ipynb` |
| Variational classifiers and QML | Direct compact counterpart | `pennylane/08_variational_classifier.ipynb` |
| Quantum kernels | Direct compact counterpart | `pennylane/09_quantum_kernel.ipynb` |
| Quantum teleportation | Deferred unitary core | `pennylane/10_teleportation_deferred.ipynb`; no native mid-circuit-control claim |
| Hamiltonian simulation | Direct compact counterpart | `pennylane/11_hamiltonian_simulation.ipynb` |
| Tensor networks / MPS | MettleQ-specific trust counterpart | `pennylane/12_mps_convergence.ipynb` |
| Devices and performance | Apple-specific counterpart | `pennylane/13_apple_gpu_scaling.ipynb` |
| Torch/JAX model integrations | QNode boundary only | MettleQ converts parameters to NumPy and supports framework-managed parameter shift, not backpropagation through MLX |
| Mixed states, channels, pulse/noise models | Unsupported model | MettleQ's active SDK path is ideal unitary statevector/MPS |
| Native dynamic circuits and postselection | Unsupported semantics | Deferred unitary circuits may work; native mid-circuit results/control do not |
| Continuous-variable tutorials | Unsupported model | MettleQ is a qubit simulator |
| Quantum hardware, error mitigation, and benchmarking | External hardware contract | Not presented as local simulator capability |
| FTQC, surface codes, and resource estimation | Analysis/tooling rather than backend execution | Out of scope for the current Qiskit/PennyLane simulator phase |
| Paper-specific research demos | Circuit-dependent | A demo is supported only after its operations and measurements pass MettleQ capability checks |

## Adding coverage

A new notebook should use one reference SDK device and one MettleQ device,
declare the comparison tolerance, time the complete calls, emit exactly one
`TUTORIAL_RESULT` record, and pass `tools/run_tutorial_notebooks.py`. Features
outside the current contract should be added to the backend and its automated
tests before a tutorial presents them as supported.
