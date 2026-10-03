# Peaked-circuit benchmark attribution

`peaked_circuit_P9_Hqap_56x1917.qasm` is redistributed from
[`alexgalda-m/peaked-mpo-solver`](https://github.com/alexgalda-m/peaked-mpo-solver)
under the Apache License 2.0 included in this directory. Its SHA-256 is
`cff3496c45d9133c1f1693f1d3b0cf1fc2da338f13cd7b339db330a4762d0f35`.

The benchmark is discussed in Quantum Advantage Tracker issue
[#153](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/issues/153).
The midpoint-MPO and greedy-unswapping method is due to David Kremer and
Nicolas Dupuis, [*Efficient Classical Simulation of Heuristic Peaked Quantum
Circuits*](https://arxiv.org/abs/2604.21908), arXiv:2604.21908 (2026), with
reference code at
[`d-kremer/peaked-circuit-simulation`](https://github.com/d-kremer/peaked-circuit-simulation).

MettleQ includes the input circuit and an adapted solver implementation under
`mettleq._vendor.peaked_mpo`. The solver's Apache-2.0 license and source notice
are included alongside the vendored code.
