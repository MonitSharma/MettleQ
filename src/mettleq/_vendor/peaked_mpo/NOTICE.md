# Third-party midpoint-MPO implementation

The files `mpo.py`, `pipeline.py`, and `qiskit_utils.py` are derived from
[`alexgalda-m/peaked-mpo-solver`](https://github.com/alexgalda-m/peaked-mpo-solver)
at commit `3bcdc1e5bfd6abb9425f71bd43e560d2b27f45c1`.

That project is Apache-2.0 licensed and builds on the midpoint-MPO plus greedy
unswapping method introduced by David Kremer and Nicolas Dupuis in
[`d-kremer/peaked-circuit-simulation`](https://github.com/d-kremer/peaked-circuit-simulation)
and *Efficient Classical Simulation of Heuristic Peaked Quantum Circuits*
(arXiv:2604.21908).

MettleQ modifications are limited to package-relative imports and integration
through the public `mettleq.midpoint_mpo` API. See
`LICENSE-APACHE-2.0.txt` in this directory for the applicable license.
