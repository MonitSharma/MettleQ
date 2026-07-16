# Step 9: MettleQ peaked-circuit evidence

This M3 Pro run contains two deliberately separate contracts:

- a 12-cell mirrored peaked regression (6/8/10 qubits; linear, grid,
  long-range, and all-to-all), matched to Qiskit Aer MPS;
- forward-MPS probes of the exact published 56-qubit P9 QASM.

All mirrored cases recovered their analytically known expected peak. Before
batched sampling, median MettleQ time across cells was 673.3 ms. Aer was faster
in every cell.

The 250-operation P9 prefix completed in 5.75 s. The full 5,807-operation P9
input completed in 111.52 s at `Dmax=64` and 100 shots, but did not recover the
published expected bitstring. It saturated the bond cap and reported 190.68
accumulated relative local discarded weight. This is negative forward-MPS
evidence, not a failed reproduction of the midpoint-MPO/unswapping reference
algorithm.

See `manifest.json` for environment, parameters, full diagnostics, and the
algorithm-mismatch declaration.
