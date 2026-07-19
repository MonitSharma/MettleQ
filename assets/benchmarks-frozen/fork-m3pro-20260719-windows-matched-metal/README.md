# Matched Windows-width MettleQ Metal campaign

This frozen run measures MettleQ commit `3b445a1` at the exact Windows/WSL
campaign widths: 15, 20, 24, 26, and 28 qubits. It covers QFT, six-layer ring
QAOA, GHZ, Grover proxy, phase estimation, and 20-step TFIM Trotter circuits.

Each workload has its own directory and process. Every cell uses one warm-up,
three measured repeats, a synchronized complex64 full state, state-norm check,
and a memory preflight requiring the two-state lower bound plus 6 GiB available
headroom. Complete-state Qiskit validation through 20 qubits passed at `5e-6`.
The per-workload manifest records the clean revision, environment, load,
packages, safety decisions, and enabled Metal optimizations.

Generate the cross-system CSV and plot with:

```bash
python windows_baseline/analyze_comparison.py
```

The comparison is honest about its scope: circuits, widths, repeats, and result
contracts match, but the machines differ and the historical Windows manifests
omit several important hardware/software fields.
