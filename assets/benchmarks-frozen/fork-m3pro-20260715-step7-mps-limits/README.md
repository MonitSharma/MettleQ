# Step 7: MPS entanglement-limit evidence

This bundle freezes the first topology-aware MPS limit campaign for the native
Qiskit `QupertinoEstimatorV2` path. It is evidence for one Apple M3 Pro (36 GB),
engine commit `7b3d2ff`, Python 3.13.2, MLX 0.32.0, and Qiskit 2.5.0. Every
manifest records an empty Git status and every case ran in a fresh child
process.

## Protocol

- CPU MPS, because the measured automatic MPS policy has no GPU crossover.
- `Dmax=64`, relative singular-value threshold `1e-10`, and a 30-second
  per-case ceiling for the completion-envelope campaigns.
- `Dmax=32/64/128` convergence probes on representative cases.
- Seven deterministic entanglement families: GHZ chain, 1D brickwork, ring,
  2D grid, rainbow, seeded random long range, and all to all.
- Local `Z` expectation through the public Qiskit Estimator interface.
- Independent Qiskit statevector references through 20 qubits with acceptance
  `|Δ⟨Z₀⟩| <= 5e-5`.
- Runtime, peak RSS, maximum/current bonds, local truncation telemetry, state
  norm, exact-reference error, timeout, and process failure are all retained.

## Interpretation

The 49 distinct `Dmax=64` cases produced 40 completions, eight process errors,
and one timeout. Completion is deliberately not equated with accuracy:

| Family | Largest demonstrated completion | Runtime | Trust signal at that case |
| --- | ---: | ---: | --- |
| GHZ chain | 10,000q d1 | 2.309 s | Bond 2, no local truncation, norm 0.99999998 |
| 1D brickwork | 1,000q d8 | 1.096 s | Bond 60, discarded-weight telemetry `6.23e-18`, norm 1.000062 |
| Ring brickwork | 1,000q d4 | 13.344 s | Bond cap reached, norm 0.0169: completed but unusable |
| 2D grid | 81q d2 | 1.467 s | Bond cap reached, norm 0.775: strongly approximate |
| Rainbow | 64q d1 | 4.025 s | Bond cap reached, norm 0.930: approximate |
| Random long range | 96q d1 | 5.935 s | Bond cap reached, norm 0.868: approximate |
| All to all | 32q d1 | 20.969 s | Bond cap reached, norm 0.573; 36q timed out |

The standard 26-case sweep completed without a timeout, but its exact checks
failed for truncated 16- and 20-qubit nonlocal cases, reaching a maximum local
observable error of `1.278e-2`. Larger 2D, rainbow, random-long-range, and ring
probes also exposed non-monotonic MLX CPU SVD failures (`sgesvdx` code 1). A
10,000-qubit depth-8 line case reached a zero numerical norm. These are current
reliability limits and are retained rather than filtered out.

The convergence campaign shows why `Dmax` must be part of the result contract.
Increasing it can improve norm or exact-reference error, but the behavior is
not uniformly monotonic and the cost can rise sharply: the 24-qubit all-to-all
case grew from 2.406 s (`Dmax=32`) to 7.530 s (`64`) and 26.126 s (`128`), while
still showing severe truncation. Local discarded weights are telemetry, not a
global fidelity guarantee.

## Files

- `mps_limit_landscape.png`: completion, error/timeout, bond growth, and local
  truncation across the combined `Dmax=64` campaigns.
- `mps_dmax_convergence.png`: runtime, norm drift, and exact small-circuit error
  across `Dmax=32/64/128`.
- `mps_limit_combined.csv` and `mps_convergence_combined.csv`: plotted source
  rows with original source paths.
- `standard_*`, `extended_*`, and `boundaries_*`: raw rows, summaries, and
  manifests for the three `Dmax=64` campaigns.
- `convergence_d32_*` and `convergence_d128_*`: raw rows, summaries, and
  manifests for the extra convergence runs; the `Dmax=64` values come from the
  standard campaign.

Recreate the standard suite:

```bash
PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/benchmark_mps_limits.py \
  --outdir bench/runs/mps-limits \
  --profile standard --dmax 64 --eps 1e-10 --timeout-seconds 30
```

Use repeatable `--case family:qubits:depth` arguments for targeted boundaries.
The complete custom case lists and commands are recorded in the manifests.
