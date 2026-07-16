# MettleQ midpoint-MPO phase evidence

Frozen on an Apple M3 Pro using implementation commit `72582c7`.

- MettleQ D=512: 1252.92 s, expected peak 100/1000.
- MettleQ D=768: 958.84 s, expected peak 103/1000.
- Published [`p9solver`](https://github.com/alexgalda-m/peaked-mpo-solver)
  core D=512: 1184.16 s, expected peak 100/1000. The direct-core arm omits
  per-cycle CLI checkpoint I/O on both sides of the compute comparison.
- Bond convergence: **converged**, peak-fraction spread 0.003.
- Cutoff 1e-3: operational non-convergence (197/1885 gates after 501 s); no peak claim.
- Native GPU MPS: remains disabled; no resident or round-trip crossover was measured through D=128.

See `manifest.json` for contracts, environments, raw-run paths, failure arms, and interpretation limits.
