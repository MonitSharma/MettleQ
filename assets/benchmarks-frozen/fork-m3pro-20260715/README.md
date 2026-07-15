# Apple M3 Pro 25-qubit evidence — 2026-07-15

This reviewed bundle records the full 29-workload benchmark at private-fork
commit `e5d9577301fa506a6f130e81ccd9968cf1a97500`. It was measured on an Apple
M3 Pro running macOS 26.5.2, Python 3.13.2, and MLX 0.32.0.

## Main paired sweep

Every workload used one warmup per arm followed by ten paired repeats. Pure MLX
and custom Metal alternated first position inside each repeat. Both checkpointing
and the dense-ablation mode were disabled. The bundle contains 290 raw timing
rows and a 29-row summary.

```bash
env -u MLXQ_METAL_CHECKPOINT_BUDGET_MB -u MLXQ_DENSE_ONLY \
PYTHONPATH=src caffeinate -i .venv/bin/python tools/shader_suite_sweep.py \
  --outdir bench/runs/shader_sweep_20260715_m3pro_e5d9577_n25_r10 \
  --qubits 25 --repeats 10
```

The paired pure-MLX/Metal ratio had a 10.17× median and 9.05× geometric mean.
Metal reached at least 1.1× on 26 of 29 workloads, at least 4× on 25, and at
least 10× on 19. Long-range Ising was the maximum at 32.65×. Amplitude
estimation, W state, and ladder Heisenberg remained at effective parity.

Compared with the prior M3 Pro fork campaign at `a73436e` (five repeats), and
using a ±0.25× ratio band, five workloads improved, 22 were unchanged, and two
regressed. The median ratio moved by only +0.0245×. Both arms ran slightly
slower in this session (pure MLX +3.93% and Metal +3.39% by geometric mean), so
absolute times should not be interpreted as a controlled code-revision effect.

The historical upstream comparison uses ratios transcribed from the original
project's committed M1 Max chart. It is a cross-machine comparison of relative
acceleration by workload, not a wall-time or commit-to-commit A/B.

## 25-qubit checkpoint crossover

The `checkpoint_n25/` subdirectory records a separate seven-repeat, six-step
TFIM crossover for lazy evaluation and five checkpoint budgets. All policies
peaked at approximately 3.00 GiB. Budgeted execution added 0.4%–1.6% runtime
and did not materially lower peak allocation. All arms matched pure MLX within
`1.41e-9` maximum amplitude error.

This negative result defines the current boundary of Step 2: safe checkpoints
between fused layers reduce long-graph memory at 20 qubits, but cannot reduce a
25-qubit peak dominated by one fused all-qubit layer. That case needs
intra-layer memory reduction or buffer reuse; checkpointing should remain
disabled for this particular 25-qubit TFIM workload.

## Files

- `shader_sweep_raw.csv` — all 290 paired observations.
- `shader_sweep_summary.csv` — per-workload means and paired-ratio range.
- `sweep_manifest.json` and `evidence_summary.json` — environment, revision,
  protocol, and aggregate claims.
- `historical_vs_current.csv` and `comparison_summary.json` — upstream M1 Max
  chart versus this M3 Pro run.
- `previous_fork_comparison/` — prior M3 Pro run versus this refreshed run.
- `checkpoint_n25/` — raw data, summary, validation, and manifest for the
  25-qubit checkpoint crossover.
- `chart_*.png` — reviewed source charts copied into `assets/perf-charts/` for
  the main README.
