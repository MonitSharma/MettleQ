# Step 3 Apple M3 Pro memory evidence — 2026-07-15

This reviewed bundle measures intra-layer Metal streaming at engine commit
`a89bbd0460cf06b83bbbd7b7580a30f359220062`. The machine was an Apple M3 Pro
running macOS 26.5.2, Python 3.13.2, and MLX 0.32.0. The unrelated local audit
Markdown file named in the manifests was untracked and excluded from every
commit.

## Why Step 3 exists

At Step 2, a 25-qubit uniform H or RX layer formed one logical fused operation
but retained 13 out-of-place Metal launches in one lazy graph. Each isolated
layer peaked at 3,072 MiB; a one-launch ZZ layer peaked at 512 MiB. Safe
checkpoints only between logical layers therefore could not lower the 25-qubit
floor.

Step 3 keeps the same kernels and operation order but allows the existing
opt-in checkpoint budget to evaluate between custom launches inside supported
multi-launch U2, RX, per-qubit U2, XX, and YY layers. It never splits one Metal
launch. Execution-plan schema 3 predicts and records every streamed chunk;
the default with no budget remains fully lazy.

## 25-qubit TFIM crossover

The six-step, 319-operation TFIM workload used one warmup and seven rotating
repeats per arm:

```bash
PYTHONPATH=src .venv/bin/python tools/checkpoint_sweep.py \
  --outdir bench/runs/checkpoint_sweep_20260715_m3pro_a89bbd0_n25_r7 \
  --qubits 25 --steps 6 --repeats 7 --warmups 1 \
  --budgets-mib 4096 2048 1024 512 256
```

| Policy | Median peak | Peak reduction | Median time | Runtime change | Checkpoints |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fully lazy | 3,072 MiB | — | 475.468 ms | — | 0 |
| 4 GiB | 2,304 MiB | 25.0% | 452.044 ms | 4.9% faster | 13 |
| 2 GiB | 1,536 MiB | 50.0% | 439.981 ms | 7.5% faster | 27 |
| **1 GiB** | **1,024 MiB** | **66.7%** | **443.556 ms** | **6.7% faster** | **48** |
| 512 MiB | 768 MiB | 75.0% | 454.365 ms | 4.4% faster | 96 |
| 256 MiB | 768 MiB | 75.0% | 453.889 ms | 4.5% faster | 97 |

The 1 GiB arm is the balanced measured policy: it cuts peak by two thirds and
keeps nearly all of the best observed runtime improvement. The 512 MiB arm is
the lowest measured peak; 256 MiB cannot go lower because one out-of-place
custom launch is the remaining floor. Every predicted checkpoint count matched
the observed count in all 42 measured arms. Every arm matched the pure-MLX
statevector within `1.4081562582646256e-09` maximum amplitude error.

## 20-qubit continuity and default-path guardrail

The equivalent seven-repeat 20-qubit crossover preserves the Step 2 result.
The 256 MiB arm peaked at 96 MiB and ran 16.9% faster than lazy. Intra-layer
streaming lets the 128 MiB arm reach 72 MiB, down from the earlier Step 2
measurement of 88 MiB. All arms remained within `5.16e-9` of pure MLX.

The fully-lazy code path was compared with Step 2 commit `e5d9577` in four
independent five-repeat, 29-workload campaigns ordered Step 3 / Step 2 / Step 2
/ Step 3. The per-revision value is the geometric mean of its two campaign
means. Step 3 Metal time was 0.43% faster at the median and 0.45% slower by
geometric mean; 24 of 29 workloads were within ±3%. The two largest slower
rows were 1–4 ms workloads with high relative noise. Pure MLX was flat at the
median. This supports no broad disabled/default-path regression, not a claim
that fully-lazy execution became faster.

## Bundle contents

- `checkpoint_n25/` and `checkpoint_n20/` contain raw rows, summaries,
  numerical validation, and exact-commit manifests.
- `guardrail_n20/{current_a,current_b,base_a,base_b}/` contains all four raw
  campaigns and manifests.
- `guardrail_n20/revision_comparison.csv` and `revision_summary.json` contain
  the drift-balanced calculation produced by `tools/compare_revision_campaigns.py`.
- `chart_checkpoint_n25.png` is generated from the tracked 25-qubit summary by
  `tools/plot_checkpoint_sweep.py`.
- `evidence_summary.json` provides the compact machine-readable result.
