# Step 4 Apple M3 Pro statevector and memory-policy evidence — 2026-07-15

This bundle freezes the cross-workload evidence for engine commit
`83940c017e372b08a236eca8dbb797e7edb50f1b`. The only untracked path recorded
by the exact-run manifest was the local technical-audit Markdown; it was not
part of the engine or campaign.

## What changed

The statevector simulator now checks one-state and two-state complex64 lower
bounds against the reported device buffer and recommended working-set limits
before allocation. State initialization stays on MLX and is materialized at
the reset boundary, avoiding both the former Python list and an initialization
expression retained in the later lazy graph. Execution-plan schema version 4
separates this preflight from custom-launch traffic estimates.

On this 36 GiB Apple M3 Pro, 22–30 qubits pass the reported lower-bound checks.
The 31-qubit request is refused before allocation: one state is 16 GiB and the
two-state lower bound is 32 GiB, above the 28.08 GiB recommended working set.
The complete 22–31 reports are in `statevector_preflight_m3pro.json`.

## Exact campaign

Six deterministic workloads—TFIM, QFT, QAOA, QCBM, Heisenberg, and SU(2)—ran
at every size from 22 through 27 qubits. Every workload/qubit cell used a fresh
Python and MLX process, one warmup per arm, and three rotating measured
repeats. This produced 36 isolated cells and 324 timing rows.

```bash
PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/memory_policy_campaign.py \
  --outdir bench/runs/memory_policy_20260715_m3pro_83940c0_q22_27_r3 \
  --qubits 22 23 24 25 26 27 \
  --workloads tfim qft qaoa qcbm heisenberg su2 \
  --repeats 3 --warmups 1 --pure-reference-max-qubits 24
```

The measured policies were:

- fully lazy: no checkpoints;
- balanced: `max(256 MiB, 4 * state bytes)`;
- minimum-memory: `max(128 MiB, 2 * state bytes)`.

## Reviewed aggregate

Runtime changes below are paired within repeat before aggregation. Peak values
are per-cell median reductions relative to fully lazy execution.

| Policy | Median peak reduction | Minimum cell reduction | Paired geometric-mean runtime | Paired cells faster | Worst paired cell |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced | 71.79% | 50.00% | 16.46% faster | 35 / 36 | 7.59% slower |
| Minimum-memory | 77.50% | 58.33% | 14.64% faster | 36 / 36 | 1.48% faster |

Balanced's single slower cell was 27-qubit Heisenberg. Its three repeats had
substantial session drift; the paired median was 7.59% slower and remains an
explicit guardrail rather than being hidden by separate arm medians.

Every one of the 324 measured executions had identical predicted and observed
checkpoint counts. Full-state validation used pure MLX through 24 qubits and
the identical fully lazy Metal kernels from 25 through 27 qubits. Maximum
amplitude error was `1.0289356850989861e-6` against a `5e-6` threshold;
maximum norm error was `6.9141387939453125e-6` against a `1e-5` threshold. The
larger-size reference validates checkpoint scheduling, not Metal algebra
independently; small-size unit tests and the pure-MLX tiers cover algebraic
parity.

## Decision

Both adaptive formulas qualify for further device testing on the measured M3
Pro. Minimum-memory is the stronger memory result and had no slower paired
cell; balanced delivered the larger paired geometric-mean speedup but retained
the 27-qubit Heisenberg outlier. Automatic checkpointing remains disabled by
default because one M3 Pro does not establish a trustworthy universal policy
for M1, M2, M3, and M4 memory sizes and GPU configurations. Users can reproduce
either formula through the existing explicit byte-budget API.

The reviewed CSVs were derived from the unchanged exact-run raw rows after the
review added paired-ratio fields and separate amplitude/norm acceptance gates.
The manifest preserves the exact engine commit and all 36 child manifests.

![Adaptive memory-policy evidence](memory_policy.png)

