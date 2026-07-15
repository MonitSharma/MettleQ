# Apple GPU engineering log

## 2026-07-15 — Step 2: memory-budgeted Metal graph evaluation

### Change and safety model

Long custom-Metal circuits previously built one fully lazy MLX graph and
evaluated only the final state. Step 2 adds an opt-in checkpoint controller
that estimates two full-state byte transfers (input plus output) for every
expected custom Metal launch. Before adding a fused operation that would cross
the configured budget, it evaluates the current state. If one fused operation
alone exceeds the budget, evaluation occurs immediately after that operation.
The controller never synchronizes inside a fused operation.

The default remains fully lazy. Users can set a positive MiB value through
`MLXQ_METAL_CHECKPOINT_BUDGET_MB` or pass an exact byte value through
`Device(..., metal_checkpoint_budget_bytes=...)`. Zero, false-like, empty, or
unset configuration disables checkpointing. Invalid, negative, non-finite, or
sub-byte values raise an explicit error.

Execution-plan schema version 2 reports the policy source, accounting basis,
budget, predicted checkpoint boundaries, actual checkpoint boundaries,
evaluated-pass and traffic estimates, evaluation durations, allocator
snapshots, and remaining pending passes. Predicted and observed schedules are
separate so runtime evidence cannot be confused with a planner claim.

Four new tests cover configuration validation, checkpoint/state parity,
boundary placement, oversized multi-launch fused layers, and inactive fallback
when Metal is unavailable. The full suite passed: 298 tests, with the same
three third-party deprecation warnings.

### Crossover measurement

Command:

```bash
PYTHONPATH=src .venv/bin/python tools/checkpoint_sweep.py \
  --outdir /tmp/qupertino-step2-checkpoints \
  --qubits 20 --steps 6 --repeats 9 --warmups 1 \
  --budgets-mib 512 384 256 192 128
```

The workload was the same 254-operation, six-step TFIM circuit used in Phase D.
The lazy arm and five budget arms rotated order every repeat. Every arm used a
fresh `Device`; the MLX allocation cache and peak counter were reset between
runs. Values are medians on the Apple M3 Pro with Python 3.13.2 and MLX 0.32.0.
The budget is the scheduling estimate described above, not an allocator cap.

| Policy | Predicted/observed checkpoints | Peak | Peak reduction | Total time | Runtime change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fully lazy | 0 / 0 | 616.0 MiB | — | 29.033 ms | — |
| 512 MiB | 2 / 2 | 264.0 MiB | 57.14% | 26.673 ms | −8.13% |
| 384 MiB | 3 / 3 | 184.0 MiB | 70.13% | 24.792 ms | −14.61% |
| 256 MiB | 6 / 6 | 96.0 MiB | 84.42% | 22.712 ms | −21.77% |
| 192 MiB | 6 / 6 | 96.0 MiB | 84.42% | 21.808 ms | −24.89% |
| 128 MiB | 13 / 13 | 88.0 MiB | 85.71% | 26.517 ms | −8.67% |

The 192 MiB arm was marginally fastest in this session but did not improve peak
memory beyond 256 MiB. The 256 MiB arm is therefore the conservative measured
operating point: it stays farther from the cliff where every multi-launch layer
is forced to synchronize while retaining the same measured peak.

All six arms had the same maximum TFIM amplitude error versus pure MLX:
`5.155240678789141e-09`. A separate 256 MiB validation produced zero error for
QFT and the affine workload. This is well inside the `5e-6` acceptance limit.

### Disabled-path guardrail

The existing 29-workload sweep compared the checkpoint-disabled working tree
with Step 1 commit `ee46a8e` at 20 qubits. Four seven-repeat campaigns ran in
working-tree/base/base/working-tree (A–B–B–A) order. The drift-balanced median
Metal change was −0.02%; 26 of 29 workloads were within ±3%, and the geometric
mean moved +0.82%. One short 3–4 ms workload was noisy at +11.1%, while its pure
path was unchanged. This shows no broad disabled-path shift, but the short row
should remain a guardrail in the next full publication campaign.

### Outcome

The acceptance targets were met: peak memory fell by more than 50%, runtime did
not regress, numerical error stayed below `5e-6`, every checkpoint is reported,
and no fused layer is split. The feature remains opt-in until a broader
workload/size campaign supports an automatic budget policy.

## 2026-07-15 — Step 1: remove repeated capability-probe overhead

### Change

Metal dispatch used to construct the full runtime-capability report every time
the simulator asked whether a custom kernel was usable. Profiling isolated the
largest part of that check to package metadata lookup (about 0.20 ms per call),
even though the MLX version and physical Metal capabilities cannot change
during a Python process.

The dispatch path now caches only process-stable facts: platform identity,
Metal availability, the custom-kernel API, MLX version, device identity, and
device memory limits. Policy and safety decisions remain live on every call:
`MLXQ_METAL_KERNELS`, the selected MLX device, backend, dtype,
`MLXQ_DENSE_ONLY`, and requested qubit count. The full inspection report and
the fast boolean selector share these rules. A public
`clear_metal_capability_cache()` hook supports tests and deliberate re-probing.

Two regression tests prove that static probes are reused and that changing
policy, GPU/CPU selection, dense ablation, dtype, or circuit size still fails
closed immediately. The complete suite passed: 294 tests, with three existing
third-party deprecation warnings.

### Controlled A/B result

The A/B ran in one Python process on the same Apple M3 Pro with MLX 0.32.0 at
20 qubits. Each workload had one warmup per arm and 15 measured repeats; arm
order alternated each repeat. The legacy arm emulated the former behavior by
clearing the static cache before every full capability report. Values below
are medians. This isolates the dispatch-check change; it is not a replacement
for a full commit-to-commit benchmark campaign.

| Workload | Legacy uncached | Cached selector | Time saved | Speedup |
| --- | ---: | ---: | ---: | ---: |
| QFT | 5.987 ms | 5.672 ms | 0.315 ms | 1.056× |
| QAOA | 9.515 ms | 9.098 ms | 0.417 ms | 1.046× |
| TFIM Trotter (2nd) | 43.908 ms | 38.418 ms | 5.490 ms | 1.143× |
| Phase estimation | 7.185 ms | 6.404 ms | 0.781 ms | 1.122× |
| Grover | 6.120 ms | 5.693 ms | 0.427 ms | 1.075× |
| GHZ | 2.966 ms | 2.549 ms | 0.418 ms | 1.164× |

In a 5,000-call microbenchmark, median selector latency fell from 0.2139 ms to
0.001583 ms, a 135× reduction. End-to-end improvement depends on how often a
workload enters the selector; the measured six-workload range was 4.6%–16.4%.

### Balanced upstream revision check

The existing 29-workload sweep then compared upstream commit `2b99d30` with
the current Step 1 working tree at 20 qubits. Four independent seven-repeat
campaigns ran in current/upstream/upstream/current (A–B–B–A) order. Pure MLX
and Metal arms were interleaved inside every repeat. To reduce the effect of
session drift, the revision estimate is the geometric mean of the two
per-campaign medians for that revision (14 observations per revision and
workload).

Across all 29 workloads, current Metal latency was lower in 28. The median
change was 2.59% faster and the geometric-mean latency ratio was 0.9592
(4.08% lower). Sixteen workloads were within ±3%. VQE was the only slower row,
at 0.17%, and no workload was more than 5% slower.

The six workloads used in the earlier focused revision check all recovered:

| Workload | Upstream Metal | Step 1 Metal | Change |
| --- | ---: | ---: | ---: |
| QFT | 4.324 ms | 4.292 ms | 0.7% faster |
| QAOA | 9.812 ms | 8.871 ms | 9.6% faster |
| TFIM Trotter (2nd) | 42.296 ms | 40.811 ms | 3.5% faster |
| Phase estimation | 6.559 ms | 6.431 ms | 1.9% faster |
| Grover | 5.074 ms | 4.879 ms | 3.8% faster |
| GHZ | 2.063 ms | 1.787 ms | 13.4% faster |

This revision comparison verifies the overall working tree against upstream;
it cannot attribute every difference to the cache because the fork also
contains correctness and observability changes. The same-process legacy/cache
A/B above is the attribution test.

## 2026-07-14 — Phase D inspection and profiling

### Scope and outcome

This phase added observability and collected measurements; it did not add a
performance optimization. Custom Metal selection is now capability-probed,
overrideable, and recorded. A `Device` execution plan reports the selected MLX
device, backend, dtype, fusion passes, matched patterns, concrete kernels,
expected low-level launches, observed dispatches, fallback reasons, process
wrapper-cache state, 32-bit indexing limits, state-memory estimates, MLX
allocator counters, and synchronized evaluation status.

`MLXQ_METAL_KERNELS` accepts explicit on/off values and `auto`. Unset remains
off. A request fails closed when the process is not on Apple Silicon/Metal, the
default device is not the GPU, `mx.fast.metal_kernel` is unavailable, the
backend is not statevector, the dtype is not `complex64`, dense ablation is on,
or the requested qubits exceed the index/device-memory limit.

On the measured Apple M3 Pro with MLX 0.32.0, the default device was
`Device(gpu, 0)`. The shader index representation has a hard 31-qubit limit;
the device's maximum single-buffer limit also permits 31 complex64 qubits, but
the recommended working set permits only 30 qubits when one input plus one
output state are budgeted. This is a capability limit, not a claim that a
30-qubit workload will always fit: lazy intermediates and lookup tables add
memory.

### Measurement protocol

Command:

```bash
PYTHONPATH=src .venv/bin/python tools/profile_apple_gpu.py \
  --qubits 20 --repeats 9 --output /tmp/qupertino_phase_d_n20_timing.json
```

Each arm used a fresh `Device`; state initialization and circuit construction
were outside the timing window. `Device.execute` host graph construction was
run with `report=True` and timed separately from `Device.synchronize`, which
calls `mx.eval(final_state)`. Normal execution does not build a report unless
`report=True` or `MLXQ_EXECUTION_REPORT=1` is set, so the inspection feature
does not add this host work to the default fast path.
The first execution in the process was recorded separately, followed by nine
synchronized warm executions. Full-state `tolist` readback was timed after
evaluation and excluded from execution time. MLX allocator active, peak, and
cache counters were captured. Pure MLX was the numerical reference.

Cold timing means first execution in that process after clearing the MLX
allocator cache. MLX does not expose per-kernel compiler-cache hits, so
"first minus warm" is an upper-bound estimate of compile/cache setup, not a
direct compiler measurement. The arms were not randomized; that belongs in
the Phase E publication benchmark system.

The complex64 state at 20 qubits is 8 MiB; the minimum custom-kernel input plus
output expectation is 16 MiB.

| Workload | Gates | Matched Metal patterns | Expected launches | Pure warm | Metal warm | Metal eval | Host graph | Readback | First−warm eval | Allocator peak | Peak/minimum | Max amplitude error |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Affine | 22 | XOR affine permutation | 1 | 13.337 ms | 3.778 ms | 2.490 ms | 1.285 ms | 20.667 ms | 5.441 ms | 40.002 MiB | 2.50× | 0 |
| QFT | 210 | 19 QFT stages | 19 | 29.023 ms | 6.622 ms | 5.336 ms | 1.507 ms | 17.827 ms | 3.975 ms | 184.002 MiB | 11.50× | 0 |
| TFIM | 254 | 6 ZZ, 6 RX, 1 uniform-U2 layer | 76 | 88.124 ms | 12.198 ms | 9.409 ms | 2.967 ms | 18.060 ms | 14.349 ms | 664.004 MiB | 41.50× | 5.16e-9 |

The profiler's traffic estimate counts one full-state read and write per
expected custom launch. It reported approximately 6.7 GB/s for the one-launch
affine gather, 59.7 GB/s for QFT, and 135.5 GB/s for TFIM. These are
algorithmic traffic divided by synchronized time, not Metal hardware-counter
bandwidth measurements.

### Metal capture

MLX Metal capture requires the capture layer when Python starts:

```bash
MTL_CAPTURE_ENABLED=1 PYTHONPATH=src .venv/bin/python \
  tools/profile_apple_gpu.py --qubits 20 --repeats 7 \
  --capture-workload qft \
  --capture-output /tmp/qupertino_phase_d_qft_n20.gputrace \
  --output /tmp/qupertino_phase_d_n20_capture.json
```

The resulting Xcode Instruments bundle recorded one captured frame. The
capture-enabled process is deliberately separate from the timing evidence
because inserting the capture layer changed first-process overhead. Its
execution plan selected and synchronously evaluated `mlxq_qft_stage_gen` for
19 matched QFT stages.

### Top three measured bottlenecks

1. **Allocator/temporary pressure.** TFIM caused 664.004 MiB of MLX allocator
   peak/cache pressure for an 8 MiB state, 41.50× the minimum two-state
   expectation. This counter includes cached buffers and does not mean 83
   statevectors were simultaneously live, but it proves the long lazy shader
   graph allocates and retains far more storage than the state alone.
2. **CPU readback.** Full-state readback took 17.827–20.667 ms, exceeding all
   three custom-Metal warm execution times. SDK adapters must not materialize
   the statevector unless the caller requests it; counts, probabilities,
   observables, and samples need device-resident paths where beneficial.
3. **First-process setup.** TFIM's first synchronized evaluation carried a
   14.349 ms first-minus-warm penalty. This is an upper bound on shader
   compilation/cache setup, but it is material for notebook and SDK workloads
   that execute a circuit only once.

Within steady-state GPU execution, full-state pass count is the main measured
cost driver: TFIM required 76 launches and 1,275,068,416 bytes of estimated
input/output traffic, versus 19 launches for QFT and one for affine.

### Smallest proposed optimization PR

Add memory-budgeted evaluation checkpoints for long custom-kernel graphs,
without changing gate fusion or kernel algebra. The PR should:

1. Track pending custom statevector passes and their input/output byte estimate
   in the execution context.
2. Evaluate at a layer boundary only when a configurable pending-memory budget
   would be exceeded; report every checkpoint in the execution plan.
3. Keep the current lazy behavior as the fallback until measurements establish
   the crossover, and never insert checkpoints inside a fused layer.
4. Differential-test final complex amplitudes against pure MLX at a maximum
   error of 5e-6, with exact dispatch-pattern assertions.
5. Benchmark below and above the crossover with first/warm time, MLX allocator
   peak, launch count, and an Xcode Metal capture.

The acceptance target is a large reduction from TFIM's 664 MiB allocator peak
without a material warm-time regression. Buffer reuse inside or across custom
kernels is a later change if checkpoints alone cannot meet that target.
