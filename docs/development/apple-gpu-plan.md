# Apple GPU engineering log

## 2026-07-15 — Step 5: native Qiskit and PennyLane execution

### Shared compatibility boundary

Qupertino now exposes its exact statevector engine through a Qiskit
`BackendV2` and a registered PennyLane `qupertino` device. Both adapters emit
the same validated operation dictionaries used by the direct API, so
statevector preflight, capability-gated Metal dispatch, checkpoint policy, and
execution-plan evidence stay inside the core engine.

The Qiskit adapter uses a dynamic-width target (`num_qubits=None`) so
transpilation does not pad a small circuit to a simulator maximum. It reverses
the external register-to-engine wire mapping to preserve Qiskit's little-endian
statevector convention, reconstructs classical memory by classical-bit index,
returns a native synchronous job/result, and materializes the state only when
`return_statevector=True`. Its initial scope is bound unitary circuits with
final measurements; unsupported dynamic semantics raise `QiskitError`.

The PennyLane adapter is discoverable through `qml.device("qupertino",
wires=...)`. It preserves declared wire order, uses PennyLane preprocessing to
decompose common gates, supports analytic and finite-shot native measurements,
shot vectors, arbitrary wire labels, and framework-managed parameter-shift
gradients. Local and Pauli-sentence observables are evaluated without a dense
full-register observable; wide non-Pauli matrices are refused.

Eight integration tests cover reference state ordering, classical-bit mapping,
dynamic-width transpilation, explicit mid-circuit rejection, plugin discovery,
wire/marginal order, gradients, shot vectors, and a ten-wire Pauli sentence.
The complete repository suite passes 316 tests with the three existing
third-party deprecation warnings.

### Exact-commit SDK evidence

Engine commit `0d0105245384bcda8a1117e9c293d2dfd5b1f8dd` ran a 20-qubit,
four-step TFIM-style circuit on an Apple M3 Pro with MLX 0.32.0, Qiskit 2.5.0,
Aer 0.17.2, and PennyLane 0.45.1. Each pair used one warmup and seven repeats,
reversing implementation order on alternating repeats. Timing includes SDK
translation, simulation, synchronization, and the requested result.

| Native SDK contract | Qupertino median | CPU reference median | Speedup | Error |
| --- | ---: | ---: | ---: | ---: |
| Qiskit full statevector | 10.10 ms | Aer 74.65 ms | 7.39× | `1.287e-8` max amplitude |
| PennyLane local `⟨Z⟩` | 18.31 ms | `default.qubit` 708.70 ms | 38.70× | `4.657e-9` absolute expectation |

The SDK result contracts differ and are not compared against each other. These
are scoped single-machine results rather than universal Apple-Silicon claims.
Raw rows, summary, thresholds, chart, and exact manifest are frozen under
[`assets/benchmarks-frozen/fork-m3pro-20260715-step5-sdk/`](../../assets/benchmarks-frozen/fork-m3pro-20260715-step5-sdk/).

## 2026-07-15 — Step 4: statevector preflight and cross-workload policy

### Pre-allocation trust boundary

The exact statevector path now computes its complex64 allocation lower bound
before constructing an MLX array. The shared report compares one state against
the device's maximum buffer length and the minimum input-plus-output pair
against the recommended working set. If either reported limit is already too
small, direct simulator and `Device` construction raise
`StatevectorMemoryError` before allocation. Missing limits produce an explicit
unverified decision instead of an invented guarantee.

The check is deliberately one-sided: passing proves only that the reported
lower bounds fit. Additional lazy intermediates, lookup buffers, allocator
cache, and other processes are listed as unmodeled costs. A false-like default
keeps the unsafe override disabled. `MLXQ_ALLOW_UNSAFE_STATEVECTOR=1` or the
boolean constructor argument can bypass a refusal, but the failure reasons,
override source, and original device limits remain observable.

State initialization no longer creates a Python list with `2**n` boxed complex
objects. It constructs the zero state on the selected MLX device, avoiding a
large unreported host allocation before simulation begins. Execution-plan
schema version 4 embeds the allocation preflight separately from the selected
custom-launch traffic estimate, which is explicitly not described as an
allocator peak.

### Cross-workload decision protocol

`tools/memory_policy_campaign.py` evaluates TFIM, QFT, QAOA, QCBM,
Heisenberg, and SU(2) circuits from 22 through 27 qubits. Fully lazy execution
is compared with two prospective state-size-aware policies:

- balanced: `max(256 MiB, 4 * state bytes)`;
- minimum-memory: `max(128 MiB, 2 * state bytes)`.

Every workload/qubit cell runs in a fresh Python and MLX process so allocator
and compiler state from a previous size cannot create a false peak floor.
Policy order rotates across repeats, qubit counts, and workloads. Every row
captures synchronized time, allocator peak, predicted and observed checkpoint
counts, predicted launch traffic, and the preflight decision. Full-state
validation uses pure MLX through 24 qubits and the identical fully lazy Metal
kernels above 24 qubits, with both maximum amplitude and norm error reported.
This larger-size comparison isolates checkpoint scheduling correctness; it is
not represented as an independent validation of the Metal kernel algebra.

```bash
PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/memory_policy_campaign.py \
  --outdir bench/runs/memory_policy_current \
  --qubits 22 23 24 25 26 27 --repeats 3 --warmups 1
```

### Exact-commit results and decision

Engine commit `83940c017e372b08a236eca8dbb797e7edb50f1b` produced 324
measured timing rows across 36 isolated cells. The review pairs each candidate
with fully lazy execution inside the same repeat before aggregation:

| Policy | Median peak reduction | Minimum cell reduction | Paired geometric-mean runtime | Paired cells faster | Worst paired cell |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced | 71.79% | 50.00% | 16.46% faster | 35 / 36 | 7.59% slower |
| Minimum-memory | 77.50% | 58.33% | 14.64% faster | 36 / 36 | 1.48% faster |

The balanced outlier was 27-qubit Heisenberg. That cell exhibited material
session drift across its three rotations; its paired median was 7.59% slower
and remains a named guardrail. No minimum-memory cell was slower by its paired
median. Neither policy increased the per-cell median allocator peak.

Predicted and observed checkpoint counts agreed for all 324 executions. The
maximum full-state amplitude error was `1.0289356850989861e-6`, below the
`5e-6` threshold. Maximum norm error was `6.9141387939453125e-6`, below the
separate `1e-5` reduction tolerance. No unsafe preflight override was used.
The engine commit passed 306 tests; the reviewed publication tree adds two
campaign-analysis regression cases and passes 308 tests, with three unchanged
third-party deprecation warnings.

The measured M3 Pro reports a 21.06 GiB maximum buffer and 28.08 GiB
recommended working set. The preflight lower bounds permit 30 qubits but
refuse 31 before allocation: a 16 GiB state requires at least 32 GiB for one
out-of-place operation. This is a lower-bound decision, not a claim that an
arbitrary 30-qubit lazy graph fits.

Both formulas qualify for wider device testing, with minimum-memory providing
the stronger peak result and no slower paired cell. Checkpointing remains
opt-in because evidence from one M3 Pro cannot justify a universal default for
M1, M2, M3, and M4 systems or different unified-memory sizes. The complete
raw rows, reviewed aggregates, validation, manifests, preflight reports, and
chart are frozen under
[`assets/benchmarks-frozen/fork-m3pro-20260715-step4/`](../../assets/benchmarks-frozen/fork-m3pro-20260715-step4/).

## 2026-07-15 — Step 3: intra-layer Metal streaming

### Measured cause and implementation

The Step 2 25-qubit crossover showed that evaluation only between logical
fused operations could not reduce the approximately 3 GiB TFIM peak. Isolated
measurements identified the cause: a uniform 25-qubit H or RX layer is one
logical operation but 13 out-of-place custom Metal launches. Each lazy layer
peaked at exactly 3,072 MiB, while a one-launch ZZ layer peaked at 512 MiB.

Step 3 retains the same operation order and Metal kernels. When the existing
opt-in checkpoint budget is configured, uniform U2, RX, per-qubit U2, XX, and
YY wrappers expose a safe boundary after each custom launch. The device
evaluates only when the next launch would exceed the scheduled pass count. One
custom launch is never split, and an unset budget preserves fully lazy
execution.

Execution-plan schema version 3 reports whether a family supports intra-layer
streaming, the maximum pending passes per streamed chunk, predicted and actual
within-layer launch indices, evaluated traffic, timing, and allocator
snapshots. Runtime launch accounting must equal the planner's expected count
or execution raises an explicit error. Four new parameterized cases cover
uniform U2, per-qubit U2, XX, and YY streaming against pure MLX. The complete
suite passed: 302 tests with three existing third-party warnings.

### Exact-commit 25-qubit crossover

Commit `a89bbd0460cf06b83bbbd7b7580a30f359220062` ran the 319-operation,
six-step TFIM workload with one warmup and seven rotating repeats per arm:

```bash
PYTHONPATH=src .venv/bin/python tools/checkpoint_sweep.py \
  --outdir bench/runs/checkpoint_sweep_20260715_m3pro_a89bbd0_n25_r7 \
  --qubits 25 --steps 6 --repeats 7 --warmups 1 \
  --budgets-mib 4096 2048 1024 512 256
```

| Policy | Intra/inter checkpoints | Median peak | Peak reduction | Median time | Runtime change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fully lazy | 0 / 0 | 3,072 MiB | — | 475.468 ms | — |
| 4 GiB | 7 / 6 | 2,304 MiB | 25.0% | 452.044 ms | 4.9% faster |
| 2 GiB | 21 / 6 | 1,536 MiB | 50.0% | 439.981 ms | 7.5% faster |
| **1 GiB** | **42 / 6** | **1,024 MiB** | **66.7%** | **443.556 ms** | **6.7% faster** |
| 512 MiB | 84 / 12 | 768 MiB | 75.0% | 454.365 ms | 4.4% faster |
| 256 MiB | 84 / 13 | 768 MiB | 75.0% | 453.889 ms | 4.5% faster |

All 42 measured budget/repeat cells had identical predicted and observed
checkpoint counts. Every arm matched the pure-MLX statevector within
`1.4081562582646256e-09` maximum amplitude error. The 1 GiB arm is the
balanced measured point; 512 MiB is the lowest measured peak. The 256 MiB arm
cannot reduce peak further because one out-of-place launch is the remaining
floor.

The seven-repeat 20-qubit continuity run preserved the Step 2 behavior. The
256 MiB arm peaked at 96 MiB and ran 16.9% faster than lazy. Streaming at 128
MiB reduced peak to 72 MiB, versus 88 MiB in the boundary-only Step 2 run. All
20-qubit arms stayed within `5.155240678789141e-09` of pure MLX.

### Fully-lazy regression guardrail

The default path was compared with Step 2 commit `e5d9577` at 20 qubits in
four independent five-repeat, 29-workload campaigns ordered Step 3 / Step 2 /
Step 2 / Step 3. Each revision's per-workload value is the geometric mean of
its two campaign means.

Step 3 Metal time was 0.43% faster at the median and 0.45% slower by geometric
mean; 24 of 29 rows were within ±3%. Pure MLX was 0.06% faster at the median.
The largest relative slower rows were phase-estimation-inexact (+19.4%, about
3.1 to 3.7 ms) and GHZ (+13.2%, about 1.6 to 1.8 ms), where sub-millisecond
movement creates large percentages. Quantum walk was +3.04%; every other row
was within the band. This supports no broad fully-lazy regression, while
retaining the two short rows as future noise/performance guardrails.

The complete raw campaigns, manifests, comparison, chart, and compact summary
are frozen under
[`assets/benchmarks-frozen/fork-m3pro-20260715-step3/`](../../assets/benchmarks-frozen/fork-m3pro-20260715-step3/).

## 2026-07-15 — refreshed 25-qubit evidence at Step 2 commit

### Exact-commit publication sweep

After Step 2 was committed as `e5d9577301fa506a6f130e81ccd9968cf1a97500`,
the full 29-workload suite ran at 25 qubits on the Apple M3 Pro with macOS
26.5.2, Python 3.13.2, and MLX 0.32.0. Each arm received one warmup and ten
paired measured repeats; pure MLX and custom Metal alternated first position
inside every repeat. Checkpointing and dense ablation were disabled. The result
contains 290 raw rows.

```bash
unset MLXQ_METAL_CHECKPOINT_BUDGET_MB MLXQ_DENSE_ONLY
PYTHONPATH=src caffeinate -i .venv/bin/python tools/shader_suite_sweep.py \
  --outdir bench/runs/shader_sweep_20260715_m3pro_e5d9577_n25_r10 \
  --qubits 25 --repeats 10
```

The median paired pure-MLX/Metal ratio was 10.1689× and the geometric mean was
9.0537×. Metal reached at least 1.1× on 26 of 29 workloads, at least 4× on 25,
and at least 10× on 19. Long-range Ising was highest at 32.6462×. Amplitude
estimation, W state, and ladder Heisenberg remained at effective parity.

Against the previous five-repeat M3 Pro campaign at `a73436e`, a ±0.25× band
classified five ratios as higher, 22 as unchanged, and two as lower. The median
ratio delta was +0.0245×. Absolute time drift affected both paths in the same
direction: pure MLX was 3.93% slower and Metal 3.39% slower by geometric mean.
The refreshed run therefore supports stable relative acceleration, but is not
a controlled attribution test for Step 2.

### 25-qubit checkpoint crossover and boundary

A separate six-step TFIM run tested fully lazy execution and 16, 12, 8, 6,
and 4 GiB scheduling budgets with one warmup and seven rotating repeats per
arm:

```bash
PYTHONPATH=src .venv/bin/python tools/checkpoint_sweep.py \
  --outdir bench/runs/checkpoint_sweep_20260715_m3pro_e5d9577_n25_r7 \
  --qubits 25 --steps 6 --repeats 7 --warmups 1 \
  --budgets-mib 16384 12288 8192 6144 4096
```

| Policy | Checkpoints | Median peak | Median runtime | Runtime change |
| --- | ---: | ---: | ---: | ---: |
| Fully lazy | 0 | 3.00 GiB | 468.204 ms | — |
| 16 GiB budget | 3 | 3.00 GiB | 470.111 ms | +0.41% |
| 12 GiB budget | 6 | 3.00 GiB | 470.324 ms | +0.45% |
| 8 GiB budget | 6 | 3.00 GiB | 471.966 ms | +0.80% |
| 6 GiB budget | 13 | 3.00 GiB | 475.464 ms | +1.55% |
| 4 GiB budget | 13 | 3.00 GiB | 473.623 ms | +1.16% |

Every arm matched pure MLX within `1.4081562582646256e-09` maximum amplitude
error, and predicted/observed checkpoint counts agreed. At this Step 2
revision, safe fused-layer boundary checkpoints did not materially lower the
25-qubit peak because one logical all-qubit layer retained 13 launches. This
negative result motivated the intra-layer streaming implemented and measured
in Step 3 above.

The reviewed data, manifests, comparisons, and plots are frozen under
[`assets/benchmarks-frozen/fork-m3pro-20260715/`](../../assets/benchmarks-frozen/fork-m3pro-20260715/).

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
