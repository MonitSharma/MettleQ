# CPU-only MPS benchmark on Apple M3 Pro

This report is deliberately scoped to the available machine: an Apple M3 Pro
MacBook Pro with 12 CPU cores and 36 GB RAM. It is not an M1/M2/M4 result and
does not support a universal Apple-Silicon claim.

## Protocol

Both campaigns use fresh Python subprocesses for every circuit case, rotate
implementation order on every repeat, run two warmups, and report the median
of five post-warmup repeats. MettleQ uses NumPy CPU tensors, SciPy/NumPy CPU
SVD/QR, `dmax=64`, `eps=1e-10`, and lookahead routing. Aer uses Qiskit Aer
`matrix_product_state` on `device=CPU` with the same truncation settings.

The machine was on AC power with no intentional foreground workload. Thermal
state was recorded as idle before the run; no external thermal instrument was
used. The recorded software versions were macOS 26.5.2, Python 3.13.2,
NumPy 2.4.6, SciPy 1.18.0, Qiskit 2.5.0, and Qiskit Aer 0.17.2.

Estimator comparison:

```sh
PYTHONPATH=src .venv/bin/python tools/benchmark_mps_phase8.py \
  --outdir bench/runs/mps-m3pro-aer-extended \
  --profile standard --dmax 64 --eps 1e-10 --svd-driver auto \
  --routing-lookahead 8 --warmups 2 --repeats 5 \
  --machine-note "M3 Pro, AC power, no intentional foreground workload" \
  --thermal-state "idle before run; no thermal measurement instrumented"
```

Sampling comparison:

```sh
PYTHONPATH=src .venv/bin/python tools/benchmark_mps_sampling.py \
  --outdir bench/runs/mps-m3pro-sampling \
  --profile standard --shots 4096 --seed 7919 \
  --warmups 2 --repeats 5
```

The sampling campaign measures the end-to-end final-measurement contract and
reports the exact `q0=1` probability error only for cases small enough for an
exact statevector reference. That error is sampling error when the MPS is not
truncated; it is not presented as a deterministic simulator error bound.

## Estimator results

The Aer/MettleQ column is Aer median runtime divided by MettleQ median runtime;
below 1 means Aer is faster, above 1 means MettleQ is faster.

The table below is the pre-optimization baseline retained for comparison.

| Workload | MettleQ CPU ms | Aer CPU MPS ms | Aer/MettleQ | MettleQ swaps |
| --- | ---: | ---: | ---: | ---: |
| GHZ chain, 1000q d1 | 148.27 | 45.75 | 0.31 | 0 |
| Line brickwork, 100q d8 | 45.93 | 21.80 | 0.47 | 0 |
| Ring brickwork, 50q d2 | 11.89 | 3.95 | 0.33 | 96 |
| 2-D grid, 36q d2 | 134.95 | 319.21 | 2.37 | 300 |
| Rainbow, 32q d1 | 390.68 | 2.68 | 0.007 | 480 |
| Random long-range, 32q d1 | 143.81 | 2.37 | 0.016 | 234 |
| All-to-all, 20q d1 | 477.67 | 18.89 | 0.040 | 384 |

The result is workload-dependent. Aer is substantially faster on the peaked
long-range/random cases, while MettleQ is faster on the 36-qubit grid. The
benchmark does not justify routing every workload to either implementation.

### Post mapping-aware/native-core run

The persisted source data and updated comparison figure are in
[`assets/benchmarks-frozen/fork-m3pro-20260809-mps-native-core/`](../../assets/benchmarks-frozen/fork-m3pro-20260809-mps-native-core/).
The same figure is also used by the main README at
[`assets/perf-charts/mps_optimization_comparison.png`](../../assets/perf-charts/mps_optimization_comparison.png).

After removing the unconditional final logical-order restore and enabling the
optional Accelerate native core, the same longer estimator campaign was
repeated on the same M3 Pro with two warmups and five post-warmup repeats:

| Workload | MettleQ routed ms | MettleQ restore ms | Aer CPU MPS ms | Aer/MettleQ | Routed swaps |
| --- | ---: | ---: | ---: | ---: | ---: |
| GHZ chain, 1000q d1 | 106.18 | 106.25 | 71.90 | 0.68 | 0 |
| Line brickwork, 100q d8 | 44.19 | 45.68 | 29.69 | 0.67 | 0 |
| Ring brickwork, 50q d2 | 8.32 | 12.46 | 6.17 | 0.74 | 48 |
| 2-D grid, 36q d2 | 133.00 | 131.79 | 424.34 | 3.19 | 300 |
| Rainbow, 32q d1 | 7.73 | 550.08 | 3.45 | 0.45 | 240 |
| Random long-range, 32q d1 | 5.07 | 83.55 | 3.12 | 0.62 | 121 |
| All-to-all, 20q d1 | 23.58 | 143.20 | 24.76 | 1.05 | 249 |

The routing change is largest for rainbow and long-range circuits: their
logical-to-physical mapping is preserved across the circuit, and readout,
sampling, and product observables now translate that mapping instead of
forcing every qubit back into logical order. The planner uses persistent
routing only when it saves more than 25% of the naive per-gate restore swaps;
the grid therefore keeps restore-per-gate routing to avoid paying a bond-growth
cost for a small swap reduction. The native core now also performs the
two-site merge, specialized gate update, Accelerate SVD/truncation, and
readout contraction. Aer remains faster on most cases because its complete
MPS executor and tensor storage are native C++; MettleQ is within about 10%
on this all-to-all case and still wins the grid workload.

## Sampling results

| Workload | MettleQ CPU ms | Aer CPU MPS ms | Aer/MettleQ | Max exact `q0=1` error |
| --- | ---: | ---: | ---: | ---: |
| GHZ chain, 32q d1 | 28.09 | 82.12 | 0.34 | n/a |
| Line brickwork, 24q d8 | 76.52 | 69.15 | 1.11 | n/a |
| Random long-range, 24q d2 | 3507.76 | 96.74 | 36.26 | n/a |
| Rainbow, 24q d1 | 3592.02 | 42.09 | 85.33 | n/a |
| All-to-all, 16q d1 | 1582.29 | 63.38 | 24.96 | 0.0120 |
| VQE-style layered ansatz, 20q d4 | 204.97 | 96.15 | 2.13 | n/a |

The long-range sampling results identify routing/swap work as the dominant
MettleQ limitation for these circuits. The all-to-all exact small-case error
is consistent with finite-shot sampling at 4096 shots and is not a reason to
loosen the configured accuracy policy.

## What changed

The CPU-only path now keeps site tensors in NumPy, uses the recoverable
SciPy/NumPy CPU SVD fallback ladder and SciPy QR, applies a measured direct
single-qubit update above the explicit M3-Pro threshold of 2048, and records
SVD/contraction/routing/bond/norm diagnostics. No GPU execution path is used
for MPS; statevector GPU support remains a separate backend.

The MPS readout contract is mapping-aware: `expectation_product`, dense
observables, probabilities, samples, and `to_statevector()` all account for
the current logical-to-physical layout. Native expectation, marginal,
statevector, and conditional-sampling paths receive that mapping directly.
The CPU path also has
specialized updates for diagonal gates, CNOT, RZZ/ZZ, SWAP, and product-state
initialization. Circuit-level SWAP exchanges logical wire labels directly,
without an unnecessary tensor SVD.

Routing-plan generation and the two-site CPU core are available through an
optional C++ native extension with deterministic Python fallbacks. The
extension is built by the normal source/wheel build; source checkouts without
a compiler continue to work and report `routing_planner="python"` in
diagnostics. The native layer now owns merge/update/SVD/split, canonical-center
QR, and mapping-aware readout; circuit dispatch, logical-map ownership, and
route-plan state remain a thin Python control layer for compatibility.

The new adjacent-swap fast path exchanges the two physical tensor legs by
transpose before the required split, avoiding a dense 4×4 SWAP contraction.
Compared with the preceding extended run, fresh post-change medians improved
by approximately 1–2% on the ring, random, all-to-all, and grid cases; the
optimization is intentionally modest because SVD time remains dominant.

The threshold sweep used 0, 512, 1024, 2048, 4096, and 8192 over eight M3 Pro
cases with five repeats and exact parity on the small cases. All settings
passed parity; aggregate medians were close, so 2048 remains the conservative
documented threshold rather than overfitting to one noisy run. Re-run it with:

```sh
PYTHONPATH=src .venv/bin/python tools/mps_opt_harness.py \
  --out bench/local/mps-threshold-2048.json \
  --repeats 5 --single-gate-threshold 2048
```

### Thread-count sweep

Thread counts were tested in fresh subprocesses so BLAS/OpenMP settings could
not leak between runs:

```sh
PYTHONPATH=src .venv/bin/python tools/benchmark_mps_cpu_threads.py \
  --out bench/local/mps-thread-scaling.json --repeats 5 \
  --thread-count 1 --thread-count 2 --thread-count 4 --thread-count 6 \
  --case rainbow:32:1 --case random_long_range:32:1 \
  --case all_to_all:20:1 --case grid_2d:36:2
```

Median MettleQ/Aer runtimes in milliseconds were:

| Workload | MettleQ 1/2/4/6 threads | Aer 1/2/4/6 threads |
| --- | ---: | ---: |
| Rainbow, 32q d1 | 8.11 / 7.88 / 7.95 / 7.65 | 2.76 / 2.77 / 2.67 / 2.73 |
| Random long-range, 32q d1 | 5.19 / 4.99 / 5.14 / 5.04 | 2.62 / 2.47 / 2.50 / 2.50 |
| All-to-all, 20q d1 | 23.69 / 23.11 / 23.12 / 23.08 | 23.97 / 23.81 / 23.45 / 23.37 |
| 2-D grid, 36q d2 | 129.74 / 129.62 / 129.27 / 129.61 | 427.24 / 392.35 / 379.89 / 375.79 |

These small-kernel workloads are mostly sequential on this machine. One or
two threads are the stable default; four threads hurt the long-range and
all-to-all cases, while the grid case benefits modestly from more Aer threads.
MettleQ should therefore default to one CPU thread, matching Aer's documented
CPU-MPS default; larger counts should be opt-in and validated against the
target circuit.

## Next optimization targets

1. Move route-plan consumption, logical-map ownership, and circuit dispatch
   behind the native core if closing the remaining Aer gap is a priority.
2. Keep SVD driver selection evidence-based. On this M3 Pro, Accelerate
   `cgesdd` is competitive for long-range cases but not uniformly faster than
   SciPy for every small grid split.
3. Add a workload-aware diagnostic/fallback decision at the integration layer
   only when Aer is available and approximation is explicitly allowed. Do not
   silently replace MettleQ or claim a universal crossover.
4. Repeat correctness and long norm/probability stability tests after any
   routing or truncation change.
