# Apple GPU engineering log

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
