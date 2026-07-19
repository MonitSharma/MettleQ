# Making MettleQ competitive with CUDA-Q at large statevector widths

## Why the 28-qubit crossover happens

A complex64 28-qubit state contains 268,435,456 amplitudes and occupies 2 GiB.
Every current MLX custom kernel is out-of-place, so one full-state pass reads
and writes at least 4 GiB. At large widths, fixed launch overhead disappears
and the number and quality of those passes dominate.

The M3 Pro provides 150 GB/s unified-memory bandwidth. Even at an unattainable
100% of that figure, one 4 GiB pass has a 28.6 ms lower bound. Six-layer 28q
ring QAOA currently needs 42 passes, so its traffic-only lower bound is about
1.20 seconds before indexing, phases, register pressure, allocation, or
synchronization. The measured 2.43 seconds corresponds to roughly 74 GB/s of
algorithmic input/output traffic. CUDA-Q finishes the same circuit in 0.464
seconds on the RTX 3070. Matching that result on this M3 Pro therefore cannot
come from launch tuning alone: MettleQ must drastically reduce state passes.

CUDA-Q's NVIDIA backend uses cuStateVec, FP32 by default, runtime gate fusion,
and automatic diagonal-gate fusion. Its documented default single-GPU fusion
size on this GPU generation is four qubits. MettleQ now matches or exceeds
that fusion width for uniform single-qubit layers, but it still constructs
mostly whole-circuit semantic layers rather than CUDA-Q-style local space-time
blocks.

## What changed in this phase

MettleQ previously recognized the gate-decomposed QFT one stage at a time,
requiring 27 full-state passes at 28 qubits. It now recognizes the complete
forward transform and selects the existing radix-4 kernel, which executes two
dependent QFT stages per pass. The measured result is 14 passes and 877.90 ms,
2.21× faster than the matched 1,936.71 ms pre-change arm. The remaining CUDA-Q
gap on QFT fell from 5.69× to 2.58×.

## Required engineering sequence

1. **Local space-time gate fusion.** Build a dependency-DAG planner that packs
   arbitrary adjacent gates across multiple circuit layers into 4–6-qubit
   blocks, compiles one dense/sparse Metal function per block, and chooses the
   block width from measured occupancy. This is the main route to fewer passes
   for QAOA, TFIM, Grover, and inverse QFT.
2. **Native persistent Metal executor.** MLX custom kernels are necessarily
   out-of-place arrays. A small native Metal runtime can own two long-lived
   buffers, precompile pipelines, encode the entire circuit into fewer command
   buffers, and enable safe in-place kernels where thread ownership permits.
3. **Inverse/subregister radix fusion.** Extend the new full-QFT selection to
   inverse and subregister QFT, cutting phase-estimation passes.
4. **Counter-guided kernel calibration.** Capture achieved bandwidth,
   occupancy, register spills, SIMD utilization, and cache behavior for
   radix-8/16/32, threadgroups 64/128/256, contiguous transforms, and gathers.
   Select per Apple GPU family rather than globally.
5. **Gather-specific layouts.** GHZ's affine permutation is one logical launch
   but an irregular gather; its roughly 2× CUDA-Q gap is not caused by launch
   count. SIMD-coalesced permutation variants and native in-place cycle/block
   algorithms need measurement.
6. **Result-contract specialization.** Full-state comparisons are necessary
   evidence, but real SDK calls often request expectations, probabilities, or
   samples. Keep those reductions GPU-resident so neither backend pays an
   unnecessary 2 GiB host-facing result cost.

The realistic target is not to promise that an 18-core M3 Pro always beats an
RTX 3070 at its bandwidth-dominated ceiling. It is to make MettleQ the fastest
trustworthy Apple-native backend, minimize the hardware gap, and scale upward
automatically on higher-bandwidth M3/M4 Max and future Apple GPUs.
