# Changelog

All notable user-facing changes are recorded here. MettleQ follows semantic
versioning once the first public release is tagged.

## 0.3.0rc2 — Unreleased

- Make MLX imports genuinely lazy so headless macOS sessions report an
  actionable error instead of triggering duplicate nanobind registration.

## 0.3.0rc1 — Unreleased

- Marked MLX as an Apple Silicon macOS dependency and made drawing/QASM
  utilities usable on platforms without MLX installed.
- Replaced exact runtime dependency pins in published extras with compatible
  ranges; keep reproducible benchmark pins in dedicated requirement files.
- Added PEP 561 typing metadata, PEP 639 license metadata, sdist manifests,
  and installed-artifact validation hooks.
- Removed the legacy `qupertino` PennyLane entry-point name in favor of the
  package-owned `mettleq.compat` alias.
- Hardened source-checkout-only benchmark and midpoint-MPO paths and stopped
  library VQE calls from writing into the caller's current directory.

## Unreleased

- Imported the Windows/WSL RTX 3070 benchmark evidence without merging stale
  branch code.
- Added exact-width 15–28-qubit Apple Metal versus NVIDIA/Windows CPU evidence
  and documented the historical Aer MPS result-contract discontinuity.
- Added radix-16 RX execution and combined chain-phase/RX Metal passes for
  controlled-phase and ZZ-heavy schedules, reducing state traversals and
  expected launches without reordering dependent gates.
- Added a safe reproducible matched-width runner with independent Qiskit
  validation, allocator isolation, memory preflight, and environment manifests.
- Added complete forward-QFT recognition so SDK-decomposed QFT circuits use
  two-stage radix-4 Metal passes, improving the matched 28q QFT by 2.21×.
- Added full 15–28q MettleQ/CUDA-Q crossover tables, a focused chart with
  numbered qubit axes, and an evidence-backed competitiveness roadmap.
- Made the recommended SDK extra install Qiskit Aer and PennyLane Lightning.
- Added clean Qiskit and PennyLane quickstarts and wheel validation in CI.
- Completed distribution metadata for repository and package-index rendering.

## 0.2.0 — 2026-07-16

- Renamed the maintained fork to MettleQ while retaining upstream attribution
  and compatibility aliases.
- Added adaptive Qiskit and PennyLane execution, statevector memory preflight,
  native Metal dispatch, MPS trust controls, and midpoint-MPO support.
- Added paired SDK tutorials, frozen benchmark evidence, and the MettleQ
  technical report.
