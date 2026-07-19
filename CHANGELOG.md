# Changelog

All notable user-facing changes are recorded here. MettleQ follows semantic
versioning once the first public release is tagged.

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
