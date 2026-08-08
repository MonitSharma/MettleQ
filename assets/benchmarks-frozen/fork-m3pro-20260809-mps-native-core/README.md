# M3 Pro native CPU MPS benchmark

This artifact records the native CPU-only MPS run used to update the
development comparison plot. The machine was an Apple M3 Pro. MettleQ used
the optional C++/Accelerate core, `dmax=64`, `eps=1e-10`, routing lookahead 8,
two warmups, and five timed repeats. Aer used its CPU MPS backend.

- `mps_native_summary.csv`: seven-workload routed/restore/Aer medians.
- `mps_thread_scaling.csv`: fresh-process 1/2/4/6-thread sweep.
- `mps_native_comparison.png`: generated comparison figure.

These are local M3 Pro results, not a cross-chip Apple Silicon claim.
