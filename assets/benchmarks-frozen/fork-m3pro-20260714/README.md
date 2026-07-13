# Private-fork M3 Pro rerun — 2026-07-14

This directory freezes the evidence used by the root README to compare the
public upstream Qupertino repository with this private fork.

## Full 25-qubit sweep

- Tested fork commit: `a73436e3d425e7b6d9bebc015165d13b21ecea62`
- Host: Apple M3 Pro, macOS 26.5.2
- Runtime: Python 3.13.2, MLX 0.32.0
- Protocol: one warmup per arm, then five paired repeats alternating pure MLX
  and custom Metal for each of 29 workloads
- Command:

  ```bash
  PYTHONPATH=src .venv/bin/python tools/shader_suite_sweep.py \
    --outdir bench/runs/shader_sweep_20260714_m3pro_current_n25_r5 \
    --qubits 25 --repeats 5
  ```

The historical ratios in `historical_chart_ratios.csv` were transcribed from
the upstream repository's committed
`assets/perf-charts/chart_speedup_sweep.png`. They were measured on an M1 Max.
The comparison therefore evaluates the shape of the pure-MLX/Metal speedup, not
absolute runtime or a controlled code-revision improvement.

## Same-machine upstream/fork check

`same_machine_upstream_vs_fork.csv` is a focused revision comparison on the
same Apple M3 Pro. The upstream worktree was detached at
`2b99d30f98cbbbf3f1ac2ae93d16a2f6394d977d`; the fork worktree used
`a73436e3d425e7b6d9bebc015165d13b21ecea62`. Each arm used the same benchmark
functions at 20 qubits, one warmup, seven paired repeats, and final-state
synchronization. The table records medians.

This check shows that the fork's correctness and runtime-observability work did
not improve raw Metal latency yet. Repeated capability probing is a measured
host-side contributor and is the next optimization target.
