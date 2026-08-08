# Configuration reference

MettleQ keeps simulator defaults conservative. Environment variables are
intended for reproducible experiments and should be recorded with benchmark
output rather than hidden in shell profiles.

| Variable | Purpose | Default |
| --- | --- | --- |
| `METTLEQ_BACKEND` | Select `sv` or `mps` for benchmark helpers | `sv` |
| `METTLEQ_BENCH_OUT_DIR` | Explicit output directory for benchmark artifacts | unset |
| `METTLEQ_MPS_PAIR_SWEEPS` | Enable pair-sweep MPS optimization | `0` |
| `METTLEQ_MPS_EARLY_STOP_BMAX` | Stop an MPS run at a bond cap | `0` (disabled) |
| `METTLEQ_MPS_STOP_ON_TRUNC` | Stop after the first truncation event | `0` |
| `METTLEQ_MPO_PYTHON` | Interpreter for the isolated midpoint-MPO worker | unset |
| `METTLEQ_QASM_LOCAL` | Directory containing local `.qasm` inputs | package data |
| `METTLEQ_MQTBENCH` | Checked-out optional MQTBench dataset directory | unset |
| `METTLEQ_SAVE_PLOTS` | Save optional VQE convergence plots | `0` |

`METTLEQ_MPO_PYTHON` executes a separate Python interpreter and must point to
an interpreter the caller trusts. The bundled midpoint-MPO requirement file is
for reproducibility, not a general-purpose runtime dependency set.
