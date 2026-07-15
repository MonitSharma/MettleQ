# Step 5 native SDK evidence

This immutable bundle measures the first native Qiskit and PennyLane adapters
at engine commit `0d0105245384bcda8a1117e9c293d2dfd5b1f8dd` on an Apple M3
Pro. The run used a 20-qubit, four-step, open-chain TFIM-style circuit with one
warmup and seven repeats. Implementation order alternated on every repeat.

The Qiskit comparison requests a full statevector from Qupertino and Qiskit
Aer CPU. The PennyLane comparison requests an analytic local Pauli-Z
expectation from Qupertino and `default.qubit`. These are separate native SDK
contracts, so their absolute times and speedups must not be compared across
SDKs.

| SDK path | Qupertino median | Reference median | Reference / Qupertino | Validation |
| --- | ---: | ---: | ---: | --- |
| Qiskit full statevector | 10.10 ms | Aer CPU 74.65 ms | 7.39× | max amplitude error `1.287e-8` |
| PennyLane local expectation | 18.31 ms | `default.qubit` 708.70 ms | 38.70× | absolute expectation error `4.657e-9` |

The measurements include SDK preprocessing/translation, simulation,
synchronization, and requested result materialization. They establish a scoped
single-machine result, not a universal claim across Apple chips, circuit
families, Aer configurations, or PennyLane devices.

Reproduce from the repository root:

```bash
MLXQ_METAL_KERNELS=auto PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/benchmark_sdk_adapters.py \
  --outdir bench/runs/sdk-adapters \
  --qubits 20 --steps 4 --warmups 1 --repeats 7

PYTHONPATH=src .venv/bin/python tools/plot_sdk_adapters.py \
  --summary bench/runs/sdk-adapters/sdk_adapter_summary.json \
  --output bench/runs/sdk-adapters/sdk_adapter_timings.png
```

Files:

- `sdk_adapter_timings.csv`: all rotating-order raw timings;
- `sdk_adapter_summary.json`: medians, scoped speedups, error metrics, and
  validation thresholds;
- `sdk_adapter_manifest.json`: exact command, engine commit, versions, chip,
  MLX device, Metal status, and tracked-worktree state;
- `sdk_adapter_timings.png`: chart rendered only from the summary JSON.
- `test_summary.json`: full-suite result for the engine/publication tree.
