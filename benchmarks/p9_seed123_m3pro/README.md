# P9 seed-123 MettleQ benchmark

This directory contains the clean-commit P9 reproduction used for the
classically-verifiable-problems submission.

## Result

- Circuit: `peaked_circuit_P9_Hqap_56x1917`
- MettleQ source commit: `e53a61a58c62acc88aed35c9eb1bb2dd47b46656`
- Hardware: Apple M3 Pro, 12-core CPU, 36 GB RAM
- CPU execution: Apple Accelerate BLAS, two threads
- Seed: `123`
- Samples: `1000`
- Tensor dtype: `complex64`
- Compression: SVD, cutoff `0.0006`, maximum bond `512`
- Consolidated work gates: `1885/1885`
- End-to-end runtime: `668.3189310829621` seconds
- Oracle match: `true`
- Peak sample count: `102/1000`

The expected 56-bit oracle is:

```text
01101110111001100000100000001010011100101101010111110111
```

The input QASM SHA-256 is:

```text
cff3496c45d9133c1f1693f1d3b0cf1fc2da338f13cd7b339db330a4762d0f35
```

## Reproduce

From the repository root, create a clean Python environment and install the
package:

```sh
python3 -m venv .venv-p9
. .venv-p9/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the exact configuration:

```sh
env VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 \
  METTLEQ_MPO_SVD_ISOLATION_MIN_ELEMENTS=131072 \
  python -m mettleq.midpoint_mpo \
  --qasm src/mettleq/datasets/peaked_circuit_P9_Hqap_56x1917.qasm \
  --output-dir benchmarks/p9_seed123_m3pro/reproduction \
  --shots 1000 \
  --expected-bitstring 01101110111001100000100000001010011100101101010111110111 \
  --max-bond 512 \
  --cutoff 0.0006 \
  --unswap-threshold 500000 \
  --center-ratio 0.5 \
  --max-unswap-iterations 20 \
  --seed 123 \
  --sabre-trials 90 \
  --post-sabre-trials 50 \
  --no-progress-limit 20 \
  --compression-method svd \
  --dtype complex64
```

The run is complete when its summary reports
`matches_expected_bitstring: true` and diagnostics show
`work_gates_consumed: 1885` and `total_work_gates: 1885`. The runtime will
vary by machine and software environment; the submitted 668.319 seconds is
the Apple M3 Pro measurement.

## Files

- `manifest.json`: environment, commit, QASM hash, command, and provenance.
- `summary.json`: runtime, oracle match, samples, and contraction diagnostics.
- `samples.tsv`: raw 1,000-shot output.
- `runtime_comparison.png`: upstream-versus-MettleQ timing comparison.
- `sampling_distribution.png`: top sampled bitstrings with the oracle marked.

The implementation uses `complex64` intentionally as an optimization variant;
it is not claimed to be bitwise-identical internally to complex128.
