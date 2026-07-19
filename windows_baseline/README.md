# Windows/WSL NVIDIA baseline

This folder preserves the benchmark scripts and measurements added by commit
`ca9a1f5` on `windows-wsl-baseline`. Only this evidence folder was imported;
the maintained source, packaging, and documentation remain those from `main`.

## Systems and measurement contract

The Windows campaign ran in WSL2 Ubuntu on an NVIDIA RTX 3070 8 GB with
Python 3.11.13. Its manifests capture PennyLane 0.45.1, Lightning 0.45.0,
NumPy 2.4.6, and Linux `6.18.33.2-microsoft-standard-WSL2`. They do not capture
the Windows CPU, RAM, CUDA-Q/Qiskit/Aer versions, NVIDIA driver, CUDA version,
power mode, or thermal state, so this is framework evidence—not a controlled
device-efficiency comparison.

The matched Apple campaign ran on a 14-inch MacBook Pro with an M3 Pro
(12 CPU cores, 18 GPU cores), 36 GB unified memory, macOS 26.5.2, Python
3.13.2, MLX 0.32.0, and MettleQ commit `3b445a1`. Each workload ran in a fresh
process so a 28-qubit MLX allocation could not contaminate the next workload's
memory gate. The preflight required the two-state lower bound plus 6 GiB of
available reserve; every 28-qubit cell passed.

Both campaigns use the same six circuit families at 15, 20, 24, 26, and 28
qubits, one warm-up, three measured repetitions, and a complete final-state
result:

| Workload | Circuit contract |
| --- | --- |
| `qft` | Explicit H/controlled-phase ladder without final swaps |
| `qaoa_ring` | Six ring layers with identical gamma/beta schedule |
| `ghz` | Hadamard followed by a nearest-neighbor CNOT chain |
| `grover_proxy` | Uniform initialization and one CZ-chain diffusion proxy |
| `phase_estimation` | Base phase 0.4 and the same inverse-QFT ladder |
| `tfim_trotter` | 20 open-boundary ZZ/RX Trotter steps |

MettleQ additionally checks norm at every width and compares the complete state
against Qiskit `Statevector` through 20 qubits after global-phase alignment.
Every accuracy check passed the `5e-6` threshold; the largest recorded checked
amplitude error is below it. TFIM's 28-qubit norm is 0.9999736 after 20
complex64 steps.

## Matched-width result

The 28-qubit endpoint is:

| Workload | MettleQ M3 Pro Metal | CUDA-Q RTX 3070 | Lightning GPU RTX 3070 | Aer CPU WSL | Lightning CPU WSL |
| --- | ---: | ---: | ---: | ---: | ---: |
| QFT | **1,916.77 ms** | 340.38 | 6,635.45 | 13,860.52 | 63,080.18 |
| Ring QAOA | **2,405.73 ms** | 463.80 | 7,254.50 | 17,438.65 | 72,095.43 |
| GHZ | **327.17 ms** | 173.24 | 1,141.82 | 3,332.40 | 6,389.85 |
| Grover proxy | **969.01 ms** | 165.94 | 4,350.20 | 3,001.91 | 45,135.60 |
| Phase estimation | **2,342.78 ms** | 363.80 | 6,881.60 | 16,894.65 | 70,061.67 |
| TFIM Trotter | **7,645.69 ms** | 1,565.92 | 27,163.63 | 54,528.06 | 276,929.24 |

At 28 qubits MettleQ is 3.10–10.19× faster than Aer CPU, 19.53–46.58× faster
than Lightning CPU, and 2.94–4.49× faster than Lightning GPU. CUDA-Q NVIDIA is
1.89–6.44× faster at that width. At smaller widths, fixed NVIDIA launch cost is
visible: MettleQ wins all six CUDA-Q rows at 15 and 20 qubits, and retains
narrow wins for Grover proxy at 24 qubits and GHZ at 26 qubits.

![Matched-width Apple Metal and Windows comparison](plots/mettleq_vs_windows_matched_widths.png)

The long-form, machine-readable table is
[`results/mettleq_windows_matched_widths.csv`](results/mettleq_windows_matched_widths.csv).
Raw Apple results and manifests are frozen under
[`assets/benchmarks-frozen/fork-m3pro-20260719-windows-matched-metal/`](../assets/benchmarks-frozen/fork-m3pro-20260719-windows-matched-metal/).

## Metal changes measured here

- All-qubit RX uses radix-16 traversal: four adjacent qubits per pass instead
  of two. Isolated RX improved 3.07× at 20q, 1.93× at 24q, and 1.65× at 26q.
- Chain/ring CPHASE or ZZ plus the next RX layer execute in one Metal pass.
  Versus radix-16 alone, this improved 24q QAOA by 1.15× and TFIM by 1.14×.
- Cumulatively, the 24q controlled-phase/ZZ-heavy paths are about 2× faster
  than the previous pair-RX schedule while preserving exact tested parity.

## Aer MPS contract warning

The historical Aer MPS script calls `save_statevector()` through 25 qubits but
switches to one-shot `measure_all()` above 25. The resulting runtime drop is a
result-contract change, not a scaling breakthrough. Those raw rows remain for
provenance but are excluded from the matched full-state plot.

## Reproduce

Install MettleQ with plotting dependencies, then run one workload per process
to ensure allocator isolation:

```bash
python -m pip install -e '.[baselines,plot]'
for workload in qft qaoa_ring ghz grover_proxy phase_estimation tfim_trotter; do
  PYTHONPATH=src python windows_baseline/mettleq_baseline.py \
    --output-dir "matched/$workload" --benchmarks "$workload"
done
python windows_baseline/analyze_comparison.py
```

The original all-backend tables are in [`results_summary.md`](results_summary.md),
and unchanged Windows raw runs and manifests are in [`results/`](results/).
