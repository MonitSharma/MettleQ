# Step 10: memory-bounded batched MPS sampling

This reruns the Step 9 mirrored peaked protocol after vectorizing conditional
MPS sampling across adaptive shot batches.

- Expected-peak recovery: 12/12 cells in every repeat
- Median MettleQ time: 28.5 ms (previously 673.3 ms)
- Ratio of median cell times: 23.6x
- Median of matched per-cell speedups: 27.2x
- Current range: 11.98–174.37 ms
- Aer result: faster in all 12 cells; approximately 5.2x faster by geometric
  mean

`peaked_sampling_improvement.csv` is the matched before/after source for
`peaked_sampling_improvement.png`. The current MettleQ/Aer comparison is in
`peaked_comparison.png`; raw rows and run parameters are in `manifest.json`.
