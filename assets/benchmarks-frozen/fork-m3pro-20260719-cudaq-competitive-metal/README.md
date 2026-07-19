# CUDA-Q competitiveness campaign after full-QFT fusion

This frozen campaign measures clean MettleQ commit `ef0b6b9` at 15, 20, 24,
26, and 28 qubits after complete decomposed QFTs began selecting the radix-4
two-stage Metal kernel. It uses the exact Windows/WSL circuits, one warm-up,
three measured runs, and a synchronized complex64 full-state contract.

Each workload ran in a fresh process. Memory preflight required its two-state
lower bound plus 6 GiB available reserve. State norms were checked at every
width and complete states were independently compared with Qiskit through 20
qubits at `5e-6`. All 30 cells and all accuracy gates passed.

The QFT change reduced the 28-qubit execution from 27 to 14 Metal passes and
from 1,936.71 ms in the matched pre-change A/B arm to 877.90 ms here. The
other five workloads were rerun rather than copied so the comparison remains
one coherent campaign.

Generate the long-form table and both comparison plots with:

```bash
python windows_baseline/analyze_comparison.py
```
