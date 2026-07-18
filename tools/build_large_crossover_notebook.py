#!/usr/bin/env python3
"""Build the explanatory notebook for the frozen large SDK crossover run."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tutorials" / "experiments" / "01_large_sdk_cpu_gpu_crossover.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def main() -> int:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    notebook["cells"] = [
        markdown(
            """
# When does MettleQ's Apple GPU path actually win?

This experiment asks a deliberately narrow question: for the **same exact
statevector workload on one Apple M3 Pro**, when is the SDK's fastest CPU path
faster, and when is MettleQ's Apple GPU path faster?

It also checks every returned amplitude. Timing without correctness is not a
useful simulator benchmark.
"""
        ),
        markdown(
            """
## Learning goals

After this notebook you should be able to:

1. distinguish a complete SDK-call benchmark from an isolated-kernel timing;
2. understand why Qiskit Aer MPS is not automatically the fastest choice;
3. read a CPU/GPU crossover chart without treating it as universal; and
4. interpret `exact_match=False` separately from numerical correctness.
"""
        ),
        markdown(
            """
## Fair comparison contract

Each framework runs the same depth-three circuit at 16–29 qubits. The circuit
contains single-qubit rotations, nearest-neighbour entanglement, and one layer
of long-range mirror CNOTs.

- **Qiskit CPU:** Aer statevector and, through 22 qubits, Aer MPS; the faster
  measured method is selected per width.
- **PennyLane CPU:** `lightning.qubit` with complex128.
- **MettleQ:** exact statevector forced onto the Apple GPU through each SDK
  adapter.
- **Output:** the complete analytic statevector, not shots or one observable.
- **Timing:** one warm-up and three measured complete calls; transpilation is
  excluded, result materialization is included.
- **Accuracy:** global-phase-aligned maximum amplitude error must be at most
  `5e-6`.
"""
        ),
        code(
            """
from pathlib import Path
import json
from IPython.display import HTML, Image, display

ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parents[1]

RUN = ROOT / "assets/benchmarks-frozen/fork-m3pro-20260718-sdk-crossover-radix16-idle"
payload = json.loads((RUN / "results.json").read_text())
print("Machine:", payload["machine"])
print("Versions:", payload["environment"])
print("Protocol:", payload["protocol"])
"""
        ),
        markdown(
            """
## Inspect the raw measurements

The ratio is CPU time divided by MettleQ GPU time. A value above 1 means
MettleQ is faster; a value below 1 means the CPU reference is faster. Keeping
the raw milliseconds prevents a ratio from hiding very small or very large
absolute runtimes.
"""
        ),
        code(
            """
rows = []
for record in payload["records"]:
    q = record["qiskit"]
    p = record["pennylane"]
    rows.append({
        "qubits": record["width"],
        "Aer fastest method": q["cpu_fastest_method"],
        "Aer CPU ms": q["cpu_fastest_ms"],
        "MettleQ via Qiskit ms": q["mettleq_gpu_ms"],
        "Aer / MettleQ": q["cpu_over_mettleq"],
        "Qiskit max error": q["max_amplitude_error"],
        "Lightning CPU ms": p["cpu_ms"],
        "MettleQ via PennyLane ms": p["mettleq_gpu_ms"],
        "Lightning / MettleQ": p["cpu_over_mettleq"],
        "PennyLane max error": p["max_amplitude_error"],
    })

headers = list(rows[0])
def formatted(key, value):
    if key.endswith("max error"):
        return f"{value:.2e}"
    if key.endswith("MettleQ"):
        return f"{value:.3f}x"
    if key.endswith("ms"):
        return f"{value:.3f}"
    return str(value)

table = ["<table><thead><tr>"]
table.extend(f"<th>{header}</th>" for header in headers)
table.append("</tr></thead><tbody>")
for row in rows:
    table.append("<tr>")
    table.extend(f"<td>{formatted(header, row[header])}</td>" for header in headers)
    table.append("</tr>")
table.append("</tbody></table>")
display(HTML("".join(table)))
"""
        ),
        markdown(
            """
## Accuracy gate first

Statevectors may differ by a physically irrelevant global phase, so the
benchmark aligns that phase before comparing amplitudes. The following cell is
the pass/fail gate; the timing plot should not be interpreted if it fails.
"""
        ),
        code(
            """
ATOL = payload["protocol"]["accuracy_atol"]
worst_error = max(
    max(row["Qiskit max error"], row["PennyLane max error"])
    for row in rows
)
assert worst_error <= ATOL, (worst_error, ATOL)
print(f"PASS: worst full-state error {worst_error:.2e} <= {ATOL:.2e}")
print(f"The worst row used {worst_error / ATOL:.1%} of the allowed tolerance.")
"""
        ),
        markdown(
            """
## The crossover plot

The top row shows absolute time on a logarithmic scale. The bottom row shows
speedup, with the dashed 1.0 line marking equality. This makes it visible both
*where* the crossover occurs and *how expensive* each state size is.
"""
        ),
        code(
            """
display(Image(filename=str(RUN / "sdk_cpu_gpu_crossover.png")))
"""
        ),
        markdown(
            """
## What the measurements say

- Against Qiskit Aer, MettleQ crosses over around 20 qubits and is 2.74–4.55×
  faster from 22 through the largest admitted 29-qubit full-state run.
- Against PennyLane Lightning, MettleQ is 16.0–21.2× faster from 22 through
  29 qubits in the clean rerun.
- Aer MPS is slower for this contract because returning a complete dense
  statevector costs exponential time and memory, eliminating the principal
  reason to use an MPS. MPS should instead be benchmarked with bounded
  observables or samples on low-entanglement circuits.
- Qiskit's ASAP serialization originally hid legal layers. The current engine
  reconstructs the per-wire dependency DAG, preserves every same-wire order,
  and fuses ready disjoint U2/U3 and affine operations into native Metal passes.
  Generic one-qubit layers now apply four different matrices per radix-16
  traversal instead of only two, reducing full-state memory traffic.
- The 30-qubit full-state comparison is a recorded safety refusal, not an
  attempted allocation.
"""
        ),
        markdown(
            """
## Why many tutorial rows say `Exact? False`

That column asks whether two outputs are bit-for-bit identical. It is stricter
than mathematical correctness. MettleQ's Apple GPU path commonly uses
complex64, whereas CPU references use complex128, so differences around
`1e-7` are normal. Finite-shot simulators also use independent random streams
and should be checked by distribution distance, not identical samples.

The project therefore records both facts: exact equality for transparency and
an explicit numerical/statistical acceptance test for correctness.
"""
        ),
        code(
            """
audit = json.loads((ROOT / "tutorials/correctness_audit.json").read_text())
summary = {key: audit[key] for key in (
    "all_passed", "notebooks", "bit_for_bit_matches",
    "tolerance_validated_matches", "largest_primary_error_over_tolerance",
)}
summary
"""
        ),
        markdown(
            """
## Reproduce the expensive benchmark

The notebook reads frozen evidence so opening it does not allocate multi-GB
states. To rerun the measurement on the current Mac:

```bash
PYTHONPATH=src caffeinate -i .venv/bin/python \\
  tools/benchmark_sdk_crossover_large.py \\
  --widths 16,18,20,22,24,26,27,28,29,30 \\
  --depth 3 --warmups 1 --repeats 3 \\
  --max-qiskit-mps-width 22 --accuracy-atol 5e-6 \\
  --output-dir bench/runs/sdk-crossover-large
```

The `caffeinate` process exits automatically when Python finishes.
"""
        ),
        markdown(
            """
## Practical conclusion

For this machine and exact full-state circuit family, a sensible policy is:

- prefer a CPU path for small circuits;
- prefer MettleQ GPU over `lightning.qubit` once the measured workload clears
  its small-circuit overhead;
- prefer MettleQ GPU over Aer for this full-state family from the measured
  20-qubit crossover, while retaining Aer as the CPU fallback and baseline; and
- choose MPS based on entanglement and output contract, not qubit count alone.

These are measured policy inputs, not permanent universal thresholds. Different
circuits, Mac models, precisions, and requested outputs require recalibration.
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
