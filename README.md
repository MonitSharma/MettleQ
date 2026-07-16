<div align="center">
  <img src="quantumstudio/assets/app_icon_source.png" alt="MettleQ logo" width="150"/>
  <h1>MettleQ</h1>
  <p><strong>Fast, inspectable local quantum-circuit simulation for Apple Silicon.</strong></p>
  <p>Qiskit and PennyLane integration, exact statevector and MPS methods, MLX/Metal execution, and reproducible evidence.</p>
  <p>
    <a href="https://github.com/MonitSharma/MettleQ/actions/workflows/ci.yml"><img src="https://github.com/MonitSharma/MettleQ/actions/workflows/ci.yml/badge.svg" alt="CI status"/></a>
    <img src="https://img.shields.io/badge/platform-Apple%20Silicon-111111" alt="Apple Silicon"/>
    <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB" alt="Python 3.9+"/>
    <img src="https://img.shields.io/badge/MLX-0.6%2B-6E56CF" alt="MLX 0.6+"/>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2EA44F" alt="MIT license"/></a>
  </p>
  <p>
    <a href="#quick-start">Quick start</a> ·
    <a href="#performance">Performance</a> ·
    <a href="#trust-correctness-and-observability">Trust &amp; correctness</a> ·
    <a href="#sdk-integration-status">SDK status</a> ·
    <a href="#reproducing-the-results">Reproduce</a>
  </p>
</div>

![MettleQ Studio dashboard](quantumstudio/assets/screenshots/screen003.png)

> **Project goal:** build the fastest trustworthy local quantum-simulation
> engine for Apple Silicon, with native Qiskit and PennyLane paths that choose
> the Mac CPU for small work and the integrated GPU only when measured overhead
> is amortized.

MettleQ exposes exact statevector and bounded matrix-product-state (MPS)
simulation through a Qiskit `BackendV2`, Qiskit SamplerV2/EstimatorV2, and a
registered PennyLane device. Every execution selects one numerical device. It
does not add CPU and GPU timings together or market them as cooperative
acceleration.

## At a glance

| Area | Current capability |
| --- | --- |
| Accelerated engine | MLX on Apple Silicon, plus opt-in hand-written Metal kernels |
| Simulation backends | Exact statevector (`sv`), forward matrix-product state (`mps`), and explicit midpoint-MPO/TNO plus unswapping |
| Circuit inputs | Native Python operations, strict unitary OpenQASM 2.0, Qiskit circuits, and PennyLane QNodes |
| Workloads | QFT, phase estimation, Grover, QAOA, VQE, QCBM, QNN, random circuits, and spin dynamics |
| Trust model | Pre-allocation statevector checks, capability-gated dispatch, recoverable SVDs, MPS accuracy thresholds and convergence reports, explicit plans, numerical parity tests, synchronized benchmarks, and safe fallbacks |
| Current test suite | **365 tests** across the simulator, SDK adapters, planner, algorithms, MPS/MPO, peaked circuits, QASM, Metal dispatch, campaign analysis, and MettleQ Studio backend |
| Desktop product | MettleQ Studio orchestration, monitoring, plotting, and export |
| SDK adapters | Native Qiskit backend and registered PennyLane device, plus the original internal `mettleq.qml` teaching wrapper |

## Project lineage

MettleQ is an independently maintained private fork of the original public
**Qupertino** project. The original name and links below are intentionally
preserved for attribution. This fork is kept in the maintainer's GitHub
account and does not open pull requests against upstream.

| | Repository |
| --- | --- |
| **Private fork** | [MonitSharma/MettleQ](https://github.com/MonitSharma/MettleQ) |
| **Original upstream** | [BoltzmannEntropy/Qupertino](https://github.com/BoltzmannEntropy/Qupertino) |
| **Original website** | [QupertinoWEB](https://boltzmannentropy.github.io/QupertinoWEB/) |
| **Original author** | Shlomo Kashani |
| **Benchmark baseline** | Upstream commit `2b99d30` |
| **25-qubit speed sweep revision** | Fork commit `e5d9577` |
| **Step 3 memory-crossover revision** | Fork commit `a89bbd0` |
| **Step 4 preflight/policy revision** | Fork commit `83940c0` |
| **Step 5 native SDK engine revision** | Fork commit `0d01052` |
| **Step 6 adaptive statevector/MPS revision** | Fork commit `3f40a47` |
| **Step 7 MPS limit campaign revision** | Fork commit `7b3d2ff` |
| **Step 8 reliable/routed MPS engine revision** | Fork commit `0691674` |
| **MettleQ 0.2 rename, peaked benchmark, and batched sampling** | Fork commit `be67897` |
| **Midpoint-MPO/TNO, CPU MPS optimization, and GPU phase gate** | Fork commit `72582c7` |
| **Qiskit 2 isolated MPO worker and recoverable native-SVD boundary** | Fork commit `a2b99fc` |

Original authorship, licensing, and citation information are retained at the
end of this README.

MettleQ is more than a cosmetic rename. It keeps the useful MLX/Metal base and
extends it toward a trustworthy SDK backend:

| Original base | MettleQ direction |
| --- | --- |
| Direct simulator and benchmark stack | Normal Qiskit `BackendV2`/V2 primitives and a registered PennyLane device |
| Primarily dense statevector execution | Exact statevector plus bounded, routed, recoverable MPS |
| Benchmark-oriented dispatch | Explicit CPU/GPU planner, statevector memory preflight, safe fallbacks, and inspectable execution reports |
| Performance plots | Frozen raw evidence, matched baselines, accuracy thresholds, and automated `Dmax` convergence |
| MLX and custom Metal paths | Capability-gated Metal, measured crossover policy, and CPU MPS until a real GPU crossover is demonstrated |
| Original desktop tooling | MettleQ Studio, synchronized with the renamed engine and SDK surfaces |

The project aim is the fastest trustworthy local quantum-simulation engine for
Apple Silicon—not merely the fastest isolated kernel. A result must preserve
SDK semantics, fit available unified memory, expose approximation evidence,
and beat its reference under a matched protocol before it is described as a
speedup.

## Quick start

### Requirements

- Apple Silicon Mac (M1, M2, M3, or M4 family)
- macOS 13.3 or newer recommended
- Python 3.9 or newer
- Xcode command-line tools recommended for development and profiling

### Install

```bash
git clone git@github.com:MonitSharma/MettleQ.git
cd MettleQ

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[plot,tests,backend]'
```

For an SDK-focused install without the development and desktop extras:

```bash
python -m pip install -e '.[sdk]'
```

The explicit midpoint-MPO method has optional tensor-network dependencies:

```bash
python -m pip install -e '.[tensor-network]'
```

The full published 56-qubit P9 implementation uses an isolated, pinned Python
3.10 worker. Keep it separate from the normal Qiskit 2.x SDK environment:

```bash
uv venv --python 3.10 .venv-mpo
uv pip install --python .venv-mpo/bin/python \
  -r tools/requirements-midpoint-mpo-p9.txt
```

`IsolatedMidpointMPOSimulator` accepts a `QuantumCircuit` from the normal
Qiskit 2.x environment, validates that it is a bound unitary circuit, exports
OpenQASM 2, and runs the tensor-network algorithm in that worker. Set
`METTLEQ_MPO_PYTHON` or pass `worker_python=` when the environment is not named
`.venv-mpo`. MettleQ never downgrades or replaces the user's main SDK
environment.

Version 0.2 makes `mettleq` and the PennyLane device name `mettleq` canonical.
The former `mlxq` import namespace, `qupertino` PennyLane entry point, and
`Qupertino*` adapter class aliases remain available as migration shims; new
code should use the MettleQ names.

### Use MettleQ from Qiskit

```python
from qiskit import QuantumCircuit, transpile
from mettleq.integrations.qiskit import (
    MettleQBackend,
    MettleQEstimatorV2,
    MettleQSamplerV2,
)

backend = MettleQBackend(method="automatic", device="auto")
circuit = QuantumCircuit(3, 3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.measure(range(3), range(3))

compiled = transpile(circuit, backend)
result = backend.run(compiled, shots=4096, seed_simulator=7).result()
print(result.get_counts())

# Native Qiskit V2 primitives use the same planner and execution engine.
sampler = MettleQSamplerV2(backend=backend)
estimator = MettleQEstimatorV2(backend=backend)
```

Pass `return_statevector=True` only when the caller needs a full state readback.
`execution_report=True` adds the execution plan and statevector preflight to
the native Qiskit result data.

For circuits that specifically benefit from midpoint cancellation, opt into
the separate MPO/TNO method instead of selecting ordinary forward MPS:

```python
from qiskit import QuantumCircuit
from mettleq.midpoint_mpo import (
    IsolatedMidpointMPOSimulator,
    MidpointMPOOptions,
)

circuit = QuantumCircuit(8)
circuit.h(0)
for qubit in range(7):
    circuit.cx(qubit, qubit + 1)

simulator = IsolatedMidpointMPOSimulator(
    MidpointMPOOptions(max_bond=512, cutoff=6e-4, seed=123),
    worker_python=".venv-mpo/bin/python",
)
result = simulator.run(circuit, shots=1000, output_dir="bench/runs/my-mpo-run")
print(result.counts)
print(result.diagnostics)
```

The isolated API keeps normal Qiskit 2.x code on the caller side while the
pinned worker owns Quimb and the long-running compression. It is deliberately
not hidden behind `method="matrix_product_state"`: midpoint MPO has different
routing, approximation, runtime, and trust controls. PennyLane exposure is not
included yet because the method's current contract is finite-shot,
whole-circuit sampling rather than general differentiable observables.

### Use MettleQ from PennyLane

```python
import pennylane as qml

device = qml.device(
    "mettleq",
    wires=3,
    method="automatic",
    device="auto",
)

@qml.qnode(device, diff_method="parameter-shift")
def circuit(theta):
    qml.Hadamard(0)
    qml.CNOT(wires=[0, 1])
    qml.RY(theta, wires=2)
    return qml.expval(qml.Z(0)), qml.probs(wires=[2, 0])

print(circuit(0.3))
```

### Choose a simulation method and device

Qiskit and PennyLane accept the same policy vocabulary:

| Option | Meaning |
| --- | --- |
| `method="automatic"` | Preserve exact statevector semantics unless approximation is explicitly allowed and the circuit passes conservative MPS checks |
| `method="statevector"` | Exact dense statevector with the allocation preflight enforced |
| `method="matrix_product_state"` | Bounded MPS using `mps_max_bond_dimension` and `mps_truncation_threshold` |
| `device="auto"` | Select one measured path: CPU below crossover, GPU above it |
| `device="cpu"` / `"gpu"` | Force one numerical path for testing or a calibrated deployment |
| `mps_svd_driver="auto"` | Recoverable CPU SVD ladder: SciPy `gesdd`, `gesvd`, then NumPy fallbacks |
| `mps_routing_strategy="lookahead"` | Persist a logical-to-MPS layout when preflight predicts no more swaps than immediate restoration |
| `mps_accuracy_policy="report"` | Attach threshold evidence; use `"warn"` or `"error"` for stricter enforcement |
| `mps_convergence_bond_dimensions=(32, 64, 128)` | Rerun analytic SDK results and report successive-`Dmax` agreement |

`IsolatedMidpointMPOSimulator` is the recommended explicit method and has its own
`max_bond`, `cutoff`, unswapping, seeded-sampling, and expected-peak evidence;
it is not an `automatic` planner target.

Approximate automatic fallback is opt-in:

```python
backend = MettleQBackend(
    method="automatic",
    device="auto",
    allow_approximation=True,
    mps_max_bond_dimension=64,
    mps_truncation_threshold=1e-10,
    mps_accuracy_policy="error",
    mps_max_relative_discarded_weight=1e-6,
)
```

The current M3 Pro calibration selects statevector GPU execution from 14
qubits. MPS tensor operations can be forced onto CPU or GPU, while the stable
SVD ladder runs on CPU. A matched seven-topology campaign found CPU faster than
GPU tensors in every case, so automatic MPS stays on CPU unless the caller
supplies a measured `mps_gpu_min_qubits` value. MPS diagnostics expose the
tensor device, SVD driver and timing, fallback attempts, routing swaps, bond
growth, canonical center, renormalization, truncation, discarded-weight
telemetry, state norm, and accuracy classification.

Analytic Qiskit Estimator and PennyLane executions can request an automated
bond-dimension convergence report:

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from mettleq.integrations.qiskit import MettleQEstimatorV2

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
observable = SparsePauliOp("IIZ")  # Z on Qiskit qubit 0

estimator = MettleQEstimatorV2(
    method="matrix_product_state",
    device="cpu",
    mps_max_bond_dimension=64,
    mps_convergence_bond_dimensions=(32, 64, 128),
    mps_convergence_atol=1e-5,
)
pub_result = estimator.run([(circuit, observable)]).result()[0]
print(pub_result.metadata["mettleq_mps_accuracy"])
print(pub_result.metadata["mettleq_mps_convergence"])
```

Accuracy thresholds are based on local truncation and norm telemetry; passing
them is not a global fidelity proof. Use convergence or an independent
reference for results that matter.

Launch the Python process with `METTLEQ_METAL_KERNELS=auto` to request custom
Metal kernels when all capability checks pass. Without it, both SDK adapters
use the safe pure-MLX path.

### Run a first accelerated circuit

```bash
METTLEQ_METAL_KERNELS=auto PYTHONPATH=src .venv/bin/python - <<'PY'
from mettleq import Device, metal_runtime_status

operations = [
    {"name": "H", "wires": [0]},
    {"name": "CNOT", "wires": [0, 1]},
    {"name": "CNOT", "wires": [1, 2]},
]

device = Device(3)
device.execute(operations, report=True)
device.synchronize()

print("probabilities:", device.sim.probabilities())
print("Metal status:", metal_runtime_status(3)["reason"])
print("execution plan:", device.last_execution_plan)
PY
```

`METTLEQ_METAL_KERNELS=auto` requests custom kernels only when the runtime proves
that the platform, GPU device, dtype, backend, indexing, and memory constraints
are compatible. The unset default remains off; unsupported configurations fall
back safely.

Every exact statevector now receives a memory preflight before MLX constructs
the array. It reports the state size, the minimum two-state out-of-place cost,
the device's maximum buffer and recommended working-set limits, and the
remaining lower-bound headroom:

```python
from mettleq import statevector_preflight

report = statevector_preflight(27)
print(report["decision"])
print(report["cost_model"])
```

`StateVectorSimulator` and the `sv` `Device` enforce the same decision before
allocation. A pass means the reported lower bounds do not already rule the
request out; it is not a promise that an arbitrary lazy circuit graph will
fit. If a lower bound exceeds a device limit, MettleQ raises
`StatevectorMemoryError` with the computed sizes and suggests the MPS backend
or fewer qubits. The last-resort `METTLEQ_ALLOW_UNSAFE_STATEVECTOR=1` override (or
`Device(..., allow_unsafe_statevector=True)`) is explicit, defaults off, and is
recorded in the preflight and execution plan.

Long lazy Metal graphs can optionally be evaluated at safe custom-launch and
fused-layer boundaries. Multi-launch single-qubit and XX/YY layers stream in
budgeted chunks; no individual Metal launch is split. The budget is an
input/output traffic estimate used to choose evaluation locations, not a
promise that allocator peak will equal the value:

```bash
METTLEQ_METAL_KERNELS=auto \
METTLEQ_METAL_CHECKPOINT_BUDGET_MB=256 \
PYTHONPATH=src .venv/bin/python your_simulation.py
```

SDK callers can configure the same policy explicitly with
`Device(..., metal_checkpoint_budget_bytes=256 * 1024 * 1024)`. Checkpointing
is opt-in; unset preserves fully lazy execution.

### Verify the checkout

```bash
./test.sh
./bench.sh --circuit ghz --simulate-limit 12 \
  --qubits 2,5,8,10,12 --warmups 1 --repeats 2 --no-save-plots
```

## How execution works

```mermaid
flowchart TB
    QISKIT["Qiskit circuits and PUBs"] --> QAPI["BackendV2 · SamplerV2 · EstimatorV2"]
    QISKIT --> MPOAPI["Isolated midpoint-MPO worker<br/>Qiskit 2 caller · pinned tensor environment"]
    PL["PennyLane QNodes and tapes"] --> PAPI["PennyLane mettleq device"]
    QAPI --> IR["Validated canonical circuit IR"]
    PAPI --> IR
    IR --> PLAN["Inspectable method and device planner"]

    PLAN --> SVSAFE["Statevector memory preflight"]
    SVSAFE --> SVCPU["Exact statevector · CPU<br/>small circuits"]
    SVSAFE --> SVGPU["Exact statevector · Apple GPU<br/>MLX + capability-gated Metal"]

    PLAN --> ROUTE["MPS whole-circuit routing preflight<br/>lookahead or restore"]
    ROUTE --> MPSCPU["Bounded MPS · CPU tensors<br/>automatic default today"]
    ROUTE --> MPSGPU["Bounded MPS · GPU tensors<br/>explicit experimental path"]
    MPSCPU --> SPLIT["Recoverable CPU SVD ladder<br/>truncate · canonicalize · renormalize"]
    MPSGPU --> SPLIT
    MPOAPI --> MIDPOINT["Consolidate and split at midpoint<br/>left/right linear Sabre routing"]
    MIDPOINT --> UNSWAP["MPO/TNO cancellation<br/>greedy unswapping + rerouting"]
    UNSWAP --> MPOSAMPLE["Materialize bounded MPS<br/>seeded sequential sampling"]
    PLAN -. future .-> STAB["Stabilizer · CPU<br/>not implemented"]

    SVCPU --> MEASURE["Native probabilities, sampling, counts, expectations"]
    SVGPU --> MEASURE
    SPLIT --> MEASURE
    MPOSAMPLE --> MEASURE
    MEASURE --> RESULTS["Qiskit Result/DataBin/BitArray or PennyLane results"]
    SPLIT --> TRUST["Accuracy threshold policy<br/>optional Dmax convergence"]
    UNSWAP --> MPOTRUST["Cutoff/bond contract<br/>expected-peak convergence"]
    TRUST --> RESULTS
    MPOTRUST --> RESULTS
    PLAN --> EVIDENCE["Selection reason · preflight · dispatch · routing/SVD evidence"]
```

The product surface is deliberately limited to Qiskit and PennyLane for this
phase. Both adapters translate once into the same validated IR, then reuse the
same planner, memory gate, simulator, measurement implementation, and evidence.
Statevector sampling remains on the selected MLX device until shot bits are
returned. MPS marginals, local expectations, and sequential samples contract
the tensor network without constructing a dense `2**n` state. CPU and GPU are
alternative numerical paths, not additive acceleration; the planner keeps MPS
on CPU until matched evidence establishes a real GPU crossover. Midpoint MPO
is a separate opt-in Qiskit-circuit method and does not pass through the
statevector/forward-MPS automatic dispatcher. Its QASM process boundary lets a
normal Qiskit 2 caller use the pinned, independently reproducible
tensor-network environment.

## Performance

### Current adaptive SDK result: Apple M3 Pro

Commit `3f40a47` was measured through Qiskit and PennyLane using the same
20-qubit, two-step local circuit and the same analytic `⟨Z₀⟩` result contract.
Each implementation received one warmup and seven rotating-order repeats.
Custom Metal was enabled for compatible statevector layers. MPS used
`Dmax=32`, `eps=1e-10`, and recorded zero truncation events and zero discarded
weight on this shallow circuit.

| SDK | Path | Median | Reference / MettleQ | Absolute error |
| --- | --- | ---: | ---: | ---: |
| Qiskit | Statevector CPU | 238.69 ms | 2.77× | `4.42e-11` |
| Qiskit | Statevector GPU | **18.11 ms** | **36.54×** | `4.42e-10` |
| Qiskit | MPS CPU | **7.80 ms** | **84.81×** | `6.22e-8` |
| Qiskit | MPS GPU tensors + CPU SVD | 24.67 ms | 26.83× | `1.12e-7` |
| PennyLane | Statevector CPU | 236.91 ms | 2.58× | `2.79e-9` |
| PennyLane | Statevector GPU | **16.28 ms** | **37.49×** | `1.12e-8` |
| PennyLane | MPS CPU | **10.79 ms** | **56.56×** | `1.95e-7` |
| PennyLane | MPS GPU tensors + CPU SVD | 27.75 ms | 21.99× | `2.26e-7` |

Qiskit rows use `StatevectorEstimator` at 661.79 ms as their reference;
PennyLane rows use `default.qubit` at 610.25 ms. MPS wins this particular
low-entanglement workload because it avoids a dense `2**n` state. It is not a
general random-circuit claim: bond growth, `Dmax`, and truncation determine MPS
cost and accuracy.

<div align="center">
  <img src="assets/benchmarks-frozen/fork-m3pro-20260716-mettleq-relabel/sdk_method_matrix.png" alt="Qiskit and PennyLane statevector CPU, statevector GPU, MPS CPU, and MPS GPU method timing matrix" width="900"/>
  <br/><em>End-to-end SDK timing on a log scale. CPU and GPU are independent alternatives, not cooperative arms.</em>
</div>

#### Why automatic selection uses CPU for small circuits

The same circuit family was swept from 4–20 qubits for statevector and 4–32
qubits for MPS. On the safe pure-MLX default, statevector GPU becomes at least
1.10× faster for consecutive sizes starting at 14 qubits. Compatible custom
Metal layers move this workload-specific crossover to 6 qubits. At 20 qubits,
pure-MLX GPU is 5.53× faster than CPU; compatible Metal GPU is 18.64× faster.

MPS GPU did not cross CPU at any measured size. At 32 qubits, MPS CPU took
10.69 ms and the explicit GPU tensor path took 28.41 ms because SVD remains on
CPU. Accordingly, automatic MPS stays on CPU; explicit GPU remains available
and auditable.

<div align="center">
  <img src="assets/benchmarks-frozen/fork-m3pro-20260715-step6-adaptive-sdk/policy_pure_mlx/execution_policy.png" alt="Pure MLX CPU versus GPU crossover for exact statevector and MPS execution" width="900"/>
  <br/><em>Safe default policy calibration. Lower is better; both panels use logarithmic time axes.</em>
</div>

The full raw rows, summaries, selection records, MPS diagnostics, manifests,
and charts are frozen in
[`fork-m3pro-20260715-step6-adaptive-sdk/`](assets/benchmarks-frozen/fork-m3pro-20260715-step6-adaptive-sdk/).

#### Unchanged Step 5 protocol: historical versus current session

The older four-step protocol was also rerun unchanged. MettleQ itself was
slower in the current session, so the larger PennyLane ratio must not be read as
an engine improvement:

| SDK contract | Previous MettleQ | Current MettleQ | Current reference | Current speedup | MettleQ change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qiskit full statevector | 10.10 ms | 11.01 ms | Aer CPU 77.61 ms | 7.05× | 8.98% slower |
| PennyLane local `⟨Z⟩` | 18.31 ms | 21.44 ms | `default.qubit` 1,075.41 ms | 50.15× | 17.12% slower |

This refresh is a session-to-session stability check. The new method-matrix
results above use a different, intentionally shared expectation contract and
must not be compared directly with the Qiskit full-state row.

### Current fork: Apple M3 Pro, 25 qubits

The 2026-07-15 campaign ran 29 workloads with one warmup per arm and ten paired
repeats, alternating pure MLX and custom Metal inside every repeat. The
environment used macOS 26.5.2, Python 3.13.2, and MLX 0.32.0 at fork commit
`e5d9577`. Checkpointing was disabled in both arms for direct comparability.

- Median Metal speedup over pure MLX: **10.17×**
- Geometric-mean Metal speedup: **9.05×**
- Workloads at or above 1.1×: **26 of 29**
- Workloads at or above 4×: **25 of 29**
- Workloads at or above 10×: **19 of 29**
- Maximum: **32.65×** on long-range Ising
- Near parity: amplitude estimation, W state, and ladder Heisenberg

| Representative workload | Pure MLX | Custom Metal | Speedup |
| --- | ---: | ---: | ---: |
| Long-range Ising | 16,439.7 ms | **503.6 ms** | **32.65×** |
| Grover | 3,101.4 ms | **137.0 ms** | **22.64×** |
| EfficientSU2 | 4,670.9 ms | **240.0 ms** | **19.46×** |
| TFIM Trotter, second order | 20,266.3 ms | **1,257.1 ms** | **16.12×** |
| GHZ | 461.9 ms | **27.7 ms** | **16.71×** |
| QFT | 2,052.5 ms | **158.2 ms** | **12.98×** |
| QAOA | 4,307.7 ms | **424.5 ms** | **10.15×** |
| VQE plus energy evaluation | 4,413.9 ms | **1,145.5 ms** | **3.86×** |

<div align="center">
  <img src="assets/perf-charts/chart_fork_m3pro_speedup_20260715.png" alt="Metal speedup over pure MLX for 29 workloads on an Apple M3 Pro" width="820"/>
  <br/><em>Paired pure-MLX divided by Metal wall time at 25 qubits; larger is better.</em>
</div>

<details>
<summary><strong>Show absolute M3 Pro runtimes</strong></summary>

<div align="center">
  <img src="assets/perf-charts/chart_fork_m3pro_runtime_20260715.png" alt="Pure MLX and custom Metal runtimes for 29 workloads on an Apple M3 Pro" width="820"/>
  <br/><em>Mean synchronized wall time on a log scale; lower is better.</em>
</div>

</details>

#### Stability versus the previous fork campaign

The previous M3 Pro campaign used five repeats at `a73436e`. With a ±0.25×
ratio band, the refreshed ten-repeat result has five higher ratios, 22
unchanged ratios, and two lower ratios. The median ratio moved only +0.0245×.
Both arms were slightly slower in the new session by geometric mean (pure MLX
+3.93%, Metal +3.39%), so this is evidence of stable relative acceleration—not
a controlled claim that Step 2 changed 25-qubit wall time.

<details>
<summary><strong>Show previous fork run versus refreshed evidence</strong></summary>

<div align="center">
  <img src="assets/perf-charts/chart_previous_vs_current_speedup_20260715.png" alt="Previous five-repeat fork speedup compared with the refreshed ten-repeat fork speedup" width="820"/>
  <br/><em>Change in paired pure-MLX/Metal speedup on the same M3 Pro; a narrow ±0.25× band is treated as unchanged.</em>
</div>

</details>

### Original upstream versus this fork

The original committed 29-workload chart was measured on an M1 Max. The fork
rerun used an M3 Pro. The comparison below is therefore useful for seeing how
the **relative pure-MLX/Metal speedup changes by workload**, but it is not a
controlled test of absolute version performance.

With a ±0.25× band, the fork rerun has 17 higher ratios, five effectively
unchanged ratios, and seven lower ratios.

<div align="center">
  <img src="assets/perf-charts/chart_original_vs_fork_speedup_20260715.png" alt="Historical upstream M1 Max speedup compared with the current fork M3 Pro speedup" width="820"/>
  <br/><em>Original published M1 Max ratio → private-fork M3 Pro ratio. Positive does not imply lower cross-machine wall time.</em>
</div>

<details>
<summary><strong>Show the complete 29-workload comparison table</strong></summary>

| Workload | Original upstream, M1 Max | Private fork, M3 Pro | Ratio difference |
| --- | ---: | ---: | ---: |
| Long-range Ising | 24.3× | 32.65× | +8.35× |
| Grover | 11.5× | 22.64× | +11.14× |
| EfficientSU2 | 10.6× | 19.46× | +8.86× |
| GHZ | 9.1× | 16.71× | +7.61× |
| TFIM Trotter (2nd) | 25.0× | 16.12× | -8.88× |
| QCBM | 12.0× | 16.06× | +4.06× |
| Variational | 14.3× | 15.93× | +1.63× |
| QFT (entangled) | 7.3× | 13.46× | +6.16× |
| cuQuantum proxy | 13.2× | 13.10× | -0.10× |
| QFT | 8.4× | 12.98× | +4.58× |
| Quantum walk (V-chain) | 11.6× | 11.49× | -0.11× |
| Phase estimation | 8.0× | 11.03× | +3.03× |
| QNN | 7.4× | 10.27× | +2.87× |
| Graph state | 8.9× | 10.22× | +1.32× |
| Phase estimation (inexact) | 8.8× | 10.17× | +1.37× |
| QAOA | 8.4× | 10.15× | +1.75× |
| Heisenberg XXZ | 13.2× | 10.15× | -3.05× |
| Heisenberg | 13.8× | 10.14× | -3.66× |
| Quantum walk | 11.4× | 10.13× | -1.27× |
| RealAmplitudes | 5.1× | 9.92× | +4.82× |
| Random circuit | 4.7× | 9.18× | +4.48× |
| Deutsch-Jozsa | 8.4× | 8.79× | +0.39× |
| TFIM random field | 13.1× | 8.51× | -4.59× |
| TFIM Trotter (1st) | 12.3× | 8.50× | -3.80× |
| Heisenberg random field | 8.9× | 7.80× | -1.10× |
| VQE plus energy evaluation | 1.3× | 3.86× | +2.56× |
| Heisenberg ladder | 1.0× | 1.00× | +0.00× |
| W state | 1.0× | 1.00× | -0.00× |
| Amplitude estimation | 1.0× | 0.99× | -0.01× |

</details>

### Controlled same-machine revision check

To isolate code changes, the same M3 Pro ran upstream commit `2b99d30` and fork
commit `a73436e` at 20 qubits with one warmup and seven paired repeats. Pure-MLX
medians were broadly stable, while Metal latency increased on all six sampled
workloads:

| Workload | Upstream Metal | Fork Metal | Fork change | Upstream → fork speedup |
| --- | ---: | ---: | ---: | ---: |
| QFT | 4.326 ms | 4.544 ms | +5.0% slower | 5.86× → 5.62× |
| QAOA | 8.902 ms | 9.625 ms | +8.1% slower | 8.86× → 8.21× |
| TFIM Trotter (2nd) | 26.355 ms | 33.028 ms | +25.3% slower | 9.02× → 7.33× |
| Phase estimation | 6.286 ms | 7.031 ms | +11.8% slower | 5.71× → 5.11× |
| Grover | 4.815 ms | 5.367 ms | +11.5% slower | 12.69× → 11.29× |
| GHZ | 1.628 ms | 2.293 ms | +40.8% slower | 5.96× → 4.06× |

The correctness and observability revision above did not improve raw Metal
latency. The first follow-up optimization now caches process-stable hardware
facts while preserving live policy, selected-device, backend, dtype, dense
ablation, and circuit-size checks. In a same-process 20-qubit A/B with 15
alternating repeats, selector latency fell from 0.2139 ms to 0.001583 ms
(135×); the six sampled workloads improved by 4.6%–16.4%. These are focused
hot-path measurements.

A separate 29-workload, 20-qubit A–B–B–A revision check used two independent
seven-repeat campaigns per revision. Against upstream `2b99d30`, the Step 1
working tree was faster on 28 of 29 Metal workloads: median latency was 2.6%
lower and geometric-mean latency was 4.1% lower. All six regressions in the
historical table above recovered; VQE was the sole slower row at 0.17%. This
validates the overall working tree but does not attribute every revision
difference to the cache. The complete protocols and per-workload medians are
in the [Apple GPU engineering log](docs/development/apple-gpu-plan.md).

### Memory-budgeted Metal graphs

Step 2 adds opt-in evaluation checkpoints between fused operations. A
20-qubit, six-step TFIM crossover sweep used nine rotating repeats per arm on
the same M3 Pro. A 256 MiB estimated-I/O budget was the conservative measured
operating point: predicted and observed checkpoint counts matched exactly.

| Policy | Checkpoints | Median peak | Peak reduction | Median runtime | Runtime change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fully lazy | 0 | 616 MiB | — | 29.03 ms | — |
| 512 MiB budget | 2 | 264 MiB | 57.1% | 26.67 ms | 8.1% faster |
| 384 MiB budget | 3 | 184 MiB | 70.1% | 24.79 ms | 14.6% faster |
| **256 MiB budget** | **6** | **96 MiB** | **84.4%** | **22.71 ms** | **21.8% faster** |
| 128 MiB budget | 13 | 88 MiB | 85.7% | 26.52 ms | 8.7% faster |

All arms matched the pure-MLX statevector within `5.16e-9` maximum amplitude
error. QFT and affine validation errors were exactly zero. The sweep is
reproducible with `tools/checkpoint_sweep.py`; full protocol and guardrail
results are in the [Apple GPU engineering log](docs/development/apple-gpu-plan.md).

The first Step 2 test at 25 qubits exposed a deeper floor: boundary-only
checkpointing left every arm near 3.00 GiB because one logical H/RX layer held
13 custom launches in a single lazy graph. Step 3 keeps the same kernels but
streams that layer at safe boundaries between launches when a budget is
configured. It never splits an individual Metal launch.

The exact-commit Step 3 crossover at `a89bbd0` used the same six-step TFIM
workload, one warmup, and seven rotating repeats per arm:

| Policy | Median peak | Peak reduction | Median runtime | Runtime change |
| --- | ---: | ---: | ---: | ---: |
| Fully lazy | 3,072 MiB | — | 475.468 ms | — |
| 4 GiB budget | 2,304 MiB | 25.0% | 452.044 ms | 4.9% faster |
| 2 GiB budget | 1,536 MiB | 50.0% | 439.981 ms | 7.5% faster |
| **1 GiB budget** | **1,024 MiB** | **66.7%** | **443.556 ms** | **6.7% faster** |
| 512 MiB budget | 768 MiB | 75.0% | 454.365 ms | 4.4% faster |
| 256 MiB budget | 768 MiB | 75.0% | 453.889 ms | 4.5% faster |

All 42 measured arms matched their predicted checkpoint counts and agreed with
pure MLX within `1.41e-9` maximum amplitude error. The 1 GiB policy is the
balanced measured point; 512 MiB minimizes measured peak. Fully lazy execution
remains the default, and the budget remains a scheduling estimate rather than
a hard allocator cap.

<div align="center">
  <img src="assets/perf-charts/chart_checkpoint_n25_step3_20260715.png" alt="Step 3 checkpoint budget versus peak memory and runtime at 25 qubits" width="820"/>
  <br/><em>Median allocator peak and synchronized wall time for the rotating 25-qubit TFIM crossover.</em>
</div>

Step 4 broadens that single workload into 36 isolated workload/qubit cells:
TFIM, QFT, QAOA, QCBM, Heisenberg, and SU(2), each from 22 through 27 qubits.
Every cell used a fresh Python/MLX process, one warmup per arm, three rotating
repeats, and paired runtime analysis. The exact engine revision was `83940c0`.

| Adaptive policy | Median peak reduction | Minimum cell reduction | Paired geometric-mean runtime | Paired cells faster | Worst paired cell |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced: `max(256 MiB, 4 × state)` | 71.79% | 50.00% | 16.46% faster | 35 / 36 | 7.59% slower |
| Minimum: `max(128 MiB, 2 × state)` | **77.50%** | **58.33%** | 14.64% faster | **36 / 36** | **1.48% faster** |

All 324 measured executions had identical predicted and observed checkpoint
counts. Maximum amplitude error was `1.03e-6` (limit `5e-6`) and maximum norm
error was `6.91e-6` (limit `1e-5`). Pure MLX was the full-state reference
through 24 qubits; 25–27 qubits used the identical fully lazy Metal kernels,
which validates scheduling parity rather than independently revalidating Metal
algebra at those sizes.

Both formulas qualify for wider hardware testing on the measured M3 Pro.
Automatic checkpointing remains disabled by default until M1/M2/M3/M4 and
different unified-memory sizes are tested; users can select the measured byte
formulas through the existing explicit budget API.

<div align="center">
  <img src="assets/perf-charts/chart_memory_policy_q22_27_step4_20260715.png" alt="Step 4 adaptive memory policy across six workloads and 22 to 27 qubits" width="820"/>
  <br/><em>Median peak reduction and paired runtime change across six isolated workloads at each size.</em>
</div>

<details>
<summary><strong>Show the original upstream four-backend M1 Max result</strong></summary>

The original project reported an interleaved M1 Max campaign against Qiskit Aer
CPU and PennyLane `lightning.qubit`. Its unpublished `paper/` evidence tree is
not distributed in this checkout, so this is retained as a historical upstream
claim rather than a newly reproduced four-backend result.

<div align="center">
  <img src="assets/perf-charts/chart_4way_25q.png" alt="Original upstream four-backend M1 Max comparison" width="820"/>
</div>

| Workload @ 25q | MettleQ Metal | MettleQ MLX | Aer CPU | PennyLane lightning |
| --- | ---: | ---: | ---: | ---: |
| QFT | **0.059 s** | 0.72 s | 2.80 s | 5.61 s |
| Ring-QAOA, 6 layers | **0.150 s** | 2.07 s | 5.35 s | 6.84 s |
| TFIM Trotter, 20 steps | **0.495 s** | 5.82 s | 17.79 s | 32.95 s |
| Phase estimation | **0.105 s** | 0.91 s | 4.05 s | 6.06 s |
| Grover proxy | **0.052 s** | 1.11 s | 1.21 s | 2.73 s |
| GHZ | **0.022 s** | 0.27 s | 0.69 s | 0.42 s |

</details>

### Native SDK paths: exact-commit M3 Pro evidence

Engine commit `0d01052` was measured through the public Qiskit and PennyLane
interfaces, not the direct operation API. The 20-qubit, four-step TFIM-style
circuit used one warmup and seven repeats with implementation order reversed on
alternating repeats. Timings include SDK translation, execution,
synchronization, and the requested native result.

| SDK-native result | MettleQ median | CPU reference median | Speedup | Numerical check |
| --- | ---: | ---: | ---: | --- |
| Qiskit full statevector | **10.10 ms** | Aer 74.65 ms | **7.39×** | max amplitude error `1.287e-8` |
| PennyLane local `⟨Z⟩` | **18.31 ms** | `default.qubit` 708.70 ms | **38.70×** | expectation error `4.657e-9` |

<div align="center">
  <img src="assets/benchmarks-frozen/fork-m3pro-20260716-mettleq-relabel/sdk_adapter_step5.png" alt="Native Qiskit and PennyLane adapter timing comparison on Apple M3 Pro" width="820"/>
  <br/><em>Scoped comparisons within each SDK. The Qiskit and PennyLane result contracts differ and are not compared to each other.</em>
</div>

This is evidence for one Apple M3 Pro, circuit family, and package set—not a
universal claim across Apple chips or SDK configurations. The raw timings,
summary, validation thresholds, exact command, versions, and clean-engine
manifest are frozen in
[`fork-m3pro-20260715-step5-sdk/`](assets/benchmarks-frozen/fork-m3pro-20260715-step5-sdk/).

### Reliable, routed MPS: current Apple M3 Pro evidence

There is no honest single MPS qubit limit: entanglement topology and bond
growth matter more than width alone. Step 8 first fixed the numerical and
trust failures exposed by Step 7, then measured the new engine at clean commit
`0691674` through Qiskit `MettleQEstimatorV2`.

The current CPU MPS path uses a recoverable SciPy/NumPy SVD ladder, an explicit
mixed-canonical center, renormalized two-site splits, local accuracy thresholds,
automated `Dmax` convergence reporting, and whole-circuit routing preflight.
The standard 26-case `Dmax=64` suite completed every case with no numerical
errors. Its worst exact `Z0` error through 20 qubits fell from `1.278e-2` in
Step 7 to `1.890e-6` now—a 6,762-fold reduction in this matched case set.

| Reliability signal | Historical Step 7 | Current Step 8 |
| --- | ---: | ---: |
| Standard `Dmax=64` cases completed | 26 / 26 | 26 / 26 |
| Worst exact-reference error through 20q | `1.278e-2` | **`1.890e-6`** |
| Matched eight-case boundary set | 2 completed, 5 errors, 1 timeout | **7 completed, 0 errors, 1 timeout** |
| All-to-all 32q d1 | 20.969 s; norm 0.5730 | **3.637 s; norm 1.00000024** |
| All-to-all 36q d1 | Timed out after 30 s | **4.935 s** |

The boundary campaign used CPU MPS, `Dmax=64`, `eps=1e-10`, topology-aware
routing, and a 60-second ceiling. The largest demonstrated completions are
test points, not universal maxima:

| Entanglement family | Largest demonstrated completion | Runtime | Trust signal |
| --- | ---: | ---: | --- |
| GHZ chain | 10,000q d1 | 10.393 s | Bond 2; no local truncation |
| 1D brickwork | 200q d4 | 0.246 s | Bond 4; 10,000q d8 reached the 60 s ceiling |
| Ring brickwork | 1,000q d4 | 11.693 s | Bond cap; within configured local thresholds |
| 2D grid | 144q d2 | 6.410 s | Bond cap; local threshold exceeded |
| Rainbow pairs | 96q d1 | 14.627 s | Bond cap; local threshold exceeded |
| Seeded long range | 160q d1 | 27.482 s | Bond cap; local threshold exceeded |
| All to all | 36q d1 | 4.935 s | Bond cap; within configured local thresholds |

<div align="center">
  <img src="assets/benchmarks-frozen/fork-m3pro-20260716-mettleq-relabel/mps_limit_landscape.png" alt="Current MettleQ MPS completion envelope, bond pressure, truncation, and timeout across seven entanglement families" width="920"/>
  <br/><em>Filled points show no observed local truncation; hollow points show truncation. The triangle retains the 60-second timeout as a failure marker, not a runtime.</em>
</div>

The representative `Dmax=32/64/128` sweep completed all 36 plotted rows. Its
worst small exact-reference error was `4.619e-5`, `1.890e-6`, and `1.341e-6`,
respectively. Runtime rises sharply with bond capacity on high-entanglement
cases, so convergence is evidence the caller must deliberately pay for.

<div align="center">
  <img src="assets/benchmarks-frozen/fork-m3pro-20260716-mettleq-relabel/mps_dmax_convergence.png" alt="Current MPS runtime, normalized state stability, and exact small-circuit error across bond dimensions 32, 64, and 128" width="920"/>
  <br/><em>Normalized state stability prevents silent norm collapse. It does not prove a truncated result is globally accurate.</em>
</div>

#### Matched MettleQ versus Qiskit Aer MPS

Only after the reliability work completed, the same Qiskit circuits and
analytic `Z0` EstimatorV2 contract were run through MettleQ CPU routed,
MettleQ CPU restore, MettleQ GPU tensors, and Qiskit Aer CPU MPS. Each case
used one warmup, three rotating-order repeats, and a fresh process.

| Circuit | MettleQ CPU routed | MettleQ CPU restore | MettleQ GPU tensors | Aer CPU MPS | Routing gain | MettleQ / Aer speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GHZ 1,000q d1 | 338.76 ms | 338.28 ms | 797.22 ms | **57.90 ms** | 1.00x | 0.17x |
| Line 100q d8 | 153.25 ms | 154.50 ms | 514.20 ms | **29.00 ms** | 1.01x | 0.19x |
| Ring 50q d2 | 38.77 ms | 65.78 ms | 139.01 ms | **5.62 ms** | 1.70x | 0.14x |
| Grid 36q d2 | **249.91 ms** | 248.75 ms | 522.18 ms | 422.57 ms | 1.00x | **1.69x** |
| Rainbow 32q d1 | 710.44 ms | 888.97 ms | 1,255.08 ms | **3.65 ms** | 1.25x | 0.005x |
| Random long range 32q d1 | 264.76 ms | 376.30 ms | 544.00 ms | **3.40 ms** | 1.42x | 0.013x |
| All to all 20q d1 | 851.13 ms | 4,700.62 ms | 1,422.03 ms | **25.17 ms** | 5.52x | 0.030x |

<div align="center">
  <img src="assets/benchmarks-frozen/fork-m3pro-20260716-mettleq-relabel/matched_aer_comparison.png" alt="Matched MettleQ CPU routed, CPU restore, GPU tensor, and Qiskit Aer CPU MPS results" width="920"/>
  <br/><em>Aer is faster on six schedules; MettleQ wins the tested 36-qubit grid by 1.69x. CPU beats GPU tensors everywhere, so automatic MPS remains on CPU.</em>
</div>

Routing is nevertheless material inside MettleQ: it improves ring by 1.70x,
rainbow by 1.25x, random long range by 1.42x, and all to all by 5.52x. The
all-to-all schedule falls from 2,280 restore-baseline swaps to 384 routed
swaps. Grid preflight correctly rejects lookahead when it predicts no saving.

The raw rows, exact commands, clean-engine manifests, summaries, and plotted
source data are frozen in
[`fork-m3pro-20260716-step8-mps-reliability/`](assets/benchmarks-frozen/fork-m3pro-20260716-step8-mps-reliability/).

### Published 56-qubit P9: midpoint MPO/TNO now recovers the peak

Quantum Advantage Tracker issue
[#153](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/issues/153)
uses the 56-qubit `peaked_circuit_P9_Hqap_56x1917` circuit: 3,890 `u` gates
and 1,917 `rzz` gates. The successful algorithm is not ordinary left-to-right
MPS. It consolidates the circuit, builds from the midpoint as an MPO/TNO,
absorbs routed layers from both sides, and uses greedy unswapping plus
rerouting. MettleQ now exposes that algorithm as the separate
`MidpointMPOSimulator` method.

The exact published QASM and expected bitstring are vendored with Apache-2.0
attribution and SHA-256
`cff3496c45d9133c1f1693f1d3b0cf1fc2da338f13cd7b339db330a4762d0f35`.
The matched M3 Pro runs below used Python 3.10.16, Qiskit 1.4.5, Quimb 1.11.2,
NumPy 2.2.6, SciPy 1.15.3, cutoff `6e-4`, routing and sampling seed 123,
90/50 initial/post-unswap Sabre trials, no parallel rewiring, and 1,000 shots.

| Same-Mac P9 arm | Max bond | End-to-end | Expected peak | Result |
| --- | ---: | ---: | ---: | --- |
| MettleQ midpoint MPO | 512 | 1,252.92 s | 100/1,000 | Recovered |
| MettleQ midpoint MPO | 768 | **958.84 s** | 103/1,000 | Recovered |
| Published [`p9solver`](https://github.com/alexgalda-m/peaked-mpo-solver) core | 512 | 1,184.16 s | 100/1,000 | Recovered |

The D=512 MettleQ and published-core arms executed the same algorithm and
identical contract. MettleQ was 5.8% slower in this single ordered pair, so the
classification is **performance parity**, not a speedup. D=768 changed the
greedy trajectory, reduced MettleQ time by 23.5% versus D=512 (1.31×), and
preserved the expected-peak fraction within 0.003. The automated bond report
therefore classifies D=512/768 as converged at a 0.03 tolerance.

The published-core arm imported the published repository's
`p9solver.pipeline` directly and ran it through the same in-memory telemetry
and seeded sampler as MettleQ. This excludes the published CLI's per-cycle
checkpoint-file overhead from both sides of the compute comparison. A separate
published-CLI attempt was externally interrupted and is retained as partial
evidence, not reported as a completed timing.

Cutoff convergence is not yet established. A matched cutoff `1e-3` arm reached
only 197/1,885 consolidated work gates after 501 seconds and was stopped as an
operationally impractical trajectory; MettleQ makes no peak claim for that arm.
The full result is therefore trustworthy at the tested `6e-4` contract, not a
claim of cutoff-independent convergence.

<div align="center">
  <img src="assets/benchmarks-frozen/fork-m3pro-20260716-step11-midpoint-mpo/midpoint_mpo_phase.png" alt="MettleQ midpoint-MPO P9 runtime, expected peak, and CPU GPU MPS phase-gate evidence" width="1120"/>
  <br/><em>Same-Mac P9 results, seeded expected-peak evidence, and the measured reason native GPU MPS remains disabled.</em>
</div>

For contrast, the historical forward-MPS boundary run at `Dmax=64`,
`eps=1e-10`, and 100 shots completed all 5,807 source gates in 111.52 s but
returned the published peak 0 times, saturated its bond cap, and accumulated
190.68 relative local discarded-weight sum. Its shorter runtime is not a
usable result and is not compared as an alternative P9 solution.

For CI and matched timing, MettleQ also includes a deterministic mirrored
peaked family. A seeded `u`/`rzz`/permutation body and its inverse create
temporary entanglement, followed by small rotations with an analytically known
unique mode. Linear, grid, long-range, and all-to-all variants at 6, 8, and 10
qubits recovered the expected peak in every repeat.

The first matched run exposed a Python loop over every shot and wire. Replacing
it with memory-bounded batched MPS conditional sampling reduced the median
MettleQ time across the 12 cells from **673.3 ms to 28.5 ms**—a **23.6x median
time reduction**. The median of the 12 matched per-cell speedups is **27.2x**.
Peak recovery was unchanged. Current MettleQ timings span
11.98–174.37 ms. Qiskit Aer MPS is still faster in all 12 small cells, by about
5.2x on geometric mean, so no competitive win is claimed.

<div align="center">
  <img src="assets/benchmarks-frozen/fork-m3pro-20260716-step10-batched-mps-sampling/peaked_sampling_improvement.png" alt="MettleQ MPS sampling before and after memory-bounded batching across 12 peaked-circuit cases" width="920"/>
  <br/><em>Matched historical versus current MettleQ timings. Both runs use the same circuits, shots, warmups, repeats, Dmax, and truncation threshold.</em>
</div>

<div align="center">
  <img src="assets/benchmarks-frozen/fork-m3pro-20260716-step10-batched-mps-sampling/peaked_comparison.png" alt="Current MettleQ versus Qiskit Aer MPS timing on the mirrored peaked family" width="920"/>
  <br/><em>Current end-to-end 1,024-shot result: correctness passes, but Aer remains faster on these small circuits.</em>
</div>

The new full MPO summaries, every sample, tensor-network stats, the interrupted
cutoff and published-CLI arms, matched published-core result, CPU/GPU profile,
manifest, and plots are frozen in
[`step11-midpoint-mpo/`](assets/benchmarks-frozen/fork-m3pro-20260716-step11-midpoint-mpo/).
The earlier forward-MPS and mirrored-family evidence remains frozen in
[`step9-mettleq-peaked/`](assets/benchmarks-frozen/fork-m3pro-20260716-step9-mettleq-peaked/)
and
[`step10-batched-mps-sampling/`](assets/benchmarks-frozen/fork-m3pro-20260716-step10-batched-mps-sampling/).

<details>
<summary><strong>Show the historical Step 7 failure envelope</strong></summary>

#### Historical Step 7 MPS limits

A 10,000-qubit GHZ chain needs bond
dimension 2, while a much smaller nonlocal circuit can saturate `Dmax`, lose
norm, or fail its SVD. Step 7 therefore swept seven deterministic topologies
through the public Qiskit `MettleQEstimatorV2` path, with every case isolated
in a fresh process.

The completion-envelope runs used CPU MPS, `Dmax=64`, `eps=1e-10`, and a
30-second per-case ceiling. CPU is the current automatic choice because the
measured GPU tensor path still returns to the CPU for every SVD and did not
cross over. These are the largest completed cases in the tested schedules,
not universal maxima:

| Entanglement family | Largest observed completion | Time | Accuracy / failure boundary |
| --- | ---: | ---: | --- |
| GHZ chain | 10,000q d1 | **2.309 s** | Bond 2, no local truncation; this was the test ceiling, not an engine limit |
| 1D brickwork | 1,000q d8 | **1.096 s** | Bond 60, norm 1.000062; 10,000q d8 later reached a zero numerical norm |
| Ring brickwork | 1,000q d4 | 13.344 s | Bond cap, norm 0.0169; completion is not a usable answer, and SVD failures were non-monotonic |
| 2D grid | 81q d2 | 1.467 s | Bond cap, norm 0.775; 100q and 144q hit MLX SVD failures |
| Rainbow pairs | 64q d1 | 4.025 s | Bond cap, norm 0.930; 80q and 96q hit MLX SVD failures |
| Seeded long range | 96q d1 | 5.935 s | Bond cap, norm 0.868; 128q and 160q hit MLX SVD failures |
| All to all | 32q d1 | 20.969 s | Bond cap, norm 0.573; 36q exceeded 30 seconds |

The standard 26-case sweep completed every case, but independent Qiskit
statevector checks through 20 qubits found up to `1.278e-2` local-observable
error after aggressive nonlocal truncation. At 16 qubits, line, ring, and grid
cases passed the `5e-5` acceptance threshold; rainbow, random-long-range, and
all-to-all cases did not all pass at `Dmax=64`.

<div align="center">
  <img src="assets/benchmarks-frozen/fork-m3pro-20260715-step7-mps-limits/mps_limit_landscape.png" alt="MettleQ MPS completion envelope, bond growth, truncation, errors, and timeout across seven entanglement families" width="920"/>
  <br/><em>Filled points show no observed local truncation; hollow points show truncation. X and triangle markers retain numerical failures and the 30-second timeout.</em>
</div>

`Dmax` convergence is part of trust, not an optional performance tweak. For
example, the 24-qubit all-to-all case rose from 2.406 s at `Dmax=32`, to 7.530 s
at 64, to 26.126 s at 128, yet remained strongly truncated. Increasing `Dmax`
did not make every observable or norm converge monotonically.

<div align="center">
  <img src="assets/benchmarks-frozen/fork-m3pro-20260715-step7-mps-limits/mps_dmax_convergence.png" alt="MPS runtime, state norm error, and exact small-circuit error across maximum bond dimensions 32, 64, and 128" width="920"/>
  <br/><em>Representative convergence probes. Wide cases without an independent reference are not labeled exact.</em>
</div>

The reviewed raw rows, manifests, summaries, plotted CSVs, commands, and
interpretation are frozen in
[`fork-m3pro-20260715-step7-mps-limits/`](assets/benchmarks-frozen/fork-m3pro-20260715-step7-mps-limits/).

These failures motivated the Step 8 work above and remain immutable historical
evidence.

</details>

## Trust, correctness, and observability

Fast simulation is useful only when dispatch, semantics, and measurements are
auditable. This fork adds explicit evidence at each layer:

- **Safe Metal selection:** platform, device, API, dtype, backend, qubit-index,
  memory, and ablation-policy checks must all pass.
- **Explicit method/device selection:** SDK results record the requested and
  selected method, selected CPU or GPU, and the reason for that decision.
- **Inspectable plans:** `Device.explain()` reports recognized patterns,
  selected kernels, fallbacks, expected dispatches, and memory estimates without
  executing the circuit.
- **Synchronized proof:** `execute(..., report=True)` plus `synchronize()`
  records concrete host dispatches and completes pending MLX work.
- **Auditable memory control:** checkpoint plans predict safe fused-layer
  boundaries, and execution reports record every actual evaluation, duration,
  estimated traffic, and allocator snapshot.
- **Numerical parity:** custom kernels are tested against pure MLX, including
  all-distinct-angle and long-product stress cases.
- **Strict QASM:** unsupported dynamic semantics are rejected with source-aware
  errors instead of being silently ignored.
- **Recoverable MPS numerics:** a scaled SciPy/NumPy SVD ladder reports failed
  attempts instead of allowing MLX `sgesvdx` to terminate the process; explicit
  canonicalization and renormalization guard finite norm.
- **Recoverable midpoint-MPO numerics:** the isolated worker bypasses Quimb's
  process-killing native path, tries unscaled NumPy first, then contains the
  Quimb-compatible SciPy `gesvd` fallback in a killable child process. Every
  child failure, scaled retry, and eigensolver fallback is reported.
- **MPS trust policy:** both SDKs attach local discarded-weight and norm
  classifications, can warn or raise on configured thresholds, and can rerun
  analytic results across requested bond dimensions.
- **Routing evidence:** whole-circuit preflight compares lookahead and immediate
  restoration, records predicted and actual swaps, and skips routing work for
  already-local circuits.
- **Bounded MPS sampling:** finite shots contract the MPS directly in adaptive
  batches instead of materializing `2**n` probabilities or looping one shot at
  a time; the batch ceiling scales down with `Dmax²` temporary storage.
- **Explicit MPO trust contract:** midpoint-MPO results record cutoff, bond
  cap, complete gate consumption, peak tensor footprint, measurement
  permutation, routing/sampling seeds, expected-peak count, and automated
  convergence classification. Bond comparisons hold cutoff fixed and cutoff
  comparisons hold bond fixed, so confounded diagonal points cannot pass.
- **Hermetic tests:** generated files use temporary directories, and successful
  tests leave the checkout unchanged.

### Test inventory

| Category | Tests |
| --- | ---: |
| Core simulator and gate algebra | 149 |
| Quantum-computing examples and algorithms | 41 |
| Internal consistency and measurement parity | 21 |
| MPS backend and correctness | 22 |
| Midpoint-MPO API, isolated worker, safe SVD, and convergence policy | 9 |
| QML wrapper, QFT, and subset semantics | 10 |
| Strict OpenQASM and silent-risk checks | 7 |
| QPE energy estimation | 2 |
| Benchmark protocol and plotting | 10 |
| Peaked-circuit fixtures, topology regressions, and backend recovery | 6 |
| Custom Metal parity and dispatch | 19 |
| Execution plans, memory policy, planner, and capability reporting | 33 |
| Native Qiskit and PennyLane integrations and rebrand compatibility | 18 |
| MettleQ Studio backend and MCP API | 18 |
| **Total** | **365** |

Run everything with:

```bash
./test.sh
```

CI runs portable backend checks on hosted Linux. Full simulator and custom-Metal
validation is available as a manually triggered job for a self-hosted Apple
Silicon runner labeled `macOS` and `ARM64`.

### Known limits

- Hand-written Metal kernels currently accelerate exact statevector patterns,
  not MPS. MPS tensors support explicit CPU/GPU MLX execution, while the stable
  SciPy/NumPy SVD ladder runs on CPU; the split is visible in diagnostics.
- MPS is bounded and may be approximate. Local discarded singular-value weight
  and normalized state stability are telemetry, not global fidelity bounds.
  Automatic MPS requires `allow_approximation=True` and conservative
  locality/depth checks. Threshold success should be paired with `Dmax`
  convergence or an independent reference for important results.
- Stable MPS SVD currently moves each two-site matrix through a CPU LAPACK path.
  Direct LAPACK dispatch and NumPy-native two-site contractions improved the
  matched CPU micro-workload by 1.116× and reduced recorded SVD time from
  56.06 ms to 45.30 ms. At D=64, SVD still represented 97.8% of measured
  contraction-plus-SVD time.
- Native GPU MPS remains disabled by evidence. On this M3 Pro, MLX GPU
  contraction was slower than NumPy CPU contraction at every tested bond from
  8 through 128, with no resident or GPU-to-CPU round-trip crossover. GPU MPS
  should be revisited only after SVD/truncation can remain resident on GPU.
- Full P9 midpoint-MPO support uses the isolated pinned Python 3.10 environment
  in `tools/requirements-midpoint-mpo-p9.txt`. A normal Qiskit 2.x caller can
  invoke it through `IsolatedMidpointMPOSimulator`; in-process execution still
  requires compatible tensor-network dependencies.
- The midpoint-MPO API currently targets whole-circuit finite-shot Qiskit
  inputs. It is not yet exposed as a Qiskit `BackendV2`, Estimator, or PennyLane
  differentiable device method.
- The bundled strict importer accepts 33 of 42 OpenQASM files. It rejects reset,
  classical control, mid-circuit measurement, opaque gates, arbitrary includes,
  and malformed declarations.
- The Qiskit backend and V2 primitives support statevector and MPS execution.
  The backend accepts unitary circuits with final measurements; reset,
  mid-circuit measurement, classical control, and noise are rejected. The
  PennyLane device supports both methods, common decomposable unitary gates, state,
  probabilities, samples, counts, expectations, variances, shot vectors, and
  framework-managed parameter-shift gradients; dynamic wires, mid-circuit
  measurement, backpropagation, and device adjoint gradients are not yet
  implemented.
- Checkpoint budgets use conservative statevector input/output traffic as a
  scheduling signal; they are not hard allocator ceilings. Supported
  multi-launch layers can evaluate between launches, but one custom kernel
  launch remains the irreducible out-of-place input/output floor.
- Cross-machine charts compare relative speedup only; absolute performance
  claims require the same machine and benchmark protocol.

## SDK integration status

| Integration | Status | Intended experience |
| --- | --- | --- |
| Native `mettleq` Python operations | Available | Execute validated operation dictionaries directly |
| OpenQASM 2.0 | Available, strict unitary subset | Import supported circuits with explicit rejection of dynamic semantics |
| `mettleq.qml` | Available, internal wrapper | PennyLane-like tapes, measurements, templates, and parameter-shift gradients |
| Qiskit `BackendV2` | Available: statevector + MPS | Transpile and run unitary circuits; receive native `Result`, device-sampled counts/memory, optional statevector, accuracy classification, and execution evidence |
| Qiskit SamplerV2 / EstimatorV2 | Available | Use PUB batching, native `BitArray` samples, device-resident Pauli expectations, and optional analytic `Dmax` convergence reports |
| Qiskit `IsolatedMidpointMPOSimulator` | Available, explicit finite-shot method | Pass a normal Qiskit 2 `QuantumCircuit`; run in the pinned tensor worker and receive seeded counts/samples plus routing, cutoff, bond, tensor-footprint, SVD, and expected-peak evidence |
| PennyLane `mettleq` device | Available: statevector + MPS | Use a normal QNode with analytic or finite-shot measurements, threshold policy, analytic `Dmax` convergence, tracking, and parameter-shift gradients |
| Other SDKs | Out of scope for the current phase | Qiskit and PennyLane are the only active integration targets |

Both adapters translate through one tested canonical operation layer. Qiskit's
little-endian state and classical-bit conventions and PennyLane's declared wire
order are covered by reference-parity tests. Unsupported semantics raise native
SDK errors, while statevector preflight, capability-gated dispatch, checkpoint
policy, MPS routing/SVD/accuracy evidence, and execution reports remain in the
core rather than being reimplemented or bypassed by an adapter.

## Running benchmarks

### Reproducible single workload

```bash
METTLEQ_METAL_KERNELS=1 ./bench.sh \
  --circuit qft --qubits 15,20,25 --warmups 1 --repeats 5
```

### Full orchestrated run

```bash
./bench_with_logging.sh
```

### Calibrate CPU/GPU selection

```bash
METTLEQ_METAL_KERNELS=auto PYTHONPATH=src .venv/bin/python \
  tools/benchmark_execution_policy.py \
  --outdir bench/runs/execution-policy \
  --warmups 1 --repeats 7
```

This paired rotating sweep reports a sustained crossover only when GPU is at
least 1.10× faster for two consecutive measured sizes. A `null` MPS crossover
means automatic MPS should remain on CPU over the tested range.

### Benchmark Qiskit and PennyLane method/device paths

```bash
METTLEQ_METAL_KERNELS=auto PYTHONPATH=src .venv/bin/python \
  tools/benchmark_sdk_method_matrix.py \
  --outdir bench/runs/sdk-method-matrix \
  --qubits 20 --steps 2 --warmups 1 --repeats 7
```

The SDK matrix uses the same analytic local `Z` expectation contract in both
frameworks. Qiskit rows compare with `StatevectorEstimator`; PennyLane rows
compare with `default.qubit`.

### Probe MPS limits by entanglement topology

```bash
PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/benchmark_mps_limits.py \
  --outdir bench/runs/mps-limits \
  --profile standard --dmax 64 --eps 1e-10 \
  --svd-driver auto --routing-strategy lookahead \
  --routing-lookahead 8 --timeout-seconds 60

# Repeat --case to narrow a boundary without editing the benchmark.
PYTHONPATH=src .venv/bin/python tools/benchmark_mps_limits.py \
  --outdir bench/runs/mps-boundary --dmax 64 --timeout-seconds 60 \
  --case grid_2d:81:2 --case rainbow:80:1 --case all_to_all:36:1
```

The campaign separates completion, timeout, process error, small exact
validation, local accuracy classification, routing, SVD fallbacks, bond growth,
normalized state stability, runtime, and peak RSS.

### Run the matched MettleQ/Aer MPS protocol

```bash
PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/benchmark_mps_phase8.py \
  --outdir bench/runs/mps-matched-aer \
  --profile standard --dmax 64 --eps 1e-10 \
  --svd-driver auto --routing-lookahead 8 \
  --warmups 1 --repeats 3 --timeout-seconds 240
```

This benchmark uses the same Qiskit circuits and analytic `Z0` result contract
for MettleQ CPU routed, CPU restore, GPU tensors, and Qiskit Aer CPU MPS. Run
it only on an otherwise idle machine; implementation order rotates and every
case starts in a fresh process.

### Run the full midpoint-MPO P9 protocol

Create the isolated environment shown in [Install](#install). The supported
Qiskit 2 caller-to-worker campaign, including reversed-order repeats and the
fixed-D512 cutoff sweep, is:

```bash
METTLEQ_MPO_PYTHON=.venv-mpo/bin/python \
PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/benchmark_midpoint_mpo_priority_phase.py \
  --output-dir bench/runs/midpoint-mpo-priorities \
  --worker-python .venv-mpo/bin/python \
  --published-repo /tmp/peaked-mpo-solver-reference \
  --repeats 2 --cutoff-repeats 2 \
  --shots 1000 --seed 123 --timeout-seconds 3600 --phase all
```

For one direct run entirely inside the pinned environment:

```bash
PYTHONPATH=src caffeinate -i .venv-mpo/bin/python -m mettleq.midpoint_mpo \
  --qasm src/mettleq/datasets/peaked_circuit_P9_Hqap_56x1917.qasm \
  --output-dir bench/runs/midpoint-mpo-p9 \
  --shots 1000 \
  --expected-bitstring 01101110111001100000100000001010011100101101010111110111 \
  --max-bond 768 --cutoff 0.0006 --seed 123 \
  --sabre-trials 90 --post-sabre-trials 50 --no-progress-limit 20
```

This is a long CPU run. It writes `summary.json`, `stats.json`, and
`samples.tsv`; success requires `termination_reason="completed"`, all 1,885
consolidated work gates consumed, and the expected bitstring recovered as the
sample mode. Use `tools/summarize_midpoint_mpo_priority_phase.py` to freeze a
completed campaign with compressed raw statistics, per-run ordering, recovery
telemetry, and plots.

To reproduce the same-Mac published-core arm, check out the recorded upstream
solver revision and keep the same pinned environment and sampling contract:

```bash
git clone https://github.com/alexgalda-m/peaked-mpo-solver \
  /tmp/peaked-mpo-solver-reference
git -C /tmp/peaked-mpo-solver-reference checkout \
  3bcdc1e5bfd6abb9425f71bd43e560d2b27f45c1

PYTHONPATH=src:/tmp/peaked-mpo-solver-reference \
  caffeinate -i .venv-mpo/bin/python \
  tools/run_published_peaked_solver_direct.py \
  --qasm src/mettleq/datasets/peaked_circuit_P9_Hqap_56x1917.qasm \
  --output-dir bench/runs/published-p9-direct \
  --shots 1000 \
  --expected-bitstring 01101110111001100000100000001010011100101101010111110111 \
  --max-bond 512 --cutoff 0.0006 --seed 123 \
  --sabre-trials 90 --post-sabre-trials 50 --no-progress-limit 20
```

### Run the forward-MPS peaked regressions

```bash
PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/benchmark_peaked_circuits.py \
  --output-dir bench/runs/peaked \
  --qubits 6,8,10 \
  --topologies linear,grid,long_range,all_to_all \
  --depth 2 --shots 1024 --warmups 1 --repeats 3 \
  --dmax 64 --eps 1e-10 --published-mode both \
  --published-prefix-operations 250 --published-timeout-s 120
```

`--published-mode full` retains the historical forward-MPS boundary probe in a
killable child. Its result is labeled `mettleq_forward_mps` and
`algorithm_matched=false`; use the mirrored family for CI and the explicit
`MidpointMPOSimulator` command above for a trustworthy full P9 result.

### Statevector or MPS backend

```bash
./bench.sh --circuit qaoa --backend sv  --qubits 8,12,16 --repro
./bench.sh --circuit qaoa --backend mps --qubits 8,12,16 --repro \
  --mps-dmax 64 --mps-eps 1e-10
```

### OpenQASM suite

```bash
./bench.sh --qasm-suite --qasm-max-qubits 18 --qasm-timeout-ms 30000
```

### Apple GPU inspection

```bash
PYTHONPATH=src .venv/bin/python tools/profile_apple_gpu.py \
  --qubits 20 --repeats 7 --output /tmp/mettleq-profile.json
```

The profiler records synchronized cold/warm timing, allocator memory, estimated
state traffic, full-state CPU readback, dispatch evidence, and pure-MLX parity.
Use the paired benchmark sweep—not a profiler run—for publication claims.

### Benchmark families

| Category | Circuit keys |
| --- | --- |
| Spin and Hamiltonian dynamics | `hamiltonian_simulation`, `time_evolution`, `trotter`, `steady_state`, `heisenberg`, `heisenberg_xxz`, `heisenberg_random_field`, `tfim`, `tfim_trotter2`, `tfim_random_field`, `long_range_ising`, `ladder_heisenberg` |
| Variational and QML | `qcbm`, `qaoa`, `vqe`, `variational_circuit` |
| Algorithms and circuits | `qft`, `phase_estimation`, `grover`, `ghz`, `random_circuit` |

Run any family with `./bench.sh --circuit <key>`. Use `./bench.sh --help` for
qubit schedules, caps, MPS controls, QASM options, and reproducibility settings.

## Reproducing the results

The complete 2026-07-15 speed-sweep evidence bundle is tracked under
[`assets/benchmarks-frozen/fork-m3pro-20260715/`](assets/benchmarks-frozen/fork-m3pro-20260715/):

- 290 paired raw timing rows
- 29-workload summary CSV
- exact-commit run manifest and aggregate evidence summary
- transcribed historical ratios with source attribution
- historical/current comparison CSV and JSON summary
- prior-fork/current comparison CSV and JSON summary
- 25-qubit checkpoint crossover raw data, validation, summary, and manifest

The Step 3 memory and default-path evidence is tracked separately under
[`assets/benchmarks-frozen/fork-m3pro-20260715-step3/`](assets/benchmarks-frozen/fork-m3pro-20260715-step3/). It contains the exact-commit 20- and
25-qubit crossovers, four raw A–B–B–A guardrail campaigns, numerical
validation, drift-balanced comparison output, and the plotted source data.

The Step 4 preflight and 22–27-qubit cross-workload evidence is frozen under
[`assets/benchmarks-frozen/fork-m3pro-20260715-step4/`](assets/benchmarks-frozen/fork-m3pro-20260715-step4/). It contains 324 raw timing rows, reviewed paired summaries, 90 full-state validation rows, the exact-commit and 36 child manifests, M3 Pro preflight reports through 31 qubits, and the plotted source data.

The Step 5 native-SDK evidence is frozen under
[`assets/benchmarks-frozen/fork-m3pro-20260715-step5-sdk/`](assets/benchmarks-frozen/fork-m3pro-20260715-step5-sdk/). It contains rotating-order Qiskit and PennyLane timings, separate native-result comparisons, numerical validation, the exact engine manifest, and a chart generated from the summary JSON.

The Step 6 adaptive statevector/MPS evidence is frozen under
[`assets/benchmarks-frozen/fork-m3pro-20260715-step6-adaptive-sdk/`](assets/benchmarks-frozen/fork-m3pro-20260715-step6-adaptive-sdk/). It contains pure-MLX and custom-Metal CPU/GPU crossover sweeps, the Qiskit/PennyLane method matrix, MPS device and truncation diagnostics, the unchanged Step 5 protocol refresh, and the 329-test result.

The Step 7 MPS entanglement-limit evidence is frozen under
[`assets/benchmarks-frozen/fork-m3pro-20260715-step7-mps-limits/`](assets/benchmarks-frozen/fork-m3pro-20260715-step7-mps-limits/). It contains 49 `Dmax=64` topology/qubit/depth probes, `Dmax=32/64/128` convergence rows, independent statevector validation through 20 qubits, retained SVD/norm failures, exact-commit manifests, and the plotted source data.

The Step 8 reliable/routed MPS evidence is frozen under
[`assets/benchmarks-frozen/fork-m3pro-20260716-step8-mps-reliability/`](assets/benchmarks-frozen/fork-m3pro-20260716-step8-mps-reliability/). It contains the clean-engine 26-case standard and 16-case boundary campaigns, `Dmax=32/64/128` convergence inputs, the matched four-path MettleQ/Aer campaign, raw rows, summaries, manifests, and charts.

The Step 9 peaked-circuit evidence is frozen under
[`fork-m3pro-20260716-step9-mettleq-peaked/`](assets/benchmarks-frozen/fork-m3pro-20260716-step9-mettleq-peaked/).
It contains the matched 12-cell mirrored family, the 250-operation prefix, the
completed full P9 forward-MPS boundary run, and its negative peak-recovery
evidence. Step 10 repeats the same 12-cell protocol after batched MPS sampling
under
[`fork-m3pro-20260716-step10-batched-mps-sampling/`](assets/benchmarks-frozen/fork-m3pro-20260716-step10-batched-mps-sampling/).

The Step 11 midpoint-MPO evidence is frozen under
[`fork-m3pro-20260716-step11-midpoint-mpo/`](assets/benchmarks-frozen/fork-m3pro-20260716-step11-midpoint-mpo/).
It contains two full MettleQ P9 runs, the same-Mac published-core arm, all
3,000 seeded samples, full tensor-network stats, bond convergence, the retained
cutoff/Qiskit-2/CLI failure arms, CPU two-site/SVD before-after profiling, and
the evidence-based decision to keep native GPU MPS disabled.

Pre-rename raw evidence retains its original schema and is not rewritten.
Current README-facing plots with MettleQ labels are reproducibly rendered from
those frozen rows under
[`fork-m3pro-20260716-mettleq-relabel/`](assets/benchmarks-frozen/fork-m3pro-20260716-mettleq-relabel/).

Recreate the current sweep and comparison:

```bash
unset METTLEQ_METAL_CHECKPOINT_BUDGET_MB METTLEQ_DENSE_ONLY
PYTHONPATH=src caffeinate -i .venv/bin/python tools/shader_suite_sweep.py \
  --outdir bench/runs/shader_sweep_current --qubits 25 --repeats 10

PYTHONPATH=src .venv/bin/python tools/compare_shader_sweeps.py \
  --historical assets/benchmarks-frozen/fork-m3pro-20260715/historical_chart_ratios.csv \
  --current bench/runs/shader_sweep_current/shader_sweep_summary.csv \
  --outdir bench/runs/shader_sweep_current \
  --historical-label "upstream M1 Max chart" \
  --current-label "fork M3 Pro current"

PYTHONPATH=src .venv/bin/python tools/checkpoint_sweep.py \
  --outdir bench/runs/checkpoint_sweep_n25 --qubits 25 --steps 6 \
  --repeats 7 --warmups 1 --budgets-mib 4096 2048 1024 512 256

PYTHONPATH=src .venv/bin/python tools/plot_checkpoint_sweep.py \
  --summary bench/runs/checkpoint_sweep_n25/checkpoint_sweep_summary.csv \
  --output bench/runs/checkpoint_sweep_n25/checkpoint_tradeoff.png \
  --title "25-qubit TFIM: intra-layer Metal streaming"

PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/memory_policy_campaign.py \
  --outdir bench/runs/memory_policy_current \
  --qubits 22 23 24 25 26 27 --repeats 3 --warmups 1

PYTHONPATH=src .venv/bin/python tools/plot_memory_policy_campaign.py \
  --summary bench/runs/memory_policy_current/memory_policy_summary.csv \
  --output bench/runs/memory_policy_current/memory_policy.png

METTLEQ_METAL_KERNELS=auto PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/benchmark_sdk_adapters.py \
  --outdir bench/runs/sdk-adapters \
  --qubits 20 --steps 4 --warmups 1 --repeats 7

PYTHONPATH=src .venv/bin/python tools/plot_sdk_adapters.py \
  --summary bench/runs/sdk-adapters/sdk_adapter_summary.json \
  --output bench/runs/sdk-adapters/sdk_adapter_timings.png

unset METTLEQ_METAL_KERNELS
PYTHONPATH=src .venv/bin/python tools/benchmark_execution_policy.py \
  --outdir bench/runs/execution-policy-pure-mlx --warmups 1 --repeats 7

METTLEQ_METAL_KERNELS=auto PYTHONPATH=src .venv/bin/python \
  tools/benchmark_execution_policy.py \
  --outdir bench/runs/execution-policy-metal --warmups 1 --repeats 7

METTLEQ_METAL_KERNELS=auto PYTHONPATH=src .venv/bin/python \
  tools/benchmark_sdk_method_matrix.py \
  --outdir bench/runs/sdk-method-matrix \
  --qubits 20 --steps 2 --warmups 1 --repeats 7

unset METTLEQ_METAL_KERNELS
PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/benchmark_mps_limits.py \
  --outdir bench/runs/mps-limits-step8 \
  --profile standard --dmax 64 --eps 1e-10 \
  --svd-driver auto --routing-strategy lookahead \
  --routing-lookahead 8 --timeout-seconds 60

PYTHONPATH=src caffeinate -i .venv/bin/python \
  tools/benchmark_mps_phase8.py \
  --outdir bench/runs/mps-matched-aer-step8 \
  --profile standard --dmax 64 --eps 1e-10 \
  --svd-driver auto --routing-lookahead 8 \
  --warmups 1 --repeats 3 --timeout-seconds 240
```

Transient runs belong under `bench/runs/`. Promote only reviewed evidence to
`assets/benchmarks-frozen/`; keep historical sample bundles immutable.

## MettleQ Studio

MettleQ Studio is the desktop companion for launching runs, monitoring progress,
viewing results, and exporting artifacts.

```bash
cd quantumstudio
./bin/appctl up
./bin/appctl status
./bin/appctl logs backend
./bin/appctl down
```

Build local UI artifacts with:

```bash
./scripts/build_flutter_app.sh --release
./scripts/build_dmg.sh
```

<details>
<summary><strong>Show the MettleQ Studio gallery</strong></summary>

![MettleQ Studio run view](quantumstudio/assets/screenshots/screen001.png)
![MettleQ Studio circuit view](quantumstudio/assets/screenshots/screen002.png)
![MettleQ Studio dashboard](quantumstudio/assets/screenshots/screen003.png)
![MettleQ Studio results view](quantumstudio/assets/screenshots/screen004.png)

</details>

## Repository map

| Path | Purpose |
| --- | --- |
| `src/mettleq/` | Simulator, gates, statevector, forward MPS, midpoint-MPO API, QASM, QML wrapper, execution plans, and Metal shaders |
| `src/mettleq/_vendor/peaked_mpo/` | Attributed Apache-2.0 midpoint-MPO/unswapping core derived from the published solver |
| `src/mettleq/integrations/` | Shared adapter layer, Qiskit `BackendV2`, and PennyLane device plugin |
| `src/mettleq/datasets/` | Published peaked-circuit input, integrity metadata, attribution, and third-party license |
| `src/tests/` | Simulator, algorithm, correctness, protocol, and Metal tests |
| `tools/` | GPU profiler, sweep runners, SDK benchmark, comparison plots, and supporting utilities |
| `bench.sh` | Main benchmark launcher |
| `bench_with_logging.sh` | Orchestrated benchmark and promotion workflow |
| `bench/runs/` | Ignored transient run output |
| `assets/benchmarks-frozen/` | Reviewed evidence and immutable sample bundles |
| `assets/perf-charts/` | README-ready performance figures |
| `datasets/qasm/local/` | Bundled OpenQASM corpus |
| `quantumstudio/` | Desktop UI, backend API, MCP server, and packaging scripts |
| `.github/workflows/ci.yml` | Portable CI and manually triggered Apple Silicon validation |

## Development

```bash
# All tests
./test.sh

# Simulator only
python -m pytest src/tests -q

# MettleQ Studio backend only
python -m pytest quantumstudio/tests -q

# One targeted module
./test.sh -- src/tests/mlxQMetalKernelsTest.py
```

Ordinary tests and generated plots use temporary directories. A successful full
test run should leave `git status --short` unchanged.

## Citation

The original Qupertino project citation remains:

```bibtex
@software{kashani_qupertino_2026,
  author       = {Shlomo Kashani},
  title        = {Qupertino / QuantumStudio: Apple Silicon Quantum Benchmarking Stack},
  year         = {2026},
  url          = {https://github.com/BoltzmannEntropy/Qupertino},
  note         = {GitHub repository}
}
```

For this independent fork:

```bibtex
@software{sharma_mettleq_2026,
  author       = {Sharma, Monit},
  title        = {MettleQ: Trustworthy Local Quantum Simulation for Apple Silicon},
  year         = {2026},
  url          = {https://github.com/MonitSharma/MettleQ},
  note         = {Independent fork of Qupertino}
}
```

The associated technical report is self-published but unpublished; its PDF and
LaTeX source are not distributed in this repository.

## License

MettleQ / MettleQ Studio is free and open source under the [MIT License](LICENSE).
The source and compiled macOS binaries may be used, modified, and redistributed,
including commercially, under the license terms. The vendored P9 benchmark
input is covered separately by the Apache License 2.0 and attribution stored in
[`src/mettleq/datasets/`](src/mettleq/datasets/). The derived midpoint-MPO core
retains its Apache-2.0 license and source notice under
[`src/mettleq/_vendor/peaked_mpo/`](src/mettleq/_vendor/peaked_mpo/).
