"""Phase 0 safety harness for the CPU MPS optimization campaign.

Two jobs, one command:

1. Correctness parity — for small circuits (n <= PARITY_MAX_QUBITS) it compares
   native MPS expectation values against an exact Qiskit statevector reference
   and fails loudly if any observable drifts beyond ``--atol``.

2. Baseline timing + diagnostics — it times each case through the native MPS
   estimator and records the full diagnostics dict (svd_calls, routing swaps,
   renormalizations, phase breakdown, bond growth).

The output JSON is the guardrail: after each optimization, re-run and diff. Any
parity failure or timing regression beyond the agreed band is rejected.

Usage:
    PYTHONPATH=src .venv/bin/python tools/mps_opt_harness.py \
        --out bench/local/mps_baseline.json [--quick]
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector, SparsePauliOp

from tools.benchmark_mps_limits import build_circuit
from mettleq.integrations.qiskit import MettleQEstimatorV2

PARITY_MAX_QUBITS = 14

# (family, qubits, depth). Small ones double as parity checks; large ones are
# timing-only. Kept representative of the two regimes we care about:
# overhead-bound (local/shallow) and SVD/swap-bound (entangled).
FULL_CASES = [
    ("ghz_chain", 12, 1),
    ("line_brickwork", 14, 4),
    ("all_to_all", 12, 1),
    ("random_long_range", 14, 2),
    ("line_brickwork", 100, 8),      # overhead-bound flagship
    ("ring_brickwork", 50, 2),
    ("all_to_all", 20, 1),           # SVD/swap-bound flagship
    ("random_long_range", 32, 1),
]
QUICK_CASES = [c for c in FULL_CASES if c[1] <= 32 and not (c[0] == "all_to_all" and c[1] == 20)]

MAX_BOND = 64
REPEATS = 3


def _observables(n: int) -> list[SparsePauliOp]:
    """A small, structure-revealing observable set: Z on each end + a few ZZ."""
    obs = []
    for q in (0, n // 2, n - 1):
        label = ["I"] * n
        label[q] = "Z"
        obs.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    for a, b in ((0, 1), (0, n - 1)):
        label = ["I"] * n
        label[a] = label[b] = "Z"
        obs.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    return obs


def _exact_expectations(circ, obs_list) -> list[float]:
    sv = Statevector(circ)
    return [float(np.real(sv.expectation_value(o))) for o in obs_list]


def _mps_run(circ, obs_list):
    est = MettleQEstimatorV2(
        method="mps", allow_approximation=True, mps_max_bond_dimension=MAX_BOND
    )
    res = est.run([(circ, obs_list)]).result()[0]
    vals = [float(v) for v in np.atleast_1d(res.data.evs)]
    diag = (est.last_mps_diagnostics or [{}])[0] or {}
    return vals, diag


def run(cases, atol: float) -> dict:
    rows = []
    all_parity_ok = True
    for family, n, depth in cases:
        circ = build_circuit(family, n, depth)
        obs_list = _observables(n)

        # timing (median of REPEATS), with one warmup
        _mps_run(circ, obs_list)
        times = []
        vals = diag = None
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            vals, diag = _mps_run(circ, obs_list)
            times.append((time.perf_counter() - t0) * 1000.0)
        median_ms = statistics.median(times)

        parity_error = None
        parity_ok = None
        if n <= PARITY_MAX_QUBITS:
            exact = _exact_expectations(circ, obs_list)
            parity_error = max(abs(a - b) for a, b in zip(vals, exact))
            parity_ok = parity_error <= atol
            all_parity_ok = all_parity_ok and parity_ok

        keep = {
            k: diag.get(k)
            for k in (
                "svd_calls", "two_site_calls", "svd_total_ms",
                "two_site_contraction_total_ms", "two_site_split_total_ms",
                "renormalization_count", "routing_swaps",
                "routing_final_restore_swaps", "routing_effective_strategy",
                "maximum_bond_dimension_reached", "tensor_device",
            )
        }
        rows.append({
            "case": f"{family}:{n}:{depth}",
            "qubits": n,
            "median_ms": round(median_ms, 3),
            "parity_error": parity_error,
            "parity_ok": parity_ok,
            "diagnostics": keep,
        })
        flag = "" if parity_ok in (True, None) else "  <<< PARITY FAIL"
        pe = "n/a" if parity_error is None else f"{parity_error:.2e}"
        print(f"{family:18} n={n:<4} d={depth:<2} "
              f"{median_ms:9.2f} ms  parity={pe}  "
              f"svd={keep['svd_calls']} swaps={keep['routing_swaps']}{flag}")

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "max_bond": MAX_BOND,
        "repeats": REPEATS,
        "atol": atol,
        "all_parity_ok": all_parity_ok,
        "cases": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("bench/local/mps_baseline.json"))
    # 1e-5 reflects complex64 (float32) accumulation over hundreds of SVDs +
    # renormalizations; tight enough to catch real correctness regressions.
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--quick", action="store_true", help="skip the slow large cases")
    args = ap.parse_args()

    cases = QUICK_CASES if args.quick else FULL_CASES
    report = run(cases, args.atol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")
    print(f"All parity OK: {report['all_parity_ok']}")
    return 0 if report["all_parity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
