# MettleQ repeated midpoint-MPO evidence

This bundle freezes the order-balanced Apple M3 Pro campaign for the isolated
midpoint-MPO/TNO + unswapping worker. The normal caller used Qiskit
2.5.0; the pinned worker used Qiskit
1.4.5 and Quimb
1.11.2.

| Arm | Repeats | Median algorithm time (s) | Expected-peak fractions | Peak recovered in every run |
|---|---:|---:|---|---|
| mettleq_d512 | 2 | 1215.56 | 0.100, 0.100 | True |
| mettleq_d768 | 2 | 1215.71 | 0.103, 0.103 | True |
| published_d512 | 2 | 1167.10 | 0.100, 0.100 | True |

- Paired median MettleQ D512 / published-core D512 runtime ratio:
  **1.041x**.
- Paired median MettleQ D512 / D768 runtime ratio:
  **1.000x**.
- Fixed-D512 cutoff classification: **operationally_incomplete**; median
  expected-peak fraction spread across 5e-4, 6e-4, and 7e-4 is
  **not measurable**. The endpoint schedule contains **2**
  recorded operational failure(s); failed arms contribute no peak estimate.
- Safe-SVD telemetry across MettleQ arms: 4648250 calls,
  168632 routed to the persistent killable
  native service, 48 service
  failures, and 48 successful
  fresh-process Quimb-compatible fallbacks; 0
  calls reached the final Hermitian-eigensolver fallback.

`recovery-evidence/` preserves the unsafe Quimb `gesvd` process failure and the
superseded all-scaled-SVD experiment. Raw contraction statistics are losslessly
compressed as `stats.json.gz`; summaries, samples, logs, ordering, hashes, and
environment versions remain directly inspectable.

These timings characterize one P9 instance and one Mac. Finite-shot peak recovery
does not establish general circuit fidelity, and no general speedup claim is made.
