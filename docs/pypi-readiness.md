# PyPI readiness audit — 3 October 2026

MettleQ **0.3.1** is prepared for submission as an alpha Apple Silicon quantum
simulation backend. Release archives are collected in `dist/pypi-ready-0.3.1/`
(ignored by Git). Publishing is postponed while the maintainer's PyPI login is
unavailable.

The release uses 0.3.1 because the repository already has a `v0.3.0` tag. The
simulator implementation is unchanged from the locally validated 0.3.0 candidate
below; GitHub release validation checks the new artifacts. The 0.3.1 source suite
also passed with 364 tests and the same two development-only skips.

## Verified locally

Tests exercised clean, non-editable installations outside the repository,
using copies of the test suite and example helpers without `src/mettleq`.

| Environment | Result |
| --- | --- |
| Python 3.11.11 native arm64 wheel | 364 passed, 2 skipped |
| Python 3.12.9 native arm64 wheel | 364 passed, 2 skipped |
| Python 3.13.2 native arm64 wheel | 364 passed, 2 skipped |
| Clean sdist installation | Native extension, imports, data, CLI verified |
| Pure-Python wheel, no MLX/SDK packages installed | 4 portable tests passed; imports/data/CLI verified |
| Missing C/C++ compiler | Wheel build succeeds using optional-extension fallback; no native binary bundled |
| NumPy 1.24.4 + SciPy 1.10.1 + MLX 0.32.0, Python 3.11 | Native extension loads; both MPS CPU optimization tests pass |
| Fresh resolved dependencies in all three SDK test environments | `pip check` passes |
| Qiskit and PennyLane documented quickstarts | Bell counts and observable/probability checks pass |
| Wheel/sdist metadata and README | `twine check --strict` passes |
| Archive contents, licenses, entry points, typing marker | `tools/check_distribution.py` passes |
| Python compilation; fatal-error lint | `compileall` and Ruff E9/F63/F7/F82 pass |
| GitHub workflows | Official actionlint v1.7.12 and YAML parsing pass |
| Whitespace | `git diff --check` passes |

Host: macOS 27.0.1, arm64, with a working Apple Metal runtime. Fresh SDK test
environments resolved MLX 0.32.3, Qiskit 2.5.2, PennyLane 0.45.1, Qiskit Aer
0.17.2, PennyLane Lightning 0.45.0, and quimb 1.15.0. Python 3.11 resolved
NumPy 2.4.6; Python 3.12/3.13 resolved NumPy 2.5.3. Isolated builds used NumPy
2 headers; the separate NumPy 1 test verifies the native ABI fallback.

The two skipped tests in each complete suite require the research priority
benchmark harness, retained on `development`. No simulator or Metal tests were
skipped. Expected warnings cover the deprecated `mlxq` alias and PennyLane's
legacy device-level shots API.

## Changes made

- Native wheels now cover Python 3.11–3.13, with a macOS 14.0 arm64 tag.
- Both publishing workflows use shared release validation, require the complete
  artifact set, test installed wheels on real Metal, and reject tag/version
  mismatches. Only the publishing job receives an OIDC token permission.
- Hosted wheel builds use `macos-15`; GPU tests use an explicitly configured
  Metal-capable runner. Portable Linux and Windows installation checks remain
  in CI; macOS builds also run on regular pushes and pull requests.
- NumPy build requirements use the 2.x headers; runtime requirements permit
  1.24 through 2.x. MLX's minimum is the validated 0.32 release family.
- Native compilation uses standard setuptools optional-extension behavior and
  Windows compiler flags. `METTLEQ_DISABLE_NATIVE=1` deliberately builds without
  the extension; release wheels must contain it.
- The sdist contains test/example helpers, public release tooling, and docs.
  Wheel checks enforce package data, notices, licenses, and SDK entry points.
- Distribution licensing accounts for Apache-2.0 vendored code and fixtures,
  and the stale dataset attribution is corrected.
- Pytest rejects unknown config/markers; the old test launcher preserves exit
  status. Runtime version tests follow installed metadata instead of a literal.
- Contributor commands, research links, and the drawing annotation are fixed.

## Remaining external setup and limits

Configure Trusted Publishers for `mettleq` on PyPI/TestPyPI and provide an
available Metal-capable runner for each release. Matching GitHub environments
and the runner selection variable are configured. Follow
[the release guide](releasing.md). The public PyPI JSON endpoint returned 404
for `mettleq` during this audit; name reservation/ownership still requires the
index account. No tag was pushed and no package was uploaded.

GitHub Actions Linux, Windows, and macOS installation checks passed on Python
3.11–3.13. Release validation uses disposable runners on the developer's Apple
Silicon Mac for actual Metal tests. Runtime behavior on macOS 14 itself remains
unverified; wheel tags alone do not prove testing on that OS version.
MLX 0.32.0 publishes macOS 14 wheels as well as newer OS variants.

Full `mypy src/mettleq` reports 166 existing annotation errors across legacy
simulator, shader, QASM, and adapter modules. This is recorded as typing debt;
it is not a successful static type check or a claim that every annotation is
correct. Fatal-error lint, runtime tests, and distribution checks pass. A full
formatting/lint rewrite, performance recalibration, and research benchmark
campaign are outside this packaging release.

## Guidance consulted

- [PyPA packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [Setuptools extension modules](https://setuptools.pypa.io/en/stable/userguide/ext_modules.html)
- [NumPy guidance for downstream authors](https://numpy.org/doc/stable/dev/depending_on_numpy.html)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [GitHub GPU-capable macOS runners](https://docs.github.com/en/actions/reference/runners/larger-runners)

## GitHub submission preparation

The release branch targets 0.3.1 and is tracked in
[PR #21](https://github.com/MonitSharma/MettleQ/pull/21). Hosted Linux, Windows,
and macOS installation checks passed. The Metal validation job uses uv-managed
Python because setup-python's hardcoded hosted-runner cache path requires
privileges on a self-hosted Mac. GitHub environments already exist, and the
Metal runner selection variable now includes the dedicated `mettleq-metal`
label. Temporary runners unregister after their jobs; no runner service was
installed on the developer's Mac.

Release builds explicitly select arm64 even when the hosted Python interpreter
supports both Intel and Apple Silicon. The archive check verifies the macOS
14 arm64 wheel tags and the native Mach-O architecture before accepting them.

Publishing is postponed at the maintainer's request while PyPI account login
is unavailable. The existing TestPyPI project has version 0.3.0rc3; the previous
PyPI attempt failed because its Trusted Publisher was not configured. Leave
`v0.3.0` intact and use `v0.3.1` only when publication is authorized again.
