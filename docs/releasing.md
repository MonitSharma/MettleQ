# Releasing MettleQ to PyPI

The distribution and import name are `mettleq`. MLX simulation targets Apple
Silicon on macOS 14+; portable drawing and QASM tools work without MLX.
Python 3.11, 3.12, and 3.13 receive native arm64 wheels. Newer Python versions
may install from the sdist but are not part of the tested release matrix.

## Local validation on an Apple Silicon Mac

Use an environment with access to Metal; headless VMs and restricted sandboxes
can prevent GPU initialization even when the host is an Apple Silicon Mac.

```bash
python3 -m venv .venv-release
source .venv-release/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[sdk,tests,plot,mpo,dev]'
python -m pip check
python -m ruff check src/mettleq src/mlxq tools setup.py --select E9,F63,F7,F82
python -m pytest src/tests
export MACOSX_DEPLOYMENT_TARGET=14.0
export ARCHFLAGS="-arch arm64"
export _PYTHON_HOST_PLATFORM=macosx-14.0-arm64
python -m build --outdir dist/release
python -m twine check --strict dist/release/*
python tools/check_distribution.py dist/release --version 0.3.1 --apple-silicon
```

Use a fresh, empty output directory for each candidate. `python -m build`
builds the wheel from the sdist so missing source files fail before upload.
Install the built wheel into a separate environment, then run the installed
checks and copied tests outside the checkout:

```bash
python3 -m venv /tmp/mettleq-release-install
/tmp/mettleq-release-install/bin/python -m pip install 'dist/release/mettleq-0.3.1-'*.whl
/tmp/mettleq-release-install/bin/python -m pip install 'mettleq[sdk,tests,plot,mpo]==0.3.1'
cd /tmp
/path/to/repo/.venv-release/bin/python /path/to/repo/tools/check_distribution.py /path/to/repo/dist/release
/tmp/mettleq-release-install/bin/python /path/to/repo/tools/smoke_installed.py --metal --require-native
```

The extras command resolves the already installed version and its dependencies;
it does not need MettleQ to be published. Also install the `.tar.gz` into another
clean environment and run `smoke_installed.py` there. Check bundled QASM,
PennyLane entry points, the `mettleq-mpo --help` command, and native-extension
loading. CI copies tests and example helpers into a temporary directory to
ensure the test suite exercises the wheel rather than `src/mettleq`.

## GitHub Actions setup

`release-build.yml` is shared by the PyPI and TestPyPI publishing workflows.
It builds and checks the sdist and all three arm64 wheels, then runs the entire
suite against the installed Python 3.11–3.13 wheels on a Metal-capable runner.
Publication waits for every job to pass. Native-extension fallback is accepted
for user source installations but rejected for published Apple Silicon wheels.

Register a runner with labels `self-hosted`, `macOS`, `ARM64` and an accessible
Apple GPU. Alternatively set the repository variable
`METTLEQ_METAL_RUNNER_JSON` to a JSON runner-label list, for example
`["macos-15-xlarge"]` for a paid GPU-capable GitHub runner. Regular hosted macOS
VMs are used for wheel builds, not as evidence of Metal correctness. Keep
self-hosted jobs restricted to trusted manual runs and release tags. The Metal
job uses standalone Python via uv, so a self-hosted Mac does not need root
access to GitHub's hosted-runner Python directory. This repository currently
selects the additional `mettleq-metal` label through that variable. A temporary
runner can be registered for validation and removed afterward; an available
Metal runner is required whenever the publishing workflow runs.

Create GitHub environments `testpypi` and `pypi`, with reviewer protection as
appropriate. Configure a Trusted Publisher on each index:

| Setting | TestPyPI | PyPI |
| --- | --- | --- |
| Project | `mettleq` | `mettleq` |
| Owner | `MonitSharma` | `MonitSharma` |
| Repository | `MettleQ` | `MettleQ` |
| Workflow | `publish-testpypi.yml` | `publish.yml` |
| Environment | `testpypi` | `pypi` |

For an unpublished project, use a pending publisher. A workflow file alone
cannot reserve the name or configure either account. Confirm name ownership
before tagging a release. No PyPI API token is needed with Trusted Publishing.

1. Update `project.version` in `pyproject.toml`, `CITATION.cff`, and the changelog. The runtime
   version comes from installed distribution metadata.
2. Commit the reviewed changes and run CI, then manually dispatch
   `publish-testpypi.yml` on the release commit. Never overwrite a version that
   already exists on either index.
3. Install the candidate from TestPyPI (install dependencies from PyPI first,
   then use `--index-url https://test.pypi.org/simple/ --no-deps` for MettleQ).
   Run the installed Metal checks on an Apple Silicon Mac.
4. Create and push `v<project.version>`, such as `v0.3.1`. The PyPI workflow
   rejects tag/version mismatches and publishes only after release validation.
5. Install from PyPI into a clean environment and repeat the smoke checks.

## Validation evidence

See [the PyPI readiness audit](pypi-readiness.md) for checks run on the release
candidate and remaining account/runner setup. Full static typing is not a
release gate: existing annotations still need cleanup, despite the PEP 561
marker that makes them available to consumers.

## Licensing and build fallback

MettleQ's own code is MIT licensed. Bundled third-party solver code and its
benchmark input use Apache-2.0; distribution metadata declares
`MIT AND Apache-2.0` and includes all licenses and notices.

NumPy 2 headers are used in isolated builds for NumPy 1/2 ABI compatibility;
runtime NumPy remains `>=1.24,<3`. Native MPS helpers use Apple's Accelerate.
A missing compiler can fall back to Python/SciPy. To deliberately omit the
extension use `METTLEQ_DISABLE_NATIVE=1 python -m build`; this produces a
pure-Python wheel, while the MLX dependency still applies on macOS arm64.
