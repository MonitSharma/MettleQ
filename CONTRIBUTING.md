# Contributing to MettleQ

MettleQ welcomes correctness fixes, Apple GPU kernels, SDK compatibility work,
tests, documentation, and reproducible benchmark evidence.

## Development setup

```bash
git clone https://github.com/MonitSharma/MettleQ.git
cd MettleQ
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[sdk,tests,plot,mpo,dev]'
python -m pytest src/tests
```

Python 3.11 or newer and an Apple Silicon Mac are required for simulator work.
Portable documentation and backend tests also run on GitHub-hosted Linux.

## Change expectations

- Add a focused test for behavioral changes and run the relevant suite.
- Preserve Qiskit and PennyLane result semantics, wire ordering, and declared
  precision contracts.
- Keep statevector memory preflight and MPS accuracy evidence intact.
- Do not describe a speedup without raw timings, warmups/repeats, hardware and
  software versions, result contract, synchronization boundary, and numerical
  or statistical agreement.
- Compare backends only when their circuits and outputs are matched. Keep
  cross-device and adjacent-width observations explicitly labelled.
- Do not commit credentials, personal paths, raw profiler captures, large
  transient runs, or private circuit data.

## Before proposing a change

```bash
python -m ruff check src/mettleq src/mlxq tools setup.py --select E9,F63,F7,F82
python -m build --outdir dist/release
python -m twine check --strict dist/release/*
python tools/check_distribution.py dist/release
git diff --check
```

Metal performance changes should additionally run the relevant matched
campaign on an otherwise idle Mac. Record rejected memory-safety cases and
accuracy failures; they are evidence, not missing rows to hide.

MettleQ is independently maintained and does not send changes to the original
Qupertino repository unless that upstream project explicitly requests them.

See [the release guide](docs/releasing.md) for installed-artifact validation and
PyPI Trusted Publisher setup. Research notebook and benchmark campaigns live on
the `development` branch.
