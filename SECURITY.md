# Security policy

## Supported versions

MettleQ is currently an alpha research package. Security fixes are applied to
the latest release and the current `main` branch. Older snapshots and frozen
benchmark environments are retained for reproducibility, not maintained as
supported runtime environments.

## Reporting a vulnerability

Please use the repository's private **Report a vulnerability** form under the
GitHub Security tab. Do not open a public issue for suspected credential
exposure, arbitrary code execution, unsafe file handling, or dependency-chain
problems.

Include the affected MettleQ version or commit, macOS/Python versions, a minimal
reproduction, and the expected impact. Avoid attaching secrets, private circuit
data, machine captures, or unredacted logs.

MettleQ executes user-supplied Python and quantum circuits locally. It is not a
sandbox. Only run notebooks, QASM files, midpoint-MPO workers, and benchmark
scripts from sources you trust.

The `METTLEQ_MPO_PYTHON` setting intentionally permits an arbitrary interpreter
path so callers can isolate the optional midpoint-MPO dependency stack. Treat
that path as executable code and set it only to an interpreter you control.
