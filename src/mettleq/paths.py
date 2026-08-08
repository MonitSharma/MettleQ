from __future__ import annotations

import os
from pathlib import Path


def _package_data_dir() -> Path:
    """The ``datasets`` directory bundled inside the installed package."""
    return Path(__file__).resolve().parent / 'datasets'


def qasm_local_dir() -> Path:
    """Directory of bundled local QASM fixtures.

    Resolves to package data so it works from an installed wheel, not only a
    source checkout. Override with ``METTLEQ_QASM_LOCAL`` to point at a larger
    external circuit set (e.g. the full collection on the development branch).
    """
    env = os.environ.get('METTLEQ_QASM_LOCAL')
    if env:
        return Path(env)
    return _package_data_dir() / 'qasm_local'


def qasm_local_path(name: str) -> str:
    return str(qasm_local_dir() / name)


def mqtbench_dir() -> Path:
    env = os.environ.get('METTLEQ_MQTBENCH')
    if env:
        return Path(env)
    candidate = _package_data_dir() / 'mqtbench'
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        "The optional MQTBench dataset is not bundled with this installation. "
        "Set METTLEQ_MQTBENCH to a checked-out MQTBench directory."
    )


__all__ = ['qasm_local_dir', 'qasm_local_path', 'mqtbench_dir']
