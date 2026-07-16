from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    # src/mettleq/paths.py → repo root is parents[2]
    # parents[0]=paths.py, [1]=mettleq, [2]=src, [3]=repo-root
    # Be defensive in case layout differs
    for up in here.parents:
        if (up / 'src').is_dir() and (up / 'paper').exists():
            return up
    # Fallback: two levels up from src/mettleq
    return here.parents[2]


def qasm_local_dir() -> Path:
    env = os.environ.get('METTLEQ_QASM_LOCAL')
    if env:
        p = Path(env)
        return p
    return _repo_root() / 'datasets' / 'qasm' / 'local'


def qasm_local_path(name: str) -> str:
    return str(qasm_local_dir() / name)


def mqtbench_dir() -> Path:
    env = os.environ.get('METTLEQ_MQTBENCH')
    if env:
        return Path(env)
    return _repo_root() / 'benchmarks' / 'mqtbench'


__all__ = ['qasm_local_dir', 'qasm_local_path', 'mqtbench_dir']

