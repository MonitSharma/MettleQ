"""Compatibility import for the former :mod:`mlxq` package name.

MettleQ is the canonical package as of version 0.2. Existing ``mlxq`` imports
continue to resolve through this namespace so downstream projects can migrate
without a flag day.
"""

import importlib
import sys
import warnings

warnings.warn(
    "The 'mlxq' package name is deprecated; use 'mettleq' instead. "
    "The compatibility package will be removed in MettleQ 1.0.",
    DeprecationWarning,
    stacklevel=2,
)

from mettleq import *  # noqa: F401,F403
from mettleq import __all__, __version__
from mettleq import __path__ as _mettleq_path

# Keep common MLX-independent modules as true aliases rather than relying only
# on a shared package path.
for _module_name in ("circuit", "draw", "pretty", "qasm", "quantikz"):
    _module = importlib.import_module(f"mettleq.{_module_name}")
    sys.modules.setdefault(f"{__name__}.{_module_name}", _module)

# Remaining imports such as ``mlxq.device`` resolve through the canonical path.
__path__ = _mettleq_path
