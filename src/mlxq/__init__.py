"""Compatibility import for the former :mod:`mlxq` package name.

MettleQ is the canonical package as of version 0.2. Existing ``mlxq`` imports
continue to resolve through this namespace so downstream projects can migrate
without a flag day.
"""

from mettleq import *  # noqa: F401,F403
from mettleq import __all__, __version__
from mettleq import __path__ as _mettleq_path

# Let imports such as ``mlxq.device`` resolve to the canonical source modules.
__path__ = _mettleq_path
