"""Back-compat shim: the Metal shaders moved to the mettleq.shaders package.

Import from mettleq.shaders directly in new code; this module re-exports the
public API so existing imports keep working.
"""
from .shaders import *  # noqa: F401,F403
from .shaders import __all__  # noqa: F401
