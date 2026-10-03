"""Optional native helpers; metadata lives in pyproject.toml."""
from __future__ import annotations

import os
import sys

import numpy as np
from setuptools import Extension, setup

# NumPy 2 headers produce extensions usable with both NumPy 1 and 2.
# Build isolation supplies these independently of the runtime NumPy version.
native_enabled = os.environ.get("METTLEQ_DISABLE_NATIVE") != "1"
compile_args = ["-O3", "-std=c++17"]
link_args = []
if sys.platform == "win32":
    compile_args = ["/O2", "/std:c++17"]
elif sys.platform == "darwin" and native_enabled:
    # Accelerate's new LAPACK API needs 13.3; publish with a 14.0 floor.
    target = os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "14.0")
    parts = target.split(".")
    if tuple(int(part) for part in (parts + ["0"])[:2]) < (14, 0):
        raise RuntimeError(
            "MettleQ native wheels require MACOSX_DEPLOYMENT_TARGET>=14.0; "
            "set METTLEQ_DISABLE_NATIVE=1 for a pure-Python build."
        )
    # Set the compiler floor explicitly: setuptools may have initialized its
    # platform settings before setup.py can update the environment.
    deployment_flag = f"-mmacosx-version-min={target}"
    compile_args += ["-DACCELERATE_NEW_LAPACK", deployment_flag]
    link_args = ["-framework", "Accelerate", deployment_flag]

setup(
    ext_modules=[
        Extension(
            "mettleq._mps_native",
            ["src/mettleq/_mps_native.cpp"],
            language="c++",
            include_dirs=[np.get_include()],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            optional=True,
        )
    ] if native_enabled else [],
)
