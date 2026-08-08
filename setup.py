"""Optional native helpers for source and wheel builds."""
from __future__ import annotations

import sys

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Keep the pure-Python fallback usable when no C++ compiler exists."""

    def run(self):
        try:
            super().run()
        except Exception as error:  # pragma: no cover - build-only path
            self.announce(
                f"optional mettleq native MPS extension unavailable: {error}",
                level=3,
            )


setup(
    ext_modules=[
        Extension(
            "mettleq._mps_native",
            ["src/mettleq/_mps_native.cpp"],
            language="c++",
            include_dirs=[np.get_include()],
            extra_compile_args=(
                ["-O3", "-std=c++17", "-DACCELERATE_NEW_LAPACK"]
                if sys.platform == "darwin"
                else ["-O3", "-std=c++17"]
            ),
            extra_link_args=(
                ["-framework", "Accelerate"] if sys.platform == "darwin" else []
            ),
        )
    ],
    cmdclass={"build_ext": OptionalBuildExt},
)
