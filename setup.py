"""Optional native helpers for source and wheel builds."""
from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    # The Accelerate entry points used by the native MPS helpers were
    # introduced in macOS 13.3. Wheel tags only encode major/minor versions,
    # so use macOS 14 as the published floor rather than claiming macOS 13.0.
    configured_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
    if configured_target is None:
        os.environ["MACOSX_DEPLOYMENT_TARGET"] = "14.0"
    else:
        major, _, minor = configured_target.partition(".")
        if (int(major), int(minor or 0)) < (14, 0):
            raise RuntimeError(
                "MettleQ published native wheels require macOS 14.0 or newer; "
                f"MACOSX_DEPLOYMENT_TARGET={configured_target!r} is too old"
            )

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
                ["-O3", "-std=c++17", "-DACCELERATE_NEW_LAPACK", "-mmacosx-version-min=14.0"]
                if sys.platform == "darwin"
                else ["-O3", "-std=c++17"]
            ),
            extra_link_args=(
                ["-framework", "Accelerate", "-mmacosx-version-min=14.0"]
                if sys.platform == "darwin" else []
            ),
        )
    ],
    cmdclass={"build_ext": OptionalBuildExt},
)
