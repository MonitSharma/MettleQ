"""Native adapters for third-party quantum SDKs.

The SDK dependencies stay optional: importing :mod:`mlxq` does not import
Qiskit or PennyLane.  Import an adapter from its dedicated module, or use the
lazy attributes exported here.
"""

from importlib import import_module

__all__ = [
    "QupertinoBackend",
    "QupertinoSamplerV2",
    "QupertinoEstimatorV2",
    "QupertinoDevice",
]


def __getattr__(name):
    if name in {
        "QupertinoBackend",
        "QupertinoSamplerV2",
        "QupertinoEstimatorV2",
    }:
        return getattr(import_module(".qiskit", __name__), name)
    if name == "QupertinoDevice":
        return import_module(".pennylane", __name__).QupertinoDevice
    raise AttributeError(name)
