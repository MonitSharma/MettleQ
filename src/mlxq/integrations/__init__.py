"""Native adapters for third-party quantum SDKs.

The SDK dependencies stay optional: importing :mod:`mlxq` does not import
Qiskit or PennyLane.  Import an adapter from its dedicated module, or use the
lazy attributes exported here.
"""

from importlib import import_module

__all__ = ["QupertinoBackend", "QupertinoDevice"]


def __getattr__(name):
    if name == "QupertinoBackend":
        return import_module(".qiskit", __name__).QupertinoBackend
    if name == "QupertinoDevice":
        return import_module(".pennylane", __name__).QupertinoDevice
    raise AttributeError(name)
