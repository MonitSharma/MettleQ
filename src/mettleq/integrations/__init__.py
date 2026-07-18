"""Native adapters for third-party quantum SDKs.

The SDK dependencies stay optional: importing :mod:`mettleq` does not import
Qiskit or PennyLane.  Import an adapter from its dedicated module, or use the
lazy attributes exported here.
"""

from importlib import import_module

__all__ = [
    "MettleQBackend",
    "MettleQSamplerV2",
    "MettleQEstimatorV2",
    "AdaptiveQiskitBackend",
    "MettleQMidpointMPOBackend",
    "MettleQDevice",
    "AdaptivePennyLaneDevice",
    "CircuitProfile",
    "SDKEngineDecision",
    "profile_operations",
    "recommend_sdk_engine",
    "QupertinoBackend",
    "QupertinoSamplerV2",
    "QupertinoEstimatorV2",
    "QupertinoDevice",
]


def __getattr__(name):
    if name in {
        "CircuitProfile", "SDKEngineDecision", "profile_operations",
        "recommend_sdk_engine",
    }:
        return getattr(import_module(".policy", __name__), name)
    if name in {
        "MettleQBackend",
        "MettleQSamplerV2",
        "MettleQEstimatorV2",
        "AdaptiveQiskitBackend",
        "MettleQMidpointMPOBackend",
    }:
        return getattr(import_module(".qiskit", __name__), name)
    if name in {"MettleQDevice", "AdaptivePennyLaneDevice"}:
        return getattr(import_module(".pennylane", __name__), name)
    legacy = {
        "QupertinoBackend": (".qiskit", "QupertinoBackend"),
        "QupertinoSamplerV2": (".qiskit", "QupertinoSamplerV2"),
        "QupertinoEstimatorV2": (".qiskit", "QupertinoEstimatorV2"),
        "QupertinoDevice": (".pennylane", "QupertinoDevice"),
    }
    if name in legacy:
        module_name, attribute = legacy[name]
        return getattr(import_module(module_name, __name__), attribute)
    raise AttributeError(name)
