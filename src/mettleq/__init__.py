"""mettleq package initializer.

This package optionally depends on MLX (Apple Silicon) for numeric kernels.
When MLX isn't available, drawing utilities (ASCII/Matplotlib/Quantikz) remain
importable so notebooks can be used for visualization without MLX.
"""

try:
    from importlib.metadata import PackageNotFoundError, version as _package_version

    __version__ = _package_version("mettleq")
except PackageNotFoundError:  # source tree without installed package metadata
    __version__ = "0+unknown"

from .pretty import info, success, warn, error, table
from .draw import circuit_ascii, circuit_mpl, schedule_columns, random_circuit
from .quantikz import circuit_to_quantikz, write_quantikz_tex

__all__ = [
    # Always available utilities
    "__version__",
    "info", "success", "warn", "error", "table",
    "circuit_ascii", "circuit_mpl", "schedule_columns", "random_circuit",
    "circuit_to_quantikz", "write_quantikz_tex",
    "MidpointMPODependencyError", "MidpointMPOError", "MidpointMPOWorkerError",
    "MidpointMPOOptions", "MidpointMPOResult", "MidpointMPOSimulator",
    "IsolatedMidpointMPOSimulator",
    "build_midpoint_mpo_convergence_report",
]

_MIDPOINT_MPO_EXPORTS = {
    "MidpointMPODependencyError",
    "MidpointMPOError",
    "MidpointMPOWorkerError",
    "MidpointMPOOptions",
    "MidpointMPOResult",
    "MidpointMPOSimulator",
    "IsolatedMidpointMPOSimulator",
    "build_midpoint_mpo_convergence_report",
}

_MLX_EXPORTS = {
    "I": ("gates", "I"), "X": ("gates", "X"), "Y": ("gates", "Y"),
    "Z": ("gates", "Z"), "H": ("gates", "H"), "S": ("gates", "S"),
    "SDG": ("gates", "SDG"), "T": ("gates", "T"), "TDG": ("gates", "TDG"),
    "RX": ("gates", "RX"), "RY": ("gates", "RY"), "RZ": ("gates", "RZ"),
    "PhaseShift": ("gates", "PhaseShift"), "U1": ("gates", "U1"),
    "U2": ("gates", "U2"), "U3": ("gates", "U3"), "SWAP": ("gates", "SWAP"),
    "iSWAP": ("gates", "iSWAP"), "CNOT": ("gates", "CNOT"),
    "CZ": ("gates", "CZ"), "CPHASE": ("gates", "CPHASE"),
    "CRX": ("gates", "CRX"), "CRY": ("gates", "CRY"), "CRZ": ("gates", "CRZ"),
    "Toffoli": ("gates", "Toffoli"), "Fredkin": ("gates", "Fredkin"),
    "CH": ("gates", "CH"), "MultiControlledX": ("gates", "MultiControlledX"),
    "MultiControlledZ": ("gates", "MultiControlledZ"),
    "StateVectorSimulator": ("sim", "StateVectorSimulator"),
    "qft": ("sim", "qft"), "iqft": ("sim", "iqft"),
    "Device": ("device", "Device"),
    "StatevectorMemoryError": ("execution", "StatevectorMemoryError"),
    "metal_runtime_status": ("execution", "metal_runtime_status"),
    "state_memory_estimate": ("execution", "state_memory_estimate"),
    "statevector_preflight": ("execution", "statevector_preflight"),
    "ExecutionSelection": ("planning", "ExecutionSelection"),
    "select_execution": ("planning", "select_execution"),
    "MPSAccuracyError": ("mps_accuracy", "MPSAccuracyError"),
    "MPSOptions": ("mps_state", "MPSOptions"),
    "MPSNumericalError": ("mps_state", "MPSNumericalError"),
    "is_hermitian": ("observables", "is_hermitian"),
    "commutator": ("observables", "commutator"),
}

_MLX_IMPORT_ERROR = None


def __getattr__(name):
    if name in _MLX_EXPORTS:
        global _MLX_IMPORT_ERROR
        if _MLX_IMPORT_ERROR is not None:
            raise _MLX_IMPORT_ERROR
        module_name, attribute = _MLX_EXPORTS[name]
        try:
            from importlib import import_module

            value = getattr(import_module(f".{module_name}", __name__), attribute)
        except Exception as exc:
            # A headless MLX import can leave its extension partially loaded.
            # Cache the failure so a later SDK import cannot re-enter MLX and
            # trigger nanobind duplicate-registration aborts.
            _MLX_IMPORT_ERROR = exc
            raise
        if name not in __all__:
            __all__.append(name)
        return value
    if name not in _MIDPOINT_MPO_EXPORTS:
        raise AttributeError(name)
    from . import midpoint_mpo as _midpoint_mpo

    if name == "build_midpoint_mpo_convergence_report":
        return _midpoint_mpo.build_convergence_report
    return getattr(_midpoint_mpo, name)
