"""mlxq package initializer.

This package optionally depends on MLX (Apple Silicon) for numeric kernels.
When MLX isn't available, drawing utilities (ASCII/Matplotlib/Quantikz) remain
importable so notebooks can be used for visualization without MLX.
"""

try:
    from importlib.metadata import version as _package_version

    __version__ = _package_version("mlxq")
except Exception:  # source tree without installed package metadata
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
]

# Detect MLX safely without importing it (to avoid side effects in non-MLX envs)
_HAS_MLX = False
try:
    import importlib.util as _ilu  # type: ignore
    _HAS_MLX = _ilu.find_spec("mlx.core") is not None
except Exception:
    _HAS_MLX = False

if _HAS_MLX:
    try:
        from .gates import (
            I, X, Y, Z, H, S, SDG, T, TDG,
            RX, RY, RZ, PhaseShift, U1, U2, U3,
            SWAP, iSWAP, CNOT, CZ, CPHASE, CRX, CRY, CRZ,
            Toffoli, Fredkin, CH, MultiControlledX, MultiControlledZ,
        )
        from .sim import StateVectorSimulator, qft, iqft
        from .device import Device
        from .execution import (
            StatevectorMemoryError,
            metal_runtime_status,
            state_memory_estimate,
            statevector_preflight,
        )
        from .planning import ExecutionSelection, select_execution
        from .mps_accuracy import MPSAccuracyError
        from .mps_state import MPSNumericalError, MPSOptions
        from .observables import is_hermitian, commutator

        __all__ += [
            "I", "X", "Y", "Z", "H", "S", "SDG", "T", "TDG",
            "RX", "RY", "RZ", "PhaseShift", "U1", "U2", "U3",
            "SWAP", "iSWAP", "CNOT", "CZ", "CPHASE", "CRX", "CRY", "CRZ",
            "Toffoli", "Fredkin", "CH", "MultiControlledX", "MultiControlledZ",
            "StateVectorSimulator", "qft", "iqft", "Device",
            "metal_runtime_status", "state_memory_estimate",
            "statevector_preflight", "StatevectorMemoryError",
            "ExecutionSelection", "select_execution",
            "MPSOptions", "MPSNumericalError", "MPSAccuracyError",
            "is_hermitian", "commutator",
        ]
    except Exception:
        # If MLX is present but imports fail, keep drawing utilities available
        pass
