"""MLX import boundary with an actionable unsupported-platform error."""

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - depends on host platform
    raise RuntimeError(
        "MettleQ's simulation kernels require MLX on macOS arm64 (Apple Silicon). "
        "Install MettleQ on an Apple Silicon Mac, or use the MLX-independent "
        "mettleq.draw and mettleq.qasm utilities on this platform."
    ) from exc

__all__ = ["mx"]
