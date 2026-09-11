"""VLM perception backends (transformers / MLX)."""

from .base import PerceptionResult, WorldPerceptionBackend, WORLD_SCHEMA_PROMPT
from .mlx_backend import MlxBackend
from .transformers_backend import TransformersBackend

BACKENDS = {
    "transformers": TransformersBackend,
    "mlx": MlxBackend,
}


def create_backend(
    backend: str = "transformers",
    model_name: str | None = None,
    **kwargs,
) -> WorldPerceptionBackend:
    """Factory: create a perception backend by name.

    Default model names per backend:
      transformers -> openbmb/MiniCPM-V-4.6
      mlx          -> mlx-community/MiniCPM-V-4.6-4bit
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Available: {list(BACKENDS)}")
    if model_name is None:
        model_name = {
            "transformers": "openbmb/MiniCPM-V-4.6",
            "mlx": "mlx-community/MiniCPM-V-4.6-4bit",
        }[backend]
    return BACKENDS[backend](model_name=model_name, **kwargs)


__all__ = [
    "PerceptionResult",
    "WorldPerceptionBackend",
    "WORLD_SCHEMA_PROMPT",
    "TransformersBackend",
    "MlxBackend",
    "BACKENDS",
    "create_backend",
]
