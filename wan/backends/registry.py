"""Backend registry for dynamic backend selection."""

from typing import Dict, Type

from .base import BackendConfig, InferenceBackend

_REGISTRY: Dict[str, Type[InferenceBackend]] = {}


def register_backend(name: str, backend_cls: Type[InferenceBackend]) -> None:
    """Register a backend class under the given name."""
    _REGISTRY[name] = backend_cls


def get_backend(config: BackendConfig) -> InferenceBackend:
    """Get a backend instance by name.

    Args:
        config: Backend configuration (config.name selects the backend).

    Returns:
        An initialized InferenceBackend instance.

    Raises:
        ValueError: If the backend name is not registered.
    """
    if config.name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backend '{config.name}'. Available: {available}"
        )
    return _REGISTRY[config.name](config)


def list_backends() -> list:
    """Return list of registered backend names."""
    return sorted(_REGISTRY.keys())


# Auto-register built-in backends
def _register_builtins():
    try:
        from .torch_mps import TorchMPSBackend
        register_backend("mps", TorchMPSBackend)
    except ImportError:
        pass
    try:
        from .mlx_backend import MLXBackend
        register_backend("mlx", MLXBackend)
    except ImportError:
        pass


_register_builtins()
