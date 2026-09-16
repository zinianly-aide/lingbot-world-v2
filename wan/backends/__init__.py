"""Backend abstraction layer for LingBot-World 2.0 inference.

Defines a uniform interface so that different compute backends
(PyTorch/MPS, MLX, future: CUDA, CPU) can be swapped without
changing the caller code.

POC stage: only VAE decode goes through this interface.
Default backend remains 'mps' (no behavior change).
"""

from .base import InferenceBackend, BackendConfig
from .registry import get_backend, register_backend, list_backends

__all__ = [
    "InferenceBackend",
    "BackendConfig",
    "get_backend",
    "register_backend",
    "list_backends",
]
