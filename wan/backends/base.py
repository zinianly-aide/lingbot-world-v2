"""Base inference backend interface.

All backends (TorchMPS, MLX, etc.) must implement this interface.
The caller does not need to know which backend is in use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BackendConfig:
    """Configuration for an inference backend.

    Attributes:
        name: Backend name ('mps', 'mlx', 'cuda', 'cpu').
        dtype: Weight/compute dtype as string ('fp32', 'bf16', 'fp16').
        device: Device identifier ('mps', 'cpu', 'cuda:0').
        sequential_load: If True, load models one at a time and unload
            between stages to reduce peak memory.
        extra: Backend-specific configuration options.
    """

    name: str = "mps"
    dtype: str = "fp32"
    device: str = "mps"
    sequential_load: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class InferenceBackend(ABC):
    """Abstract base class for inference backends.

    A backend manages model lifecycle (load/unload) and provides
    inference methods. Each backend is responsible for its own
    memory management and synchronization.
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self._loaded_models: Dict[str, Any] = {}

    @abstractmethod
    def load_model(self, model_type: str, checkpoint_path: str, **kwargs) -> Any:
        """Load a model into this backend.

        Args:
            model_type: Type of model ('vae', 'dit', 'text_encoder').
            checkpoint_path: Path to model checkpoint.
            **kwargs: Backend-specific loading options.

        Returns:
            The loaded model object (backend-specific).
        """
        pass

    @abstractmethod
    def unload_model(self, model_type: str) -> None:
        """Unload a model and free its memory.

        Args:
            model_type: Type of model to unload.
        """
        pass

    @abstractmethod
    def unload_all(self) -> None:
        """Unload all models and free all backend memory."""
        pass

    @abstractmethod
    def sync(self) -> None:
        """Synchronize the backend device (wait for all ops to complete).

        For MPS: torch.mps.synchronize()
        For MLX: mx.eval() / mx.synchronize()
        For CPU: no-op
        """
        pass

    @abstractmethod
    def empty_cache(self) -> None:
        """Free unused memory from the backend allocator.

        For MPS: torch.mps.empty_cache()
        For MLX: mx.metal.clear_cache()
        """
        pass

    @abstractmethod
    def current_memory_mb(self) -> float:
        """Return currently allocated memory in MB."""
        pass

    @abstractmethod
    def driver_memory_mb(self) -> float:
        """Return driver-allocated memory in MB (for unified memory arch)."""
        pass

    @abstractmethod
    def vae_decode(self, latents, model=None, **kwargs) -> Any:
        """Run VAE decode on the given latents.

        Args:
            latents: Latent tensor (backend-specific format).
            model: Pre-loaded VAE model (if None, loads it).
            **kwargs: Additional decode options.

        Returns:
            Decoded video tensor.
        """
        pass

    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        return True

    def get_loaded_models(self) -> Dict[str, Any]:
        """Return dict of currently loaded models."""
        return dict(self._loaded_models)
