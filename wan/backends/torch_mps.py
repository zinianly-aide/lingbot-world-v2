"""PyTorch MPS backend implementation.

Wraps the existing PyTorch/MPS inference path (Wan2_1_VAE, etc.)
behind the InferenceBackend interface. This is the default backend
and preserves all existing behavior.
"""

import gc
import logging
from typing import Any, Dict, Optional

import torch

from .base import BackendConfig, InferenceBackend

logger = logging.getLogger(__name__)


class TorchMPSBackend(InferenceBackend):
    """PyTorch MPS inference backend.

    Uses the existing Wan2_1_VAE and related models with PyTorch MPS.
    This is the default backend and produces identical results to the
    existing generate.py pipeline.
    """

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._device = torch.device(config.device or "mps")
        self._dtype_map = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }
        self._dtype = self._dtype_map.get(config.dtype, torch.float32)

    def is_available(self) -> bool:
        return torch.backends.mps.is_available()

    def load_model(self, model_type: str, checkpoint_path: str, **kwargs) -> Any:
        if model_type in self._loaded_models:
            return self._loaded_models[model_type]

        if model_type == "vae":
            from wan.modules.vae2_1 import Wan2_1_VAE
            # MPS default uses FP16 for VAE (existing behavior)
            vae_dtype = torch.float16 if self._device.type == "mps" else self._dtype
            model = Wan2_1_VAE(
                vae_pth=checkpoint_path,
                dtype=vae_dtype,
                device=self._device,
            )
        elif model_type == "dit":
            # DiT loading is complex; delegate to existing pipeline
            raise NotImplementedError(
                "DiT loading via backend interface is POC-stage only. "
                "Use the existing generate.py pipeline for DiT."
            )
        elif model_type == "text_encoder":
            raise NotImplementedError(
                "Text encoder loading via backend interface is POC-stage only."
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self._loaded_models[model_type] = model
        logger.info(f"TorchMPSBackend: loaded {model_type} from {checkpoint_path}")
        return model

    def unload_model(self, model_type: str) -> None:
        if model_type in self._loaded_models:
            del self._loaded_models[model_type]
            gc.collect()
            if self._device.type == "mps":
                torch.mps.empty_cache()
            logger.info(f"TorchMPSBackend: unloaded {model_type}")

    def unload_all(self) -> None:
        self._loaded_models.clear()
        gc.collect()
        if self._device.type == "mps":
            torch.mps.empty_cache()

    def sync(self) -> None:
        if self._device.type == "mps":
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        if self._device.type == "mps":
            torch.mps.empty_cache()

    def current_memory_mb(self) -> float:
        if self._device.type == "mps":
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        return 0.0

    def driver_memory_mb(self) -> float:
        if self._device.type == "mps":
            return torch.mps.driver_allocated_memory() / (1024 ** 2)
        return 0.0

    def vae_decode(self, latents, model=None, **kwargs) -> Any:
        """Decode latents using the VAE.

        Args:
            latents: Latent tensor [C, T, H, W] (will be moved to device).
            model: Pre-loaded VAE model. If None, must be loaded first.
            **kwargs: Additional options (scale, etc.).

        Returns:
            Decoded video tensor [C, T, H, W] in [-1, 1], on CPU.
        """
        if model is None:
            if "vae" not in self._loaded_models:
                raise RuntimeError("VAE model not loaded. Call load_model('vae', ...) first.")
            model = self._loaded_models["vae"]

        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(latents)}")

        z = latents.to(device=self._device)
        self.sync()
        videos = model.decode([z])
        self.sync()
        result = videos[0].cpu()
        return result
