"""MLX backend implementation (POC stage).

Uses mlx-diffuser's AutoencoderKLWan for VAE decode as a POC.
This backend is experimental and not enabled by default.

To use:
  1. Convert the Wan2.1 native VAE to MLX format once:
       python scripts/convert_vae_to_mlx.py
         --pth /Volumes/ssd/lingbot-assets/Wan2.1_VAE.pth
         --out eval/bench_vae_mlx/vae_mlx
  2. Select it via BackendConfig(name="mlx") and load the VAE from that folder.

POC scope: only VAE decode is implemented. DiT and text encoder raise
NotImplementedError. Everything else stays on the MPS backend.
"""

import json
import logging
from typing import Any, Optional

import numpy as np

from .base import BackendConfig, InferenceBackend

logger = logging.getLogger(__name__)

# MLX is optional; import lazily so the module still imports (and registry
# registration becomes a no-op) on machines without mlx installed.
_MLX_AVAILABLE = False
_MLX_DIFFUSER_AVAILABLE = False
try:
    import mlx.core as mx  # noqa: F401
    _MLX_AVAILABLE = True
except ImportError:
    mx = None  # type: ignore

try:
    from mlx_diffuser import AutoencoderKLWan  # noqa: F401
    from mlx_diffuser.models.autoencoder_kl_wan import AutoencoderKLWanConfig  # noqa: F401
    _MLX_DIFFUSER_AVAILABLE = True
except ImportError:
    pass


class MLXBackend(InferenceBackend):
    """MLX inference backend (POC stage).

    Uses mlx-diffuser's AutoencoderKLWan for VAE decode. MLX operates on
    unified memory, so there is no separate device memory allocation; peak
    memory is approximated by the process RSS.
    """

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        if not _MLX_AVAILABLE:
            raise ImportError("mlx is not installed. Install with: pip install mlx")
        if not _MLX_DIFFUSER_AVAILABLE:
            raise ImportError(
                "mlx-diffuser is not installed. Install with: pip install mlx-diffuser"
            )
        self._dtype_map = {
            "fp32": mx.float32,
            "bf16": mx.bfloat16,
            "fp16": mx.float16,
        }
        self._dtype = self._dtype_map.get(config.dtype, mx.float32)

    def is_available(self) -> bool:
        return _MLX_AVAILABLE and _MLX_DIFFUSER_AVAILABLE

    def load_model(self, model_type: str, checkpoint_path: str, **kwargs) -> Any:
        if model_type in self._loaded_models:
            return self._loaded_models[model_type]

        if model_type == "vae":
            from mlx_diffuser import AutoencoderKLWan
            from mlx_diffuser.models.autoencoder_kl_wan import AutoencoderKLWanConfig
            from mlx.utils import tree_flatten
            from pathlib import Path

            folder = Path(checkpoint_path)
            with open(folder / "config.json") as f:
                raw_cfg = json.load(f)
            # Drop non-constructor keys (e.g. class_name).
            cfg = AutoencoderKLWanConfig(**{
                k: list(v) if isinstance(v, tuple) else v
                for k, v in raw_cfg.items() if k != "class_name"
            })
            model = AutoencoderKLWan(cfg)
            weights = mx.load(str(folder / "model.safetensors"))
            model.load_weights(list(weights.items()), strict=True)
            model.set_dtype(self._dtype)
            mx.eval(model.parameters())
            del weights  # drop source references
            mx.clear_cache()
            model.eval()
            self._vae_cfg = cfg
        elif model_type == "dit":
            raise NotImplementedError(
                "DiT via MLX backend is not implemented in POC stage."
            )
        elif model_type == "text_encoder":
            raise NotImplementedError(
                "Text encoder via MLX backend is not implemented in POC stage."
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self._loaded_models[model_type] = model
        logger.info(f"MLXBackend: loaded {model_type} from {checkpoint_path}")
        return model

    def unload_model(self, model_type: str) -> None:
        if model_type in self._loaded_models:
            del self._loaded_models[model_type]
            import gc
            gc.collect()
            mx.clear_cache()
            logger.info(f"MLXBackend: unloaded {model_type}")

    def unload_all(self) -> None:
        self._loaded_models.clear()
        import gc
        gc.collect()
        mx.clear_cache()

    def sync(self) -> None:
        # Force full evaluation of any pending lazy graph.
        mx.synchronize() if hasattr(mx, "synchronize") else mx.eval()

    def empty_cache(self) -> None:
        mx.clear_cache()

    def current_memory_mb(self) -> float:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)

    def driver_memory_mb(self) -> float:
        # MLX uses unified memory; report the Metal cache footprint when available.
        try:
            return mx.get_cache_memory() / (1024 ** 2)
        except Exception:
            return 0.0

    def vae_decode(self, latents, model=None, **kwargs) -> np.ndarray:
        """Decode latents using the MLX VAE.

        Args:
            latents: Latent tensor in *normalized* latent space. Accepts:
                - torch.Tensor [C, T, H, W] (will be moved to CPU numpy)
                - numpy ndarray [C, T, H, W]
                - mx.array [C, T, H, W] (torch layout) or already channels-last
            model: Pre-loaded MLX VAE. If None, must be loaded first.

        Returns:
            Decoded video as float32 numpy ndarray [C, T, H, W] in [-1, 1].
        """
        if model is None:
            if "vae" not in self._loaded_models:
                raise RuntimeError("VAE model not loaded. Call load_model('vae', ...) first.")
            model = self._loaded_models["vae"]

        # Normalize input -> numpy [C,T,H,W] float32
        import torch
        if isinstance(latents, torch.Tensor):
            z_np = latents.detach().float().cpu().numpy()
        elif isinstance(latents, np.ndarray):
            z_np = latents.astype(np.float32)
        elif isinstance(latents, mx.array):
            z_np = np.asarray(latents).astype(np.float32)
        else:
            raise TypeError(f"Unsupported latent type: {type(latents)}")

        # torch layout [C,T,H,W] -> channels-last [B,T,H,W,C]
        if z_np.ndim == 4:
            z_cl = np.transpose(z_np, (1, 2, 3, 0))[None, ...]
        elif z_np.ndim == 5:
            # Already channels-last [B,T,H,W,C]
            z_cl = z_np
        else:
            raise ValueError(f"Unexpected latent ndim={z_np.ndim}: {z_np.shape}")

        z_mx = mx.array(z_cl, dtype=self._dtype)
        # Denormalize (z*std+mean) to match native Wan2_1_VAE.decode scale.
        z_denorm = model.denormalize_latents(z_mx)
        video_mx = model.decode(z_denorm)
        mx.eval(video_mx)

        # [B,T,H,W,C] -> [C,T,H,W]
        video_np = np.asarray(video_mx[0], dtype=np.float32)
        return np.ascontiguousarray(np.transpose(video_np, (3, 0, 1, 2)))
