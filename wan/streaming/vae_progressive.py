"""Stateful progressive decoder for Wan2.1 VAE.

WanVAE_.decode already processes latent time one slice at a time with causal
feature caches, but it clears those caches at the beginning and end of every
call. This adapter keeps the same cache alive across multiple latent chunks so
chunked decode follows the exact same temporal order as one full decode.

This module does not change the VAE weights or math. It is intentionally
single-stream and not thread-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from wan.utils.device import autocast_ctx


@dataclass(frozen=True)
class ProgressiveDecodeStats:
    latent_frames: int
    output_frames: int
    chunks: int


class ProgressiveWanVaeDecoder:
    """Decode sequential latent chunks while preserving Wan causal caches."""

    def __init__(self, vae) -> None:
        if vae is None or getattr(vae, "model", None) is None:
            raise ValueError("a loaded Wan2_1_VAE instance is required")
        self.vae = vae
        self.model = vae.model
        self._active = False
        self._closed = False
        self._latent_frames = 0
        self._output_frames = 0
        self._chunks = 0

    def start(self) -> "ProgressiveWanVaeDecoder":
        if self._closed:
            raise RuntimeError("decoder is closed")
        if not self._active:
            self.model.clear_cache()
            self._active = True
        return self

    def decode_chunk(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode one chronological latent chunk.

        Args:
            latent: ``[C, T, H, W]`` tensor. Calls must be made in original
                temporal order and must not overlap or skip latent frames.

        Returns:
            RGB tensor ``[3, Tout, Hout, Wout]`` in ``[-1, 1]``.
        """
        if self._closed:
            raise RuntimeError("decoder is closed")
        if latent.ndim != 4:
            raise ValueError(f"latent must be [C,T,H,W], got {tuple(latent.shape)}")
        if latent.shape[1] <= 0:
            raise ValueError("latent chunk must contain at least one time step")
        self.start()

        with autocast_ctx(dtype=self.vae.dtype):
            z = latent.unsqueeze(0)
            scale = self.vae.scale
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, self.model.z_dim, 1, 1, 1) + scale[0].view(
                    1, self.model.z_dim, 1, 1, 1
                )
            else:
                z = z / scale[1] + scale[0]

            # conv2 is a 1x1x1 causal convolution, so processing a chronological
            # chunk is equivalent to processing the same slices in one tensor.
            x = self.model.conv2(z)
            decoded = []
            for i in range(x.shape[2]):
                self.model._conv_idx = [0]
                out = self.model.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self.model._feat_map,
                    feat_idx=self.model._conv_idx,
                )
                decoded.append(out)

            result = torch.cat(decoded, dim=2).float().clamp_(-1, 1).squeeze(0)

        self._latent_frames += int(latent.shape[1])
        self._output_frames += int(result.shape[1])
        self._chunks += 1
        return result

    def decode_chunks(self, chunks: Iterable[torch.Tensor]) -> torch.Tensor:
        outputs = [self.decode_chunk(chunk) for chunk in chunks]
        if not outputs:
            raise ValueError("no chunks supplied")
        return torch.cat(outputs, dim=1)

    def stats(self) -> ProgressiveDecodeStats:
        return ProgressiveDecodeStats(
            latent_frames=self._latent_frames,
            output_frames=self._output_frames,
            chunks=self._chunks,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._active:
            self.model.clear_cache()
        self._active = False
        self._closed = True

    def __enter__(self) -> "ProgressiveWanVaeDecoder":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
