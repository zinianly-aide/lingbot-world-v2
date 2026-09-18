"""Utilities that publish decoded RGB tensors to ``LatestFrameStore``."""
from __future__ import annotations

from io import BytesIO

import torch
from PIL import Image

from .frame_bridge import LatestFrameStore


class FrameBridgePublisher:
    """Encode decoded Wan frames as JPEG and publish the latest frame locally."""

    def __init__(
        self,
        store: LatestFrameStore,
        *,
        fps: float = 24.0,
        jpeg_quality: int = 90,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self.store = store
        self.fps = float(fps)
        self.jpeg_quality = int(jpeg_quality)
        self.sequence = 0

    def _jpeg(self, frame: torch.Tensor) -> bytes:
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(f"frame must be [3,H,W], got {tuple(frame.shape)}")
        image = (
            frame.detach()
            .float()
            .clamp(-1, 1)
            .add(1.0)
            .mul(127.5)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        out = BytesIO()
        Image.fromarray(image, mode="RGB").save(
            out,
            format="JPEG",
            quality=self.jpeg_quality,
            optimize=False,
        )
        return out.getvalue()

    def publish_chunk(self, frames: torch.Tensor) -> int:
        """Publish ``[3,T,H,W]`` frames in temporal order."""
        if frames.ndim != 4 or frames.shape[0] != 3:
            raise ValueError(f"frames must be [3,T,H,W], got {tuple(frames.shape)}")
        published = 0
        for i in range(frames.shape[1]):
            sequence = self.sequence
            pts_ms = sequence * 1000.0 / self.fps
            self.store.publish(self._jpeg(frames[:, i]), sequence, pts_ms)
            self.sequence += 1
            published += 1
        return published
