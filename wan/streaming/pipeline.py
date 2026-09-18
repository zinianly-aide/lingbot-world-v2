"""Small composable sinks for the Quest streaming POC."""
from __future__ import annotations

from dataclasses import dataclass

from .events import LatentChunkEvent
from .frame_publisher import FrameBridgePublisher
from .vae_progressive import ProgressiveWanVaeDecoder


@dataclass(frozen=True)
class ProgressivePipelineStats:
    latent_chunks: int
    rgb_frames: int


class ProgressiveVaeFrameSink:
    """Decode each completed latent chunk and publish RGB frames immediately.

    This sink is intentionally synchronous. Use it only after Q1.5 proves
    progressive VAE equivalence and after memory measurements confirm that the
    DiT and VAE can coexist on the target machine. A future worker/queue can
    move decode off the generation thread without changing this contract.
    """

    def __init__(
        self,
        decoder: ProgressiveWanVaeDecoder,
        publisher: FrameBridgePublisher,
    ) -> None:
        self.decoder = decoder
        self.publisher = publisher
        self._latent_chunks = 0
        self._rgb_frames = 0

    def on_latent_chunk(self, event: LatentChunkEvent, latent) -> None:
        frames = self.decoder.decode_chunk(latent)
        published = self.publisher.publish_chunk(frames)
        self._latent_chunks += 1
        self._rgb_frames += published

    def stats(self) -> ProgressivePipelineStats:
        return ProgressivePipelineStats(
            latent_chunks=self._latent_chunks,
            rgb_frames=self._rgb_frames,
        )
