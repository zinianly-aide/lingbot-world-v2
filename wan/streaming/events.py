"""Small, model-agnostic contracts for progressive video generation.

The streaming path deliberately lives outside the DiT/VAE implementation so the
model math stays unchanged. The causal generator can emit these events at the
existing latent-chunk boundary and a downstream worker can decide whether to
persist, decode, or stream the chunk.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

GenerationPhase = Literal[
    "starting",
    "latent_chunk",
    "decoding",
    "frame_chunk",
    "completed",
    "failed",
]


@dataclass(frozen=True)
class GenerationEvent:
    generation_id: str
    phase: GenerationPhase
    monotonic_ms: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatentChunkEvent:
    """Metadata for one completed causal latent chunk.

    ``latent_start`` and ``latent_count`` are latent-time indices. They are
    intentionally *not* converted to RGB-frame indices here because Wan's VAE
    temporal decoder has context requirements that must be validated before we
    claim arbitrary chunks are independently decodable.
    """

    generation_id: str
    chunk_index: int
    total_chunks: int
    latent_start: int
    latent_count: int
    shape: tuple[int, ...]
    dtype: str
    seed: int
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data


class GenerationEventSink(Protocol):
    """Metadata-only, non-blocking event sink."""

    def on_event(self, event: GenerationEvent | LatentChunkEvent) -> None:
        ...


class LatentChunkSink(Protocol):
    """Optional data sink called only when progressive output is enabled.

    ``latent`` is the live device tensor for the completed chunk. Implementors
    that need to retain it after this method returns must explicitly detach/copy
    it. The default generator path has no sink, so it pays no CPU-copy or I/O
    cost.
    """

    def on_latent_chunk(self, event: LatentChunkEvent, latent: Any) -> None:
        ...
