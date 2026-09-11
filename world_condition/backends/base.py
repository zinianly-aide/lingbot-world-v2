"""Abstract backend interface for VLM world perception.

Both transformers and MLX backends implement this interface so the
WorldPromptComposer and LingBot pipeline never need to know which VLM
stack produced the structured world description.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..schemas import WorldDescription, parse_world_description


WORLD_SCHEMA_PROMPT = """Analyze this current image only. Return JSON and nothing else using exactly this shape:
{"environment":"", "scene_layout":"", "main_entities":[{"name":"", "appearance":"", "position":"", "state":""}], "lighting":"", "weather":"", "camera":"", "motion":"", "persistent_constraints":[], "user_intent":""}
Describe observable facts, relative positions, appearance, state, camera and motion. Keep every value concise. Do not invent objects. Do not predict a future frame or describe an action that has not happened. If uncertain, use an empty string or an empty list."""


@dataclass(frozen=True)
class PerceptionResult:
    world: WorldDescription
    raw_text: str = ""
    error: str | None = None

    @property
    def used_fallback(self) -> bool:
        return self.error is not None


class WorldPerceptionBackend(ABC):
    """Load, query and release a VLM around one image."""

    def __init__(self, model_name: str, max_new_tokens: int = 384) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def analyze(self, image: object, user_prompt: str | None = None) -> PerceptionResult: ...

    @abstractmethod
    def release(self) -> None: ...

    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    def _build_prompt(self, user_prompt: str | None) -> str:
        prompt = WORLD_SCHEMA_PROMPT
        if user_prompt:
            prompt += (
                "\nUse this user request only to resolve relevance; "
                "do not override observed facts:\n" + user_prompt
            )
        return prompt

    def _parse_result(self, raw: str) -> PerceptionResult:
        world = parse_world_description(raw)
        if world == WorldDescription() and raw.strip():
            return PerceptionResult(
                world=world,
                raw_text=raw,
                error="VLM returned malformed or empty world JSON",
            )
        return PerceptionResult(world=world, raw_text=raw)
