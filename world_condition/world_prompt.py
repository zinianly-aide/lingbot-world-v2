"""Compose a concise, deterministic LingBot text prompt from observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import WorldDescription


def _clip(value: str, length: int) -> str:
    value = value.strip()
    return value if len(value) <= length else value[: length - 1].rstrip() + "…"


def compose_world_prompt(
    world: WorldDescription | dict[str, Any],
    user_prompt: str,
    max_chars: int = 1800,
) -> str:
    """Return a stable prompt while preserving the complete user request.

    World observations are bounded before composition; the user request is
    intentionally appended verbatim so an explicit instruction cannot be
    silently replaced by a VLM hallucination.
    """
    observation = world if isinstance(world, WorldDescription) else WorldDescription.from_mapping(world)
    entities = []
    for entity in observation.main_entities:
        entities.append(
            f"{entity.name} ({entity.appearance}; position: {entity.position}; state: {entity.state})"
        )
    entity_text = "; ".join(entities) or "none confidently observed"
    constraints = "; ".join(observation.persistent_constraints) or "none"
    user_prompt = user_prompt or ""
    fixed = (
        "Requested evolution (authoritative user intent): {user}\n"
        "Current world: {environment}\n"
        "Persistent entities: {entities}\n"
        "Spatial layout: {layout}\n"
        "Lighting/weather: {lighting}; {weather}\n"
        "Camera: {camera}\n"
        "Observed motion: {motion}\n"
        "Observed intent (context only): {intent}\n"
        "Constraints: preserve identity; preserve object layout unless the requested action changes it; "
        "preserve environment consistency; {constraints}\n"
        "Use the observed world only as continuity context. Follow the requested evolution as the authoritative user intent."
    ).format(
        environment=_clip(observation.environment, 400),
        entities=_clip(entity_text, 900),
        layout=_clip(observation.scene_layout, 650),
        lighting=_clip(observation.lighting, 180),
        weather=_clip(observation.weather, 180),
        camera=_clip(observation.camera, 350),
        motion=_clip(observation.motion, 250),
        intent=_clip(observation.user_intent, 250) or "none",
        constraints=_clip(constraints, 450),
        user=user_prompt,
    )
    if len(fixed) <= max_chars:
        return fixed
    # Never truncate the original request. It is deliberately at the front so
    # UMT5's existing 512-token limit cannot hide explicit user intent.
    header = "Requested evolution (authoritative user intent): " + user_prompt
    return header + "\n" + fixed[len(header) : len(header) + max(0, max_chars - len(header) - 1)].rstrip()


def load_world_condition(path: str | Path) -> WorldDescription:
    with Path(path).open("r", encoding="utf-8") as handle:
        return WorldDescription.from_mapping(json.load(handle))


def save_world_condition(world: WorldDescription, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(world.to_dict(), handle, ensure_ascii=False, indent=2)


def save_world_prompt(prompt: str, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(prompt, encoding="utf-8")
