"""Small, defensive schema for a single-frame world observation.

The schema deliberately contains observations only.  It has no field for a
future frame or an action prediction, which keeps perception separate from
LingBot's generation task.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


MAX_TEXT = 480
MAX_ENTITIES = 12
MAX_CONSTRAINTS = 12


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (str, int, float, bool)):
        return str(value).strip()[:MAX_TEXT]
    return default


@dataclass(frozen=True)
class EntityDescription:
    name: str = "unknown entity"
    appearance: str = ""
    position: str = ""
    state: str = ""

    @classmethod
    def from_value(cls, value: Any) -> "EntityDescription":
        if not isinstance(value, Mapping):
            return cls(name=_text(value, "unknown entity") or "unknown entity")
        return cls(
            name=_text(value.get("name"), "unknown entity") or "unknown entity",
            appearance=_text(value.get("appearance")),
            position=_text(value.get("position")),
            state=_text(value.get("state")),
        )


@dataclass(frozen=True)
class WorldDescription:
    environment: str = "unknown"
    scene_layout: str = "unknown"
    main_entities: tuple[EntityDescription, ...] = field(default_factory=tuple)
    lighting: str = "unknown"
    weather: str = "unknown"
    camera: str = "unknown"
    motion: str = "unknown"
    persistent_constraints: tuple[str, ...] = field(default_factory=tuple)
    user_intent: str = ""

    @classmethod
    def from_mapping(cls, value: Any) -> "WorldDescription":
        if not isinstance(value, Mapping):
            return cls()
        entities = value.get("main_entities", [])
        if not isinstance(entities, list):
            entities = []
        constraints = value.get("persistent_constraints", [])
        if not isinstance(constraints, list):
            constraints = []
        return cls(
            environment=_text(value.get("environment"), "unknown") or "unknown",
            scene_layout=_text(value.get("scene_layout"), "unknown") or "unknown",
            main_entities=tuple(EntityDescription.from_value(item) for item in entities[:MAX_ENTITIES]),
            lighting=_text(value.get("lighting"), "unknown") or "unknown",
            weather=_text(value.get("weather"), "unknown") or "unknown",
            camera=_text(value.get("camera"), "unknown") or "unknown",
            motion=_text(value.get("motion"), "unknown") or "unknown",
            persistent_constraints=tuple(
                _text(item) for item in constraints[:MAX_CONSTRAINTS] if _text(item)
            ),
            user_intent=_text(value.get("user_intent")),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["main_entities"] = [asdict(entity) for entity in self.main_entities]
        value["persistent_constraints"] = list(self.persistent_constraints)
        return value


def _extract_json(raw: str) -> str:
    """Extract the first balanced, parseable JSON object from a VLM response."""
    cleaned = raw.strip()
    # Strip chain-of-thought blocks (MLX / reasoning models wrap analysis in <think>)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    # Try each '{' as a potential JSON start; return the first that parses as valid JSON
    starts = [m.start() for m in re.finditer(r"\{", cleaned)]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : index + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except (json.JSONDecodeError, ValueError):
                        break  # this { didn't yield valid JSON, try the next
    return cleaned


def parse_world_description(raw: Any) -> WorldDescription:
    """Parse model output, returning a safe empty observation on bad output."""
    if isinstance(raw, Mapping):
        return WorldDescription.from_mapping(raw)
    if not isinstance(raw, str):
        return WorldDescription()
    try:
        decoded = json.loads(_extract_json(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return WorldDescription()
    return WorldDescription.from_mapping(decoded)

