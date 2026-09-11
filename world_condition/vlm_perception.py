"""Optional MiniCPM-V 4.6 single-frame perception.

Imports for torch/transformers are lazy so the original LingBot path does not
gain a VLM dependency or change its startup behavior.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable

from .schemas import WorldDescription, parse_world_description


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


class MiniCPMVPerceiver:
    """Load, query and explicitly release MiniCPM-V around one image."""

    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-V-4.6",
        device: str = "auto",
        max_new_tokens: int = 384,
        loader: Callable[[str, str], tuple[Any, Any]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._loader = loader
        self.model: Any = None
        self.processor: Any = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def load(self) -> None:
        if self.model is not None:
            return
        device = self._resolve_device()
        if self._loader is not None:
            self.model, self.processor = self._loader(self.model_name, device)
            return
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "MiniCPM-V requires a recent transformers build (the upstream 4.x "
                "LingBot environment may need an isolated VLM environment)."
            ) from exc
        kwargs: dict[str, Any] = {"torch_dtype": "auto"}
        if device == "cuda":
            kwargs["device_map"] = "auto"
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, trust_remote_code=True, **kwargs
        )
        if hasattr(self.model, "eval"):
            self.model.eval()
        if device != "cuda" and hasattr(self.model, "to"):
            self.model.to(device)

    def _generate(self, image: Any, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        # Official 4.6 Transformers path.
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        try:
            import torch

            target_device = getattr(self.model, "device", None)
            if target_device is None:
                try:
                    target_device = next(self.model.parameters()).device
                except (AttributeError, StopIteration):
                    target_device = self._resolve_device()
            if hasattr(inputs, "to"):
                inputs = inputs.to(target_device)
            elif isinstance(inputs, dict):
                inputs = {
                    key: value.to(target_device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        except ImportError:
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        input_ids = inputs["input_ids"]
        trimmed = [out[len(inp):] for inp, out in zip(input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def analyze(self, image: Any, user_prompt: str | None = None) -> PerceptionResult:
        prompt = WORLD_SCHEMA_PROMPT
        if user_prompt:
            prompt += "\nUse this user request only to resolve relevance; do not override observed facts:\n" + user_prompt
        try:
            self.load()
            raw = self._generate(image, prompt)
            world = parse_world_description(raw)
            if world == WorldDescription() and raw.strip():
                return PerceptionResult(
                    world=world,
                    raw_text=raw,
                    error="MiniCPM-V returned malformed or empty world JSON",
                )
            return PerceptionResult(world=world, raw_text=raw)
        except Exception as exc:  # VLM is best-effort; generation must continue.
            return PerceptionResult(world=WorldDescription(), error=f"{type(exc).__name__}: {exc}")

    def release(self) -> None:
        model, processor = self.model, self.processor
        self.model = None
        self.processor = None
        del model, processor
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
