"""Transformers backend for MiniCPM-V 4.6 world perception.

Uses AutoModelForImageTextToText + AutoProcessor. Requires a recent
transformers build (5.x) with built-in minicpmv4_6 support.
"""

from __future__ import annotations

import gc
from typing import Any

from ..schemas import WorldDescription
from .base import PerceptionResult, WorldPerceptionBackend


class TransformersBackend(WorldPerceptionBackend):
    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-V-4.6",
        device: str = "auto",
        max_new_tokens: int = 384,
    ) -> None:
        super().__init__(model_name, max_new_tokens)
        self.device = device
        self.model: Any = None
        self.processor: Any = None
        self._loader: Any = None  # test injection hook

    @property
    def backend_name(self) -> str:
        return "transformers"

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
        if self._loader is not None:
            self.model, self.processor = self._loader(self.model_name, self._resolve_device())
            return
        device = self._resolve_device()
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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
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
        prompt = self._build_prompt(user_prompt)
        try:
            self.load()
            raw = self._generate(image, prompt)
            return self._parse_result(raw)
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
