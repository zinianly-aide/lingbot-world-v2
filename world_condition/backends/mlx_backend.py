"""MLX backend for MiniCPM-V 4.6 world perception on Apple Silicon.

Uses mlx-vlm with 4-bit quantised checkpoints (e.g.
mlx-community/MiniCPM-V-4.6-4bit) for low memory footprint on Mac.
"""

from __future__ import annotations

import gc
from typing import Any

from ..schemas import WorldDescription
from .base import PerceptionResult, WorldPerceptionBackend


class MlxBackend(WorldPerceptionBackend):
    def __init__(
        self,
        model_name: str = "mlx-community/MiniCPM-V-4.6-4bit",
        max_new_tokens: int = 384,
    ) -> None:
        super().__init__(model_name, max_new_tokens)
        self.model: Any = None
        self.processor: Any = None
        self._config: Any = None

    @property
    def backend_name(self) -> str:
        return "mlx"

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise RuntimeError(
                "MLX backend requires mlx-vlm. Install with: pip install mlx-vlm"
            ) from exc
        self.model, self.processor = load(self.model_name)
        self._config = load_config(self.model_name)

    def _generate(self, image: Any, prompt: str) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted_prompt = apply_chat_template(
            self.processor, self._config, messages, tokenize=False
        )
        output = generate(
            self.model,
            self.processor,
            formatted_prompt,
            image=image,
            max_tokens=self.max_new_tokens,
            verbose=False,
        )
        # mlx-vlm >= 0.7 returns a GenerationResult dataclass; extract .text.
        if hasattr(output, "text"):
            return output.text
        return str(output)

    def analyze(self, image: Any, user_prompt: str | None = None) -> PerceptionResult:
        prompt = self._build_prompt(user_prompt)
        try:
            self.load()
            raw = self._generate(image, prompt)
            return self._parse_result(raw)
        except Exception as exc:  # VLM is best-effort; generation must continue.
            return PerceptionResult(world=WorldDescription(), error=f"{type(exc).__name__}: {exc}")

    def release(self) -> None:
        model, processor, config = self.model, self.processor, self._config
        self.model = None
        self.processor = None
        self._config = None
        del model, processor, config
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass
