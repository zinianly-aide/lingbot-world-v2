"""Optional MiniCPM-V 4.6 single-frame perception.

Backwards-compatible facade that delegates to a pluggable backend
(transformers or MLX). Imports for heavy frameworks are lazy so the
original LingBot path does not gain a VLM dependency or change its
startup behavior.
"""

from __future__ import annotations

from typing import Any, Callable

from .backends import (
    PerceptionResult,
    TransformersBackend,
    create_backend,
)
from .schemas import WorldDescription, parse_world_description

# Re-export for backwards compatibility
WORLD_SCHEMA_PROMPT = __import__("world_condition.backends.base", fromlist=["WORLD_SCHEMA_PROMPT"]).WORLD_SCHEMA_PROMPT


class MiniCPMVPerceiver:
    """Load, query and explicitly release MiniCPM-V around one image.

    Supports ``backend="transformers"`` (default, full BF16 via
    AutoModelForImageTextToText) or ``backend="mlx"`` (4-bit quantised
    via mlx-vlm, recommended on Apple Silicon).
    """

    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-V-4.6",
        device: str = "auto",
        max_new_tokens: int = 384,
        backend: str = "transformers",
        loader: Callable[[str, str], tuple[Any, Any]] | None = None,
    ) -> None:
        self.backend_name = backend
        if loader is not None:
            # Legacy custom loader path — wrap into transformers backend.
            self._backend = TransformersBackend(
                model_name=model_name, device=device, max_new_tokens=max_new_tokens
            )
            self._backend._loader = loader  # type: ignore[attr-defined]
        else:
            kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
            if backend == "transformers":
                kwargs["device"] = device
            self._backend = create_backend(backend, model_name, **kwargs)

    @property
    def model(self) -> Any:
        return getattr(self._backend, "model", None)

    @property
    def processor(self) -> Any:
        return getattr(self._backend, "processor", None)

    def load(self) -> None:
        self._backend.load()

    def analyze(self, image: Any, user_prompt: str | None = None) -> PerceptionResult:
        return self._backend.analyze(image, user_prompt)

    def release(self) -> None:
        self._backend.release()
