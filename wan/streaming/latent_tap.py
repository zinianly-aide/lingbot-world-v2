"""Non-invasive latent chunk tap for the causal-fast generator.

This POC avoids rewriting ``wan/image2video.py``. It observes the existing
post-chunk KV-cache update call by temporarily wrapping two bound methods:

1. ``_convert_flow_pred_to_x0`` remembers the latest x0 object.
2. ``model.forward`` recognizes the later zero-timestep context-update call
   that receives that exact x0 object.

The sink is called *after* that context update succeeds, once per completed
chunk. Original methods are restored on exit, even when generation fails.
"""
from __future__ import annotations

import logging
import time
import types
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch

from .events import LatentChunkEvent, LatentChunkSink


@dataclass
class _TapState:
    latest_x0: torch.Tensor | None = None
    chunk_index: int = 0
    latent_start: int = 0


def _is_zero_timestep(value) -> bool:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return False
    return bool(torch.count_nonzero(value).item() == 0)


@contextmanager
def tap_causal_latent_chunks(
    pipe,
    sink: LatentChunkSink,
    *,
    generation_id: str | None = None,
    seed: int = -1,
    total_chunks: int = -1,
    fail_open: bool = True,
) -> Iterator[str]:
    """Emit each completed causal-fast latent chunk without changing model math.

    This is a POC seam for Q1. It requires a loaded ``pipe.model`` and should
    wrap exactly one ``pipe.generate(...)`` call. The underlying generator still
    owns and stores its normal x0 tensors; the tap only passes the live tensor
    reference to ``sink``.
    """
    if getattr(pipe, "model", None) is None:
        raise ValueError("pipe.model must be loaded before installing the latent tap")
    if getattr(pipe, "infer_mode", None) != "causal_fast":
        raise ValueError("latent tap currently supports infer_mode='causal_fast' only")

    generation_id = generation_id or uuid.uuid4().hex
    state = _TapState()
    started = time.monotonic()

    original_convert = pipe._convert_flow_pred_to_x0
    original_forward = pipe.model.forward

    def wrapped_convert(_pipe_self, *args, **kwargs):
        x0 = original_convert(*args, **kwargs)
        state.latest_x0 = x0
        return x0

    def wrapped_forward(_model_self, *args, **kwargs):
        result = original_forward(*args, **kwargs)

        x_arg = kwargs.get("x")
        if x_arg is None and args:
            x_arg = args[0]
        t_arg = kwargs.get("t")
        if t_arg is None and len(args) > 1:
            t_arg = args[1]

        latent = None
        if isinstance(x_arg, (list, tuple)) and len(x_arg) == 1:
            candidate = x_arg[0]
            if isinstance(candidate, torch.Tensor):
                latent = candidate

        # The generator performs one context/KV update after appending each x0:
        # self.model(x=[x0], t=[0], cross_attn_first_call=False, ...)
        # Object identity avoids confusing this with regular denoising forwards.
        if latent is state.latest_x0 and _is_zero_timestep(t_arg):
            event = LatentChunkEvent(
                generation_id=generation_id,
                chunk_index=state.chunk_index,
                total_chunks=total_chunks,
                latent_start=state.latent_start,
                latent_count=int(latent.shape[1]),
                shape=tuple(int(v) for v in latent.shape),
                dtype=str(latent.dtype),
                seed=int(seed),
                elapsed_ms=(time.monotonic() - started) * 1000.0,
            )
            try:
                sink.on_latent_chunk(event, latent)
            except Exception:
                if not fail_open:
                    raise
                logging.exception("progressive latent sink failed; continuing generation")
            state.latent_start += int(latent.shape[1])
            state.chunk_index += 1
            state.latest_x0 = None

        return result

    pipe._convert_flow_pred_to_x0 = types.MethodType(wrapped_convert, pipe)
    pipe.model.forward = types.MethodType(wrapped_forward, pipe.model)
    try:
        yield generation_id
    finally:
        pipe._convert_flow_pred_to_x0 = original_convert
        pipe.model.forward = original_forward
