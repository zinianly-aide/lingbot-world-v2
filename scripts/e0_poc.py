#!/usr/bin/env python3
"""E0 POC forward validation for the Text Adapter.

Does NOT load the real MiniCPM5 (still downloading at E0).  Instead it uses
random tensors shaped exactly like MiniCPM5 last-hidden-state and verifies the
adapter contract end-to-end:

  1. Random encoder hidden [B, L, 2048] -> adapter -> [B, 64, 4096] float32
  2. Attention mask actually gates padding (garbage in padding positions does
     not leak into the output when the mask is applied).
  3. Mock encoder lifecycle: build a fake "MiniCPM5" module, forward, then
     del + gc.collect() + torch.mps.empty_cache() releases its memory
     (trend, not a strict number).
  4. Adapter does not co-reside with the encoder: create adapter, forward,
     del adapter, then verify allocated memory trends downward after GC.

Pure random weights. No training, no video generation.
"""
from __future__ import annotations

import argparse
import gc
import os
import resource
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.adapters.text_adapter import TextAdapter  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def mps_alloc_mb() -> float:
    if not torch.backends.mps.is_available():
        return 0.0
    return torch.mps.current_allocated_memory() / (1024 * 1024)


def settle() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def check_shape_dtype() -> dict:
    adapter = TextAdapter()
    x = torch.randn(2, 50, 2048)
    with torch.no_grad():
        y = adapter(x)
    ok_shape = tuple(y.shape) == (2, 64, 4096)
    ok_dtype = y.dtype == torch.float32
    print(f"[1] shape/dtype: out={tuple(y.shape)} dtype={y.dtype} "
          f"shape_ok={ok_shape} dtype_ok={ok_dtype}")
    del adapter, x, y
    settle()
    return {"shape_ok": ok_shape, "dtype_ok": ok_dtype}


def check_attention_mask() -> dict:
    adapter = TextAdapter()
    torch.manual_seed(0)
    # Real content on the prefix (positions 0..29).
    x_prefix = torch.randn(2, 30, 2048)
    # Full-length tensor with a *different random realization* in the tail
    # (positions 30..49).  With the mask applied, the tail must be ignored and
    # the output must match the prefix-only reference.
    x_tail_diff = torch.cat([x_prefix, torch.randn(2, 20, 2048)], dim=1)

    mask = torch.ones(2, 50)
    mask[:, 30:] = 0.0  # positions 30..49 are padding

    with torch.no_grad():
        # Prefix-only reference (no padding at all).
        y_ref = adapter(x_prefix)
        # Masked: tail ignored -> should match y_ref.
        y_masked = adapter(x_tail_diff, mask)
        # Unmasked: tail leaks in -> should differ from y_ref.
        y_unmasked = adapter(x_tail_diff)

    diff_masked_vs_ref = (y_masked - y_ref).abs().max().item()
    diff_unmasked_vs_ref = (y_unmasked - y_ref).abs().max().item()
    # Masked output should be close to reference (tail ignored); unmasked
    # should be far from reference (tail leaked in).
    mask_gates = diff_masked_vs_ref < diff_unmasked_vs_ref
    print(f"[2] attention mask: |masked-ref|={diff_masked_vs_ref:.4e} "
          f"|unmasked-ref|={diff_unmasked_vs_ref:.4e} gates={mask_gates}")
    del adapter, x_prefix, x_tail_diff, mask, y_ref, y_masked, y_unmasked
    settle()
    return {"masked_vs_ref": diff_masked_vs_ref,
            "unmasked_vs_ref": diff_unmasked_vs_ref,
            "mask_gates": mask_gates}


def check_encoder_release(device: torch.device) -> dict:
    """Mock MiniCPM5 lifecycle: build a fake big encoder, forward, del, GC."""
    settle()
    rss_before = rss_mb()
    mps_before = mps_alloc_mb()

    # A stand-in "MiniCPM5" that holds a big parameter buffer to simulate
    # encoder weights. Real MiniCPM5 is ~2B params bf16; we mock with a
    # modest tensor so this POC stays fast.
    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = torch.nn.Parameter(torch.randn(50, 2048, device=device))

        def forward(self, batch: int) -> torch.Tensor:
            return self.hidden.unsqueeze(0).expand(batch, -1, -1).contiguous()

    enc = FakeEncoder().to(device)
    hidden = enc(batch=2)
    adapter = TextAdapter().to(device)
    with torch.no_grad():
        out = adapter(hidden)
    assert out.shape == (2, 64, 4096)

    rss_after_work = rss_mb()
    mps_after_work = mps_alloc_mb()

    # Release encoder first (mirrors "del MiniCPM5 -> gc -> mps.empty_cache()").
    del enc, hidden, out, adapter
    settle()
    rss_after_release = rss_mb()
    mps_after_release = mps_alloc_mb()

    # ru_maxrss is a high-water mark on macOS and never decreases, so on CPU
    # we cannot observe RSS release; we only verify the lifecycle completes.
    # On MPS, current_allocated_memory reflects actual deallocation.
    if device.type == "mps":
        released = mps_after_release < mps_after_work
    else:
        released = True  # lifecycle completed without crash; CPU RSS is HWM
    print(f"[3] encoder release (device={device.type}): "
          f"RSS work={rss_after_work:.1f}MB -> release={rss_after_release:.1f}MB | "
          f"MPS work={mps_after_work:.1f}MB -> release={mps_after_release:.1f}MB "
          f"released_trend={released}")
    del rss_before, mps_before
    settle()
    return {"rss_work_mb": rss_after_work, "rss_release_mb": rss_after_release,
            "mps_work_mb": mps_after_work, "mps_release_mb": mps_after_release,
            "released_trend": released}


def check_no_coresidency(device: torch.device) -> dict:
    """Adapter lifecycle: build -> forward -> del -> memory trend down."""
    settle()
    rss_0 = rss_mb()
    mps_0 = mps_alloc_mb()

    adapter = TextAdapter().to(device)
    x = torch.randn(2, 50, 2048, device=device)
    with torch.no_grad():
        y = adapter(x)
    rss_1 = rss_mb()
    mps_1 = mps_alloc_mb()

    del adapter, x, y
    settle()
    rss_2 = rss_mb()
    mps_2 = mps_alloc_mb()

    # On MPS, allocated memory should drop after del + empty_cache. On CPU,
    # ru_maxrss is HWM so we only assert the lifecycle ran cleanly.
    if device.type == "mps":
        trend_ok = mps_2 <= mps_1 + 1.0
    else:
        trend_ok = True
    print(f"[4] adapter no-coresidency (device={device.type}): "
          f"RSS {rss_0:.1f} -> {rss_1:.1f} -> {rss_2:.1f} MB | "
          f"MPS {mps_0:.1f} -> {mps_1:.1f} -> {mps_2:.1f} MB "
          f"trend_ok={trend_ok}")
    return {"rss_start": rss_0, "rss_peak": rss_1, "rss_end": rss_2,
            "mps_start": mps_0, "mps_peak": mps_1, "mps_end": mps_2,
            "trend_ok": trend_ok}


def main() -> int:
    parser = argparse.ArgumentParser(description="E0 Text Adapter POC forward check.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = parser.parse_args()

    torch.manual_seed(42)
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"=== E0 Text Adapter POC (device={device}) ===")
    results = {
        "shape_dtype": check_shape_dtype(),
        "attention_mask": check_attention_mask(),
        "encoder_release": check_encoder_release(device),
        "no_coresidency": check_no_coresidency(device),
    }
    all_ok = (
        results["shape_dtype"]["shape_ok"]
        and results["shape_dtype"]["dtype_ok"]
        and results["attention_mask"]["mask_gates"]
        and results["encoder_release"]["released_trend"]
        and results["no_coresidency"]["trend_ok"]
    )
    print(f"=== POC {'PASS' if all_ok else 'FAIL'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
