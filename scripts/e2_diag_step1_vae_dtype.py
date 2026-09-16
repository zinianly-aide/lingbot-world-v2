#!/usr/bin/env python3
"""E2 root-cause Step 1: VAE dtype diagnosis on the SAME E2 latent.

Decodes the cached E2 latent (single_subject, seed=42) with:
  (a) FP16 VAE   (LINGBOT_VAE_DTYPE=fp16)
  (b) BF16 VAE   (LINGBOT_VAE_DTYPE=bf16)
and also decodes the g0.7 UMT5-baseline latent with FP16 as a quality reference.

For each decode we save a frame-grid PNG (for visual inspection) and compute
frame mean/std + temporal MAD.  If FP16 and BF16 decode of the SAME minicpm
latent are both bad (high temporal MAD, unrecognizable) -> the problem is the
latents themselves (conditioning), not VAE dtype.  If FP16 is clean but BF16 is
garbage -> BF16 VAE is the culprit.

Does NOT touch the DiT, does NOT re-run generation, does NOT load MiniCPM5.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wan.modules.vae2_1 import Wan2_1_VAE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e2_step1_vae")

VAE_CHECKPOINT = "/Volumes/ssd/lingbot-assets/Wan2.1_VAE.pth"
E2_LATENT = str(REPO_ROOT / "eval" / "e2" / ".cache" / "single_subject_42" / "latents.safetensors")
G07_LATENT = str(REPO_ROOT / "eval" / "g0.7" / ".cache" / "single_subject_A_42" / "latents.safetensors")
OUT_DIR = REPO_ROOT / "eval" / "e2_diag" / "step1_vae_dtype"


def load_latent(path: str) -> torch.Tensor:
    sd = load_file(path)
    z = sd["latents"] if "latents" in sd else list(sd.values())[0]
    if z.dim() == 5:
        z = z.squeeze(0)
    return z.float().contiguous()


def load_vae(dtype: torch.dtype, device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    vae = Wan2_1_VAE(vae_pth=VAE_CHECKPOINT, dtype=dtype, device=device)
    if dtype != torch.float32:
        vae.model = vae.model.to(dtype)
        vae.mean = vae.mean.to(dtype)
        vae.std = vae.std.to(dtype)
    return vae


def unload(vae, device):
    del vae
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


def decode(vae, z: torch.Tensor, device) -> torch.Tensor:
    z_d = z.to(device=device, dtype=next(vae.model.parameters()).dtype)
    with torch.no_grad():
        vid = vae.decode([z_d])[0]
    return vid.cpu().float()


def postprocess_rgb(vid: torch.Tensor) -> torch.Tensor:
    """vid [C,T,H,W] in [-1,1] -> uint8 RGB [T,H,W,3]."""
    rgb = ((vid + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    # [C,T,H,W] -> [T,H,W,C]
    rgb = rgb.permute(1, 2, 3, 0).contiguous()
    return rgb


def stats(rgb: torch.Tensor) -> dict:
    """rgb uint8 [T,H,W,3] -> per-frame mean/std + temporal MAD."""
    x = rgb.float() / 255.0
    per_frame_mean = x.mean(dim=(1, 2, 3))
    per_frame_std = x.std(dim=(1, 2, 3))
    # temporal MAD = mean over frames of |x_t - x_{t-1}|
    diffs = (x[1:] - x[:-1]).abs()
    return {
        "global_mean": float(x.mean().item()),
        "global_std": float(x.std().item()),
        "per_frame_mean": [round(float(v), 4) for v in per_frame_mean.tolist()],
        "per_frame_std": [round(float(v), 4) for v in per_frame_std.tolist()],
        "temporal_mad_mean": float(diffs.mean().item()),
        "temporal_mad_max": float(diffs.max().item()),
        "frame_count": int(x.shape[0]),
    }


def save_frame_grid(rgb: torch.Tensor, path: str):
    """Save a horizontal grid of up to 6 frames as PNG."""
    from PIL import Image
    n = min(6, rgb.shape[0])
    idxs = [int(i * (rgb.shape[0] - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    frames = [rgb[i].numpy() for i in idxs]
    h, w = frames[0].shape[:2]
    grid = Image.new("RGB", (w * n, h))
    for i, fr in enumerate(frames):
        grid.paste(Image.fromarray(fr), (i * w, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid.save(path)
    return [int(i) for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    e2_z = load_latent(E2_LATENT)
    g07_z = load_latent(G07_LATENT)
    log.info("E2 latent %s, g0.7 latent %s", tuple(e2_z.shape), tuple(g07_z.shape))

    report = {"e2_latent": E2_LATENT, "g07_latent": G07_LATENT, "decodes": {}}

    # (1) E2 latent -> FP16 VAE
    vae = load_vae(torch.float16, device)
    vid = decode(vae, e2_z, device)
    rgb = postprocess_rgb(vid)
    s = stats(rgb)
    idxs = save_frame_grid(rgb, str(OUT_DIR / "e2_fp16.png"))
    report["decodes"]["e2_fp16"] = {"stats": s, "grid_frames": idxs, "video_shape": list(vid.shape)}
    unload(vae, device)
    del vid, rgb

    # (2) E2 latent -> BF16 VAE
    vae = load_vae(torch.bfloat16, device)
    vid = decode(vae, e2_z, device)
    rgb = postprocess_rgb(vid)
    s = stats(rgb)
    idxs = save_frame_grid(rgb, str(OUT_DIR / "e2_bf16.png"))
    report["decodes"]["e2_bf16"] = {"stats": s, "grid_frames": idxs, "video_shape": list(vid.shape)}
    unload(vae, device)
    del vid, rgb

    # (3) g0.7 UMT5 baseline latent -> FP16 VAE (quality reference)
    vae = load_vae(torch.float16, device)
    vid = decode(vae, g07_z, device)
    rgb = postprocess_rgb(vid)
    s = stats(rgb)
    idxs = save_frame_grid(rgb, str(OUT_DIR / "g07_umt5_fp16.png"))
    report["decodes"]["g07_umt5_fp16"] = {"stats": s, "grid_frames": idxs, "video_shape": list(vid.shape)}
    unload(vae, device)
    del vid, rgb

    # Compare e2_fp16 vs e2_bf16 numerically (decode the same latent twice more
    # to get both videos on CPU simultaneously for a direct diff).
    vae = load_vae(torch.float16, device)
    v_fp16 = postprocess_rgb(decode(vae, e2_z, device)).float()
    unload(vae, device)
    vae = load_vae(torch.bfloat16, device)
    v_bf16 = postprocess_rgb(decode(vae, e2_z, device)).float()
    unload(vae, device)
    diff = (v_fp16 - v_bf16).abs()
    report["fp16_vs_bf16_on_e2_latent"] = {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "fp16_temporal_mad": report["decodes"]["e2_fp16"]["stats"]["temporal_mad_mean"],
        "bf16_temporal_mad": report["decodes"]["e2_bf16"]["stats"]["temporal_mad_mean"],
    }

    with open(OUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved to %s", OUT_DIR / "report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
