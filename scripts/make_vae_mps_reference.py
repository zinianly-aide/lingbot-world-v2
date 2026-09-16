#!/usr/bin/env python3
"""Produce the MPS BF16 VAE-decode reference video for the e2 single_subject seed=42 latent.

Decodes eval/e2/.cache/single_subject_42/latents.safetensors with the production
Wan2_1_VAE (BF16 on MPS) and saves the resulting [-1,1] float32 video to a .npy
file [C,T,H,W] for numerical comparison against the MLX backend.

Run in its own subprocess to keep MPS/MLX memory disjoint.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wan.modules.vae2_1 import Wan2_1_VAE  # noqa: E402

VAE_CKPT = "/Volumes/ssd/lingbot-assets/Wan2.1_VAE.pth"
LATENT = str(REPO_ROOT / "eval/e2/.cache/single_subject_42/latents.safetensors")
OUT_NPY = str(REPO_ROOT / "eval/bench_vae_mlx/ref_mps_bf16.npy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", default=LATENT)
    ap.add_argument("--out", default=OUT_NPY)
    args = ap.parse_args()

    device = torch.device("mps")
    torch.mps.empty_cache()

    z = load_file(args.latent)["latents"].float().contiguous()  # [C,T,H,W]
    print(f"[ref] latent {tuple(z.shape)} {z.dtype}")

    gc.collect()
    torch.mps.empty_cache()
    vae = Wan2_1_VAE(vae_pth=VAE_CKPT, dtype=torch.bfloat16, device=device)
    vae.model = vae.model.to(torch.bfloat16)
    vae.mean = vae.mean.to(torch.bfloat16)
    vae.std = vae.std.to(torch.bfloat16)

    z_dev = z.to(device=device, dtype=torch.bfloat16)
    torch.mps.synchronize()
    videos = vae.decode([z_dev])
    torch.mps.synchronize()
    video = videos[0].float().cpu().numpy()  # [C,T,H,W] in [-1,1]
    print(f"[ref] video {video.shape} range [{video.min():.3f},{video.max():.3f}]")
    np.save(args.out, video)
    print(f"[ref] saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
