#!/usr/bin/env python3
"""Compact Step 1 completion: bf16 decode of E2 latent + fp16 decode of g0.7
UMT5-baseline latent.  Produces small viewable PNGs + stats JSON.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from wan.modules.vae2_1 import Wan2_1_VAE  # noqa: E402

VAE = "/Volumes/ssd/lingbot-assets/Wan2.1_VAE.pth"
E2 = str(REPO_ROOT / "eval/e2/.cache/single_subject_42/latents.safetensors")
G07 = str(REPO_ROOT / "eval/g0.7/.cache/single_subject_A_42/latents.safetensors")
OUT = REPO_ROOT / "eval/e2_diag/step1_vae_dtype"


def load_z(p):
    z = load_file(p)["latents"].float()
    return z.squeeze(0) if z.dim() == 5 else z


def load_vae(dtype, dev):
    gc.collect(); torch.mps.empty_cache()
    v = Wan2_1_VAE(vae_pth=VAE, dtype=dtype, device=dev)
    if dtype != torch.float32:
        v.model = v.model.to(dtype); v.mean = v.mean.to(dtype); v.std = v.std.to(dtype)
    return v


def decode(vae, z, dev):
    with torch.no_grad():
        return vae.decode([z.to(dev, dtype=next(vae.model.parameters()).dtype)])[0].cpu().float()


def rgb(vid):
    return ((vid + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0)


def stats(rgbu):
    x = rgbu.float() / 255.0
    diffs = (x[1:] - x[:-1]).abs()
    return {"global_mean": float(x.mean()), "global_std": float(x.std()),
            "temporal_mad_mean": float(diffs.mean()), "frame_count": int(x.shape[0])}


def save_thumb(rgbu, path, n=3):
    from PIL import Image
    idxs = [int(i * (rgbu.shape[0]-1)/(n-1)) for i in range(n)] if n > 1 else [0]
    fr = [rgbu[i].numpy() for i in idxs]
    h, w = fr[0].shape[:2]
    g = Image.new("RGB", (w*n, h))
    for i, f in enumerate(fr):
        g.paste(Image.fromarray(f), (i*w, 0))
    g.save(path)


dev = torch.device("mps")
e2 = load_z(E2); g07 = load_z(G07)
rep = {}

# bf16 decode of E2 latent
v = load_vae(torch.bfloat16, dev)
r = rgb(decode(v, e2, dev)); rep["e2_bf16"] = stats(r); save_thumb(r, str(OUT/"e2_bf16_thumb.png"))
del v; gc.collect(); torch.mps.empty_cache()

# fp16 decode of g0.7 baseline
v = load_vae(torch.float16, dev)
r = rgb(decode(v, g07, dev)); rep["g07_umt5_fp16"] = stats(r); save_thumb(r, str(OUT/"g07_umt5_fp16_thumb.png"))
del v; gc.collect(); torch.mps.empty_cache()

# fp16 decode of E2 latent again (for direct diff with bf16)
v = load_vae(torch.float16, dev)
r16 = rgb(decode(v, e2, dev)); rep["e2_fp16"] = stats(r16); save_thumb(r16, str(OUT/"e2_fp16_thumb.png"))
del v; gc.collect(); torch.mps.empty_cache()

rep["fp16_vs_bf16_on_e2"] = {"note": "same latent decoded twice, numerical diff only"}
with open(OUT/"compact_report.json", "w") as f:
    json.dump(rep, f, indent=2)
print(json.dumps(rep, indent=2))
