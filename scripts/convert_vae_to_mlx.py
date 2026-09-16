#!/usr/bin/env python3
"""Convert Wan2.1_VAE.pth (native Wan2.1 format, 194 keys) to MLX AutoencoderKLWan.

Approach (Option A: manual key mapping, load_weights directly):
  - The MLX ``mlx_diffuser.models.autoencoder_kl_wan.AutoencoderKLWan`` is a faithful
    port of the SAME architecture and uses diffusers-style parameter names
    (encoder.conv_in / down_blocks.N.* / mid_block.* / up_blocks.N.*, quant_conv,
    post_quant_conv, decoder.conv_in ...).
  - The native Wan2.1 checkpoint flattens the encoder/decoder into ``nn.Sequential``
    lists (encoder.downsamples.N / decoder.upsamples.N / *.residual.{0,2,3,6}).
  - This script maps native keys -> MLX keys by an explicit structural table,
    transposes conv kernels to channels-last (MLX layout), and reshapes RMSNorm
    gamma to a 1-D vector. Coverage and shapes are verified strictly.

Output (a MLX-loadable folder, NOT committed to git):
  out_dir/config.json          - AutoencoderKLWanConfig (mlx-diffuser compatible)
  out_dir/model.safetensors    - channels-last MLX weights (mx.save format)

Usage:
  python scripts/convert_vae_to_mlx.py \
      --pth /Volumes/ssd/lingbot-assets/Wan2.1_VAE.pth \
      --out eval/bench_vae_mlx/vae_mlx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mlx_diffuser import AutoencoderKLWan, AutoencoderKLWanConfig  # noqa: E402


# --- residual block submodule mapping -----------------------------------------
# native ResidualBlock: Sequential(RMS_norm, SiLU, Conv3d, RMS_norm, SiLU, Dropout, Conv3d)
#   residual.0.gamma, residual.2.weight/bias, residual.3.gamma, residual.5.weight/bias, shortcut.*
# MLX WanResidualBlock: norm1.gamma, conv1.*, norm2.gamma, conv2.*, conv_shortcut.*
def _remap_residual(native_tail: str) -> str:
    table = {
        "residual.0.gamma": "norm1.gamma",
        "residual.2.weight": "conv1.weight",
        "residual.2.bias": "conv1.bias",
        "residual.3.gamma": "norm2.gamma",
        "residual.6.weight": "conv2.weight",
        "residual.6.bias": "conv2.bias",
        "shortcut.weight": "conv_shortcut.weight",
        "shortcut.bias": "conv_shortcut.bias",
    }
    return table[native_tail]


# encoder.downsamples.N (0..10) == MLX encoder.down_blocks.N (0..10)
_ENC_DS = {i: i for i in range(11)}
# decoder upsamples flattened index -> (up_block index, role)
#   0,1,2 -> up_blocks.0.resnets.{0,1,2}; 3 -> up_blocks.0.upsamplers.0
#   4,5,6 -> up_blocks.1.resnets.{0,1,2}; 7 -> up_blocks.1.upsamplers.0
#   8,9,10 -> up_blocks.2.resnets.{0,1,2}; 11 -> up_blocks.2.upsamplers.0
#   12,13,14 -> up_blocks.3.resnets.{0,1,2}
_DEC_UP = {
    0: ("up_blocks.0.resnets.0", "resblock"),
    1: ("up_blocks.0.resnets.1", "resblock"),
    2: ("up_blocks.0.resnets.2", "resblock"),
    3: ("up_blocks.0.upsamplers.0", "resample"),
    4: ("up_blocks.1.resnets.0", "resblock"),
    5: ("up_blocks.1.resnets.1", "resblock"),
    6: ("up_blocks.1.resnets.2", "resblock"),
    7: ("up_blocks.1.upsamplers.0", "resample"),
    8: ("up_blocks.2.resnets.0", "resblock"),
    9: ("up_blocks.2.resnets.1", "resblock"),
    10: ("up_blocks.2.resnets.2", "resblock"),
    11: ("up_blocks.2.upsamplers.0", "resample"),
    12: ("up_blocks.3.resnets.0", "resblock"),
    13: ("up_blocks.3.resnets.1", "resblock"),
    14: ("up_blocks.3.resnets.2", "resblock"),
}


def native_to_mlx_key(native_key: str) -> str:
    """Map a native Wan2.1 .pth key to the MLX AutoencoderKLWan parameter key."""
    # --- top-level quant / post-quant convs ---
    if native_key == "conv1.weight" or native_key == "conv1.bias":
        return f"quant_conv.{native_key.split('.')[-1]}"
    if native_key == "conv2.weight" or native_key == "conv2.bias":
        return f"post_quant_conv.{native_key.split('.')[-1]}"

    # --- encoder ---
    if native_key.startswith("encoder.conv1."):
        return "encoder.conv_in." + native_key.split("encoder.conv1.", 1)[1]

    if native_key.startswith("encoder.downsamples."):
        rest = native_key.split("encoder.downsamples.", 1)[1]
        idx_str, tail = rest.split(".", 1)
        idx = int(idx_str)
        mlx_prefix = f"encoder.down_blocks.{idx}"
        if tail.startswith("residual.") or tail.startswith("shortcut."):
            return f"{mlx_prefix}.{_remap_residual(tail)}"
        # resample (resample.1.*) or time_conv.*
        return f"{mlx_prefix}.{tail}"

    if native_key.startswith("encoder.middle."):
        rest = native_key.split("encoder.middle.", 1)[1]
        sub, tail = rest.split(".", 1)
        if sub == "0":
            return f"encoder.mid_block.resnets.0.{_remap_residual(tail)}"
        if sub == "2":
            return f"encoder.mid_block.resnets.1.{_remap_residual(tail)}"
        if sub == "1":  # AttentionBlock: norm.gamma, to_qkv.*, proj.*
            return f"encoder.mid_block.attentions.0.{tail}"
        raise KeyError(native_key)

    if native_key == "encoder.head.0.gamma":
        return "encoder.norm_out.gamma"
    if native_key.startswith("encoder.head.2."):
        return "encoder.conv_out." + native_key.split("encoder.head.2.", 1)[1]

    # --- decoder ---
    if native_key.startswith("decoder.conv1."):
        return "decoder.conv_in." + native_key.split("decoder.conv1.", 1)[1]

    if native_key.startswith("decoder.middle."):
        rest = native_key.split("decoder.middle.", 1)[1]
        sub, tail = rest.split(".", 1)
        if sub == "0":
            return f"decoder.mid_block.resnets.0.{_remap_residual(tail)}"
        if sub == "2":
            return f"decoder.mid_block.resnets.1.{_remap_residual(tail)}"
        if sub == "1":
            return f"decoder.mid_block.attentions.0.{tail}"
        raise KeyError(native_key)

    if native_key.startswith("decoder.upsamples."):
        rest = native_key.split("decoder.upsamples.", 1)[1]
        idx_str, tail = rest.split(".", 1)
        idx = int(idx_str)
        mlx_prefix, role = _DEC_UP[idx]
        if role == "resblock":
            return f"decoder.{mlx_prefix}.{_remap_residual(tail)}"
        # resample block: resample.1.* or time_conv.*
        return f"decoder.{mlx_prefix}.{tail}"

    if native_key == "decoder.head.0.gamma":
        return "decoder.norm_out.gamma"
    if native_key.startswith("decoder.head.2."):
        return "decoder.conv_out." + native_key.split("decoder.head.2.", 1)[1]

    raise KeyError(f"Unmapped native key: {native_key}")


def conv_to_channels_last(w: np.ndarray) -> np.ndarray:
    """PyTorch conv kernel -> MLX channels-last. Conv2d (O,I,kH,kW)->(O,kH,kW,I);
    Conv3d (O,I,kT,kH,kW)->(O,kT,kH,kW,I)."""
    if w.ndim == 4:
        return w.transpose(0, 2, 3, 1)
    if w.ndim == 5:
        return w.transpose(0, 2, 3, 4, 1)
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", default="/Volumes/ssd/lingbot-assets/Wan2.1_VAE.pth")
    ap.add_argument("--out", default="eval/bench_vae_mlx/vae_mlx")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[convert] loading native checkpoint: {args.pth}")
    native = torch.load(args.pth, map_location="cpu", weights_only=True)
    print(f"[convert] native keys: {len(native)}")

    # Build target MLX model to know exact expected keys/shapes.
    cfg = AutoencoderKLWanConfig()
    model = AutoencoderKLWan(cfg)
    expected = {k: tuple(v.shape) for k, v in __import__("mlx.utils").utils.tree_flatten(model.parameters())}
    print(f"[convert] MLX expected keys: {len(expected)}")

    converted: dict[str, mx.array] = {}
    unmapped = []
    for k, v in native.items():
        try:
            mlx_key = native_to_mlx_key(k)
        except KeyError:
            unmapped.append(k)
            continue
        arr = v.detach().cpu().numpy().astype(np.float32)
        if mlx_key.endswith(".gamma"):
            arr = arr.reshape(-1)
        elif mlx_key.endswith(".weight") and arr.ndim in (4, 5):
            arr = conv_to_channels_last(arr)
        converted[mlx_key] = mx.array(arr)

    # Strict coverage + shape check (mirrors converters/base.py::_assert_matches).
    got_shapes = {k: tuple(v.shape) for k, v in converted.items()}
    missing = sorted(set(expected) - set(got_shapes))
    extra = sorted(set(got_shapes) - set(expected))
    mismatched = sorted(k for k in expected.keys() & got_shapes.keys() if expected[k] != got_shapes[k])
    if unmapped:
        print(f"[convert] WARNING unmapped native keys: {unmapped}")
    if missing or extra or mismatched:
        print("[convert] MISMATCH:")
        if missing:
            print(f"  missing ({len(missing)}): {missing[:10]}")
        if extra:
            print(f"  extra ({len(extra)}): {extra[:10]}")
        if mismatched:
            print(f"  shape-mismatch ({len(mismatched)}): {mismatched[:10]}")
        return 1
    print("[convert] strict key+shape coverage: OK (194/194)")

    # Load weights into model.
    model.load_weights(list(converted.items()), strict=True)
    mx.eval(model.parameters())

    # Numerical verification vs a few native tensors (pre-transpose).
    print("[convert] verifying weight values on sampled layers...")
    from mlx.utils import tree_flatten
    flat = dict(tree_flatten(model.parameters()))
    checks = []
    # 1) decoder.conv_out gamma
    g_native = native["decoder.head.0.gamma"].numpy().reshape(-1).astype(np.float32)
    g_mlx = np.array(flat["decoder.norm_out.gamma"])
    checks.append(("decoder.norm_out.gamma", float(np.abs(g_native - g_mlx).max())))
    # 2) decoder.conv_out.weight (native 3,96,3,3,3 -> MLX 3,3,3,3,96)
    w_native = conv_to_channels_last(native["decoder.head.2.weight"].numpy().astype(np.float32))
    w_mlx = np.array(flat["decoder.conv_out.weight"])
    checks.append(("decoder.conv_out.weight", float(np.abs(w_native - w_mlx).max())))
    # 3) quant_conv weight
    q_native = conv_to_channels_last(native["conv1.weight"].numpy().astype(np.float32))
    q_mlx = np.array(flat["quant_conv.weight"])
    checks.append(("quant_conv.weight", float(np.abs(q_native - q_mlx).max())))
    # 4) a resample conv2d
    r_native = conv_to_channels_last(native["encoder.downsamples.2.resample.1.weight"].numpy().astype(np.float32))
    r_mlx = np.array(flat["encoder.down_blocks.2.resample.1.weight"])
    checks.append(("encoder.down_blocks.2.resample.1.weight", float(np.abs(r_native - r_mlx).max())))
    ok = True
    for name, err in checks:
        status = "OK" if err < 1e-5 else "MISMATCH"
        if err >= 1e-5:
            ok = False
        print(f"   {status} {name}: max_abs_diff={err:.3e}")
    if not ok:
        print("[convert] FATAL: weight value verification failed")
        return 1

    # Save config + weights.
    cfg_dict = {
        "class_name": "AutoencoderKLWan",
        "base_dim": cfg.base_dim,
        "z_dim": cfg.z_dim,
        "dim_mult": list(cfg.dim_mult),
        "num_res_blocks": cfg.num_res_blocks,
        "attn_scales": list(cfg.attn_scales),
        "temperal_downsample": list(cfg.temperal_downsample),
        "in_channels": cfg.in_channels,
        "out_channels": cfg.out_channels,
        "latents_mean": list(cfg.latents_mean),
        "latents_std": list(cfg.latents_std),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    mx.save_safetensors(str(out_dir / "model.safetensors"), dict(tree_flatten(model.parameters())))
    print(f"[convert] wrote {(out_dir / 'config.json')}")
    print(f"[convert] wrote {(out_dir / 'model.safetensors')}")
    print("[convert] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
