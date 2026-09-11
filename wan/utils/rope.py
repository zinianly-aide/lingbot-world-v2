"""FP32 real-valued RoPE implementation for MPS/CUDA/CPU compatibility.

The original implementation uses torch.float64 + torch.view_as_complex,
which is not supported on Apple MPS (float64 not available). This module
provides a drop-in FP32 real-valued replacement using even/odd split +
cos/sin, with numerically equivalent results.
"""

from __future__ import annotations

import torch


def rope_apply_real(x, grid_sizes, freqs, start_frame=0):
    """FP32 real-valued RoPE (drop-in replacement for complex-based rope_apply).

    The original implementation uses complex multiplication: x_complex * freqs_complex,
    where freqs = exp(i*angle) from torch.polar(). This is mathematically equivalent to
    a 2D rotation by angle. We implement it directly with real arithmetic to avoid
    float64/complex128 (unsupported on MPS).

    Args:
        x: [B, seq_len, n_heads, head_dim] (last dim must be even)
        grid_sizes: [B, 3] (f, h, w)
        freqs: precomputed complex frequency tensor from rope_params()
        start_frame: frame offset for causal RoPE

    Returns:
        Rotated tensor with same shape/dtype as input.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs into temporal / height / width components
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # Build frequency grid: [seq_len, 1, c] (complex from rope_params)
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)

        # Extract cos/sin from complex freqs (freqs = cos(angle) + i*sin(angle))
        # Move to CPU for complex ops if on MPS (MPS supports complex64 but be safe)
        freqs_i = freqs_i.to(torch.complex64)
        cos_f = freqs_i.real.to(torch.float32)  # [seq_len, 1, c]
        sin_f = freqs_i.imag.to(torch.float32)  # [seq_len, 1, c]

        # FP32 real-valued RoPE: even/odd split + rotation
        x_i = x[i, :seq_len].to(torch.float32)  # [seq_len, n, 2c]
        x_even = x_i[..., 0::2]  # [seq_len, n, c]
        x_odd = x_i[..., 1::2]   # [seq_len, n, c]

        # 2D rotation: [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
        x_rot = torch.stack([
            x_even * cos_f - x_odd * sin_f,
            x_even * sin_f + x_odd * cos_f,
        ], dim=-1).flatten(-2)  # [seq_len, n, 2c]

        # Append unrotated part (if head_dim > 2c)
        x_rot = torch.cat([x_rot, x[i, seq_len:]], dim=0)
        output.append(x_rot)

    return torch.stack(output).type_as(x)


def causal_rope_apply_real(x, grid_sizes, freqs, start_frame=0):
    """Alias for rope_apply_real (causal version with start_frame)."""
    return rope_apply_real(x, grid_sizes, freqs, start_frame=start_frame)
