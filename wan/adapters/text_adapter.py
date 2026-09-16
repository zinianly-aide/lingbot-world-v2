"""E0 Text Adapter: MiniCPM5 hidden states -> Wan DiT cross-attention context.

Architecture (random weights, no training in E0):

    MiniCPM5 hidden  [B, L, hidden_dim]
      -> LayerNorm(hidden_dim)
      -> Transformer Resampler  (num_resampler_layers, default 2)
           - learned query tokens  [1, num_queries, hidden_dim]
           - per layer:
               * cross-attention  (queries attend to encoder hidden)
               * self-attention   (queries attend to each other)
               * feed-forward
           - residual + pre-layer-norm blocks
           - padding-aware via attention_mask (1=valid, 0=padding)
      -> [B, num_queries, hidden_dim]
      -> Linear/MLP hidden_dim -> output_dim
      -> LayerNorm(output_dim)
      -> [B, num_queries, output_dim]   (default output_dim=4096 to match UMT5)

The output tensor is intended to *replace* the UMT5 context
``prompt_embeds.safetensors`` at the [B, L, 4096] boundary.  Downstream
``WanModel.text_embedding`` (4096 -> 1536) and the DiT cross-attention are
untouched.

The resampler is intentionally generic: it takes any encoder hidden states
(padding-aware) and a learned query bank, so it can be reused for G1
visual / world hidden conditioning later.

E0 contract:
    * pure random weights, forward-only, no training
    * MPS-compatible (uses nn.MultiheadAttention, no custom CUDA kernels)
    * parameters are float32 by default; input hidden may be bf16/fp32 and is
      cast to the parameter dtype internally
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ResamplerBlock(nn.Module):
    """One Perceiver-style resampler block.

    Order (pre-norm residuals, batch_first):
        1. cross-attn: queries <- encoder hidden (padding-aware)
        2. self-attn: queries <- queries
        3. FFN
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm_q_cross = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm_q_self = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Cross-attention (queries attend to encoder hidden)
        qn = self.norm_q_cross(queries)
        ctx, _ = self.cross_attn(
            qn, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        queries = queries + ctx

        # 2. Self-attention among queries (no padding mask needed)
        qn = self.norm_q_self(queries)
        self_ctx, _ = self.self_attn(qn, qn, qn, need_weights=False)
        queries = queries + self_ctx

        # 3. FFN
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class TextAdapter(nn.Module):
    """MiniCPM5 hidden -> Wan-compatible context adapter (E0).

    Args:
        hidden_dim: encoder hidden size (MiniCPM5-2B = 2048).
        output_dim: adapter output width (must stay 4096 to match UMT5 context).
        num_queries: number of learned latent query tokens (default 64).
        num_resampler_layers: depth of the resampler stack (default 2).
        num_heads: attention heads (default 8; divides hidden_dim).
        ffn_mult: FFN hidden expansion factor (default 4).
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        output_dim: int = 4096,
        num_queries: int = 64,
        num_resampler_layers: int = 2,
        num_heads: int = 8,
        ffn_mult: int = 4,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_queries = num_queries
        self.num_resampler_layers = num_resampler_layers

        # Input norm over encoder hidden
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Learned query bank: [1, num_queries, hidden_dim], broadcast over batch
        self.queries = nn.Parameter(torch.empty(1, num_queries, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        ffn_dim = hidden_dim * ffn_mult
        self.layers = nn.ModuleList([
            ResamplerBlock(hidden_dim, num_heads, ffn_dim)
            for _ in range(num_resampler_layers)
        ])

        # Project hidden_dim -> output_dim (UMT5 context width = 4096)
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        input_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the adapter.

        Args:
            input_hidden: encoder last hidden state, shape [B, L, hidden_dim].
            attention_mask: optional padding mask, shape [B, L] with
                1 = valid token, 0 = padding.  Padding positions are ignored
                by cross-attention.

        Returns:
            Tensor of shape [B, num_queries, output_dim] in float32 (the
            module's parameter dtype), regardless of input dtype.
        """
        if input_hidden.dim() != 3:
            raise ValueError(
                f"input_hidden must be [B, L, D], got shape {tuple(input_hidden.shape)}"
            )
        bsz, seq_len, hidden = input_hidden.shape
        if hidden != self.hidden_dim:
            raise ValueError(
                f"input_hidden last dim {hidden} != adapter hidden_dim {self.hidden_dim}"
            )

        # Run adapter in parameter dtype (float32 by default); cast input up
        # so bf16 MiniCPM5 hidden does not crash float32 Linear/LayerNorm.
        param_dtype = self.proj.weight.dtype
        memory = self.input_norm(input_hidden.to(param_dtype))

        # Build key_padding_mask for cross-attn: True = *ignore* this key.
        # attention_mask convention: 1=valid, 0=padding -> ignore where == 0.
        kpm = None
        if attention_mask is not None:
            if attention_mask.shape != (bsz, seq_len):
                raise ValueError(
                    f"attention_mask must be [B, L]=[{bsz}, {seq_len}], "
                    f"got {tuple(attention_mask.shape)}"
                )
            kpm = attention_mask.to(torch.bool).logical_not()

        queries = self.queries.expand(bsz, -1, -1).contiguous()
        for layer in self.layers:
            queries = layer(queries, memory, memory_key_padding_mask=kpm)

        out = self.proj(queries)
        out = self.output_norm(out)
        return out


__all__ = ["TextAdapter", "ResamplerBlock"]
