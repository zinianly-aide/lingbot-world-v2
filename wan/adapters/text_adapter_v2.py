"""E1.2 Text Adapter v2.

MiniCPM5 hidden states are resampled to the *UMT5 token length* for the same
prompt, so Wan sees the same token geometry as the teacher before its fixed
512-token padding step.

Key differences from v1:
- variable target length instead of 64 always-on learned queries
- 2048 -> bottleneck -> 4096 projection to reduce trainable parameters
- no output LayerNorm (preserve the teacher embedding distribution)
- explicit query padding mask for mixed-length batches
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ResamplerBlockV2(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm_q_cross = nn.LayerNorm(dim)
        self.norm_mem = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm_q_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

    @staticmethod
    def _zero_padded(x: torch.Tensor, query_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if query_padding_mask is None:
            return x
        return x.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        query_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qn = self.norm_q_cross(queries)
        mn = self.norm_mem(memory)
        ctx, _ = self.cross_attn(
            qn,
            mn,
            mn,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        queries = self._zero_padded(queries + ctx, query_padding_mask)

        qn = self.norm_q_self(queries)
        self_ctx, _ = self.self_attn(
            qn,
            qn,
            qn,
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        queries = self._zero_padded(queries + self_ctx, query_padding_mask)
        queries = self._zero_padded(queries + self.ffn(self.norm_ffn(queries)), query_padding_mask)
        return queries


class VariableLengthTextAdapter(nn.Module):
    """MiniCPM5 [B,L,2048] -> Wan-compatible [B,T,4096].

    ``target_lengths`` must be the token lengths produced by the UMT5 tokenizer
    for the same prompts. The adapter emits only T=max(target_lengths) tokens in
    a batch and returns an output mask; padded query slots are exactly zero.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 768,
        output_dim: int = 4096,
        max_queries: int = 512,
        num_resampler_layers: int = 2,
        num_heads: int = 8,
        ffn_mult: int = 4,
    ):
        super().__init__()
        if bottleneck_dim % num_heads != 0:
            raise ValueError("bottleneck_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.max_queries = max_queries
        self.num_resampler_layers = num_resampler_layers

        self.input_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.queries = nn.Parameter(torch.empty(1, max_queries, bottleneck_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        ffn_dim = bottleneck_dim * ffn_mult
        self.layers = nn.ModuleList([
            ResamplerBlockV2(bottleneck_dim, num_heads, ffn_dim)
            for _ in range(num_resampler_layers)
        ])
        self.output_proj = nn.Linear(bottleneck_dim, output_dim)
        # Intentionally NO output LayerNorm. Wan's pretrained text_embedding was
        # trained against the native UMT5 context distribution.

    def forward(
        self,
        input_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_hidden.dim() != 3:
            raise ValueError(f"input_hidden must be [B,L,D], got {tuple(input_hidden.shape)}")
        bsz, seq_len, hidden = input_hidden.shape
        if hidden != self.hidden_dim:
            raise ValueError(f"expected hidden_dim={self.hidden_dim}, got {hidden}")
        if target_lengths.dim() != 1 or target_lengths.numel() != bsz:
            raise ValueError("target_lengths must be [B]")

        target_lengths = target_lengths.to(device=input_hidden.device, dtype=torch.long)
        if torch.any(target_lengths <= 0):
            raise ValueError("target_lengths must be positive")
        max_target = int(target_lengths.max().item())
        if max_target > self.max_queries:
            raise ValueError(f"target length {max_target} exceeds max_queries={self.max_queries}")

        param_dtype = self.output_proj.weight.dtype
        memory = self.input_proj(input_hidden.to(param_dtype))

        memory_kpm = None
        if attention_mask is not None:
            if attention_mask.shape != (bsz, seq_len):
                raise ValueError("attention_mask must match [B,L]")
            memory_kpm = ~attention_mask.to(torch.bool)

        pos = torch.arange(max_target, device=input_hidden.device).unsqueeze(0)
        output_mask = pos < target_lengths.unsqueeze(1)  # [B,T], True=valid
        query_kpm = ~output_mask

        queries = self.queries[:, :max_target].expand(bsz, -1, -1).contiguous()
        queries = queries.masked_fill(query_kpm.unsqueeze(-1), 0.0)
        for layer in self.layers:
            queries = layer(
                queries,
                memory,
                memory_key_padding_mask=memory_kpm,
                query_padding_mask=query_kpm,
            )

        out = self.output_proj(queries)
        out = out.masked_fill(query_kpm.unsqueeze(-1), 0.0)
        return out, output_mask


__all__ = ["VariableLengthTextAdapter", "ResamplerBlockV2"]
