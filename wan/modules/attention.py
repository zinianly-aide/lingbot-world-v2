import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'sdpa_attention',
]


def _build_padding_mask(q_lens, k_lens, lq, lk, device, dtype=torch.float32):
    """Build a boolean padding mask from q_lens/k_lens for SDPA.

    Returns a mask of shape [B, 1, Lq, Lk] where True = attend, False = ignore.
    If q_lens and k_lens are both None, returns None (no padding).
    """
    if q_lens is None and k_lens is None:
        return None

    b = q_lens.shape[0] if q_lens is not None else k_lens.shape[0]
    if q_lens is None:
        q_lens = torch.full((b,), lq, dtype=torch.int32, device=device)
    if k_lens is None:
        k_lens = torch.full((b,), lk, dtype=torch.int32, device=device)

    # [B, Lq]
    q_idx = torch.arange(lq, device=device).unsqueeze(0) < q_lens.unsqueeze(1)
    # [B, Lk]
    k_idx = torch.arange(lk, device=device).unsqueeze(0) < k_lens.unsqueeze(1)
    # [B, 1, Lq, Lk]
    mask = (q_idx.unsqueeze(2) & k_idx.unsqueeze(1)).unsqueeze(1)
    return mask


def sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    """Scaled Dot-Product Attention fallback for non-CUDA devices (MPS/CPU).

    q: [B, Lq, Nq, C1]
    k: [B, Lk, Nk, C1]
    v: [B, Lk, Nk, C2]
    q_lens/k_lens: [B] optional padding lengths (used to build real mask)
    """
    b, lq, lk = q.size(0), q.size(1), k.size(1)

    # q_scale
    if q_scale is not None:
        q = q * q_scale

    # Build padding mask from q_lens/k_lens
    attn_mask = _build_padding_mask(q_lens, k_lens, lq, lk, q.device)

    # SDPA expects [B, N, L, C]
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    # Handle Nq != Nk (GQA): repeat k/v heads
    nq = q.size(1)
    nk = k.size(1)
    if nq != nk:
        assert nq % nk == 0, f"nq={nq} must be divisible by nk={nk}"
        repeat = nq // nk
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    # softmax_scale: SDPA uses scale parameter in newer PyTorch
    sdpa_kwargs = dict(
        attn_mask=attn_mask,
        is_causal=causal and attn_mask is None,
        dropout_p=dropout_p,
    )
    if softmax_scale is not None:
        sdpa_kwargs['scale'] = softmax_scale

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
    out = out.transpose(1, 2).contiguous()
    return out


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """Device-aware attention dispatch.

    CUDA + FlashAttention available: flash_attention (FA2/FA3)
    MPS / CPU / no FlashAttention: sdpa_attention with real padding mask
    """
    is_cuda = q.device.type == 'cuda'
    has_flash = FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE

    if is_cuda and has_flash:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if not is_cuda:
            logging_msg = f"Using SDPA attention (device={q.device.type})"
        else:
            logging_msg = "FlashAttention not available, using SDPA fallback"
        if q_lens is not None or k_lens is not None:
            logging_msg += " with real padding mask from q_lens/k_lens"
        warnings.warn(logging_msg)

        return sdpa_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
        )
