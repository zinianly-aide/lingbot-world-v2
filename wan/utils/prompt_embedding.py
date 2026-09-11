"""Prompt embedding cache: save/load pre-computed UMT5 text embeddings.

This module enables skipping the UMT5 encoder entirely during generation by
loading a pre-computed prompt embedding from a safetensors file. The
embedding is produced by ``T5EncoderModel([prompt], device)`` and must be
semantically identical to what the pipeline would compute on the fly.

Format:
    prompt_embeds.safetensors contains:
        - "context": tensor of shape [seq_len, hidden_dim] (the truncated
          encoder output, exactly as returned by T5EncoderModel.__call__)
    Metadata (stored in a sidecar .json or in safetensors metadata):
        - prompt_sha256: sha256 of the original prompt text
        - dtype: tensor dtype (e.g. "bfloat16", "float16")
        - shape: [seq_len, hidden_dim]
        - hidden_dim: must be 4096 for umt5-xxl
        - text_len: max sequence length the encoder was configured for
        - model_id: "umt5-xxl"
        - format_version: "1.0"
"""

import hashlib
import json
import logging
import os
from typing import Optional

import torch
from safetensors.torch import load_file, save_file

FORMAT_VERSION = "1.0"
MODEL_ID = "umt5-xxl"
EXPECTED_HIDDEN_DIM = 4096


def prompt_sha256(prompt: str) -> str:
    """Compute sha256 hex digest of a prompt string (UTF-8)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def save_prompt_embedding(
    output_path: str,
    context: torch.Tensor,
    prompt: str,
    text_len: int,
    dtype: Optional[torch.dtype] = None,
) -> str:
    """Save a pre-computed prompt embedding to a safetensors file.

    Args:
        output_path: Path to output .safetensors file.
        context: Tensor of shape [seq_len, hidden_dim], as returned by
            T5EncoderModel.__call__ (already truncated to actual seq length).
        prompt: The original prompt text (used for metadata verification).
        text_len: Max sequence length the T5 encoder was configured for.
        dtype: Optional dtype override; defaults to context.dtype.

    Returns:
        The output path.
    """
    if context.dim() != 2:
        raise ValueError(
            f"Expected context of shape [seq_len, hidden_dim], got {context.shape}"
        )

    dtype = dtype or context.dtype
    dtype_str = str(dtype).replace("torch.", "")

    metadata = {
        "prompt_sha256": prompt_sha256(prompt),
        "dtype": dtype_str,
        "shape": list(context.shape),
        "hidden_dim": context.shape[1],
        "text_len": text_len,
        "model_id": MODEL_ID,
        "format_version": FORMAT_VERSION,
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Save with metadata (safetensors only accepts str values, so convert)
    tensors = {"context": context.detach().cpu()}
    safetensors_metadata = {k: str(v) for k, v in metadata.items()}
    save_file(tensors, output_path, metadata=safetensors_metadata)

    # Also write a sidecar .json for easy inspection (preserves types)
    json_path = output_path.replace(".safetensors", ".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(
        f"Saved prompt embedding to {output_path} "
        f"(shape={list(context.shape)}, dtype={dtype_str}, "
        f"prompt_sha256={metadata['prompt_sha256'][:16]}...)"
    )
    return output_path


def load_prompt_embedding(
    path: str,
    expected_prompt: Optional[str] = None,
    expected_hidden_dim: int = EXPECTED_HIDDEN_DIM,
    max_text_len: Optional[int] = None,
) -> tuple[torch.Tensor, dict]:
    """Load a pre-computed prompt embedding from a safetensors file.

    Validates metadata format, hidden dimension, sequence length, and
    optionally the prompt sha256.

    Args:
        path: Path to .safetensors file.
        expected_prompt: If provided, verify the stored prompt_sha256 matches.
        expected_hidden_dim: Expected hidden size (default 4096 for umt5-xxl).
        max_text_len: If provided, verify seq_len <= max_text_len.

    Returns:
        Tuple of (context tensor on CPU, metadata dict).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If metadata validation fails (wrong format, hidden dim,
            sequence length, or prompt mismatch).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt embedding file not found: {path}")

    tensors = load_file(path)
    if "context" not in tensors:
        raise ValueError(
            f"Prompt embedding file {path} does not contain a 'context' tensor. "
            f"Available keys: {list(tensors.keys())}"
        )

    context = tensors["context"]

    # Read metadata from sidecar JSON (safetensors metadata is not always
    # accessible via load_file in all versions)
    json_path = path.replace(".safetensors", ".json")
    metadata = {}
    if os.path.isfile(json_path):
        with open(json_path) as f:
            metadata = json.load(f)
    else:
        logging.warning(
            f"No sidecar metadata JSON found at {json_path}; "
            f"will infer metadata from tensor shape."
        )
        metadata = {
            "dtype": str(context.dtype).replace("torch.", ""),
            "shape": list(context.shape),
            "hidden_dim": context.shape[1] if context.dim() == 2 else None,
            "format_version": "unknown",
        }

    # Validate format version
    fmt_ver = metadata.get("format_version", "unknown")
    if fmt_ver != FORMAT_VERSION:
        raise ValueError(
            f"Prompt embedding format version mismatch: expected {FORMAT_VERSION}, "
            f"got {fmt_ver}. The file may have been created by an incompatible version."
        )

    # Validate hidden dim
    hidden_dim = metadata.get("hidden_dim")
    if hidden_dim is not None and hidden_dim != expected_hidden_dim:
        raise ValueError(
            f"Prompt embedding hidden_dim mismatch: expected {expected_hidden_dim}, "
            f"got {hidden_dim}. This embedding may be from a different T5 model."
        )
    if context.dim() == 2 and context.shape[1] != expected_hidden_dim:
        raise ValueError(
            f"Context tensor hidden_dim mismatch: expected {expected_hidden_dim}, "
            f"got {context.shape[1]}."
        )

    # Validate sequence length
    if context.dim() == 2 and max_text_len is not None:
        if context.shape[0] > max_text_len:
            raise ValueError(
                f"Prompt embedding seq_len ({context.shape[0]}) exceeds "
                f"max_text_len ({max_text_len})."
            )

    # Validate prompt sha256 if expected prompt provided
    if expected_prompt is not None:
        stored_hash = metadata.get("prompt_sha256")
        expected_hash = prompt_sha256(expected_prompt)
        if stored_hash is not None and stored_hash != expected_hash:
            raise ValueError(
                f"Prompt embedding prompt_sha256 mismatch. "
                f"The embedding was created for a different prompt. "
                f"Stored: {stored_hash[:16]}..., expected: {expected_hash[:16]}..."
            )

    logging.info(
        f"Loaded prompt embedding from {path} "
        f"(shape={list(context.shape)}, dtype={context.dtype}, "
        f"format_version={fmt_ver})"
    )
    return context, metadata


def verify_prompt_embedding_file(path: str) -> dict:
    """Verify a prompt embedding file and return its metadata.

    Does not raise for recoverable issues; returns a dict with 'valid' bool
    and 'issues' list.
    """
    result = {"valid": True, "issues": [], "metadata": {}}
    try:
        _, metadata = load_prompt_embedding(path)
        result["metadata"] = metadata
    except FileNotFoundError as e:
        result["valid"] = False
        result["issues"].append(str(e))
    except ValueError as e:
        result["valid"] = False
        result["issues"].append(str(e))
    return result
