"""Low-memory streaming safetensors checkpoint loader.

Designed for M4 16GB unified memory. Instead of loading all shards into a
single state dict (which can peak at >2x the checkpoint size), this loader:

1. Creates the model on the ``meta`` device (no real parameter allocation).
2. Reads the safetensors index to map keys -> shards.
3. Opens each shard with ``safetensors.safe_open`` and loads tensors one at
   a time, casting F32 -> target dtype and moving to target device.
4. Assigns each tensor directly into the meta model via
   ``load_state_dict(partial, strict=False, assign=True)``.
5. Releases the source tensor immediately after assignment.
6. Verifies no meta tensors remain and reports missing/unexpected keys.

This avoids constructing a full state dict and avoids a second copy during
``model.to(dtype=...)``.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)

# Candidate index / single-file names (in priority order)
_INDEX_NAMES = (
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
)
_SINGLE_NAMES = (
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
)


@dataclass
class StreamingLoadStats:
    """Statistics and diagnostics from a streaming checkpoint load."""

    target_dtype: torch.dtype = torch.float32
    target_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    shard_count: int = 0
    tensor_count: int = 0
    source_tensor_bytes: int = 0
    target_tensor_bytes: int = 0
    load_duration_sec: float = 0.0
    rss_before_mb: Optional[float] = None
    rss_peak_mb: Optional[float] = None
    rss_after_mb: Optional[float] = None
    mps_allocated_before_mb: Optional[float] = None
    mps_allocated_after_mb: Optional[float] = None
    expected_keys: List[str] = field(default_factory=list)
    loaded_keys: List[str] = field(default_factory=list)
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
    shape_mismatch: Dict[str, Tuple[torch.Size, torch.Size]] = field(default_factory=dict)
    meta_params_remaining: int = 0
    meta_buffers_remaining: int = 0

    @property
    def success(self) -> bool:
        return (
            self.tensor_count > 0
            and len(self.expected_keys) > 0
            and len(self.missing_keys) == 0
            and len(self.unexpected_keys) == 0
            and len(self.shape_mismatch) == 0
            and self.meta_params_remaining == 0
            and self.meta_buffers_remaining == 0
        )

    def summary(self) -> str:
        lines = [
            f"Streaming load {'SUCCESS' if self.success else 'FAILED'}",
            f"  target dtype: {self.target_dtype}",
            f"  target device: {self.target_device}",
            f"  shards: {self.shard_count}",
            f"  tensors loaded: {self.tensor_count}",
            f"  source bytes: {self.source_tensor_bytes / 1e9:.3f} GB",
            f"  target bytes: {self.target_tensor_bytes / 1e9:.3f} GB",
            f"  duration: {self.load_duration_sec:.2f}s",
            f"  expected keys: {len(self.expected_keys)}",
            f"  loaded keys: {len(self.loaded_keys)}",
            f"  missing: {len(self.missing_keys)}",
            f"  unexpected: {len(self.unexpected_keys)}",
            f"  shape mismatch: {len(self.shape_mismatch)}",
            f"  meta params remaining: {self.meta_params_remaining}",
            f"  meta buffers remaining: {self.meta_buffers_remaining}",
        ]
        if self.rss_before_mb is not None:
            lines.append(f"  RSS before: {self.rss_before_mb:.1f} MB")
        if self.rss_peak_mb is not None:
            lines.append(f"  RSS peak: {self.rss_peak_mb:.1f} MB")
        if self.rss_after_mb is not None:
            lines.append(f"  RSS after: {self.rss_after_mb:.1f} MB")
        if self.missing_keys:
            lines.append(f"  MISSING (first 10): {self.missing_keys[:10]}")
        if self.unexpected_keys:
            lines.append(f"  UNEXPECTED (first 10): {self.unexpected_keys[:10]}")
        if self.shape_mismatch:
            lines.append(f"  SHAPE MISMATCH (first 5): {list(self.shape_mismatch.items())[:5]}")
        return "\n".join(lines)


def _get_rss_mb() -> Optional[float]:
    """Return current process RSS in MB, or None if unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _get_mps_allocated_mb() -> Optional[float]:
    """Return current MPS allocated memory in MB, or None if unavailable."""
    try:
        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / (1024 * 1024)
    except Exception:
        pass
    return None


def _read_safetensors_index(dit_dir: str) -> Optional[Dict[str, str]]:
    """Read a sharded safetensors index and return weight_map (key -> shard)."""
    for index_name in _INDEX_NAMES:
        index_path = os.path.join(dit_dir, index_name)
        if not os.path.isfile(index_path):
            continue
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        logger.info(
            f"Read sharded index {index_name}: {len(weight_map)} keys, "
            f"{len(set(weight_map.values()))} shards"
        )
        return weight_map
    return None


def _find_single_safetensors(dit_dir: str) -> Optional[str]:
    """Find a single (non-sharded) safetensors file in dit_dir."""
    for name in _SINGLE_NAMES:
        path = os.path.join(dit_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _get_model_state_keys(model: torch.nn.Module) -> Set[str]:
    """Return all parameter and buffer keys in the model's state_dict."""
    keys = set()
    for name, _ in model.named_parameters():
        keys.add(name)
    for name, _ in model.named_buffers():
        keys.add(name)
    return keys


def _assign_tensor_to_model(
    model: torch.nn.Module,
    key: str,
    tensor: torch.Tensor,
    expected_keys: Set[str],
    shape_mismatch: Dict[str, Tuple[torch.Size, torch.Size]],
) -> bool:
    """Assign a single tensor to the model via load_state_dict(assign=True).

    Returns True if the key was found in the model and assigned successfully.
    Records shape mismatches in shape_mismatch.
    """
    if key not in expected_keys:
        return False

    # Check shape via meta parameter/buffer
    try:
        ref = model.get_parameter(key)
    except AttributeError:
        try:
            ref = model.get_buffer(key)
        except AttributeError:
            return False

    if ref.shape != tensor.shape:
        shape_mismatch[key] = (ref.shape, tensor.shape)
        logger.warning(
            f"Shape mismatch for {key}: expected {ref.shape}, got {tensor.shape}"
        )
        return False

    # Assign via partial state dict with assign=True (works for meta tensors)
    partial = {key: tensor}
    model.load_state_dict(partial, strict=False, assign=True)
    return True


def load_sharded_safetensors_streaming(
    model: torch.nn.Module,
    dit_dir: str,
    target_dtype: torch.dtype = torch.float16,
    target_device: Optional[torch.device] = None,
    collect_garbage: bool = True,
) -> StreamingLoadStats:
    """Load a (possibly sharded) safetensors checkpoint into a meta model.

    This is a low-memory streaming loader:
    - Opens each shard with ``safetensors.safe_open``
    - Loads one tensor at a time
    - Casts to target_dtype and moves to target_device
    - Assigns directly into the (meta) model
    - Releases the source tensor immediately

    Args:
        model: A model, ideally created on the ``meta`` device.
        dit_dir: Directory containing the safetensors checkpoint.
        target_dtype: Target dtype for loaded tensors (default: float16 for MPS).
        target_device: Target device. If None, uses the model's first parameter's
            device, or CPU if all params are meta.
        collect_garbage: If True, call gc.collect() after each shard.

    Returns:
        StreamingLoadStats with full diagnostics.
    """
    from safetensors import safe_open

    stats = StreamingLoadStats(target_dtype=target_dtype)
    t0 = time.time()

    # Determine target device
    if target_device is None:
        target_device = torch.device("cpu")
        for p in model.parameters():
            if not p.is_meta:
                target_device = p.device
                break
    stats.target_device = target_device

    # Record baseline memory
    stats.rss_before_mb = _get_rss_mb()
    stats.mps_allocated_before_mb = _get_mps_allocated_mb()
    rss_peak = stats.rss_before_mb

    # Get expected keys from model
    expected_keys = _get_model_state_keys(model)
    stats.expected_keys = sorted(expected_keys)
    logger.info(f"Model has {len(expected_keys)} expected keys (params + buffers)")

    loaded_keys: Set[str] = set()
    unexpected_keys: Set[str] = set()
    shape_mismatch: Dict[str, Tuple[torch.Size, torch.Size]] = {}

    # Try sharded index first
    weight_map = _read_safetensors_index(dit_dir)

    if weight_map is not None:
        # Sharded checkpoint: group keys by shard
        shard_to_keys: Dict[str, List[str]] = {}
        for key, shard in weight_map.items():
            shard_to_keys.setdefault(shard, []).append(key)
        stats.shard_count = len(shard_to_keys)
        logger.info(f"Loading {len(weight_map)} keys from {len(shard_to_keys)} shards")

        for shard_name in sorted(shard_to_keys.keys()):
            shard_path = os.path.join(dit_dir, shard_name)
            if not os.path.isfile(shard_path):
                logger.error(f"Shard file not found: {shard_path}")
                for key in shard_to_keys[shard_name]:
                    unexpected_keys.add(key)
                continue

            keys_in_shard = shard_to_keys[shard_name]
            logger.debug(f"Opening shard {shard_name} ({len(keys_in_shard)} keys)")

            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in keys_in_shard:
                    try:
                        source_tensor = f.get_tensor(key)
                    except Exception as e:
                        logger.error(f"Failed to load tensor {key} from {shard_name}: {e}")
                        unexpected_keys.add(key)
                        continue

                    stats.source_tensor_bytes += source_tensor.numel() * source_tensor.element_size()
                    stats.tensor_count += 1

                    # Cast and move to target
                    tensor = source_tensor.to(dtype=target_dtype, device=target_device)
                    stats.target_tensor_bytes += tensor.numel() * tensor.element_size()

                    # Assign to model
                    assigned = _assign_tensor_to_model(
                        model, key, tensor, expected_keys, shape_mismatch
                    )
                    if assigned:
                        loaded_keys.add(key)
                    else:
                        if key not in shape_mismatch:
                            unexpected_keys.add(key)

                    # Release source tensor immediately
                    del source_tensor
                    del tensor

                    # Track RSS peak
                    rss_now = _get_rss_mb()
                    if rss_now is not None and (rss_peak is None or rss_now > rss_peak):
                        rss_peak = rss_now

            if collect_garbage:
                gc.collect()

    else:
        # Single file checkpoint
        single_path = _find_single_safetensors(dit_dir)
        if single_path is None:
            raise FileNotFoundError(
                f"No safetensors weights found in {dit_dir}. Expected a sharded "
                "index (model.safetensors.index.json) or a single model.safetensors."
            )

        stats.shard_count = 1
        logger.info(f"Loading single safetensors file: {single_path}")

        with safe_open(single_path, framework="pt", device="cpu") as f:
            all_keys = list(f.keys())
            for key in all_keys:
                source_tensor = f.get_tensor(key)
                stats.source_tensor_bytes += source_tensor.numel() * source_tensor.element_size()
                stats.tensor_count += 1

                tensor = source_tensor.to(dtype=target_dtype, device=target_device)
                stats.target_tensor_bytes += tensor.numel() * tensor.element_size()

                assigned = _assign_tensor_to_model(
                    model, key, tensor, expected_keys, shape_mismatch
                )
                if assigned:
                    loaded_keys.add(key)
                else:
                    if key not in shape_mismatch:
                        unexpected_keys.add(key)

                del source_tensor
                del tensor

                rss_now = _get_rss_mb()
                if rss_now is not None and (rss_peak is None or rss_now > rss_peak):
                    rss_peak = rss_now

        if collect_garbage:
            gc.collect()

    # Compute missing keys
    missing_keys = expected_keys - loaded_keys

    # Check for remaining meta tensors
    meta_params = sum(1 for p in model.parameters() if p.is_meta)
    meta_buffers = sum(1 for b in model.buffers() if b.is_meta)

    # Final garbage collection
    if collect_garbage:
        gc.collect()

    # Record final memory
    stats.rss_peak_mb = rss_peak
    stats.rss_after_mb = _get_rss_mb()
    stats.mps_allocated_after_mb = _get_mps_allocated_mb()
    stats.load_duration_sec = time.time() - t0

    stats.loaded_keys = sorted(loaded_keys)
    stats.missing_keys = sorted(missing_keys)
    stats.unexpected_keys = sorted(unexpected_keys)
    stats.shape_mismatch = shape_mismatch
    stats.meta_params_remaining = meta_params
    stats.meta_buffers_remaining = meta_buffers

    # Log summary
    logger.info(stats.summary())

    if not stats.success:
        logger.warning("Streaming load completed with issues (see summary above)")

    return stats


def create_meta_model(model_cls, **kwargs) -> torch.nn.Module:
    """Create a model on the meta device (no real parameter allocation).

    Args:
        model_cls: The model class to instantiate.
        **kwargs: Arguments to pass to the model constructor.

    Returns:
        A model with all parameters/buffers on the meta device.
    """
    with torch.device("meta"):
        model = model_cls(**kwargs)
    return model
