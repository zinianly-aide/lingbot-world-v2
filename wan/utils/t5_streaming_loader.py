"""Low-memory UMT5 checkpoint loader for M4 16GB environments.

The standard T5EncoderModel.__init__ does:
    model = umt5_xxl(...)  # full model in memory
    state = torch.load(checkpoint)  # full state_dict in memory
    model.load_state_dict(state)

This can produce ~2x memory peak (full model + full state_dict). For
umt5-xxl (~11GB BF16), this is problematic on M4 16GB.

This module provides:
1. Meta-init UMT5 encoder (no real parameter allocation)
2. mmap-based state_dict loading (if .pth supports it)
3. Tensor-by-tensor assign to target device/dtype
4. Full integrity verification
5. Memory instrumentation

If the .pth file does not support mmap, use convert_t5_checkpoint.py to
convert it to safetensors first (one-time operation).
"""

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class T5LoadStats:
    """Statistics from a low-memory T5 load."""
    success: bool = False
    tensor_count: int = 0
    source_tensor_bytes: int = 0
    target_tensor_bytes: int = 0
    load_duration_sec: float = 0.0
    rss_before_mb: Optional[float] = None
    rss_peak_mb: Optional[float] = None
    rss_after_mb: Optional[float] = None
    target_device: Optional[str] = None
    target_dtype: Optional[str] = None
    used_mmap: bool = False
    missing_keys: list = field(default_factory=list)
    unexpected_keys: list = field(default_factory=list)
    shape_mismatch: list = field(default_factory=list)
    meta_params_remaining: int = 0
    meta_buffers_remaining: int = 0

    def summary(self) -> str:
        lines = [
            f"T5 low-memory load {'SUCCESS' if self.success else 'FAILED'}",
            f"  target dtype: {self.target_dtype}",
            f"  target device: {self.target_device}",
            f"  used mmap: {self.used_mmap}",
            f"  tensors loaded: {self.tensor_count}",
            f"  source bytes: {self.source_tensor_bytes / 1e9:.3f} GB",
            f"  target bytes: {self.target_tensor_bytes / 1e9:.3f} GB",
            f"  duration: {self.load_duration_sec:.2f}s",
        ]
        if self.rss_before_mb:
            lines.append(f"  RSS before: {self.rss_before_mb:.0f} MB")
        if self.rss_peak_mb:
            lines.append(f"  RSS peak: {self.rss_peak_mb:.0f} MB")
        if self.rss_after_mb:
            lines.append(f"  RSS after: {self.rss_after_mb:.0f} MB")
        lines.append(f"  missing keys: {len(self.missing_keys)}")
        lines.append(f"  unexpected keys: {len(self.unexpected_keys)}")
        lines.append(f"  shape mismatch: {len(self.shape_mismatch)}")
        lines.append(f"  meta params remaining: {self.meta_params_remaining}")
        lines.append(f"  meta buffers remaining: {self.meta_buffers_remaining}")
        if self.missing_keys:
            lines.append(f"  MISSING (first 10): {self.missing_keys[:10]}")
        if self.unexpected_keys:
            lines.append(f"  UNEXPECTED (first 10): {self.unexpected_keys[:10]}")
        if self.shape_mismatch:
            lines.append(f"  SHAPE MISMATCH (first 10): {self.shape_mismatch[:10]}")
        return "\n".join(lines)


def _get_rss_mb() -> Optional[float]:
    """Get current process RSS in MB, or None if psutil unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def _get_model_state_keys(model: torch.nn.Module) -> set:
    """Get all state dict keys from a model (parameters + buffers)."""
    return set(model.state_dict().keys())


def _assign_tensor_to_model(
    model: torch.nn.Module,
    key: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    target_device: torch.device,
    stats: T5LoadStats,
) -> bool:
    """Assign a single tensor to the model, casting dtype and moving device.

    Uses load_state_dict with a single-key partial state dict and assign=True
    to directly replace meta tensors.
    """
    # Check if key exists in model
    model_keys = _get_model_state_keys(model)
    if key not in model_keys:
        stats.unexpected_keys.append(key)
        return False

    # Get expected shape
    current = model.state_dict()[key]
    if current.shape != tensor.shape:
        stats.shape_mismatch.append(
            f"{key}: expected {current.shape}, got {tensor.shape}"
        )
        return False

    # Cast and move
    tensor = tensor.to(dtype=target_dtype, device=target_device)

    # Assign via partial state dict
    try:
        model.load_state_dict({key: tensor}, strict=False, assign=True)
        stats.tensor_count += 1
        stats.source_tensor_bytes += tensor.numel() * 4  # source assumed F32/BF16
        stats.target_tensor_bytes += tensor.numel() * tensor.element_size()
        return True
    except Exception as e:
        logger.warning(f"Failed to assign {key}: {e}")
        return False


def check_pth_mmap_support(checkpoint_path: str) -> dict:
    """Check if a .pth checkpoint supports mmap loading.

    Returns a dict with:
        - supports_mmap: bool
        - is_zipfile: bool
        - format: str ('zipfile' | 'legacy_pickle' | 'unknown')
        - tensor_count: int (if inspectable)
        - error: str (if any)
    """
    result = {
        "supports_mmap": False,
        "is_zipfile": False,
        "format": "unknown",
        "tensor_count": 0,
        "error": None,
    }

    try:
        import zipfile
        result["is_zipfile"] = zipfile.is_zipfile(checkpoint_path)

        if result["is_zipfile"]:
            result["format"] = "zipfile"
            # PyTorch zipfile format supports mmap in 2.1+
            result["supports_mmap"] = True

            # Try to inspect tensor count without full load
            try:
                with zipfile.ZipFile(checkpoint_path, "r") as zf:
                    # data.pkl contains the pickle, 0/, 1/, etc. are tensors
                    tensor_files = [
                        n for n in zf.namelist()
                        if n.startswith("data/") or n.split("/")[0].isdigit()
                    ]
                    result["tensor_count"] = len(tensor_files)
            except Exception:
                pass
        else:
            # Could be legacy pickle format
            result["format"] = "legacy_pickle"
            result["supports_mmap"] = False

    except Exception as e:
        result["error"] = str(e)

    return result


def load_t5_low_memory(
    model_factory: Callable[[], torch.nn.Module],
    checkpoint_path: str,
    target_device: torch.device,
    target_dtype: torch.dtype = torch.bfloat16,
    use_mmap: bool = True,
) -> tuple[torch.nn.Module, T5LoadStats]:
    """Load a T5 model with minimal memory peak.

    1. Create model on meta device (no real parameter allocation)
    2. Load checkpoint with mmap (if supported)
    3. Assign tensors one at a time, casting to target dtype/device
    4. Verify completeness

    Args:
        model_factory: Callable that creates the model (will be called under
            torch.device('meta') context).
        checkpoint_path: Path to .pth checkpoint.
        target_device: Target device (cpu, mps, cuda).
        target_dtype: Target dtype (bfloat16, float16, float32).
        use_mmap: Whether to use mmap for .pth loading (if supported).

    Returns:
        Tuple of (loaded model, T5LoadStats).
    """
    stats = T5LoadStats(
        target_device=str(target_device),
        target_dtype=str(target_dtype),
    )
    start_time = time.time()
    stats.rss_before_mb = _get_rss_mb()

    # Step 1: Meta-init model
    logger.info(f"Creating T5 model on meta device...")
    with torch.device("meta"):
        model = model_factory()
    expected_keys = _get_model_state_keys(model)
    logger.info(f"Meta model created: {len(expected_keys)} keys")

    # Step 2: Check mmap support
    mmap_info = check_pth_mmap_support(checkpoint_path)
    stats.used_mmap = use_mmap and mmap_info["supports_mmap"]
    logger.info(
        f"Checkpoint format: {mmap_info['format']}, "
        f"mmap supported: {mmap_info['supports_mmap']}, "
        f"using mmap: {stats.used_mmap}"
    )

    # Step 3: Load state dict (mmap if possible)
    logger.info(f"Loading checkpoint state dict...")
    load_kwargs = {"map_location": "cpu"}
    if stats.used_mmap:
        load_kwargs["mmap"] = True
        # weights_only may not work with all .pth files; try first
        try:
            state_dict = torch.load(checkpoint_path, weights_only=True, **load_kwargs)
        except Exception:
            logger.warning("weights_only=True failed, retrying without...")
            state_dict = torch.load(checkpoint_path, **load_kwargs)
    else:
        state_dict = torch.load(checkpoint_path, **load_kwargs)

    # Handle nested state dicts (some checkpoints wrap in 'model' or 'state_dict')
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "module"):
            if key in state_dict and isinstance(state_dict[key], dict):
                logger.info(f"Found nested state dict under '{key}'")
                state_dict = state_dict[key]
                break

    checkpoint_keys = set(state_dict.keys())
    logger.info(f"Checkpoint loaded: {len(checkpoint_keys)} keys")

    # Step 4: Assign tensors one at a time
    loaded_keys = set()
    for key in checkpoint_keys:
        if key not in expected_keys:
            stats.unexpected_keys.append(key)
            continue

        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            logger.warning(f"Skipping non-tensor key: {key} (type={type(tensor)})")
            continue

        success = _assign_tensor_to_model(
            model, key, tensor, target_dtype, target_device, stats
        )
        if success:
            loaded_keys.add(key)

        # Release source tensor reference
        del tensor

        # Periodic GC to keep memory peak low
        if stats.tensor_count % 100 == 0:
            gc.collect()
            stats.rss_peak_mb = max(
                stats.rss_peak_mb or 0,
                _get_rss_mb() or 0,
            )

    # Release state dict
    del state_dict
    gc.collect()

    # Step 5: Verify completeness
    stats.missing_keys = list(expected_keys - loaded_keys)
    stats.meta_params_remaining = sum(1 for p in model.parameters() if p.is_meta)
    stats.meta_buffers_remaining = sum(1 for b in model.buffers() if b.is_meta)

    stats.success = (
        len(stats.missing_keys) == 0
        and len(stats.unexpected_keys) == 0
        and len(stats.shape_mismatch) == 0
        and stats.meta_params_remaining == 0
        and stats.meta_buffers_remaining == 0
    )

    stats.load_duration_sec = time.time() - start_time
    stats.rss_after_mb = _get_rss_mb()
    if stats.rss_peak_mb is None:
        stats.rss_peak_mb = max(
            stats.rss_before_mb or 0,
            stats.rss_after_mb or 0,
        )

    if stats.success:
        logger.info(
            f"T5 low-memory load complete: {stats.tensor_count} tensors, "
            f"{stats.load_duration_sec:.1f}s, "
            f"RSS peak {stats.rss_peak_mb:.0f} MB"
        )
    else:
        logger.error(f"T5 low-memory load failed:\n{stats.summary()}")

    return model, stats


def convert_pth_to_safetensors(
    pth_path: str,
    output_path: str,
    target_dtype: Optional[torch.dtype] = None,
) -> dict:
    """Convert a .pth checkpoint to safetensors format (one-time operation).

    This allows subsequent loads to use safetensors.safe_open for true
    tensor-by-tensor streaming without loading the full state dict.

    Args:
        pth_path: Input .pth file path.
        output_path: Output .safetensors file path.
        target_dtype: If set, cast all tensors to this dtype before saving.

    Returns:
        Dict with tensor_count, total_bytes, output_path.
    """
    from safetensors.torch import save_file

    logger.info(f"Converting {pth_path} to safetensors...")
    state_dict = torch.load(pth_path, map_location="cpu")

    # Handle nested state dicts
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "module"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    tensors = {}
    total_bytes = 0
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            if target_dtype is not None:
                tensor = tensor.to(dtype=target_dtype)
            tensors[key] = tensor.contiguous()
            total_bytes += tensor.numel() * tensor.element_size()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(tensors, output_path)

    result = {
        "tensor_count": len(tensors),
        "total_bytes": total_bytes,
        "total_gb": total_bytes / 1e9,
        "output_path": output_path,
    }
    logger.info(
        f"Conversion complete: {result['tensor_count']} tensors, "
        f"{result['total_gb']:.3f} GB -> {output_path}"
    )
    return result
