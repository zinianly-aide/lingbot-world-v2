"""Memory management utilities for low-memory environments (M4 16GB, etc.).

Provides unified component release and memory instrumentation. On Apple
Silicon unified memory, model.cpu() does NOT free memory — you must del
all references, gc.collect(), and torch.mps.empty_cache().
"""

import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_memory_stats(device: Optional[torch.device] = None) -> dict:
    """Get current memory statistics for the given device.

    Returns a dict with available keys (some may be None if unsupported):
        - rss_mb: process RSS in MB (if psutil available)
        - mps_allocated_mb: MPS current allocated memory (if MPS)
        - mps_recommended_mb: MPS recommended max memory (if MPS)
        - cuda_allocated_mb: CUDA current allocated memory (if CUDA)
        - cuda_reserved_mb: CUDA reserved memory (if CUDA)
    """
    stats = {}

    # Process RSS
    try:
        import psutil
        stats["rss_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        stats["rss_mb"] = None

    # Device-specific
    if device is not None:
        if device.type == "mps":
            try:
                stats["mps_allocated_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)
            except Exception:
                stats["mps_allocated_mb"] = None
            try:
                stats["mps_recommended_mb"] = torch.mps.recommended_max_memory() / (1024 * 1024)
            except Exception:
                stats["mps_recommended_mb"] = None
        elif device.type == "cuda":
            try:
                stats["cuda_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)
                stats["cuda_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 * 1024)
            except Exception:
                stats["cuda_allocated_mb"] = None
                stats["cuda_reserved_mb"] = None

    return stats


def log_memory_stats(prefix: str = "", device: Optional[torch.device] = None):
    """Log current memory statistics."""
    stats = get_memory_stats(device)
    parts = []
    if stats.get("rss_mb") is not None:
        parts.append(f"RSS={stats['rss_mb']:.0f}MB")
    if stats.get("mps_allocated_mb") is not None:
        parts.append(f"MPS_alloc={stats['mps_allocated_mb']:.0f}MB")
    if stats.get("cuda_allocated_mb") is not None:
        parts.append(f"CUDA_alloc={stats['cuda_allocated_mb']:.0f}MB")
    msg = f"[Memory] {prefix} {' '.join(parts)}" if prefix else f"[Memory] {' '.join(parts)}"
    logger.info(msg)
    return stats


def release_component(
    obj,
    name: str = "component",
    device: Optional[torch.device] = None,
    extra_refs: Optional[list] = None,
) -> bool:
    """Release a model/component and free associated memory.

    On Apple Silicon unified memory, simply calling model.cpu() does NOT free
    memory. This function:
    1. Deletes the object and any extra references
    2. Calls gc.collect() to break reference cycles
    3. Calls torch.mps.empty_cache() or torch.cuda.empty_cache()
    4. Synchronizes the device

    Args:
        obj: The object to release (model, tensor, etc.). Set to None after.
        name: Human-readable name for logging.
        device: Device to empty cache for (auto-detected if None).
        extra_refs: Additional references to delete (e.g. cached tensors).

    Returns:
        True if release completed, False if obj was already None.
    """
    if obj is None:
        logger.debug(f"[Memory] {name} already released (None)")
        return False

    logger.info(f"[Memory] Releasing {name}...")

    # Delete extra references first
    if extra_refs:
        for ref in extra_refs:
            try:
                del ref
            except Exception:
                pass

    # Delete the main object
    try:
        del obj
    except Exception as e:
        logger.warning(f"[Memory] Error deleting {name}: {e}")

    # Force garbage collection to break reference cycles
    gc.collect()

    # Empty device cache
    if device is not None:
        if device.type == "mps":
            try:
                torch.mps.empty_cache()
                torch.mps.synchronize()
            except Exception as e:
                logger.debug(f"[Memory] MPS cache empty error: {e}")
        elif device.type == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
            except Exception as e:
                logger.debug(f"[Memory] CUDA cache empty error: {e}")

    logger.info(f"[Memory] {name} released")
    log_memory_stats(f"after {name} release", device)
    return True


def release_model_and_setattr(
    parent_obj,
    attr_name: str,
    device: Optional[torch.device] = None,
) -> bool:
    """Release a model stored as an attribute on parent_obj and set it to None.

    This is the preferred way to release pipeline components (text_encoder,
    vae, model) because it ensures the attribute reference is cleared.

    Args:
        parent_obj: The object holding the model attribute.
        attr_name: Name of the attribute (e.g. "text_encoder", "vae", "model").
        device: Device to empty cache for.

    Returns:
        True if released, False if already None.
    """
    obj = getattr(parent_obj, attr_name, None)
    if obj is None:
        return False

    # First clear the attribute reference
    setattr(parent_obj, attr_name, None)

    # Then release
    return release_component(obj, name=attr_name, device=device)


def check_no_references(obj_name: str, globals_dict: dict, locals_dict: dict) -> list:
    """Check for lingering references to an object (debug utility).

    Returns a list of variable names that still reference the object.
    """
    # This is a best-effort check; gc.get_referrers is more reliable
    # but can be slow. Use for debugging only.
    lingering = []
    for name, val in list(globals_dict.items()):
        if hasattr(val, "__class__") and val.__class__.__name__ == obj_name:
            lingering.append(f"global:{name}")
    for name, val in list(locals_dict.items()):
        if hasattr(val, "__class__") and val.__class__.__name__ == obj_name:
            lingering.append(f"local:{name}")
    return lingering
