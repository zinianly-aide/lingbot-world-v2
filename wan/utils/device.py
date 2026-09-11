"""Unified device abstraction for CUDA / MPS / CPU.

This module provides device-aware helpers so that business logic does not
directly call torch.cuda.* or hardcode 'cuda' in autocast contexts.

Device resolution priority for 'auto':
    CUDA available > MPS available > CPU
"""

from __future__ import annotations

import contextlib
from typing import Optional, Union

import torch


DeviceLike = Union[str, torch.device]


def resolve_device(device: DeviceLike = "auto") -> torch.device:
    """Resolve a device string to a torch.device.

    Args:
        device: 'auto', 'cuda', 'mps', 'cpu', or a torch.device.

    Returns:
        Resolved torch.device.

    Raises:
        ValueError: If the requested device is not available.
    """
    if isinstance(device, torch.device):
        return device

    device = device.lower().strip()

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this system.")
        return torch.device("cuda")

    if device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                "MPS is not available. Requires macOS 12.3+ with Apple Silicon "
                "and a PyTorch build with MPS support."
            )
        return torch.device("mps")

    if device == "cpu":
        return torch.device("cpu")

    # Allow explicit device indices like 'cuda:0', 'mps:0'
    if ":" in device:
        base, idx = device.split(":", 1)
        if base == "cuda" and torch.cuda.is_available():
            return torch.device(f"cuda:{idx}")
        if base == "mps" and torch.backends.mps.is_available():
            return torch.device(f"mps:{idx}")

    raise ValueError(f"Unknown device: '{device}'. Use 'auto', 'cuda', 'mps', or 'cpu'.")


def is_cuda(device: DeviceLike) -> bool:
    """Check if device is CUDA."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    return dev.type == "cuda"


def is_mps(device: DeviceLike) -> bool:
    """Check if device is MPS."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    return dev.type == "mps"


def is_cpu(device: DeviceLike) -> bool:
    """Check if device is CPU."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    return dev.type == "cpu"


def device_synchronize(device: Optional[DeviceLike] = None) -> None:
    """Synchronize the current device.

    For CUDA: torch.cuda.synchronize()
    For MPS: torch.mps.synchronize()
    For CPU: no-op
    """
    if device is not None:
        dev = device if isinstance(device, torch.device) else torch.device(device)
    else:
        # Auto-detect from current default device
        if torch.cuda.is_initialized():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")

    if is_cuda(dev):
        torch.cuda.synchronize()
    elif is_mps(dev):
        torch.mps.synchronize()
    # CPU: no-op


def device_empty_cache(device: Optional[DeviceLike] = None) -> None:
    """Empty the device cache.

    For CUDA: torch.cuda.empty_cache()
    For MPS: torch.mps.empty_cache()
    For CPU: gc.collect() (called by caller if needed)
    """
    if device is not None:
        dev = device if isinstance(device, torch.device) else torch.device(device)
    else:
        if torch.cuda.is_initialized():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")

    if is_cuda(dev):
        torch.cuda.empty_cache()
    elif is_mps(dev):
        torch.mps.empty_cache()
    # CPU: no-op


@contextlib.contextmanager
def device_autocast(
    device: DeviceLike,
    dtype: Optional[torch.dtype] = None,
    enabled: bool = True,
):
    """Device-aware autocast context manager.

    For CUDA: torch.amp.autocast('cuda', dtype=dtype, enabled=enabled)
    For MPS: torch.amp.autocast('mps', dtype=dtype, enabled=enabled)
    For CPU: torch.amp.autocast('cpu', dtype=dtype, enabled=enabled)

    Note: MPS autocast may have limited operator support in some PyTorch
    versions. If operators fail, set enabled=False or use explicit casting.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    device_type = dev.type

    if dtype is None:
        if device_type == "cuda":
            dtype = torch.float16
        elif device_type == "mps":
            dtype = torch.float16
        else:
            dtype = torch.bfloat16

    with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled):
        yield


def get_device_name(device: DeviceLike) -> str:
    """Get a human-readable device name."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if is_cuda(dev):
        try:
            return torch.cuda.get_device_name(dev)
        except Exception:
            return "CUDA"
    if is_mps(dev):
        return "Apple MPS"
    return "CPU"


def get_device_memory(device: DeviceLike) -> Optional[int]:
    """Get total device memory in bytes, or None if not available."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if is_cuda(dev):
        try:
            return torch.cuda.get_device_properties(dev).total_mem
        except Exception:
            return None
    # MPS and CPU don't have a straightforward "total memory" query
    # (unified memory / system RAM). Return None.
    return None
