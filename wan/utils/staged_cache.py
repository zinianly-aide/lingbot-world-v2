"""Staged cache utilities for M4 low-memory inference pipeline.

Allows splitting inference into independent stages:
  1. encode-image: VAE encode input image -> image_condition.safetensors
  2. generate-latents: DiT generation -> generated_latents.safetensors
  3. decode: VAE decode -> mp4

Each stage can run in a separate process, with large models loaded/unloaded
independently. Caches include strict metadata validation to prevent silent
reuse of incompatible caches.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from safetensors.torch import save_file, load_file


FORMAT_VERSION = "1.0"


@dataclass
class ImageConditionMetadata:
    """Metadata for image condition cache."""
    format_version: str = FORMAT_VERSION
    source_image_sha256: str = ""
    requested_frame_num: int = 0
    aligned_frame_num: int = 0
    chunk_size: int = 0
    h: int = 0
    w: int = 0
    lat_f: int = 0
    lat_h: int = 0
    lat_w: int = 0
    vae_stride: tuple = (4, 8, 8)
    patch_size: tuple = (1, 2, 2)
    dtype: str = "float32"

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["vae_stride"] = list(self.vae_stride)
        d["patch_size"] = list(self.patch_size)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ImageConditionMetadata":
        d = d.copy()
        d["vae_stride"] = tuple(d.get("vae_stride", (4, 8, 8)))
        d["patch_size"] = tuple(d.get("patch_size", (1, 2, 2)))
        return cls(**d)

    def validate(self, expected: "ImageConditionMetadata") -> list[str]:
        """Validate against expected metadata. Returns list of issues."""
        issues = []
        if self.format_version != expected.format_version:
            issues.append(f"format_version: {self.format_version} != {expected.format_version}")
        if self.source_image_sha256 != expected.source_image_sha256:
            issues.append("source_image_sha256 mismatch")
        if self.aligned_frame_num != expected.aligned_frame_num:
            issues.append(f"aligned_frame_num: {self.aligned_frame_num} != {expected.aligned_frame_num}")
        if self.lat_f != expected.lat_f or self.lat_h != expected.lat_h or self.lat_w != expected.lat_w:
            issues.append(f"latent shape: ({self.lat_f},{self.lat_h},{self.lat_w}) != ({expected.lat_f},{expected.lat_h},{expected.lat_w})")
        if self.vae_stride != expected.vae_stride:
            issues.append(f"vae_stride: {self.vae_stride} != {expected.vae_stride}")
        if self.patch_size != expected.patch_size:
            issues.append(f"patch_size: {self.patch_size} != {expected.patch_size}")
        return issues


@dataclass
class GeneratedLatentsMetadata:
    """Metadata for generated latents cache."""
    format_version: str = FORMAT_VERSION
    checkpoint_id: str = ""
    seed: int = -1
    requested_frame_num: int = 0
    aligned_frame_num: int = 0
    chunk_size: int = 0
    h: int = 0
    w: int = 0
    lat_f: int = 0
    lat_h: int = 0
    lat_w: int = 0
    dtype: str = "float32"
    prompt_embedding_sha256: str = ""
    image_condition_sha256: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "GeneratedLatentsMetadata":
        return cls(**d)

    def validate(self, expected: "GeneratedLatentsMetadata") -> list[str]:
        """Validate against expected metadata. Returns list of issues."""
        issues = []
        if self.format_version != expected.format_version:
            issues.append(f"format_version: {self.format_version} != {expected.format_version}")
        if self.checkpoint_id != expected.checkpoint_id:
            issues.append("checkpoint_id mismatch")
        if self.aligned_frame_num != expected.aligned_frame_num:
            issues.append(f"aligned_frame_num: {self.aligned_frame_num} != {expected.aligned_frame_num}")
        if self.lat_f != expected.lat_f or self.lat_h != expected.lat_h or self.lat_w != expected.lat_w:
            issues.append(f"latent shape mismatch")
        if self.dtype != expected.dtype:
            issues.append(f"dtype: {self.dtype} != {expected.dtype}")
        return issues


def sha256_file(path: str) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Compute SHA256 of a tensor's data."""
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def save_image_condition(
    output_path: str,
    image_condition: torch.Tensor,
    metadata: ImageConditionMetadata,
):
    """Save image condition tensor with metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tensors = {"image_condition": image_condition.detach().cpu().contiguous()}
    safetensors_metadata = {k: str(v) for k, v in metadata.to_dict().items()}
    save_file(tensors, output_path, metadata=safetensors_metadata)

    # Also write sidecar JSON for easy inspection
    json_path = output_path.replace(".safetensors", ".json")
    with open(json_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)


def load_image_condition(
    path: str,
    expected_metadata: Optional[ImageConditionMetadata] = None,
) -> tuple[torch.Tensor, ImageConditionMetadata]:
    """Load image condition tensor with metadata validation.

    Raises RuntimeError if metadata validation fails.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image condition file not found: {path}")

    tensors = load_file(path)
    if "image_condition" not in tensors:
        raise RuntimeError(f"image_condition tensor not found in {path}")

    # Load metadata from sidecar JSON (safetensors metadata only has str values)
    json_path = path.replace(".safetensors", ".json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            metadata = ImageConditionMetadata.from_dict(json.load(f))
    else:
        metadata = ImageConditionMetadata()

    if expected_metadata is not None:
        issues = metadata.validate(expected_metadata)
        if issues:
            raise RuntimeError(
                "Image condition metadata validation failed:\n  - "
                + "\n  - ".join(issues)
            )

    return tensors["image_condition"], metadata


def save_generated_latents(
    output_path: str,
    latents: torch.Tensor,
    metadata: GeneratedLatentsMetadata,
):
    """Save generated latents tensor with metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tensors = {"latents": latents.detach().cpu().contiguous()}
    safetensors_metadata = {k: str(v) for k, v in metadata.to_dict().items()}
    save_file(tensors, output_path, metadata=safetensors_metadata)

    json_path = output_path.replace(".safetensors", ".json")
    with open(json_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)


def load_generated_latents(
    path: str,
    expected_metadata: Optional[GeneratedLatentsMetadata] = None,
) -> tuple[torch.Tensor, GeneratedLatentsMetadata]:
    """Load generated latents tensor with metadata validation.

    Raises RuntimeError if metadata validation fails.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Generated latents file not found: {path}")

    tensors = load_file(path)
    if "latents" not in tensors:
        raise RuntimeError(f"latents tensor not found in {path}")

    json_path = path.replace(".safetensors", ".json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            metadata = GeneratedLatentsMetadata.from_dict(json.load(f))
    else:
        metadata = GeneratedLatentsMetadata()

    if expected_metadata is not None:
        issues = metadata.validate(expected_metadata)
        if issues:
            raise RuntimeError(
                "Generated latents metadata validation failed:\n  - "
                + "\n  - ".join(issues)
            )

    return tensors["latents"], metadata
