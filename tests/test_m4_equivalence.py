import torch

from scripts.compare_m4_latents import compare_files
from wan.utils.staged_cache import GeneratedLatentsMetadata, save_generated_latents


def _write(path, tensor, *, seed=42):
    meta = GeneratedLatentsMetadata(
        checkpoint_id="transformers",
        seed=seed,
        requested_frame_num=13,
        aligned_frame_num=13,
        chunk_size=4,
        h=464,
        w=832,
        lat_f=4,
        lat_h=58,
        lat_w=104,
        dtype=str(tensor.dtype),
    )
    save_generated_latents(str(path), tensor, meta)


def test_m4_equivalence_exact_match_passes(tmp_path):
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    staged = tmp_path / "staged.safetensors"
    full = tmp_path / "full.safetensors"
    _write(staged, tensor)
    _write(full, tensor.clone())

    result = compare_files(str(staged), str(full))

    assert result["equivalent"] is True
    assert result["exact"] is True
    assert result["max_abs_diff"] == 0.0


def test_m4_equivalence_value_mismatch_fails_by_default(tmp_path):
    staged_tensor = torch.zeros(8, dtype=torch.float32)
    full_tensor = staged_tensor.clone()
    full_tensor[3] = 1e-6
    staged = tmp_path / "staged.safetensors"
    full = tmp_path / "full.safetensors"
    _write(staged, staged_tensor)
    _write(full, full_tensor)

    result = compare_files(str(staged), str(full))

    assert result["equivalent"] is False
    assert result["exact"] is False
    assert result["max_abs_diff"] > 0.0


def test_m4_equivalence_metadata_mismatch_fails(tmp_path):
    tensor = torch.ones(8, dtype=torch.float32)
    staged = tmp_path / "staged.safetensors"
    full = tmp_path / "full.safetensors"
    _write(staged, tensor, seed=42)
    _write(full, tensor.clone(), seed=43)

    result = compare_files(str(staged), str(full))

    assert result["equivalent"] is False
    assert any("metadata seed mismatch" in issue for issue in result["issues"])


def test_m4_equivalence_explicit_tolerance_is_opt_in(tmp_path):
    staged_tensor = torch.zeros(8, dtype=torch.float32)
    full_tensor = staged_tensor.clone()
    full_tensor[0] = 1e-6
    staged = tmp_path / "staged.safetensors"
    full = tmp_path / "full.safetensors"
    _write(staged, staged_tensor)
    _write(full, full_tensor)

    strict = compare_files(str(staged), str(full))
    tolerant = compare_files(str(staged), str(full), atol=1e-5, rtol=0.0)

    assert strict["equivalent"] is False
    assert tolerant["equivalent"] is True
    assert tolerant["exact"] is False
