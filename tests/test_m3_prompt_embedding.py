"""M3 tests: prompt embedding cache, memory release, lazy component loading."""

import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from wan.utils.prompt_embedding import (
    EXPECTED_HIDDEN_DIM,
    FORMAT_VERSION,
    load_prompt_embedding,
    prompt_sha256,
    save_prompt_embedding,
    verify_prompt_embedding_file,
)
from wan.utils.memory import (
    get_memory_stats,
    release_component,
    release_model_and_setattr,
)


class TestPromptEmbeddingRoundTrip(unittest.TestCase):
    """Test save/load round-trip of prompt embeddings."""

    def test_save_load_round_trip(self):
        """Saved embedding should load back with same shape and values."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test_embeds.safetensors")
        context = torch.randn(128, 4096, dtype=torch.bfloat16)
        prompt = "A cat walking on the beach"

        save_prompt_embedding(path, context, prompt, text_len=512)
        loaded, metadata = load_prompt_embedding(path)

        self.assertEqual(loaded.shape, context.shape)
        self.assertEqual(loaded.dtype, context.dtype)
        self.assertTrue(torch.equal(loaded, context))
        self.assertEqual(metadata["prompt_sha256"], prompt_sha256(prompt))
        self.assertEqual(metadata["hidden_dim"], 4096)
        self.assertEqual(metadata["format_version"], FORMAT_VERSION)

    def test_prompt_hash_correct(self):
        """prompt_sha256 should match hashlib.sha256 of UTF-8 encoded prompt."""
        prompt = "Test prompt for hashing"
        expected = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.assertEqual(prompt_sha256(prompt), expected)

    def test_different_prompts_different_hashes(self):
        """Different prompts should have different hashes."""
        self.assertNotEqual(prompt_sha256("cat"), prompt_sha256("dog"))

    def test_metadata_sidecar_json_created(self):
        """Saving should create a sidecar .json with metadata."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test_embeds.safetensors")
        context = torch.randn(64, 4096)
        save_prompt_embedding(path, context, "test", text_len=512)

        json_path = path.replace(".safetensors", ".json")
        self.assertTrue(os.path.isfile(json_path))
        with open(json_path) as f:
            metadata = json.load(f)
        self.assertEqual(metadata["format_version"], FORMAT_VERSION)
        self.assertEqual(metadata["hidden_dim"], 4096)


class TestPromptEmbeddingValidation(unittest.TestCase):
    """Test validation of loaded prompt embeddings."""

    def _make_embedding(self, tmpdir, hidden_dim=4096, seq_len=128,
                         format_version=FORMAT_VERSION, prompt="test"):
        """Helper to create a test embedding file."""
        path = os.path.join(tmpdir, "test.safetensors")
        context = torch.randn(seq_len, hidden_dim)
        # Save with correct metadata first
        save_prompt_embedding(path, context, prompt, text_len=512)
        # If format_version needs to be wrong, overwrite the JSON
        if format_version != FORMAT_VERSION:
            json_path = path.replace(".safetensors", ".json")
            with open(json_path) as f:
                meta = json.load(f)
            meta["format_version"] = format_version
            with open(json_path, "w") as f:
                json.dump(meta, f)
        return path

    def test_wrong_hidden_dim_rejected(self):
        """Embedding with hidden_dim != 4096 should be rejected."""
        tmpdir = tempfile.mkdtemp()
        path = self._make_embedding(tmpdir, hidden_dim=2048)
        with self.assertRaises(ValueError):
            load_prompt_embedding(path, expected_hidden_dim=4096)

    def test_seq_len_exceeds_max_rejected(self):
        """Embedding with seq_len > max_text_len should be rejected."""
        tmpdir = tempfile.mkdtemp()
        path = self._make_embedding(tmpdir, seq_len=1024)
        with self.assertRaises(ValueError):
            load_prompt_embedding(path, max_text_len=512)

    def test_wrong_format_version_rejected(self):
        """Embedding with wrong format_version should be rejected."""
        tmpdir = tempfile.mkdtemp()
        path = self._make_embedding(tmpdir, format_version="0.1")
        with self.assertRaises(ValueError):
            load_prompt_embedding(path)

    def test_prompt_mismatch_rejected(self):
        """Loading with expected_prompt that doesn't match should be rejected."""
        tmpdir = tempfile.mkdtemp()
        path = self._make_embedding(tmpdir, prompt="original prompt")
        with self.assertRaises(ValueError):
            load_prompt_embedding(path, expected_prompt="different prompt")

    def test_matching_prompt_accepted(self):
        """Loading with matching expected_prompt should succeed."""
        tmpdir = tempfile.mkdtemp()
        path = self._make_embedding(tmpdir, prompt="matching prompt")
        context, _ = load_prompt_embedding(path, expected_prompt="matching prompt")
        self.assertIsNotNone(context)

    def test_file_not_found_raises(self):
        """Loading non-existent file should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_prompt_embedding("/nonexistent/path.safetensors")

    def test_verify_file_valid(self):
        """verify_prompt_embedding_file should return valid=True for good file."""
        tmpdir = tempfile.mkdtemp()
        path = self._make_embedding(tmpdir)
        result = verify_prompt_embedding_file(path)
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["issues"]), 0)

    def test_verify_file_invalid(self):
        """verify_prompt_embedding_file should return valid=False for bad file."""
        result = verify_prompt_embedding_file("/nonexistent.safetensors")
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["issues"]), 0)


class TestT5SkipOnPromptEmbedsFile(unittest.TestCase):
    """Test that providing prompt_embeds_file skips T5 encoder construction."""

    def test_t5_constructor_not_called_with_prompt_embeds_file(self):
        """When prompt_embeds_file is set, T5EncoderModel should not be constructed."""
        tmpdir = tempfile.mkdtemp()
        # Create a dummy prompt embeds file
        embeds_path = os.path.join(tmpdir, "embeds.safetensors")
        context = torch.randn(64, 4096)
        save_prompt_embedding(embeds_path, context, "test prompt", text_len=512)

        # Create dummy config
        config = MagicMock()
        config.text_len = 512
        config.t5_dtype = torch.bfloat16
        config.t5_checkpoint = "t5.pth"
        config.t5_tokenizer = "tokenizer"
        config.vae_checkpoint = "vae.pth"
        config.vae_stride = (4, 8, 8)
        config.patch_size = (1, 2, 2)
        config.num_train_timesteps = 1000
        config.boundary = 0.5
        config.param_dtype = torch.bfloat16
        config.fast_checkpoint = "fast"
        config.causal_checkpoint = "causal"
        config.sample_neg_prompt = "bad"
        config.text_dim = 4096

        with patch("wan.image2video.T5EncoderModel") as mock_t5, \
             patch("wan.image2video.Wan2_1_VAE") as mock_vae, \
             patch("wan.image2video.load_dit_model") as mock_load_dit, \
             patch("wan.image2video._resolve_asset_path") as mock_resolve, \
             patch("wan.image2video._configure_model", create=True):
            mock_load_dit.return_value = MagicMock()
            mock_resolve.return_value = "/dummy/path"
            from wan.image2video import WanI2VCausal
            pipe = WanI2VCausal(
                config=config,
                checkpoint_dir=tmpdir,
                device_id=torch.device("cpu"),
                infer_mode="causal_fast",
                prompt_embeds_file=embeds_path,
            )
            # T5 constructor should never be called
            mock_t5.assert_not_called()
            self.assertIsNone(pipe.text_encoder)
            self.assertEqual(pipe.prompt_embeds_file, embeds_path)


class TestMemoryRelease(unittest.TestCase):
    """Test memory release utilities."""

    def test_release_component_none(self):
        """Releasing None should return False without error."""
        result = release_component(None, "test")
        self.assertFalse(result)

    def test_release_component_actual(self):
        """Releasing an actual object should return True."""
        obj = torch.randn(100, 100)
        result = release_component(obj, "test_tensor")
        self.assertTrue(result)

    def test_release_model_and_setattr(self):
        """release_model_and_setattr should clear the attribute and release."""
        class Holder:
            pass
        holder = Holder()
        holder.model = nn.Linear(10, 10)
        self.assertIsNotNone(holder.model)

        result = release_model_and_setattr(holder, "model")
        self.assertTrue(result)
        self.assertIsNone(holder.model)

    def test_release_model_and_setattr_already_none(self):
        """Releasing an already-None attribute should return False."""
        class Holder:
            pass
        holder = Holder()
        holder.model = None
        result = release_model_and_setattr(holder, "model")
        self.assertFalse(result)

    def test_get_memory_stats_returns_rss(self):
        """get_memory_stats should return RSS if psutil is available."""
        stats = get_memory_stats()
        self.assertIn("rss_mb", stats)
        # psutil is installed in the test env
        self.assertIsNotNone(stats["rss_mb"])
        self.assertGreater(stats["rss_mb"], 0)


class TestUnloadMethods(unittest.TestCase):
    """Test WanI2VCausal unload methods."""

    def _make_pipe(self):
        """Create a minimal WanI2VCausal with mocked components."""
        tmpdir = tempfile.mkdtemp()
        config = MagicMock()
        config.text_len = 512
        config.t5_dtype = torch.bfloat16
        config.t5_checkpoint = "t5.pth"
        config.t5_tokenizer = "tokenizer"
        config.vae_checkpoint = "vae.pth"
        config.vae_stride = (4, 8, 8)
        config.patch_size = (1, 2, 2)
        config.num_train_timesteps = 1000
        config.boundary = 0.5
        config.param_dtype = torch.bfloat16
        config.fast_checkpoint = "fast"
        config.causal_checkpoint = "causal"
        config.sample_neg_prompt = "bad"
        config.text_dim = 4096

        with patch("wan.image2video.T5EncoderModel") as mock_t5, \
             patch("wan.image2video.Wan2_1_VAE") as mock_vae, \
             patch("wan.image2video.load_dit_model") as mock_load_dit, \
             patch("wan.image2video._resolve_asset_path") as mock_resolve:
            mock_t5.return_value = MagicMock()
            mock_vae.return_value = MagicMock()
            mock_load_dit.return_value = MagicMock()
            mock_resolve.return_value = "/dummy/path"
            from wan.image2video import WanI2VCausal
            pipe = WanI2VCausal(
                config=config,
                checkpoint_dir=tmpdir,
                device_id=torch.device("cpu"),
                infer_mode="causal_fast",
                sequential_load=False,
            )
        return pipe

    def test_unload_text_encoder(self):
        """unload_text_encoder should set text_encoder to None."""
        pipe = self._make_pipe()
        self.assertIsNotNone(pipe.text_encoder)
        result = pipe.unload_text_encoder()
        self.assertTrue(result)
        self.assertIsNone(pipe.text_encoder)

    def test_unload_vae(self):
        """unload_vae should set vae to None."""
        pipe = self._make_pipe()
        self.assertIsNotNone(pipe.vae)
        result = pipe.unload_vae()
        self.assertTrue(result)
        self.assertIsNone(pipe.vae)

    def test_unload_dit(self):
        """unload_dit should set model to None."""
        pipe = self._make_pipe()
        self.assertIsNotNone(pipe.model)
        result = pipe.unload_dit()
        self.assertTrue(result)
        self.assertIsNone(pipe.model)

    def test_memory_summary(self):
        """memory_summary should return component load status."""
        pipe = self._make_pipe()
        summary = pipe.memory_summary()
        self.assertTrue(summary["text_encoder_loaded"])
        self.assertTrue(summary["vae_loaded"])
        self.assertTrue(summary["dit_loaded"])


if __name__ == "__main__":
    unittest.main()
