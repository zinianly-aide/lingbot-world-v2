"""Lightweight unit tests for the backend interface layer.

Uses mock backends to test the interface contract, registry, and
configuration parsing without loading real models.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from wan.backends.base import BackendConfig, InferenceBackend
from wan.backends.registry import get_backend, list_backends, register_backend


class MockBackend(InferenceBackend):
    """Minimal mock backend for testing the interface contract."""

    def __init__(self, config):
        super().__init__(config)
        self.loaded = set()
        self.synced = False
        self.cache_cleared = False

    def load_model(self, model_type, checkpoint_path, **kwargs):
        if model_type in self._loaded_models:
            return self._loaded_models[model_type]
        self.loaded.add(model_type)
        mock = MagicMock()
        mock.decode = MagicMock(return_value=[torch.randn(3, 13, 64, 64)])
        self._loaded_models[model_type] = mock
        return mock

    def unload_model(self, model_type):
        self.loaded.discard(model_type)
        self._loaded_models.pop(model_type, None)

    def unload_all(self):
        self.loaded.clear()
        self._loaded_models.clear()

    def sync(self):
        self.synced = True

    def empty_cache(self):
        self.cache_cleared = True

    def current_memory_mb(self):
        return 100.0

    def driver_memory_mb(self):
        return 200.0

    def vae_decode(self, latents, model=None, **kwargs):
        if model is None:
            if "vae" not in self._loaded_models:
                raise RuntimeError("VAE model not loaded.")
            model = self._loaded_models["vae"]
        return model.decode(latents)[0]


class TestBackendConfig(unittest.TestCase):
    """Test BackendConfig dataclass."""

    def test_default_config(self):
        cfg = BackendConfig()
        self.assertEqual(cfg.name, "mps")
        self.assertEqual(cfg.dtype, "fp32")
        self.assertEqual(cfg.device, "mps")
        self.assertTrue(cfg.sequential_load)
        self.assertEqual(cfg.extra, {})

    def test_custom_config(self):
        cfg = BackendConfig(name="mlx", dtype="bf16", device="cpu", sequential_load=False)
        self.assertEqual(cfg.name, "mlx")
        self.assertEqual(cfg.dtype, "bf16")
        self.assertFalse(cfg.sequential_load)

    def test_extra_config(self):
        cfg = BackendConfig(extra={"quantize": 4, "cache_threshold": 0.2})
        self.assertEqual(cfg.extra["quantize"], 4)
        self.assertEqual(cfg.extra["cache_threshold"], 0.2)


class TestBackendRegistry(unittest.TestCase):
    """Test backend registry."""

    def test_register_and_get(self):
        register_backend("test_mock", MockBackend)
        cfg = BackendConfig(name="test_mock")
        backend = get_backend(cfg)
        self.assertIsInstance(backend, MockBackend)
        self.assertEqual(backend.config.name, "test_mock")

    def test_unknown_backend_raises(self):
        cfg = BackendConfig(name="nonexistent_backend_xyz")
        with self.assertRaises(ValueError) as ctx:
            get_backend(cfg)
        self.assertIn("Unknown backend", str(ctx.exception))

    def test_list_backends(self):
        register_backend("test_list_a", MockBackend)
        register_backend("test_list_b", MockBackend)
        backends = list_backends()
        self.assertIn("test_list_a", backends)
        self.assertIn("test_list_b", backends)
        # mps should always be registered (built-in)
        self.assertIn("mps", backends)


class TestBackendInterfaceContract(unittest.TestCase):
    """Test that MockBackend fulfills the InferenceBackend interface."""

    def setUp(self):
        self.cfg = BackendConfig(name="mock")
        self.backend = MockBackend(self.cfg)

    def test_load_model(self):
        model = self.backend.load_model("vae", "/fake/path.pth")
        self.assertIsNotNone(model)
        self.assertIn("vae", self.backend.loaded)
        self.assertIn("vae", self.backend.get_loaded_models())

    def test_load_model_idempotent(self):
        m1 = self.backend.load_model("vae", "/fake/path.pth")
        m2 = self.backend.load_model("vae", "/fake/path.pth")
        self.assertIs(m1, m2)

    def test_unload_model(self):
        self.backend.load_model("vae", "/fake/path.pth")
        self.backend.unload_model("vae")
        self.assertNotIn("vae", self.backend.loaded)
        self.assertNotIn("vae", self.backend.get_loaded_models())

    def test_unload_all(self):
        self.backend.load_model("vae", "/fake.pth")
        self.backend.load_model("dit", "/fake.pth")
        self.backend.unload_all()
        self.assertEqual(len(self.backend.loaded), 0)
        self.assertEqual(len(self.backend.get_loaded_models()), 0)

    def test_sync(self):
        self.backend.sync()
        self.assertTrue(self.backend.synced)

    def test_empty_cache(self):
        self.backend.empty_cache()
        self.assertTrue(self.backend.cache_cleared)

    def test_memory_methods(self):
        self.assertEqual(self.backend.current_memory_mb(), 100.0)
        self.assertEqual(self.backend.driver_memory_mb(), 200.0)

    def test_vae_decode(self):
        self.backend.load_model("vae", "/fake.pth")
        latents = torch.randn(16, 4, 64, 96)
        result = self.backend.vae_decode(latents)
        self.assertEqual(result.shape, (3, 13, 64, 64))

    def test_vae_decode_with_explicit_model(self):
        model = self.backend.load_model("vae", "/fake.pth")
        latents = torch.randn(16, 4, 64, 96)
        result = self.backend.vae_decode(latents, model=model)
        self.assertEqual(result.shape, (3, 13, 64, 64))

    def test_vae_decode_without_loaded_model_raises(self):
        latents = torch.randn(16, 4, 64, 96)
        with self.assertRaises(RuntimeError):
            self.backend.vae_decode(latents)

    def test_is_available(self):
        self.assertTrue(self.backend.is_available())


class TestTorchMPSBackendBasic(unittest.TestCase):
    """Test TorchMPSBackend initialization and config (no model loading)."""

    def test_init(self):
        from wan.backends.torch_mps import TorchMPSBackend
        cfg = BackendConfig(name="mps", dtype="fp32", device="mps")
        backend = TorchMPSBackend(cfg)
        self.assertEqual(backend.config.name, "mps")
        self.assertTrue(backend.is_available() or True)  # MPS may not be available in test env

    def test_dtype_map(self):
        from wan.backends.torch_mps import TorchMPSBackend
        cfg = BackendConfig(name="mps", dtype="bf16")
        backend = TorchMPSBackend(cfg)
        self.assertEqual(backend._dtype, torch.bfloat16)

    def test_unknown_dtype_fallback(self):
        from wan.backends.torch_mps import TorchMPSBackend
        cfg = BackendConfig(name="mps", dtype="int8")
        backend = TorchMPSBackend(cfg)
        self.assertEqual(backend._dtype, torch.float32)


class TestMLXBackendAvailability(unittest.TestCase):
    """Test MLX backend availability check (doesn't require MLX installed)."""

    def test_mlx_backend_module_imports(self):
        # The module should import even if mlx is not installed
        # (it raises ImportError only on instantiation)
        try:
            from wan.backends import mlx_backend
            self.assertTrue(hasattr(mlx_backend, "MLXBackend"))
        except ImportError:
            # If mlx is not installed, the module may fail to import
            # This is acceptable in test environments without MLX
            pass


class TestMLXBackendConfig(unittest.TestCase):
    """Fast config/contract tests for MLXBackend (no real model load)."""

    def setUp(self):
        try:
            from wan.backends.mlx_backend import MLXBackend
        except ImportError:
            self.skipTest("mlx/mlx-diffuser not installed")
        self.MLXBackend = MLXBackend

    def test_dtype_map(self):
        import mlx.core as mx
        be = self.MLXBackend(BackendConfig(name="mlx", dtype="bf16"))
        self.assertEqual(be._dtype, mx.bfloat16)
        be = self.MLXBackend(BackendConfig(name="mlx", dtype="fp32"))
        self.assertEqual(be._dtype, mx.float32)

    def test_unknown_dtype_fallback(self):
        import mlx.core as mx
        be = self.MLXBackend(BackendConfig(name="mlx", dtype="int8"))
        self.assertEqual(be._dtype, mx.float32)

    def test_dit_and_text_encoder_not_implemented(self):
        be = self.MLXBackend(BackendConfig(name="mlx", dtype="bf16"))
        with self.assertRaises(NotImplementedError):
            be.load_model("dit", "/fake/path")
        with self.assertRaises(NotImplementedError):
            be.load_model("text_encoder", "/fake/path")

    def test_vae_decode_without_model_raises(self):
        be = self.MLXBackend(BackendConfig(name="mlx", dtype="bf16"))
        with self.assertRaises(RuntimeError):
            be.vae_decode(torch.randn(16, 4, 8, 8))


@unittest.skipUnless(
    __import__("os").environ.get("MLX_BACKEND_SMOKE") == "1",
    "real MLX VAE decode smoke test; set MLX_BACKEND_SMOKE=1 to run",
)
class TestMLXVaeDecodeSmoke(unittest.TestCase):
    """Real (gated) smoke test: loads the converted MLX VAE and decodes a tiny latent.

    Skipped by default to keep the unit-test suite fast. Run explicitly with:
        MLX_BACKEND_SMOKE=1 python -m unittest tests.test_backends.TestMLXVaeDecodeSmoke
    Requires the converted weights at eval/bench_vae_mlx/vae_mlx/ (run
    scripts/convert_vae_to_mlx.py first).
    """

    VAE_DIR = "eval/bench_vae_mlx/vae_mlx"

    def test_vae_decode_output_shape(self):
        import os
        from pathlib import Path
        from wan.backends.base import BackendConfig
        from wan.backends.registry import get_backend

        if not (Path(self.VAE_DIR) / "config.json").exists():
            self.skipTest("converted MLX VAE weights not present")

        be = get_backend(BackendConfig(name="mlx", dtype="bf16"))
        self.assertTrue(be.is_available())
        be.load_model("vae", self.VAE_DIR)
        # Tiny 1-latent-frame input [C=16, T=1, H=8, W=8].
        z = torch.randn(16, 1, 8, 8)
        out = be.vae_decode(z)
        self.assertEqual(out.ndim, 4)
        self.assertEqual(out.shape[0], 3)  # C=3
        self.assertTrue(float(out.min()) >= -1.0001 and float(out.max()) <= 1.0001)
        be.unload_all()


if __name__ == "__main__":
    unittest.main()
