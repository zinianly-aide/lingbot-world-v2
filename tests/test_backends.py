"""Tests for the pluggable VLM perception backends."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from world_condition import (
    BACKENDS,
    MlxBackend,
    TransformersBackend,
    WorldDescription,
    create_backend,
)
from world_condition.backends.base import PerceptionResult, WorldPerceptionBackend


class BackendFactoryTests(unittest.TestCase):
    def test_factory_returns_transformers(self):
        backend = create_backend("transformers")
        self.assertIsInstance(backend, TransformersBackend)
        self.assertEqual(backend.backend_name, "transformers")
        self.assertEqual(backend.model_name, "openbmb/MiniCPM-V-4.6")

    def test_factory_returns_mlx(self):
        backend = create_backend("mlx")
        self.assertIsInstance(backend, MlxBackend)
        self.assertEqual(backend.backend_name, "mlx")
        self.assertEqual(backend.model_name, "mlx-community/MiniCPM-V-4.6-4bit")

    def test_factory_custom_model(self):
        backend = create_backend("transformers", model_name="custom/model")
        self.assertEqual(backend.model_name, "custom/model")

    def test_factory_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            create_backend("nonexistent")

    def test_backends_registry(self):
        self.assertIn("transformers", BACKENDS)
        self.assertIn("mlx", BACKENDS)
        self.assertEqual(len(BACKENDS), 2)


class BackendInterfaceTests(unittest.TestCase):
    def test_base_class_is_abstract(self):
        with self.assertRaises(TypeError):
            WorldPerceptionBackend("test")

    def test_perception_result_fallback_property(self):
        ok = PerceptionResult(world=WorldDescription())
        self.assertFalse(ok.used_fallback)
        fail = PerceptionResult(world=WorldDescription(), error="boom")
        self.assertTrue(fail.used_fallback)

    def test_mlx_backend_has_required_methods(self):
        backend = MlxBackend()
        self.assertTrue(hasattr(backend, "load"))
        self.assertTrue(hasattr(backend, "analyze"))
        self.assertTrue(hasattr(backend, "release"))
        self.assertEqual(backend.backend_name, "mlx")

    def test_transformers_backend_loader_injection(self):
        """Test that the _loader hook works for test injection."""
        calls = []

        def fake_loader(model_name, device):
            calls.append((model_name, device))
            return object(), object()

        backend = TransformersBackend(model_name="test/model", device="cpu")
        backend._loader = fake_loader
        backend.load()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "test/model")
        self.assertIsNotNone(backend.model)
        self.assertIsNotNone(backend.processor)


class GenerateWorldPromptTests(unittest.TestCase):
    """Integration tests for generate._prepare_world_prompt."""

    @classmethod
    def setUpClass(cls):
        # Mac has no CUDA; mock it before importing generate (which imports wan).
        import torch

        torch.cuda.is_available = lambda: False
        torch.cuda.current_device = lambda: 0
        torch.cuda.device_count = lambda: 0
        import generate

        cls._generate = generate

    def _make_args(self, **overrides):
        import argparse

        defaults = dict(
            vlm_world_prompt=False,
            world_condition_file=None,
            vlm_image=None,
            vlm_model=None,
            vlm_device="cpu",
            vlm_backend="transformers",
            image="examples/00/image.jpg",
            save_dir="output",
            dump_world_prompt=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_disabled_returns_original_prompt(self):
        args = self._make_args()
        original = "Keep the main subject and move forward"
        prompt, world = self._generate._prepare_world_prompt(args, original)
        self.assertEqual(prompt, original)
        self.assertIsNone(world)

    def test_cached_file_loads_world(self):
        from world_condition import save_world_condition, WorldDescription

        world = WorldDescription(environment="test room", scene_layout="desk center")
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "wc.json"
            save_world_condition(world, cache_path)
            args = self._make_args(world_condition_file=str(cache_path), save_dir=directory)
            prompt, loaded_world = self._generate._prepare_world_prompt(args, "move camera left")
        self.assertIsNotNone(loaded_world)
        self.assertEqual(loaded_world.environment, "test room")
        self.assertIn("Requested evolution (authoritative user intent): move camera left", prompt)
        self.assertIn("test room", prompt)

    def test_vlm_failure_falls_back_to_original(self):
        args = self._make_args(vlm_world_prompt=True, vlm_image="examples/00/image.jpg")
        original = "keep subject, move forward"

        with patch.object(self._generate, "MiniCPMVPerceiver") as MockPerceiver:
            mock_instance = MockPerceiver.return_value
            mock_instance.analyze.return_value = PerceptionResult(
                world=WorldDescription(),
                error="model offline",
            )
            prompt, world = self._generate._prepare_world_prompt(args, original)

        self.assertEqual(prompt, original)
        self.assertIsNotNone(world)
        mock_instance.release.assert_called_once()

    def test_vlm_success_composes_prompt(self):
        args = self._make_args(vlm_world_prompt=True, vlm_image="examples/00/image.jpg")
        original = "pan right slowly"
        world = WorldDescription(
            environment="forest",
            main_entities=(
                type("E", (), {"name": "deer", "appearance": "brown", "position": "left", "state": "standing"})(),
            ),
        )

        with patch.object(self._generate, "MiniCPMVPerceiver") as MockPerceiver:
            mock_instance = MockPerceiver.return_value
            mock_instance.analyze.return_value = PerceptionResult(world=world, raw_text="{}")
            prompt, returned_world = self._generate._prepare_world_prompt(args, original)

        self.assertIsNotNone(returned_world)
        self.assertIn("pan right slowly", prompt)
        self.assertIn("forest", prompt)
        self.assertIn("deer", prompt)
        mock_instance.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
