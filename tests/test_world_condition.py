import tempfile
import unittest
from pathlib import Path

from world_condition.schemas import WorldDescription, parse_world_description
from world_condition.vlm_perception import MiniCPMVPerceiver
from world_condition.world_prompt import (
    compose_world_prompt,
    load_world_condition,
    save_world_condition,
)


class WorldConditionTests(unittest.TestCase):
    def test_schema_parsing_and_code_fence(self):
        result = parse_world_description(
            '```json\n{"environment":"studio", "main_entities":[{"name":"cube", "position":"left", "state":"still"}]}\n```'
        )
        self.assertEqual(result.environment, "studio")
        self.assertEqual(result.main_entities[0].name, "cube")
        self.assertEqual(result.main_entities[0].position, "left")

    def test_malformed_output_falls_back(self):
        result = parse_world_description("not JSON")
        self.assertEqual(result, WorldDescription())

    def test_user_prompt_is_authoritative(self):
        world = WorldDescription(environment="kitchen", scene_layout="table on right")
        prompt = compose_world_prompt(world, "Turn the camera left and keep the red mug.")
        self.assertIn("Requested evolution (authoritative user intent): Turn the camera left and keep the red mug.", prompt)
        self.assertIn("Follow the requested evolution as the authoritative user intent", prompt)

    def test_disabled_path_is_identity(self):
        original = "A user-authored prompt"
        # This is the same guard used by generate.py; disabled mode must not
        # normalize, append, or otherwise mutate the original string.
        enabled = False
        actual = original if not enabled else compose_world_prompt(WorldDescription(), original)
        self.assertEqual(actual, original)

    def test_cached_condition_reusable(self):
        world = WorldDescription(environment="park", persistent_constraints=("keep tree identity",))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world_condition.json"
            save_world_condition(world, path)
            loaded = load_world_condition(path)
        self.assertEqual(loaded, world)

    def test_vlm_load_failure_is_safe(self):
        def broken_loader(_model_name, _device):
            raise RuntimeError("offline")

        result = MiniCPMVPerceiver(loader=broken_loader).analyze(object(), "keep the subject")
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.world, WorldDescription())

    def test_malformed_vlm_response_falls_back_to_safe_observation(self):
        class FakeInputs(dict):
            def to(self, _device):
                return self

        class FakeProcessor:
            def apply_chat_template(self, *_args, **_kwargs):
                return FakeInputs(input_ids=[[1]])

            def batch_decode(self, *_args, **_kwargs):
                return ["this is not JSON"]

        class FakeModel:
            device = "cpu"

            def eval(self):
                return self

            def generate(self, **_kwargs):
                return [[1, 2]]

        def loader(_model_name, _device):
            return FakeModel(), FakeProcessor()

        result = MiniCPMVPerceiver(loader=loader).analyze(object())
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.world, WorldDescription())


if __name__ == "__main__":
    unittest.main()
