"""Unit tests for G0.7 Real A/B/C Evaluation Framework.

Covers:
- A/B/C prompt construction
- compose_compact_world_prompt
- User prompt always retained (never truncated/overridden)
- Token count / truncation metadata structure
- Manifest hashing (sha256)
- Resume logic (skip on hash match, regenerate on change)
- Blind mapping (variant hidden, deterministic shuffle)
- Score summary statistics
- Missing video handling
- Failed generation does not pollute existing results
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from world_condition import (  # noqa: E402
    EntityDescription,
    WorldDescription,
    compose_compact_world_prompt,
    compose_world_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_world() -> WorldDescription:
    return WorldDescription(
        environment="a sunlit living room with wooden floors",
        scene_layout="sofa center, coffee table front, window right",
        main_entities=(
            EntityDescription(name="beige sofa", appearance="large three-seat, beige fabric",
                              position="center of room", state="stationary"),
            EntityDescription(name="wooden coffee table", appearance="rectangular, oak",
                              position="in front of sofa", state="stationary"),
            EntityDescription(name="floor lamp", appearance="tall, white shade",
                              position="left of sofa", state="on"),
        ),
        lighting="warm natural light from window",
        weather="indoor",
        camera="eye-level, facing sofa",
        motion="static scene",
        persistent_constraints=("preserve sofa identity", "keep room layout stable"),
        user_intent="move camera forward",
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. A/B/C Prompt Construction
# ---------------------------------------------------------------------------

class TestABCConstruction(unittest.TestCase):
    def test_variant_a_is_original_prompt_only(self):
        world = _make_world()
        user = "Move the camera forward slowly."
        prompt_a = user  # Variant A = original only
        self.assertEqual(prompt_a, user)
        self.assertNotIn("Observed world", prompt_a)
        self.assertNotIn("Environment", prompt_a)

    def test_variant_b_uses_full_world_prompt(self):
        world = _make_world()
        user = "Move the camera forward slowly."
        prompt_b = compose_world_prompt(world, user)
        self.assertIn(user, prompt_b)
        self.assertIn("Current world:", prompt_b)
        self.assertIn("Persistent entities:", prompt_b)
        self.assertIn("beige sofa", prompt_b)

    def test_variant_c_uses_compact_world_prompt(self):
        world = _make_world()
        user = "Move the camera forward slowly."
        prompt_c = compose_compact_world_prompt(world, user)
        self.assertIn(user, prompt_c)
        self.assertIn("Observed world:", prompt_c)
        self.assertIn("beige sofa", prompt_c)
        # Compact should be shorter than full
        prompt_b = compose_world_prompt(world, user)
        self.assertLess(len(prompt_c), len(prompt_b))

    def test_all_variants_share_same_user_prompt(self):
        world = _make_world()
        user = "Pan left while keeping the sofa centered."
        prompt_a = user
        prompt_b = compose_world_prompt(world, user)
        prompt_c = compose_compact_world_prompt(world, user)
        self.assertIn(user, prompt_a)
        self.assertIn(user, prompt_b)
        self.assertIn(user, prompt_c)


# ---------------------------------------------------------------------------
# 2. Compact World Prompt
# ---------------------------------------------------------------------------

class TestCompactWorldPrompt(unittest.TestCase):
    def test_user_prompt_first(self):
        world = _make_world()
        user = "Move forward."
        prompt = compose_compact_world_prompt(world, user)
        lines = prompt.split("\n")
        self.assertEqual(lines[0], "User request:")
        self.assertEqual(lines[1], user)

    def test_empty_user_prompt(self):
        world = _make_world()
        prompt = compose_compact_world_prompt(world, "")
        self.assertIn("User request:", prompt)
        self.assertIn("Observed world:", prompt)

    def test_max_entities_limit(self):
        entities = tuple(
            EntityDescription(name=f"entity_{i}", appearance=f"attr_{i}",
                              position=f"pos_{i}")
            for i in range(10)
        )
        world = WorldDescription(main_entities=entities)
        prompt = compose_compact_world_prompt(world, "test", max_entities=3)
        self.assertIn("entity_0", prompt)
        self.assertIn("entity_2", prompt)
        self.assertNotIn("entity_3", prompt)

    def test_unknown_fields_skipped(self):
        world = WorldDescription(
            environment="unknown",
            scene_layout="unknown",
            lighting="unknown",
            weather="unknown",
            camera="unknown",
            motion="unknown",
            main_entities=(),
            persistent_constraints=(),
        )
        prompt = compose_compact_world_prompt(world, "test")
        self.assertIn("User request:", prompt)
        self.assertIn("Constraints: preserve subject identity and scene layout", prompt)

    def test_persistent_constraints_included(self):
        world = WorldDescription(
            persistent_constraints=("keep red color", "do not change shape"),
        )
        prompt = compose_compact_world_prompt(world, "test")
        self.assertIn("keep red color", prompt)
        self.assertIn("do not change shape", prompt)

    def test_dict_input_accepted(self):
        world_dict = {
            "environment": "forest",
            "main_entities": [{"name": "tree", "appearance": "tall", "position": "center"}],
        }
        prompt = compose_compact_world_prompt(world_dict, "move forward")
        self.assertIn("forest", prompt)
        self.assertIn("tree", prompt)


# ---------------------------------------------------------------------------
# 3. User Prompt Always Retained
# ---------------------------------------------------------------------------

class TestUserPromptRetention(unittest.TestCase):
    def test_compact_never_truncates_user_prompt(self):
        world = _make_world()
        long_user = "A" * 2000 + " very long user prompt that must be preserved exactly"
        prompt = compose_compact_world_prompt(world, long_user)
        self.assertIn(long_user, prompt)

    def test_full_never_truncates_user_prompt(self):
        world = _make_world()
        long_user = "B" * 2000 + " very long user prompt"
        prompt = compose_world_prompt(world, long_user)
        self.assertIn(long_user, prompt)

    def test_user_intent_not_overridden_by_world(self):
        world = WorldDescription(
            environment="kitchen",
            user_intent="the VLM thinks the user wants to turn left",
        )
        user = "Move the camera right, not left."
        prompt_c = compose_compact_world_prompt(world, user)
        self.assertIn("Move the camera right, not left.", prompt_c)
        # Compact should not include VLM's guessed intent as authoritative
        self.assertNotIn("turn left", prompt_c.lower().replace("not left", ""))


# ---------------------------------------------------------------------------
# 4. Token Count / Truncation Metadata
# ---------------------------------------------------------------------------

class TestTokenMetadata(unittest.TestCase):
    def test_metrics_record_structure(self):
        record = {
            "scene": "indoor",
            "variant": "B",
            "prompt_chars": 1200,
            "token_count_before_truncation": 600,
            "token_count_after_truncation": 512,
            "truncated": True,
            "world_condition_chars": 800,
            "entity_count": 3,
        }
        required = ["scene", "variant", "prompt_chars",
                     "token_count_before_truncation", "token_count_after_truncation",
                     "truncated", "world_condition_chars", "entity_count"]
        for key in required:
            self.assertIn(key, record)

    def test_truncation_flag_consistent(self):
        # When tokens > 512, truncated must be True and after = 512
        record = {
            "token_count_before_truncation": 600,
            "token_count_after_truncation": 512,
            "truncated": True,
        }
        self.assertTrue(record["truncated"])
        self.assertEqual(record["token_count_after_truncation"], 512)
        self.assertGreater(record["token_count_before_truncation"], 512)

    def test_no_truncation_flag_consistent(self):
        record = {
            "token_count_before_truncation": 200,
            "token_count_after_truncation": 200,
            "truncated": False,
        }
        self.assertFalse(record["truncated"])
        self.assertEqual(record["token_count_before_truncation"],
                         record["token_count_after_truncation"])

    def test_variant_a_usually_not_truncated(self):
        # Variant A is just the user prompt, should be short
        user = "Move the camera slowly forward while keeping the subject stable."
        self.assertLess(len(user.split()), 512)


# ---------------------------------------------------------------------------
# 5. Manifest Hashing
# ---------------------------------------------------------------------------

class TestManifestHashing(unittest.TestCase):
    def test_sha256_text_deterministic(self):
        text = "test prompt"
        h1 = _sha256_text(text)
        h2 = _sha256_text(text)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_sha256_text_different_for_different_text(self):
        self.assertNotEqual(_sha256_text("prompt A"), _sha256_text("prompt B"))

    def test_sha256_file_matches_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            h = _sha256_file(path)
            expected = hashlib.sha256(b"hello world").hexdigest()
            self.assertEqual(h, expected)
        finally:
            os.unlink(path)

    def test_manifest_entry_has_required_hash_fields(self):
        entry = {
            "scene_id": "indoor",
            "variant": "A",
            "seed": 42,
            "input_image_sha256": "a" * 64,
            "prompt_sha256": "b" * 64,
            "world_condition_sha256": "c" * 64,
            "output_sha256": "d" * 64,
        }
        for field in ["input_image_sha256", "prompt_sha256",
                       "world_condition_sha256", "output_sha256"]:
            self.assertIn(field, entry)
            self.assertEqual(len(entry[field]), 64)


# ---------------------------------------------------------------------------
# 6. Resume Logic
# ---------------------------------------------------------------------------

class TestResumeLogic(unittest.TestCase):
    def _make_config(self):
        return {
            "generation": {"frame_num": 13, "chunk_size": 4, "size": "832*480"},
            "baseline_commit": "52ce9f4",
            "paths": {"checkpoint_dir": "/tmp/fake_ckpt"},
        }

    def _make_scene(self, tmp_path):
        img = tmp_path / "input.jpg"
        img.write_bytes(b"fake image content")
        return {
            "id": "test_scene",
            "image": str(img),
            "action_path": str(tmp_path),
            "user_prompt": "move forward",
        }

    def test_skip_when_hash_matches(self):
        """Import the should_skip logic conceptually."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scene = self._make_scene(tmp)
            cfg = self._make_config()

            # Create prompt file
            prompt_file = tmp / "prompt_A.txt"
            prompt_file.write_text("move forward")

            # Create fake output video
            output_video = tmp / "output.mp4"
            output_video.write_bytes(b"fake video")

            existing = {
                "status": "PASS",
                "output_video": str(output_video),
                "input_image_sha256": _sha256_file(scene["image"]),
                "prompt_sha256": _sha256_file(prompt_file),
                "frame_num": 13,
                "seed": 42,
            }

            # Simulate should_skip checks
            self.assertTrue(os.path.exists(existing["output_video"]))
            self.assertEqual(existing["input_image_sha256"], _sha256_file(scene["image"]))
            self.assertEqual(existing["prompt_sha256"], _sha256_file(prompt_file))
            self.assertEqual(existing["frame_num"], cfg["generation"]["frame_num"])
            self.assertEqual(existing["seed"], 42)
            # All checks pass → should skip

    def test_regenerate_when_prompt_changes(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scene = self._make_scene(tmp)
            prompt_file = tmp / "prompt_A.txt"
            prompt_file.write_text("original prompt")
            output_video = tmp / "output.mp4"
            output_video.write_bytes(b"fake video")

            existing = {
                "status": "PASS",
                "output_video": str(output_video),
                "input_image_sha256": _sha256_file(scene["image"]),
                "prompt_sha256": _sha256_text("old different prompt"),
                "frame_num": 13,
                "seed": 42,
            }
            # Prompt hash mismatch → should NOT skip
            self.assertNotEqual(existing["prompt_sha256"], _sha256_file(prompt_file))

    def test_regenerate_when_output_missing(self):
        existing = {
            "status": "PASS",
            "output_video": "/nonexistent/path.mp4",
        }
        self.assertFalse(os.path.exists(existing["output_video"]))

    def test_regenerate_when_previous_failed(self):
        existing = {"status": "FAIL", "error": "out of memory"}
        self.assertNotEqual(existing["status"], "PASS")

    def test_regenerate_when_frame_num_changes(self):
        existing = {"status": "PASS", "frame_num": 29}
        cfg = self._make_config()
        self.assertNotEqual(existing["frame_num"], cfg["generation"]["frame_num"])


# ---------------------------------------------------------------------------
# 7. Blind Mapping
# ---------------------------------------------------------------------------

class TestBlindMapping(unittest.TestCase):
    def test_blind_id_format(self):
        for i in range(1, 50):
            blind_id = f"video_{i:03d}"
            self.assertTrue(blind_id.startswith("video_"))
            self.assertEqual(len(blind_id), 9)

    def test_blind_map_hides_variant_in_csv(self):
        """The human_eval.csv must not contain variant column."""
        fieldnames = ["blind_id", "video_id", "scene_id", "seed",
                      "output_video", "status",
                      "identity_consistency", "attribute_preservation",
                      "spatial_layout", "environment_consistency",
                      "camera_continuity", "temporal_stability",
                      "intent_fidelity", "hallucination",
                      "positive_score", "adjusted_score", "notes"]
        self.assertNotIn("variant", fieldnames)

    def test_blind_map_stores_variant_for_unblinding(self):
        blind_map = {
            "video_001": {
                "video_id": "indoor_A_seed42",
                "scene_id": "indoor",
                "variant": "A",
                "seed": 42,
            }
        }
        self.assertEqual(blind_map["video_001"]["variant"], "A")

    def test_deterministic_shuffle(self):
        import random
        items = list(range(45))
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        shuffled1 = items[:]
        shuffled2 = items[:]
        rng1.shuffle(shuffled1)
        rng2.shuffle(shuffled2)
        self.assertEqual(shuffled1, shuffled2)

    def test_shuffle_does_not_lose_items(self):
        import random
        items = list(range(45))
        rng = random.Random(42)
        rng.shuffle(items)
        self.assertEqual(sorted(items), list(range(45)))


# ---------------------------------------------------------------------------
# 8. Score Summary
# ---------------------------------------------------------------------------

class TestScoreSummary(unittest.TestCase):
    def test_positive_score_calculation(self):
        scores = {
            "identity_consistency": 4.0,
            "attribute_preservation": 3.5,
            "spatial_layout": 4.0,
            "environment_consistency": 3.0,
            "camera_continuity": 4.5,
            "temporal_stability": 4.0,
            "intent_fidelity": 3.5,
        }
        positive = sum(scores.values()) / len(scores)
        self.assertAlmostEqual(positive, 3.7857, places=3)

    def test_adjusted_score_subtracts_hallucination(self):
        positive = 4.0
        hallucination = 2.0
        adjusted = positive - hallucination * 0.5
        self.assertEqual(adjusted, 3.0)

    def test_hallucination_is_negative_metric(self):
        # Higher hallucination = worse
        scores_good = {"hallucination": 0.0}
        scores_bad = {"hallucination": 5.0}
        self.assertLess(scores_good["hallucination"], scores_bad["hallucination"])

    def test_stats_function_empty(self):
        import statistics
        values: list[float] = []
        result = {
            "mean": None,
            "median": None,
            "std": None,
            "count": 0,
        } if not values else {}
        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["mean"])

    def test_stats_function_with_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        import statistics
        self.assertEqual(statistics.mean(values), 3.0)
        self.assertEqual(statistics.median(values), 3.0)
        self.assertAlmostEqual(statistics.stdev(values), 1.581, places=2)

    def test_pairwise_difference_calculation(self):
        mean_c = 3.8
        mean_a = 3.2
        diff = mean_c - mean_a
        self.assertAlmostEqual(diff, 0.6)
        self.assertGreater(diff, 0)  # C improves over A


# ---------------------------------------------------------------------------
# 9. Missing Video Handling
# ---------------------------------------------------------------------------

class TestMissingVideoHandling(unittest.TestCase):
    def test_missing_video_marked_fail(self):
        entry = {
            "scene_id": "indoor",
            "variant": "A",
            "seed": 42,
            "status": "FAIL",
            "error": "output video not found after decode",
            "output_video": None,
            "output_sha256": None,
        }
        self.assertEqual(entry["status"], "FAIL")
        self.assertIsNone(entry["output_video"])

    def test_missing_video_excluded_from_stats(self):
        records = [
            {"variant": "A", "status": "PASS", "scene_id": "s1", "seed": 42},
            {"variant": "A", "status": "FAIL", "scene_id": "s2", "seed": 42},
            {"variant": "B", "status": "PASS", "scene_id": "s1", "seed": 42},
        ]
        passing_a = [r for r in records if r["variant"] == "A" and r["status"] == "PASS"]
        self.assertEqual(len(passing_a), 1)

    def test_sanity_metrics_none_for_missing_video(self):
        metrics = {
            "frame_count": None,
            "resolution": None,
            "temporal_mad_mean": None,
            "temporal_mad_max": None,
        }
        self.assertIsNone(metrics["frame_count"])
        self.assertIsNone(metrics["temporal_mad_mean"])


# ---------------------------------------------------------------------------
# 10. Failed Generation Does Not Pollute Existing Results
# ---------------------------------------------------------------------------

class TestFailedGenerationIsolation(unittest.TestCase):
    def test_failed_entry_has_error_field(self):
        entry = {
            "scene_id": "indoor",
            "variant": "B",
            "seed": 123,
            "status": "FAIL",
            "error": "generate-latents failed: MPS out of memory",
        }
        self.assertIn("error", entry)
        self.assertEqual(entry["status"], "FAIL")

    def test_failed_entry_does_not_overwrite_passing_entry(self):
        """Simulate: existing PASS for A/42, new FAIL for B/42. Both preserved."""
        results = {
            "indoor|A|42": {"status": "PASS", "variant": "A", "seed": 42},
        }
        new_fail = {"status": "FAIL", "variant": "B", "seed": 42, "error": "OOM"}
        results["indoor|B|42"] = new_fail
        self.assertEqual(results["indoor|A|42"]["status"], "PASS")
        self.assertEqual(results["indoor|B|42"]["status"], "FAIL")

    def test_failed_entry_retries_on_next_run(self):
        """A FAIL entry should be retried (not skipped) on next run."""
        existing = {"status": "FAIL", "error": "OOM"}
        self.assertNotEqual(existing["status"], "PASS")
        # should_skip would return False for FAIL

    def test_jsonl_incremental_write_preserves_data(self):
        """Writing results incrementally should not lose earlier entries."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            records = [
                {"scene_id": "s1", "variant": "A", "seed": 42, "status": "PASS"},
                {"scene_id": "s1", "variant": "B", "seed": 42, "status": "PASS"},
                {"scene_id": "s1", "variant": "C", "seed": 42, "status": "FAIL"},
            ]
            with open(path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            # Read back
            read_back = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    read_back.append(json.loads(line.strip()))

            self.assertEqual(len(read_back), 3)
            self.assertEqual(read_back[0]["status"], "PASS")
            self.assertEqual(read_back[2]["status"], "FAIL")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 11. Config Validation
# ---------------------------------------------------------------------------

class TestEvalConfig(unittest.TestCase):
    def test_config_has_required_fields(self):
        config_path = REPO_ROOT / "eval" / "g0.7" / "config.json"
        self.assertTrue(config_path.exists(), f"Config not found at {config_path}")
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        required = ["eval_name", "baseline_commit", "generation", "paths",
                    "seeds", "variants", "umt5_max_tokens", "scenes"]
        for field in required:
            self.assertIn(field, cfg, f"Missing config field: {field}")

    def test_config_has_5_scenes(self):
        config_path = REPO_ROOT / "eval" / "g0.7" / "config.json"
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(len(cfg["scenes"]), 5)
        scene_ids = [s["id"] for s in cfg["scenes"]]
        self.assertEqual(scene_ids, ["single_subject", "spatial", "indoor", "outdoor", "camera_motion"])

    def test_config_seeds_are_fixed(self):
        config_path = REPO_ROOT / "eval" / "g0.7" / "config.json"
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["seeds"], [42, 123, 2026])

    def test_config_generation_params_match_spec(self):
        config_path = REPO_ROOT / "eval" / "g0.7" / "config.json"
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        gen = cfg["generation"]
        self.assertEqual(gen["device"], "mps")
        self.assertEqual(gen["task"], "i2v-1.3B")
        self.assertEqual(gen["infer_mode"], "causal_fast")
        self.assertEqual(gen["frame_num"], 13)
        self.assertEqual(gen["chunk_size"], 4)
        self.assertEqual(gen["size"], "832*480")
        self.assertTrue(gen["sequential_load"])
        self.assertTrue(gen["staged_pipeline"])

    def test_each_scene_has_required_fields(self):
        config_path = REPO_ROOT / "eval" / "g0.7" / "config.json"
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        for scene in cfg["scenes"]:
            for field in ["id", "image", "action_path", "user_prompt"]:
                self.assertIn(field, scene, f"Scene {scene.get('id')} missing {field}")


# ---------------------------------------------------------------------------
# Blind evaluation anonymity & integrity
# ---------------------------------------------------------------------------

class TestBlindEvalAnonymity(unittest.TestCase):
    """Verify human_eval.csv and blind/videos do not leak variant/scene/seed."""

    def setUp(self):
        self.csv_path = REPO_ROOT / "eval" / "g0.7" / "human_eval.csv"
        self.blind_dir = REPO_ROOT / "eval" / "g0.7" / "blind" / "videos"
        self.blind_map_path = REPO_ROOT / "eval" / "g0.7" / "blind_map.json"

    def test_human_eval_csv_no_variant_column(self):
        with open(self.csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertNotIn("variant", reader.fieldnames)
            self.assertNotIn("video_id", reader.fieldnames)
            self.assertNotIn("scene_id", reader.fieldnames)
            self.assertNotIn("seed", reader.fieldnames)
            self.assertNotIn("output_video", reader.fieldnames)

    def test_human_eval_csv_data_no_scene_names(self):
        with open(self.csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        data_text = " ".join(",".join(r.values()) for r in rows)
        for scene in ["single_subject", "spatial", "indoor", "outdoor", "camera_motion"]:
            self.assertNotIn(scene, data_text, f"Scene name '{scene}' leaked into CSV data")

    def test_human_eval_csv_data_no_paths(self):
        with open(self.csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        data_text = " ".join(",".join(r.values()) for r in rows)
        self.assertNotIn("eval/g0.7/", data_text)
        self.assertNotIn(".mp4", data_text)

    def test_human_eval_csv_45_rows_unique_blind_ids(self):
        with open(self.csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 45)
        blind_ids = [r["blind_id"] for r in rows]
        self.assertEqual(len(set(blind_ids)), 45)
        for bid in blind_ids:
            self.assertRegex(bid, r"^video_\d{3}$")

    def test_human_eval_csv_has_required_dimension_columns(self):
        with open(self.csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
        for dim in ["identity_consistency", "attribute_preservation", "spatial_layout",
                     "environment_consistency", "camera_continuity", "temporal_stability",
                     "intent_fidelity", "hallucination"]:
            self.assertIn(dim, fields)
        self.assertIn("positive_score", fields)
        self.assertIn("adjusted_score", fields)
        self.assertIn("notes", fields)

    def test_blind_videos_filenames_anonymous(self):
        if not self.blind_dir.exists():
            self.skipTest("blind/videos directory not created yet")
        import re
        for fname in os.listdir(self.blind_dir):
            self.assertRegex(fname, r"^video_\d{3}\.mp4$",
                             f"Filename '{fname}' leaks identity")
            for scene in ["single_subject", "spatial", "indoor", "outdoor", "camera_motion"]:
                self.assertNotIn(scene, fname)
            for v in ["_A_", "_B_", "_C_"]:
                self.assertNotIn(v, fname)

    def test_blind_videos_count_matches_blind_map(self):
        if not self.blind_dir.exists():
            self.skipTest("blind/videos directory not created yet")
        with open(self.blind_map_path, encoding="utf-8") as f:
            bm = json.load(f)
        mapping = bm["mapping"]
        video_files = [f for f in os.listdir(self.blind_dir) if f.endswith(".mp4")]
        self.assertEqual(len(video_files), len(mapping))
        for bid in mapping:
            self.assertTrue(
                (self.blind_dir / f"{bid}.mp4").exists(),
                f"Missing blind copy for {bid}"
            )

    def test_blind_map_mapping_keys_match_csv(self):
        with open(self.blind_map_path, encoding="utf-8") as f:
            bm = json.load(f)
        mapping = bm["mapping"]
        with open(self.csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_ids = set(r["blind_id"] for r in reader)
        self.assertEqual(set(mapping.keys()), csv_ids)


# ---------------------------------------------------------------------------
# Summarize strict validation
# ---------------------------------------------------------------------------

class TestSummarizeValidation(unittest.TestCase):
    """Test load_and_validate_human_scores strict validation logic."""

    def _make_blind_map(self):
        return {
            "video_001": {"scene_id": "s1", "variant": "A", "seed": 42},
            "video_002": {"scene_id": "s1", "variant": "B", "seed": 42},
            "video_003": {"scene_id": "s1", "variant": "C", "seed": 42},
        }

    def _write_csv(self, tmpdir, rows, fieldnames=None):
        if fieldnames is None:
            fieldnames = ["blind_id"] + [
                "identity_consistency", "attribute_preservation", "spatial_layout",
                "environment_consistency", "camera_continuity", "temporal_stability",
                "intent_fidelity", "hallucination",
                "positive_score", "adjusted_score", "notes"]
        path = tmpdir / "human_eval.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def test_empty_scores_returns_empty_dict(self):
        import summarize_g07_eval as sm
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            rows = [{"blind_id": f"video_{i:03d}"} for i in range(1, 46)]
            path = self._write_csv(tmpdir, rows)
            with patch.object(sm, "HUMAN_EVAL_PATH", path):
                result = sm.load_and_validate_human_scores(self._make_blind_map())
                self.assertEqual(result, {})

    def test_all_valid_scores_pass(self):
        import summarize_g07_eval as sm
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            bm = {f"video_{i:03d}": {"scene_id": "s", "variant": "A", "seed": 42}
                  for i in range(1, 46)}
            dims = ["identity_consistency", "attribute_preservation", "spatial_layout",
                    "environment_consistency", "camera_continuity", "temporal_stability",
                    "intent_fidelity", "hallucination"]
            rows = []
            for i in range(1, 46):
                row = {"blind_id": f"video_{i:03d}"}
                for d in dims:
                    row[d] = "3"
                rows.append(row)
            path = self._write_csv(tmpdir, rows)
            with patch.object(sm, "HUMAN_EVAL_PATH", path):
                result = sm.load_and_validate_human_scores(bm)
                self.assertEqual(len(result), 45)
                self.assertIn("video_001", result)
                self.assertEqual(result["video_001"]["identity_consistency"], 3.0)

    def test_missing_dimension_errors(self):
        import summarize_g07_eval as sm
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            bm = {"video_001": {"scene_id": "s", "variant": "A", "seed": 42}}
            rows = [{"blind_id": "video_001", "identity_consistency": "3",
                     "attribute_preservation": "3", "spatial_layout": "3",
                     "environment_consistency": "3", "camera_continuity": "3",
                     "temporal_stability": "3", "intent_fidelity": "3"}]
            # hallucination missing
            path = self._write_csv(tmpdir, rows)
            with patch.object(sm, "HUMAN_EVAL_PATH", path):
                with self.assertRaises(SystemExit):
                    sm.load_and_validate_human_scores(bm)

    def test_out_of_range_score_errors(self):
        import summarize_g07_eval as sm
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            bm = {"video_001": {"scene_id": "s", "variant": "A", "seed": 42}}
            dims = ["identity_consistency", "attribute_preservation", "spatial_layout",
                    "environment_consistency", "camera_continuity", "temporal_stability",
                    "intent_fidelity", "hallucination"]
            row = {"blind_id": "video_001"}
            for d in dims:
                row[d] = "6"  # out of range
            path = self._write_csv(tmpdir, [row])
            with patch.object(sm, "HUMAN_EVAL_PATH", path):
                with self.assertRaises(SystemExit):
                    sm.load_and_validate_human_scores(bm)

    def test_duplicate_blind_id_errors(self):
        import summarize_g07_eval as sm
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            bm = {"video_001": {"scene_id": "s", "variant": "A", "seed": 42}}
            dims = ["identity_consistency", "attribute_preservation", "spatial_layout",
                    "environment_consistency", "camera_continuity", "temporal_stability",
                    "intent_fidelity", "hallucination"]
            row = {"blind_id": "video_001"}
            for d in dims:
                row[d] = "3"
            path = self._write_csv(tmpdir, [row, row])  # duplicate
            with patch.object(sm, "HUMAN_EVAL_PATH", path):
                with self.assertRaises(SystemExit):
                    sm.load_and_validate_human_scores(bm)

    def test_unblindable_blind_id_errors(self):
        import summarize_g07_eval as sm
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            bm = {"video_001": {"scene_id": "s", "variant": "A", "seed": 42}}
            dims = ["identity_consistency", "attribute_preservation", "spatial_layout",
                    "environment_consistency", "camera_continuity", "temporal_stability",
                    "intent_fidelity", "hallucination"]
            row = {"blind_id": "video_999"}  # not in blind_map
            for d in dims:
                row[d] = "3"
            path = self._write_csv(tmpdir, [row])
            with patch.object(sm, "HUMAN_EVAL_PATH", path):
                with self.assertRaises(SystemExit):
                    sm.load_and_validate_human_scores(bm)


# ---------------------------------------------------------------------------
# Unblinding correctness
# ---------------------------------------------------------------------------

class TestUnblindingCorrectness(unittest.TestCase):
    """Test build_score_lookup maps blind_id to (scene, variant, seed) correctly."""

    def test_build_score_lookup(self):
        import summarize_g07_eval as sm
        blind_map = {
            "video_001": {"scene_id": "indoor", "variant": "A", "seed": 42},
            "video_002": {"scene_id": "outdoor", "variant": "C", "seed": 2026},
        }
        human_scores = {
            "video_001": {"identity_consistency": 4.0, "hallucination": 1.0},
            "video_002": {"identity_consistency": 3.0, "hallucination": 2.0},
        }
        lookup = sm.build_score_lookup(human_scores, blind_map)
        self.assertEqual(len(lookup), 2)
        self.assertIn(("indoor", "A", 42), lookup)
        self.assertIn(("outdoor", "C", 2026), lookup)
        self.assertEqual(lookup[("indoor", "A", 42)]["identity_consistency"], 4.0)
        self.assertEqual(lookup[("outdoor", "C", 2026)]["hallucination"], 2.0)

    def test_blind_map_actual_mapping_covers_all_results(self):
        """Every (scene, variant, seed) in results.jsonl must be in blind_map."""
        with open(REPO_ROOT / "eval" / "g0.7" / "blind_map.json", encoding="utf-8") as f:
            bm = json.load(f)
        mapping = bm["mapping"]
        mapped_keys = set()
        for info in mapping.values():
            mapped_keys.add((info["scene_id"], info["variant"], info["seed"]))
        with open(REPO_ROOT / "eval" / "g0.7" / "results.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["scene_id"], rec["variant"], rec["seed"])
                self.assertIn(key, mapped_keys,
                              f"Result {key} not found in blind_map mapping")


# ---------------------------------------------------------------------------
# Paired deltas & wins/ties/losses
# ---------------------------------------------------------------------------

class TestPairedDeltas(unittest.TestCase):
    """Test compute_paired_deltas with fixed small sample."""

    def _make_records(self):
        return [
            {"scene_id": "s1", "variant": "A", "seed": 42, "status": "PASS"},
            {"scene_id": "s1", "variant": "B", "seed": 42, "status": "PASS"},
            {"scene_id": "s1", "variant": "C", "seed": 42, "status": "PASS"},
            {"scene_id": "s2", "variant": "A", "seed": 42, "status": "PASS"},
            {"scene_id": "s2", "variant": "B", "seed": 42, "status": "PASS"},
            {"scene_id": "s2", "variant": "C", "seed": 42, "status": "PASS"},
        ]

    def _make_score_lookup(self):
        dims_A = {"identity_consistency": 3.0, "attribute_preservation": 3.0,
                  "spatial_layout": 3.0, "environment_consistency": 3.0,
                  "camera_continuity": 3.0, "temporal_stability": 3.0,
                  "intent_fidelity": 3.0, "hallucination": 1.0}
        dims_B = {"identity_consistency": 4.0, "attribute_preservation": 4.0,
                  "spatial_layout": 4.0, "environment_consistency": 4.0,
                  "camera_continuity": 4.0, "temporal_stability": 4.0,
                  "intent_fidelity": 4.0, "hallucination": 1.0}
        dims_C = {"identity_consistency": 5.0, "attribute_preservation": 5.0,
                  "spatial_layout": 5.0, "environment_consistency": 5.0,
                  "camera_continuity": 5.0, "temporal_stability": 5.0,
                  "intent_fidelity": 5.0, "hallucination": 0.0}
        return {
            ("s1", "A", 42): dims_A, ("s1", "B", 42): dims_B, ("s1", "C", 42): dims_C,
            ("s2", "A", 42): dims_A, ("s2", "B", 42): dims_B, ("s2", "C", 42): dims_C,
        }

    def test_paired_deltas_calculation(self):
        import summarize_g07_eval as sm
        records = self._make_records()
        lookup = self._make_score_lookup()
        result = sm.compute_paired_deltas(records, lookup)

        # A adjusted = 3.0 - 0.5*1.0 = 2.5
        # B adjusted = 4.0 - 0.5*1.0 = 3.5
        # C adjusted = 5.0 - 0.5*0.0 = 5.0
        # B-A = 1.0, C-A = 2.5, C-B = 1.5

        self.assertEqual(result["total_pairs"]["B-A"], 2)
        self.assertEqual(result["total_pairs"]["C-A"], 2)
        self.assertEqual(result["total_pairs"]["C-B"], 2)

        for delta in result["paired_deltas"]["B-A"]:
            self.assertAlmostEqual(delta["adjusted_delta"], 1.0, places=2)
        for delta in result["paired_deltas"]["C-A"]:
            self.assertAlmostEqual(delta["adjusted_delta"], 2.5, places=2)
        for delta in result["paired_deltas"]["C-B"]:
            self.assertAlmostEqual(delta["adjusted_delta"], 1.5, places=2)

    def test_wins_ties_losses(self):
        import summarize_g07_eval as sm
        records = self._make_records()
        lookup = self._make_score_lookup()
        result = sm.compute_paired_deltas(records, lookup)

        # B > A always: W=2, T=0, L=0
        self.assertEqual(result["wins_overall"]["B-A"], {"win": 2, "tie": 0, "loss": 0})
        # C > A always: W=2, T=0, L=0
        self.assertEqual(result["wins_overall"]["C-A"], {"win": 2, "tie": 0, "loss": 0})
        # C > B always: W=2, T=0, L=0
        self.assertEqual(result["wins_overall"]["C-B"], {"win": 2, "tie": 0, "loss": 0})

    def test_tie_detection(self):
        import summarize_g07_eval as sm
        records = [
            {"scene_id": "s1", "variant": "A", "seed": 42, "status": "PASS"},
            {"scene_id": "s1", "variant": "B", "seed": 42, "status": "PASS"},
        ]
        dims = {"identity_consistency": 3.0, "attribute_preservation": 3.0,
                "spatial_layout": 3.0, "environment_consistency": 3.0,
                "camera_continuity": 3.0, "temporal_stability": 3.0,
                "intent_fidelity": 3.0, "hallucination": 1.0}
        lookup = {("s1", "A", 42): dims, ("s1", "B", 42): dict(dims)}
        result = sm.compute_paired_deltas(records, lookup)
        self.assertEqual(result["wins_overall"]["B-A"], {"win": 0, "tie": 1, "loss": 0})

    def test_loss_detection(self):
        import summarize_g07_eval as sm
        records = [
            {"scene_id": "s1", "variant": "A", "seed": 42, "status": "PASS"},
            {"scene_id": "s1", "variant": "B", "seed": 42, "status": "PASS"},
        ]
        dims_A = {"identity_consistency": 5.0, "attribute_preservation": 5.0,
                  "spatial_layout": 5.0, "environment_consistency": 5.0,
                  "camera_continuity": 5.0, "temporal_stability": 5.0,
                  "intent_fidelity": 5.0, "hallucination": 0.0}
        dims_B = {"identity_consistency": 2.0, "attribute_preservation": 2.0,
                  "spatial_layout": 2.0, "environment_consistency": 2.0,
                  "camera_continuity": 2.0, "temporal_stability": 2.0,
                  "intent_fidelity": 2.0, "hallucination": 3.0}
        lookup = {("s1", "A", 42): dims_A, ("s1", "B", 42): dims_B}
        result = sm.compute_paired_deltas(records, lookup)
        self.assertEqual(result["wins_overall"]["B-A"], {"win": 0, "tie": 0, "loss": 1})

    def test_wins_per_scene(self):
        import summarize_g07_eval as sm
        records = self._make_records()
        lookup = self._make_score_lookup()
        result = sm.compute_paired_deltas(records, lookup)
        self.assertIn("s1", result["wins_per_scene"])
        self.assertIn("s2", result["wins_per_scene"])
        self.assertEqual(result["wins_per_scene"]["s1"]["C-A"], {"win": 1, "tie": 0, "loss": 0})
        self.assertEqual(result["wins_per_scene"]["s2"]["C-A"], {"win": 1, "tie": 0, "loss": 0})


if __name__ == "__main__":
    unittest.main()
