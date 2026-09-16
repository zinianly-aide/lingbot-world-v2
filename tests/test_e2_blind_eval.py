"""Tests for E2 blind eval materials: anonymity, pairing, summarize validation."""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

E2 = REPO / "eval" / "e2"
BLIND_DIR = E2 / "blind"


class TestBlindEvalAnonymity(unittest.TestCase):
    """Scoring CSV must not leak encoder/scene/seed identity."""

    def setUp(self):
        self.csv_path = BLIND_DIR / "human_eval_paired.csv"
        self.text = self.csv_path.read_text()

    def test_no_encoder_leak(self):
        for term in ["umt5", "minicpm", "adapter"]:
            self.assertNotIn(term, self.text.lower(), f"encoder name leaked: {term}")

    def test_no_scene_name_leak(self):
        # Dimension names spatial_consistency/camera_motion_fidelity are allowed
        # Check actual scene names as standalone tokens
        import re
        for scene in ["single_subject", "indoor", "outdoor"]:
            self.assertNotIn(scene, self.text.lower())
        # "spatial" as standalone (not in spatial_consistency)
        lines = self.text.split("\n")
        for line in lines[1:]:  # skip header
            for token in re.split(r'[^a-z_]+', line.lower()):
                self.assertNotEqual(token, "spatial", "scene 'spatial' leaked in data row")
                self.assertNotEqual(token, "camera_motion", "scene 'camera_motion' leaked in data row")

    def test_no_seed_leak(self):
        lines = self.text.split("\n")
        for line in lines[1:]:
            for token in line.split(","):
                token = token.strip()
                if token.isdigit():
                    self.assertNotIn(token, ["42", "123", "2026"], f"seed leaked: {token}")

    def test_no_real_filename_leak(self):
        for term in ["_minicpm", "A_seed", "single_subject_seed", "spatial_seed"]:
            self.assertNotIn(term, self.text)

    def test_15_rows_unique_pair_ids(self):
        with open(self.csv_path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 15)
        pids = [r["pair_id"] for r in rows]
        self.assertEqual(len(set(pids)), 15)

    def test_columns_correct(self):
        expected = [
            "pair_id", "video_a", "video_b",
            "intent_fidelity", "identity_consistency", "spatial_consistency",
            "temporal_stability", "camera_motion_fidelity", "hallucination",
            "overall_preference", "notes"
        ]
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, expected)


class TestBlindMapIntegrity(unittest.TestCase):
    def setUp(self):
        self.map_path = BLIND_DIR / "blind_map.json"
        self.map = json.load(open(self.map_path))

    def test_30_entries(self):
        self.assertEqual(len(self.map), 30)

    def test_unique_blind_ids(self):
        ids = [e["blind_id"] for e in self.map]
        self.assertEqual(len(set(ids)), 30)

    def test_all_blind_videos_exist(self):
        for e in self.map:
            p = BLIND_DIR / "videos" / f"{e['blind_id']}.mp4"
            self.assertTrue(p.exists(), f"missing {p}")
            self.assertGreater(p.stat().st_size, 0)

    def test_blind_ids_format(self):
        import re
        for e in self.map:
            self.assertRegex(e["blind_id"], r"^video_\d{3}$")

    def test_15_pairs_same_scene_seed(self):
        from collections import defaultdict
        pairs = defaultdict(list)
        for e in self.map:
            pairs[(e["scene_id"], e["seed"])].append(e)
        self.assertEqual(len(pairs), 15)
        for key, entries in pairs.items():
            self.assertEqual(len(entries), 2, f"pair {key} has {len(entries)} entries")
            encoders = {e["encoder"] for e in entries}
            self.assertEqual(encoders, {"umt5", "minicpm+adapter"})

    def test_csv_pairs_align_with_map(self):
        """Every CSV row's A/B resolve to same scene/seed and different encoders."""
        by_id = {e["blind_id"]: e for e in self.map}
        with open(BLIND_DIR / "human_eval_paired.csv") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            a_id = r["video_a"].split("/")[-1].replace(".mp4", "")
            b_id = r["video_b"].split("/")[-1].replace(".mp4", "")
            a, b = by_id[a_id], by_id[b_id]
            self.assertEqual(a["scene_id"], b["scene_id"])
            self.assertEqual(a["seed"], b["seed"])
            self.assertNotEqual(a["encoder"], b["encoder"])


class TestSummarizeValidation(unittest.TestCase):
    """Test summarize script strict validation logic."""

    def _make_csv(self, tmpdir, rows):
        path = Path(tmpdir) / "test.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "pair_id", "video_a", "video_b",
                "intent_fidelity", "identity_consistency", "spatial_consistency",
                "temporal_stability", "camera_motion_fidelity", "hallucination",
                "overall_preference", "notes"
            ])
            for row in rows:
                w.writerow(row)
        return path

    def test_missing_rows_rejected(self):
        """Fewer than 15 rows should be caught by validate()."""
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            csv_path = self._make_csv(td, [
                ["pair_001", "blind/videos/video_001.mp4", "blind/videos/video_002.mp4",
                 "3", "3", "3", "3", "3", "3", "0", ""]
            ])
            result = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "summarize_e2_paired.py"),
                 "--csv", str(csv_path)],
                capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected 15", result.stderr)

    def test_out_of_range_score_rejected(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            rows = [
                [f"pair_{i:03d}", "blind/videos/video_001.mp4", "blind/videos/video_002.mp4",
                 "9", "3", "3", "3", "3", "3", "0", ""]
                for i in range(1, 16)
            ]
            csv_path = self._make_csv(td, rows)
            result = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "summarize_e2_paired.py"),
                 "--csv", str(csv_path)],
                capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("out of range", result.stderr)

    def test_dup_pair_id_rejected(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            rows = [
                ["pair_001", "blind/videos/video_001.mp4", "blind/videos/video_002.mp4",
                 "3", "3", "3", "3", "3", "3", "0", ""]
                for _ in range(15)
            ]
            csv_path = self._make_csv(td, rows)
            result = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "summarize_e2_paired.py"),
                 "--csv", str(csv_path)],
                capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate", result.stderr)

    def test_dry_mode_unfilled(self):
        """Empty scores should produce dry mode, not conclusions."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "summarize_e2_paired.py")],
            capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("DRY MODE", result.stdout)


class TestObjectiveMetricsExist(unittest.TestCase):
    def test_file_exists(self):
        p = E2 / "metrics_objective.json"
        self.assertTrue(p.exists())
        data = json.load(open(p))
        self.assertIn("_note", data)
        self.assertEqual(len(data["pairs"]), 15)

    def test_note_says_auxiliary(self):
        data = json.load(open(E2 / "metrics_objective.json"))
        self.assertIn("auxiliary", data["_note"].lower())


if __name__ == "__main__":
    unittest.main()
