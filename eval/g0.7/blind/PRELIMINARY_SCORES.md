# G0.7 partial blind-score archive — NOT FINAL

This branch preserves three pre-existing `partial_scores_*.json` files as
**preliminary assessments**, together with the original visual-agent rubric
and two supporting G0.7 scripts.

Mechanical validation (2026-10-13):
- 45 rows across three JSON arrays (15 + 15 + 15).
- `video_001`–`video_045` appear exactly once each.
- All eight scores are integers within 0–5; no missing reasons.
- Eight repeated reason strings were observed.

Not verified: whether each score was produced by reviewing the corresponding
full video and image or whether repeated comments faithfully reflect frames.
These files are not the canonical `eval/g0.7/human_eval.csv`.
DO NOT mark human visual validation, G1 entry, or a production VLM
quality improvement as PASS from this archival commit. Blind map stays
separate and must be used only for a properly validated comparison.

Syntax checks passed for scripts/encode_prompts_batch.py (Python) and
scripts/build_g07_montage.sh (bash). No T5 model, LingBot generation,
or video scoring was run while preparing this archival commit.
