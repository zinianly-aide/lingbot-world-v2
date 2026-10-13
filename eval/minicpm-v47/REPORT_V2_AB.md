# MiniCPM-V 4.7 / 4.6 World Perception follow-up
Date: 2026-10-13. Original G0.7 input images, no composite frames. No LingBot video generation or repository baseline edits.

## Experimental settings
4.7: 1B GGUF Q4_K_M, 4x f16 mmproj, llama.cpp b11551 / Metal, existing JSON grammar, n=550, temp=0, seed=42, -c 4096, no warmup.
4.6: cached MLX 4-bit, mlx-vlm 0.7.0, same images and V2b observation prompt, n=550, temp=0, seed=42; schema supplied as prompt text, NOT as constrained decoding. Thinking was still emitted despite attempts to disable it.

Prompts: world_perception_prompt_v2.txt (first draft), world_perception_prompt_v2b.txt (refined).
Raw outputs: logs/g07_v2_*.stdout.log, logs/g07_v2b_*.stdout.log, g07_46_mlx_v2b_*.json.
Detailed results including untouched raw and separately derived fields: g07_v2b_ab_results.json.

## V2b five original scenes (4.7)
Scene | Wall seconds | Recognized entities | JSON | Empty user_intent | motion exactly unknown
--- | ---: | --- | --- | --- | ---
single_subject | 11.07 | tree, mountains, sky | PASS | PASS | PASS
spatial | 6.56 | stone circle, isolated stone, archway, people* | PASS | PASS | PASS
indoor | 8.04 | sofa, coffee table, bookshelf, lamp, window | PASS | PASS | PASS
outdoor | 7.67 | Great Wall, Watchtower | PASS | FAIL | FAIL
camera_motion | 6.60 | dragon rider, wings, castle, trees | PASS | PASS | PASS

*People need manual visual verification, as do all hallucination judgments.
Aggregate 4.7 V2b: JSON 5/5, required keys 5/5, nonempty environment 5/5, layout 5/5, entities 5/5, exact empty intent 4/5, exact unknown motion 4/5, mean wall 7.99 seconds. First V2: JSON 5/5, nonempty entities 4/5, exact empty intent 0/5, mean wall 8.78 seconds.

Semantic defects persist: all five camera fields assert static/fixed although a still cannot establish camera movement; outdoor user_intent is '(empty)' and Great Wall state is 'Active'; some persistent_constraints echo instructions rather than observable facts. Spatial-accuracy scoring and hallucination scoring versus image ground truth are NOT VERIFIED.

## 4.6 versus 4.7
With exactly V2b observation text, 4.6 MLX returned complete JSON for 1/5 scenes; other outputs spent tokens on thinking and/or truncated JSON. Mean 4.6 inference 8.36 seconds, mean generated throughput 76.8 tokens/s. 4.7 complete JSON 5/5 via grammar, but this is an end-to-end interface comparison, NOT a fair vision-accuracy ranking. 4.6 received schema as extra prompt text and lacks the same constrained decoder; tokenizer, engine and quantizer also differ.

## Mac M4 memory and timing
4.7 V2b process max RSS 2,197,422,080 bytes (2.20 GB decimal); largest time-l peak memory footprint 1,509,509,568 bytes (1.51 GB). Average visual encoding log 3414 ms.
4.6 instrumented single-image run: MLX allocator peak 3,128,241,552 bytes (3.13 GB), allocator active 2,160,688,726 bytes; process max RSS 3,175,596,032 bytes; peak memory footprint 4,961,307,744 bytes. Earlier standalone load first observed 4.96 sec, later cached processes ~1 sec.
These indicators are NOT a complete hardware GPU/unified-memory peak. True hardware cold start, 4.7 decode TPS, first-token times and long-lived session stability are NOT RUN.

## Provenance and recommended gate
VLM input must contain only the image and observation prompt. Keep G0.7 user request separately and pass it as authoritative request to the existing world-prompt composer. Per-scene JSON results include a separately labeled derived_world_semantic_gate that normalizes user_intent to the empty string, motion to unknown, removes static/fixed assertions and obvious prompt-echo constraints. This is a deterministic downstream adapter, NOT a claim that the raw model produced correct semantics.

Decisions:
- 4.7 Metal runtime: PASS.
- Five-scene grammar-constrained structure: PASS.
- Raw observation vs intent isolation: FAIL.
- Visual hallucination / true spatial precision: NOT VERIFIED.
- Fair backend-matched 4.6 / 4.7 benchmark: NOT RUN.
- Replace existing production 4.6: NOT APPROVED.
- Next: validate images manually, implement source provenance gate on isolated experiment branch, match decoder conditions, and instrument sustained hot inference and GPU unified-memory peak.

No modifications to original REPORT.md, world_schema.json, G0.5/G0.7 baseline, or LingBot repository tree.
