# E2 OLD PRODUCTS — INVALID (do not reuse)

**Status: INVALID as of E1.2 root-cause fix (branch feat/e1.2-adapter-v2, d88528e).**

## What is invalid
All artifacts under `eval/e2/` produced with the OLD E1 adapters
(epoch5 best `eval/e1/best_adapter.safetensors`, SHA256 `d68ab40a...`, and
epoch20 `eval/e1/adapter_final/adapter.safetensors`, SHA256 `7dae9a57...`):

- `embeddings/*_minicpm.safetensors` — MiniCPM5+epoch5/20 adapter contexts
- `videos/*_minicpm.mp4` (15 files) — ALL unrecognizable color noise
- `results_minicpm.jsonl`, `summary.json`, `metrics_objective.json`
- `.cache/*/latents.safetensors`

## Why invalid
Root cause found in E2 diagnosis (see `eval/e2_diag/`):
- Old adapter output LayerNorm(4096) left student 4096 features at the wrong
  direction; feeding the FROZEN LingBot text_embedding(4096->1536) saturated
  the GELU -> student 1536 activations 25x too large (per-token L2 85 vs
  teacher 4.2) -> DiT cross-attention destabilized -> pure-noise latents.
- VAE decode was ruled out: FP16 == BF16 on the same latent (temporal_mad
  0.151 vs 0.152); UMT5 baseline on the same pipeline is clean.

## Replacement
Use the E1.2 adapter (`eval/e1.2/adapter_best.safetensors`, SHA256
`bd73963a...`), trained with REAL downstream LingBot cross-attention K/V
distillation (blocks 0/14/29) + block-0 SDPA output distillation.
E1.2 smoke (single_subject) recovered a recognizable Wanaka-tree scene
(frame stats now match UMT5 baseline: mean 0.698 vs 0.678, temporal_mad
0.043 vs 0.048).

## Do not
- Do not reuse `eval/e2/embeddings/*.safetensors` in any new generation.
- Do not cite `eval/e2/videos/*.mp4` as E2 quality evidence.
- Do not delete this directory (kept for forensics); new work lives in
  `eval/e1.2/` (gate3) and `eval/e1.2/embeddings/`.
