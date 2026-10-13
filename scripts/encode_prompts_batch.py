#!/usr/bin/env python3
"""Batch-encode all G0.7 evaluation prompts with UMT5 in a single session.

Loads the T5 encoder once, encodes all scene×variant prompts, and saves
each embedding to the eval cache.  This avoids reloading the 10.5GB T5
model 15 times.

Usage:
    python scripts/encode_prompts_batch.py
    python scripts/encode_prompts_batch.py --scene single_subject
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from world_condition import (  # noqa: E402
    WorldDescription,
    compose_compact_world_prompt,
    compose_world_prompt,
    load_world_condition,
)
from wan.utils.prompt_embedding import save_prompt_embedding  # noqa: E402

EVAL_DIR = REPO_ROOT / "eval" / "g0.7"
CONFIG_PATH = EVAL_DIR / "config.json"
CACHE_DIR = EVAL_DIR / ".cache"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("batch-encode")


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_prompts_for_scene(scene: dict, world: WorldDescription) -> dict[str, str]:
    """Build A/B/C prompts for a scene."""
    user = scene["user_prompt"]
    return {
        "A": user,
        "B": compose_world_prompt(world, user),
        "C": compose_compact_world_prompt(world, user),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-encode all G0.7 prompts.")
    parser.add_argument("--scene", type=str, default=None, help="Only encode for one scene.")
    args = parser.parse_args()

    cfg = load_config()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all prompts to encode
    encode_jobs: list[dict] = []
    for scene in cfg["scenes"]:
        if args.scene and scene["id"] != args.scene:
            continue
        wc_path = EVAL_DIR / scene["id"] / "world_condition.json"
        if not wc_path.exists():
            log.warning("World condition not found for %s, skipping (run VLM first).", scene["id"])
            continue
        world = load_world_condition(str(wc_path))
        prompts = build_prompts_for_scene(scene, world)

        for variant, prompt_text in prompts.items():
            output_file = CACHE_DIR / f"prompt_embeds_{scene['id']}_{variant}.safetensors"
            if output_file.exists():
                log.info("  Already cached: %s/%s", scene["id"], variant)
                continue
            encode_jobs.append({
                "scene": scene["id"],
                "variant": variant,
                "prompt": prompt_text,
                "output": str(output_file),
            })

    if not encode_jobs:
        log.info("All prompts already cached. Nothing to encode.")
        return 0

    log.info("Encoding %d prompts in one T5 session...", len(encode_jobs))

    # Load T5 encoder once
    from wan.modules.t5 import T5EncoderModel
    from wan.configs.wan_i2v_1_3B import i2v_1_3B as config

    assets_dir = cfg["paths"]["assets_dir"]
    t5_checkpoint = cfg["paths"]["t5_checkpoint"]
    tokenizer_path = cfg["paths"]["tokenizer_path"]

    log.info("Loading T5 encoder (mmap, ~10.5GB address space)...")
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=__import__("torch").bfloat16,
        device=__import__("torch").device("cpu"),
        checkpoint_path=t5_checkpoint,
        tokenizer_path=tokenizer_path,
    )
    log.info("T5 encoder loaded.")

    import torch
    success = 0
    for job in encode_jobs:
        log.info("  Encoding %s/%s (%d chars)...", job["scene"], job["variant"], len(job["prompt"]))
        try:
            context_list = text_encoder([job["prompt"]], torch.device("cpu"))
            context = context_list[0]
            save_prompt_embedding(
                output_path=job["output"],
                context=context,
                prompt=job["prompt"],
                text_len=config.text_len,
                dtype=torch.bfloat16,
            )
            log.info("    -> shape=%s, saved to %s", list(context.shape), job["output"])
            success += 1
        except Exception as exc:
            log.error("    FAILED: %s: %s", type(exc).__name__, exc)

    # Cleanup
    log.info("Releasing T5 encoder...")
    del text_encoder
    import gc
    gc.collect()

    log.info("Done: %d/%d prompts encoded successfully.", success, len(encode_jobs))
    return 0 if success == len(encode_jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
