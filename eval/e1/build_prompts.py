#!/usr/bin/env python3
"""Build eval/e1/prompts.jsonl from the 15 G0.7 prompts + synthetic prompts.

Deterministic: fixed seed=42 for the 80/20 train/val split.  The val split is
guaranteed to contain at least one G0.7 prompt so alignment on known prompts
can be monitored.  Run once; the output prompts.jsonl is the committed artifact.
"""
from __future__ import annotations

import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(REPO_ROOT, "eval", "e1")
G07_ROOT = os.path.join(REPO_ROOT, "eval", "g0.7")

SCENES = ["single_subject", "spatial", "indoor", "outdoor", "camera_motion"]
VARIANTS = ["A", "B", "C"]

# ---------------------------------------------------------------------------
# Synthetic prompts: cover subjects / scenes / actions / camera / style.
# Each is 10-40 English words, video-generation style.
# ---------------------------------------------------------------------------
SYNTHETIC = [
    # animals
    "A golden retriever runs across a sunlit beach, waves crashing behind it, camera tracking alongside.",
    "A flock of seagulls glides over turquoise ocean water, pan right following their flight path.",
    "A red fox leaps through a snow-covered forest, dolly in close on its paws and bushy tail.",
    "A sea turtle swims slowly through a coral reef, soft underwater light, slow orbit around its shell.",
    "An eagle soars above snow-capped mountains, tilt down as it circles on a rising thermal.",
    # vehicles
    "A vintage sports car speeds down a desert highway, dust trailing, camera dolly forward alongside.",
    "A cargo plane flies through golden clouds at dusk, crane up revealing its full silver silhouette.",
    "A surfer carves a wave on a bright blue beach, camera tracking the spray and board motion.",
    "A sleek spaceship drifts past a ringed planet in deep space, slow zoom toward its cockpit window.",
    "A steam train winds through an autumn forest, pan left following the curve of the rusty track.",
    # people
    "A dancer spins in a studio with soft warm light, orbit around her flowing dress and arm gestures.",
    "A hiker climbs a rocky ridge at sunrise, tilt up to reveal the wide valley below.",
    "A child blows dandelion seeds in a green meadow, macro close-up on the floating white seeds.",
    "A chef tosses vegetables in a flaming wok in a warm kitchen, camera zoom in on the flames.",
    "A cyclist races through a neon-lit city street at night, tracking shot alongside at speed.",
    # objects
    "A red flower blooms in time-lapse on a windowsill, zoom in on its slowly unfurling petals.",
    "A glass ballerina figurine rotates on a wooden shelf, slow orbit around it in warm light.",
    "A hot air balloon rises over a misty green valley, crane up gently following its ascent.",
    "A folded paper boat floats down a rainy gutter, slow dolly forward as it bobs along.",
    "A beeswax candle flame flickers in a dark room, macro push-in on melted wax and wick.",
    # nature
    "A wide waterfall cascades into a turquoise pool in a lush forest, pan across the falling water.",
    "A desert sand dune shifts under steady wind, wide orbit around the rippled golden ridges.",
    "A glacier cracks and calves into a blue fjord, slow zoom in on the falling chunks of ice.",
    "Green northern lights ripple across a starry sky over snowy mountains, tilt up toward the aurora.",
    "A thunderstorm rolls over a city skyline at night, bolts of lightning illuminating the towers.",
    # styles
    "Cartoon style: a small friendly robot walks through a candy-colored town, pan right following its steps.",
    "Anime style: a schoolgirl runs across a bridge beneath falling cherry blossoms, tracking forward.",
    "Ink wash painting style: a lone fisherman floats on a misty river, slow dolly through tall reeds.",
    "Cinematic style: a lone cowboy walks into a dusty western town at golden hour, dolly back.",
    "Realistic macro: clear dewdrops slide off a fern leaf, slow tilt down following each drop.",
    # more action / camera variety
    "A cluster of hot air balloons lifts off at dawn, crane up over the colorful drifting fleet.",
    "A surfer waits inside a barreling wave, underwater shot, dolly forward into the green tube.",
    "A flock of white birds takes off from a golden field, orbit around the swirling cloud of wings.",
    "A lava lamp bubbles on a desk in a dark study, zoom in on the rising orange blobs.",
    "A kite surfer jumps over choppy sea waves, slow motion tracking the airborne arcing jump.",
    "A futuristic train glides through a neon megacity tunnel, forward dolly into bright light.",
    "A small garden grows in fast time-lapse from bare soil to ripe tomatoes, tilt down across the bed.",
    "A translucent jellyfish pulses in dark ocean water, soft bioluminescence, slow orbit around tentacles.",
    "An old covered market street bustles with people and umbrellas, pan right across the wooden stalls.",
    "A drone follows a mountain biker down a rocky dirt trail, chase camera over the handlebars.",
    "A tattered flag waves on a windy cliff at sunset, slow zoom out revealing the wide ocean horizon.",
]


def load_g07_prompts() -> list[dict]:
    rows = []
    for scene in SCENES:
        for v in VARIANTS:
            path = os.path.join(G07_ROOT, scene, f"prompt_{v}.txt")
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read().strip()
            rows.append({
                "id": f"g07_{scene}_{v}",
                "text": text,
                "source": f"g07_{v}",
            })
    return rows


def main() -> int:
    rows = load_g07_prompts()
    for i, text in enumerate(SYNTHETIC):
        rows.append({
            "id": f"synthetic_{i:03d}",
            "text": text,
            "source": "synthetic",
        })

    total = len(rows)
    n_val = max(1, round(total * 0.20))  # ~20% held out
    rng = random.Random(42)
    order = list(range(total))
    rng.shuffle(order)
    val_idx = set(order[:n_val])
    # Guarantee at least one G0.7 prompt in val.
    has_g07_val = any(rows[i]["source"].startswith("g07") for i in val_idx)
    if not has_g07_val:
        g07_indices = [i for i in range(total) if rows[i]["source"].startswith("g07")]
        # pick a g07 not already in val (all are in train right now)
        for gi in g07_indices:
            if gi not in val_idx:
                # swap with a val member
                swap = next(iter(val_idx))
                val_idx.discard(swap)
                val_idx.add(gi)
                break

    for i, row in enumerate(rows):
        row["split"] = "val" if i in val_idx else "train"

    out_path = os.path.join(EVAL_DIR, "prompts.jsonl")
    os.makedirs(EVAL_DIR, exist_ok=True)
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = total - n_train
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")
    print(f"total={total}  train={n_train}  val={n_val}")
    g07_val = [r["id"] for r in rows if r["split"] == "val" and r["source"].startswith("g07")]
    print(f"G0.7 in val: {g07_val}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
