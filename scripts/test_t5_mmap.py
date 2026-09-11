#!/usr/bin/env python
"""Test T5 checkpoint mmap support and low-memory loading.

This script validates M3.3 UMT5 low-memory loading on the real
11GB checkpoint:
  1. Check file exists and size
  2. Test torch.load(mmap=True) - verify RSS doesn't explode
  3. Construct meta UMT5 encoder
  4. Compare expected keys vs checkpoint keys
  5. Materialize weights tensor-by-tensor with assign=True
  6. Run real prompt forward
  7. Save prompt embedding
  8. Unload and verify memory release

Usage:
    python scripts/test_t5_mmap.py --checkpoint /path/to/models_t5_umt5-xxl-enc-bf16.pth
"""

import argparse
import gc
import os
import sys
import time

import torch


def get_rss_mb():
    """Get current process RSS in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    except ImportError:
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except Exception:
            pass
    return -1


def log(msg, rss_before=None):
    """Log message with current RSS."""
    rss = get_rss_mb()
    delta = ""
    if rss_before is not None:
        delta = f" (delta={rss - rss_before:+.1f}MB)"
    print(f"[RSS={rss:8.1f}MB] {msg}{delta}")
    return rss


def test_mmap(checkpoint_path):
    """Stage 1: Test torch.load with mmap=True."""
    print("\n" + "=" * 60)
    print("Stage 1: Test torch.load(mmap=True)")
    print("=" * 60)

    rss_before = log("Before torch.load")

    start = time.time()
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except Exception as e:
        print(f"ERROR: torch.load(mmap=True) failed: {e}")
        print("Trying without mmap...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        log("After torch.load (NO mmap)", rss_before)
        return state_dict, False

    load_time = time.time() - start
    rss_after = log(f"After torch.load (mmap=True, {load_time:.1f}s)", rss_before)

    # Count keys and tensors
    if isinstance(state_dict, dict):
        key_count = len(state_dict)
        tensor_count = sum(1 for v in state_dict.values() if isinstance(v, torch.Tensor))
        total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values() if isinstance(v, torch.Tensor))
        print(f"  Keys: {key_count}")
        print(f"  Tensors: {tensor_count}")
        print(f"  Total logical tensor bytes: {total_bytes / 1024**3:.2f} GB")
    else:
        print(f"  state_dict type: {type(state_dict)}")
        key_count = 0
        tensor_count = 0

    rss_delta = rss_after - rss_before
    mmap_effective = rss_delta < 2000  # If RSS increased < 2GB, mmap is working
    print(f"\n  RSS increase: {rss_delta:+.1f} MB")
    print(f"  mmap effective: {'YES' if mmap_effective else 'NO (RSS increased too much)'}")

    return state_dict, mmap_effective


def construct_meta_umt5():
    """Stage 2: Construct UMT5 encoder on meta device."""
    print("\n" + "=" * 60)
    print("Stage 2: Construct meta UMT5 encoder")
    print("=" * 60)

    rss_before = log("Before meta UMT5 construction")

    from transformers import UMT5EncoderModel, UMT5Config

    config = UMT5Config(
        vocab_size=256384,
        d_model=4096,
        d_ff=10240,
        num_heads=64,
        num_layers=24,
        is_encoder_decoder=False,
    )

    with torch.device("meta"):
        model = UMT5EncoderModel(config)

    rss_after = log("After meta UMT5 construction", rss_before)

    # Verify all parameters are meta
    param_count = sum(p.numel() for p in model.parameters())
    meta_params = sum(1 for p in model.parameters() if p.is_meta)
    total_params = sum(1 for p in model.parameters())
    meta_buffers = sum(1 for b in model.buffers() if b.is_meta)
    total_buffers = sum(1 for b in model.buffers())

    print(f"  Parameter count: {param_count:,} ({param_count / 1e9:.2f}B)")
    print(f"  Meta parameters: {meta_params}/{total_params}")
    print(f"  Meta buffers: {meta_buffers}/{total_buffers}")

    all_meta = (meta_params == total_params) and (meta_buffers == total_buffers)
    print(f"  All meta: {'YES' if all_meta else 'NO'}")

    return model, config


def compare_keys(model, state_dict):
    """Stage 3: Compare expected keys vs checkpoint keys."""
    print("\n" + "=" * 60)
    print("Stage 3: Key mapping comparison")
    print("=" * 60)

    expected_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    missing = expected_keys - checkpoint_keys
    unexpected = checkpoint_keys - expected_keys
    matching = expected_keys & checkpoint_keys

    print(f"  Expected keys: {len(expected_keys)}")
    print(f"  Checkpoint keys: {len(checkpoint_keys)}")
    print(f"  Matching: {len(matching)}")
    print(f"  Missing: {len(missing)}")
    print(f"  Unexpected: {len(unexpected)}")

    if missing:
        print(f"\n  Missing keys (first 10):")
        for k in sorted(missing)[:10]:
            print(f"    - {k}")

    if unexpected:
        print(f"\n  Unexpected keys (first 10):")
        for k in sorted(unexpected)[:10]:
            print(f"    - {k}")

    # Shape mismatch check
    shape_mismatches = []
    for k in matching:
        expected_shape = model.state_dict()[k].shape
        actual_shape = state_dict[k].shape
        if expected_shape != actual_shape:
            shape_mismatches.append((k, expected_shape, actual_shape))

    print(f"\n  Shape mismatches: {len(shape_mismatches)}")
    if shape_mismatches:
        for k, exp, act in shape_mismatches[:10]:
            print(f"    - {k}: expected {exp}, got {act}")

    return {
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
    }


def materialize_weights(model, state_dict, target_dtype=None):
    """Stage 4: Materialize weights tensor-by-tensor with assign=True."""
    print("\n" + "=" * 60)
    print("Stage 4: Materialize weights (tensor-by-tensor)")
    print("=" * 60)

    rss_before = log("Before materialization")
    start = time.time()

    matching_keys = set(model.state_dict().keys()) & set(state_dict.keys())
    total = len(matching_keys)
    peak_rss = rss_before

    for i, key in enumerate(sorted(matching_keys)):
        # Get tensor from mmap state_dict (lazy)
        source_tensor = state_dict[key]

        # Cast to target dtype if specified
        if target_dtype is not None and source_tensor.dtype != target_dtype:
            source_tensor = source_tensor.to(target_dtype)

        # Assign to model using load_state_dict with assign=True
        partial_state = {key: source_tensor}
        model.load_state_dict(partial_state, strict=False, assign=True)

        # Release source tensor
        del source_tensor
        del partial_state

        # Track peak RSS
        current_rss = get_rss_mb()
        if current_rss > peak_rss:
            peak_rss = current_rss

        if (i + 1) % 50 == 0 or i == total - 1:
            log(f"  Materialized {i+1}/{total} tensors")

    load_time = time.time() - start
    rss_after = log(f"After materialization ({load_time:.1f}s)", rss_before)

    # Verify no meta tensors remain
    meta_params = sum(1 for p in model.parameters() if p.is_meta)
    meta_buffers = sum(1 for b in model.buffers() if b.is_meta)
    print(f"\n  Remaining meta parameters: {meta_params}")
    print(f"  Remaining meta buffers: {meta_buffers}")
    print(f"  Peak RSS during materialization: {peak_rss:.1f} MB")

    success = (meta_params == 0) and (meta_buffers == 0)
    print(f"  Materialization success: {'YES' if success else 'NO'}")

    return model, success


def test_prompt_forward(model, assets_dir, prompt="A red car is parked beside a tree."):
    """Stage 5: Run real prompt forward."""
    print("\n" + "=" * 60)
    print("Stage 5: Real prompt forward")
    print("=" * 60)

    rss_before = log("Before prompt forward")

    from transformers import AutoTokenizer

    tokenizer_path = os.path.join(assets_dir, "google", "umt5-xxl")
    if not os.path.exists(tokenizer_path):
        # Try alternative paths
        for candidate in [
            os.path.join(assets_dir, "umt5-xxl"),
            "google/umt5-xxl",
        ]:
            if os.path.exists(candidate) or candidate.startswith("google/"):
                tokenizer_path = candidate
                break

    print(f"  Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    print(f"  Input IDs shape: {inputs['input_ids'].shape}")

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)

    forward_time = time.time() - start
    context = outputs.last_hidden_state[0]  # [L, 4096]

    rss_after = log(f"After prompt forward ({forward_time:.1f}s)", rss_before)

    print(f"  Context shape: {context.shape}")
    print(f"  Context dtype: {context.dtype}")
    print(f"  Sequence length: {context.shape[0]}")
    print(f"  Hidden dim: {context.shape[1]}")
    print(f"  All finite: {torch.isfinite(context).all().item()}")
    print(f"  Has NaN: {torch.isnan(context).any().item()}")
    print(f"  Has Inf: {torch.isinf(context).any().item()}")

    success = (
        context.shape[1] == 4096
        and context.shape[0] > 0
        and context.shape[0] <= 512
        and torch.isfinite(context).all().item()
    )
    print(f"  Forward success: {'YES' if success else 'NO'}")

    return context, tokenizer, success


def save_prompt_embedding(context, prompt, output_path):
    """Stage 6: Save prompt embedding to safetensors."""
    print("\n" + "=" * 60)
    print("Stage 6: Save prompt embedding")
    print("=" * 60)

    from wan.utils.prompt_embedding import save_prompt_embedding

    rss_before = log("Before save")

    save_prompt_embedding(
        output_path,
        context,
        prompt=prompt,
        model_id="umt5-xxl-encoder",
    )

    log("After save", rss_before)
    print(f"  Saved to: {output_path}")

    # Verify reload
    from wan.utils.prompt_embedding import load_prompt_embedding
    loaded, meta = load_prompt_embedding(output_path, expected_hidden_dim=4096)
    print(f"  Reload shape: {loaded.shape}")
    print(f"  Reload dtype: {loaded.dtype}")
    print(f"  Round-trip match: {torch.allclose(context, loaded)}")


def unload_everything(model, tokenizer, state_dict):
    """Stage 7: Unload everything and verify memory release."""
    print("\n" + "=" * 60)
    print("Stage 7: Unload and verify memory release")
    print("=" * 60)

    rss_before = log("Before unload")

    del state_dict
    del model
    del tokenizer
    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()

    rss_after = log("After unload + gc.collect()", rss_before)

    print(f"  RSS released: {rss_before - rss_after:+.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Test T5 checkpoint mmap and low-memory loading")
    parser.add_argument("--checkpoint", required=True, help="Path to models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--assets_dir", required=True, help="Path to 14B assets dir (for tokenizer)")
    parser.add_argument("--prompt", default="A red car is parked beside a tree.", help="Test prompt")
    parser.add_argument("--output", default="m3-validation/prompt_embeds.safetensors", help="Output embedding path")
    parser.add_argument("--skip_forward", action="store_true", help="Skip prompt forward test")
    args = parser.parse_args()

    print("=" * 60)
    print("M3.3 UMT5 Low-Memory Loading Validation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Assets dir: {args.assets_dir}")
    print(f"Prompt: {args.prompt}")

    # Check file
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    file_size = os.path.getsize(args.checkpoint)
    print(f"File size: {file_size / 1024**3:.2f} GB ({file_size:,} bytes)")

    # Check disk space
    stat = os.statvfs(os.path.dirname(args.checkpoint))
    free_space = stat.f_bavail * stat.f_frsize
    print(f"Free disk space: {free_space / 1024**3:.2f} GB")

    results = {}

    # Stage 1: mmap test
    state_dict, mmap_effective = test_mmap(args.checkpoint)
    results["mmap_effective"] = mmap_effective

    if not mmap_effective:
        print("\nWARNING: mmap not effective. Continuing but memory may be high.")

    # Stage 2: meta UMT5
    model, config = construct_meta_umt5()
    results["meta_construction_success"] = True

    # Stage 3: key comparison
    key_results = compare_keys(model, state_dict)
    results["missing_keys"] = len(key_results["missing"])
    results["unexpected_keys"] = len(key_results["unexpected"])
    results["shape_mismatches"] = len(key_results["shape_mismatches"])

    if key_results["missing"] or key_results["unexpected"] or key_results["shape_mismatches"]:
        print("\nERROR: Key mapping issues detected. Cannot proceed with materialization.")
        results["overall"] = "FAIL"
        print_results(results)
        sys.exit(1)

    # Stage 4: materialize
    model, materialize_success = materialize_weights(model, state_dict, target_dtype=None)
    results["materialize_success"] = materialize_success

    if not materialize_success:
        print("\nERROR: Materialization failed (meta tensors remain).")
        results["overall"] = "FAIL"
        print_results(results)
        sys.exit(1)

    # Stage 5: prompt forward
    if not args.skip_forward:
        context, tokenizer, forward_success = test_prompt_forward(model, args.assets_dir, args.prompt)
        results["forward_success"] = forward_success

        # Stage 6: save embedding
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_prompt_embedding(context, args.prompt, args.output)

        # Stage 7: unload
        unload_everything(model, tokenizer, state_dict)
    else:
        print("\nSkipping prompt forward (--skip_forward)")
        results["forward_success"] = "SKIPPED"

    # Final results
    results["overall"] = "PASS" if all([
        results.get("mmap_effective", False),
        results.get("materialize_success", False),
        results.get("forward_success", False) is True or results.get("forward_success") == "SKIPPED",
    ]) else "FAIL"

    print_results(results)


def print_results(results):
    print("\n" + "=" * 60)
    print("M3.3 Validation Results Summary")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\n  Overall: {results['overall']}")


if __name__ == "__main__":
    main()
