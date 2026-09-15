import gc
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
# import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from wan.utils.device import device_autocast, device_empty_cache, device_synchronize
from wan.utils.streaming_loader import (
    create_meta_model,
    load_sharded_safetensors_streaming,
)

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward_causal, sp_dit_forward_causal
from .distributed.util import get_world_size
from .modules.model_fast import WanModelFast
from .modules.model_causal import WanModelCausal
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE

from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.cam_utils import (
    compute_relative_poses,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)
from einops import rearrange


def _resolve_asset_path(filename, checkpoint_dir, assets_dir=None):
    """Resolve T5 / VAE / tokenizer paths, falling back to ``assets_dir``.

    The 1.3B Hugging Face upload currently ships only DiT weights; T5, VAE,
    and the tokenizer can be reused from the 14B release via ``--assets_dir``.
    """
    candidates = [os.path.join(checkpoint_dir, filename)]
    if assets_dir:
        candidates.append(os.path.join(assets_dir, filename))
    for path in candidates:
        if os.path.exists(path):
            return path
    searched = ", ".join(candidates)
    raise FileNotFoundError(
        f"Required asset {filename!r} not found. Looked in: {searched}. "
        "Pass --assets_dir pointing at a 14B (or Wan) checkpoint that contains "
        "models_t5_umt5-xxl-enc-bf16.pth, Wan2.1_VAE.pth, and google/umt5-xxl."
    )


def _resolve_dit_dir(checkpoint_dir, subfolder):
    """Prefer ``checkpoint_dir/subfolder`` when it exists, else the root.

    14B causal-fast stores DiT weights under ``transformers/``. The 1.3B
    causal-fast upload currently places shards at the repository root.
    """
    if subfolder:
        candidate = os.path.join(checkpoint_dir, subfolder)
        if os.path.isdir(candidate):
            return candidate
    return checkpoint_dir


def _load_safetensors_state_dict(dit_dir):
    """Load a (possibly sharded) safetensors dump from ``dit_dir``."""
    from safetensors.torch import load_file

    for index_name in (
            "model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors.index.json",
    ):
        index_path = os.path.join(dit_dir, index_name)
        if not os.path.isfile(index_path):
            continue
        with open(index_path) as f:
            index = json.load(f)
        state = {}
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(os.path.join(dit_dir, shard)))
        return state

    for single_name in (
            "model.safetensors",
            "diffusion_pytorch_model.safetensors",
    ):
        single_path = os.path.join(dit_dir, single_name)
        if os.path.isfile(single_path):
            return load_file(single_path)

    raise FileNotFoundError(
        f"No safetensors weights found in {dit_dir}. Expected a sharded "
        "index (model.safetensors.index.json) or a single model.safetensors."
    )


def _dit_kwargs_from_config(config, extra=None):
    kwargs = dict(
        model_type="i2v",
        patch_size=tuple(config.patch_size),
        text_len=config.text_len,
        in_dim=getattr(config, "in_dim", 36),
        dim=config.dim,
        ffn_dim=config.ffn_dim,
        freq_dim=config.freq_dim,
        text_dim=getattr(config, "text_dim", 4096),
        out_dim=getattr(config, "out_dim", 16),
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        qk_norm=config.qk_norm,
        cross_attn_norm=config.cross_attn_norm,
        eps=config.eps,
    )
    if extra:
        kwargs.update(extra)
    return kwargs


def load_dit_model(model_cls, checkpoint_dir, subfolder, config, torch_dtype,
                   extra=None, use_streaming=True, target_device=None,
                   allow_legacy_fallback=True):
    """Load a DiT from ``transformers/`` or the checkpoint root.

    Uses ``from_pretrained`` when ``config.json`` is present. Otherwise builds
    the module from the task EasyDict and loads sharded safetensors — the
    layout of the current 1.3B Hugging Face upload.

    When ``use_streaming=True`` (default), uses a low-memory streaming loader:
    - Creates the model on the ``meta`` device (no real parameter allocation)
    - Loads tensors one at a time via ``safetensors.safe_open``
    - Casts F32 -> target dtype and assigns directly into the meta model
    - Avoids constructing a full state dict and a second ``model.to(dtype)`` copy

    Args:
        target_device: If set, streaming loader assigns tensors directly to this
            device (e.g. torch.device("mps")). Avoids a full CPU->device copy.
            If None, tensors are loaded to CPU and the caller may move the model.
        allow_legacy_fallback: If True (default), streaming failure falls back to
            the legacy full-state loader. Set False for MPS to fail fast and
            avoid memory explosion from the legacy loader.

    Falls back to the legacy full-state loader if streaming fails (only when
    allow_legacy_fallback=True).
    """
    extra = extra or {}
    dit_dir = _resolve_dit_dir(checkpoint_dir, subfolder)
    logging.info(f"Loading {model_cls.__name__} from {dit_dir}")
    if os.path.isfile(os.path.join(dit_dir, "config.json")):
        return model_cls.from_pretrained(
            dit_dir, torch_dtype=torch_dtype, **extra)

    logging.info(
        f"config.json not found in {dit_dir}; building {model_cls.__name__} "
        "from the task config and loading safetensors weights."
    )

    if use_streaming:
        try:
            logging.info(
                f"Using low-memory streaming safetensors loader "
                f"(target_device={target_device}, allow_legacy_fallback={allow_legacy_fallback})"
            )
            kwargs = _dit_kwargs_from_config(config, extra)
            model = create_meta_model(model_cls, **kwargs)
            stats = load_sharded_safetensors_streaming(
                model,
                dit_dir,
                target_dtype=torch_dtype,
                target_device=target_device,
            )
            if stats.success:
                rss_str = f", RSS peak {stats.rss_peak_mb:.0f} MB" if stats.rss_peak_mb else ""
                logging.info(
                    f"Streaming load complete: {stats.tensor_count} tensors, "
                    f"{stats.load_duration_sec:.1f}s{rss_str}"
                )
                # Verify all tensors are on target device (if specified)
                if target_device is not None:
                    wrong_device = [
                        n for n, p in model.named_parameters()
                        if p.device.type != target_device.type
                    ]
                    if wrong_device:
                        logging.warning(
                            f"{len(wrong_device)} params not on target device {target_device}; "
                            f"first: {wrong_device[:3]}"
                        )
                return model
            else:
                msg = "Streaming load had issues.\n" + stats.summary()
                if allow_legacy_fallback:
                    logging.warning(msg + "\nFalling back to legacy loader.")
                else:
                    logging.error(msg + "\nLegacy fallback disabled; failing fast.")
                    del model
                    gc.collect()
                    raise RuntimeError(
                        f"Streaming DiT load failed and legacy fallback is disabled. "
                        f"Missing={len(stats.missing_keys)}, unexpected={len(stats.unexpected_keys)}, "
                        f"shape_mismatch={len(stats.shape_mismatch)}, "
                        f"meta_remaining={stats.meta_params_remaining}"
                    )
                del model
                gc.collect()
        except Exception as e:
            if allow_legacy_fallback:
                logging.warning(f"Streaming loader failed ({e}); falling back to legacy loader")
                gc.collect()
            else:
                logging.error(f"Streaming loader failed ({e}); legacy fallback disabled; re-raising")
                gc.collect()
                raise

    # Legacy full-state loader (fallback)
    logging.info("Using legacy full-state safetensors loader")
    model = model_cls(**_dit_kwargs_from_config(config, extra))
    state = _load_safetensors_state_dict(dit_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logging.warning(f"Missing keys when loading DiT: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys when loading DiT: {unexpected}")
    return model.to(dtype=torch_dtype)


class WanI2VCausal:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        pipe_dtype=torch.bfloat16,
        local_attn_size=-1,
        sink_size=0,
        infer_mode="causal_fast",
        assets_dir=None,
        prompt_embeds_file=None,
        sequential_load=None,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            infer_mode (`str`, *optional*, defaults to "causal_fast"):
                Inference mode. "causal_fast" uses the distilled few-step
                model (config.fast_checkpoint) with KV-cache windowing
                (local_attn_size / sink_size). "causal_pretrain" uses the
                pretrained causal model (config.causal_checkpoint) with
                40-step CFG sampling.
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int` or `torch.device`, *optional*, defaults to 0):
                Id of target GPU device, or a torch.device (e.g. torch.device('mps')).
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
            assets_dir (`str`, *optional*):
                Directory that holds shared T5 / VAE / tokenizer files when
                they are not packaged with ``checkpoint_dir`` (the 1.3B DiT
                upload). Typically the 14B causal-fast checkpoint directory.
        """
        assert infer_mode in ("causal_fast", "causal_pretrain"), \
            f"Unsupported infer_mode: {infer_mode}"
        self.infer_mode = infer_mode

        # Resolve device: accept int (CUDA index) or torch.device
        if isinstance(device_id, torch.device):
            self.device = device_id
        else:
            self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype
        self.pipe_dtype = pipe_dtype
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        # Resolve sequential_load: MPS→True, CUDA→False, CPU→True
        if sequential_load is None:
            self.sequential_load = self.device.type in ("mps", "cpu")
        else:
            self.sequential_load = sequential_load

        # Save construction config for lazy loading
        self._checkpoint_dir = checkpoint_dir
        self._assets_dir = assets_dir
        self._t5_fsdp = t5_fsdp
        self._dit_fsdp = dit_fsdp
        self._use_sp = use_sp
        self._convert_model_dtype = convert_model_dtype
        self._shard_fn = partial(shard_model, device_id=device_id)
        self.prompt_embeds_file = prompt_embeds_file

        # Scheduler and other non-model state can be initialized immediately
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

        # T5 prompt-embedding cache
        self._t5_cache: dict[str, list] = {}

        # Reset per generate()
        self._cross_attn_initialized: bool = False

        if self.sequential_load:
            # Lazy mode: don't load any models yet
            logging.info(
                f"Sequential/lazy load mode enabled (device={self.device}). "
                f"Models will be loaded on demand via load_text_encoder/load_vae/load_dit."
            )
            self.vae_stride = config.vae_stride
            self.patch_size = config.patch_size
            self.text_encoder = None
            self.vae = None
            self.model = None
            return

        # Eager mode: load all models immediately (original behavior)
        shard_fn = self._shard_fn
        if prompt_embeds_file is not None:
            # Skip T5 encoder entirely — context will be loaded from
            # the pre-computed prompt_embeds.safetensors file.
            logging.info(
                f"prompt_embeds_file provided; skipping T5 encoder init. "
                f"Will load context from {prompt_embeds_file}"
            )
            self.text_encoder = None
        else:
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=_resolve_asset_path(
                    config.t5_checkpoint, checkpoint_dir, assets_dir),
                tokenizer_path=_resolve_asset_path(
                    config.t5_tokenizer, checkpoint_dir, assets_dir),
                shard_fn=shard_fn if t5_fsdp else None,
            )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=_resolve_asset_path(
                config.vae_checkpoint, checkpoint_dir, assets_dir),
            device=self.device)

        # MPS mode: stream weights directly to MPS, no legacy fallback, no
        # second full-model CPU->MPS copy. MPS uses float16 (bf16 support
        # is limited on Apple GPU).
        is_mps = self.device.type == "mps"
        dit_target_device = self.device if is_mps else None
        dit_allow_fallback = not is_mps
        dit_dtype = torch.float16 if is_mps else torch.bfloat16

        if self.infer_mode == "causal_fast":
            self.model = load_dit_model(
                WanModelFast,
                checkpoint_dir,
                config.fast_checkpoint,
                config,
                dit_dtype,
                extra=dict(
                    local_attn_size=self.local_attn_size,
                    sink_size=self.sink_size),
                target_device=dit_target_device,
                allow_legacy_fallback=dit_allow_fallback,
            )
        else:
            self.model = load_dit_model(
                WanModelCausal,
                checkpoint_dir,
                config.causal_checkpoint,
                config,
                dit_dtype,
                target_device=dit_target_device,
                allow_legacy_fallback=dit_allow_fallback,
            )

        # Configure model (eval, requires_grad, SP, FSDP, dtype).
        # For MPS with streaming load, weights are already on target device
        # and correct dtype — skip the external .to(self.device) to avoid a
        # full-model temporary copy.
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        if not is_mps:
            self.model = self.model.to(self.device)

        # Verify device placement for MPS streaming load — FAIL FAST
        # Any wrong-device or meta tensor means the load was incomplete and
        # inference would crash or silently produce garbage.
        if is_mps:
            wrong_param_device = [
                n for n, p in self.model.named_parameters()
                if p.device != self.device
            ]
            wrong_buffer_device = [
                n for n, b in self.model.named_buffers()
                if b.device != self.device
            ]
            meta_params = sum(1 for p in self.model.parameters() if p.is_meta)
            meta_buffers = sum(1 for b in self.model.buffers() if b.is_meta)

            issues = []
            if wrong_param_device:
                issues.append(
                    f"{len(wrong_param_device)} params not on {self.device} "
                    f"(first: {wrong_param_device[:3]})"
                )
            if wrong_buffer_device:
                issues.append(
                    f"{len(wrong_buffer_device)} buffers not on {self.device} "
                    f"(first: {wrong_buffer_device[:3]})"
                )
            if meta_params:
                issues.append(f"{meta_params} meta parameters remain")
            if meta_buffers:
                issues.append(f"{meta_buffers} meta buffers remain")

            if issues:
                raise RuntimeError(
                    "MPS DiT load verification failed:\n  - "
                    + "\n  - ".join(issues)
                )
            logging.info(
                f"MPS DiT load verified: all params/buffers on {self.device}, no meta tensors"
            )

    def clear_text_cache(self):
        """Drop all cached T5 prompt embeddings. Frees ~4 MB per entry."""
        self._t5_cache.clear()

    def unload_text_encoder(self):
        """Release the T5 text encoder to free memory (~11GB for umt5-xxl).

        After calling this, generate() will fail unless prompt_embeds_file
        was provided or the text encoder is reloaded.
        """
        from wan.utils.memory import release_model_and_setattr
        return release_model_and_setattr(self, "text_encoder", self.device)

    def unload_vae(self):
        """Release the VAE to free memory (~0.5GB).

        After calling this, image encoding/decoding will fail unless the VAE
        is reloaded.
        """
        from wan.utils.memory import release_model_and_setattr
        return release_model_and_setattr(self, "vae", self.device)

    def unload_dit(self):
        """Release the DiT model to free memory (~3.4GB for 1.3B FP16).

        After calling this, generation will fail unless the DiT is reloaded.
        """
        from wan.utils.memory import release_model_and_setattr
        return release_model_and_setattr(self, "model", self.device)

    def load_text_encoder(self):
        """Load the T5 text encoder (idempotent).

        No-op if already loaded or if prompt_embeds_file is provided.
        """
        if self.text_encoder is not None:
            return
        if self.prompt_embeds_file is not None:
            logging.info("prompt_embeds_file provided; skipping text encoder load.")
            return

        logging.info("Loading T5 text encoder...")
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=_resolve_asset_path(
                self.config.t5_checkpoint, self._checkpoint_dir, self._assets_dir),
            tokenizer_path=_resolve_asset_path(
                self.config.t5_tokenizer, self._checkpoint_dir, self._assets_dir),
            shard_fn=self._shard_fn if self._t5_fsdp else None,
        )
        logging.info("T5 text encoder loaded.")

    def load_vae(self):
        """Load the VAE (idempotent)."""
        if self.vae is not None:
            return

        logging.info("Loading VAE...")
        # MPS: use FP16 for VAE to reduce memory pressure (VAE decode is memory-heavy)
        vae_dtype = torch.float16 if self.device.type == "mps" else torch.float32
        self.vae = Wan2_1_VAE(
            vae_pth=_resolve_asset_path(
                self.config.vae_checkpoint, self._checkpoint_dir, self._assets_dir),
            device=self.device,
            dtype=vae_dtype)
        logging.info(f"VAE loaded (dtype={vae_dtype}).")

    def load_dit(self):
        """Load the DiT model (idempotent).

        For MPS, uses streaming loader with direct device placement and
        fail-fast verification. For CUDA/CPU, uses standard loading with
        legacy fallback.
        """
        if self.model is not None:
            return

        is_mps = self.device.type == "mps"
        dit_target_device = self.device if is_mps else None
        dit_allow_fallback = not is_mps
        # MPS: use FP32 to avoid MPSNDArrayMatrixMultiplication dtype mismatch
        # (MPS requires destination and accumulator to have the same dtype).
        # FP16 mixed-precision matmul is not fully supported on MPS.
        dit_dtype = torch.float32 if is_mps else torch.bfloat16

        logging.info(f"Loading DiT model (infer_mode={self.infer_mode})...")

        if self.infer_mode == "causal_fast":
            self.model = load_dit_model(
                WanModelFast,
                self._checkpoint_dir,
                self.config.fast_checkpoint,
                self.config,
                dit_dtype,
                extra=dict(
                    local_attn_size=self.local_attn_size,
                    sink_size=self.sink_size),
                target_device=dit_target_device,
                allow_legacy_fallback=dit_allow_fallback,
            )
        else:
            self.model = load_dit_model(
                WanModelCausal,
                self._checkpoint_dir,
                self.config.causal_checkpoint,
                self.config,
                dit_dtype,
                target_device=dit_target_device,
                allow_legacy_fallback=dit_allow_fallback,
            )

        # Configure model
        self.model = self._configure_model(
            model=self.model,
            use_sp=self._use_sp,
            dit_fsdp=self._dit_fsdp,
            shard_fn=self._shard_fn,
            convert_model_dtype=self._convert_model_dtype)
        if not is_mps:
            self.model = self.model.to(self.device)

        # MPS fail-fast verification
        if is_mps:
            wrong_param_device = [
                n for n, p in self.model.named_parameters()
                if p.device.type != self.device.type
            ]
            wrong_buffer_device = [
                n for n, b in self.model.named_buffers()
                if b.device.type != self.device.type
            ]
            meta_params = sum(1 for p in self.model.parameters() if p.is_meta)
            meta_buffers = sum(1 for b in self.model.buffers() if b.is_meta)

            issues = []
            if wrong_param_device:
                issues.append(f"{len(wrong_param_device)} params not on {self.device}")
            if wrong_buffer_device:
                issues.append(f"{len(wrong_buffer_device)} buffers not on {self.device}")
            if meta_params:
                issues.append(f"{meta_params} meta parameters remain")
            if meta_buffers:
                issues.append(f"{meta_buffers} meta buffers remain")
            if issues:
                raise RuntimeError("MPS DiT load verification failed: " + "; ".join(issues))

        logging.info("DiT model loaded.")

    def memory_summary(self) -> dict:
        """Return current memory usage summary."""
        from wan.utils.memory import get_memory_stats
        stats = get_memory_stats(self.device)
        stats["text_encoder_loaded"] = self.text_encoder is not None
        stats["vae_loaded"] = getattr(self, "vae", None) is not None
        stats["dit_loaded"] = getattr(self, "model", None) is not None
        return stats

    def prewarm(
        self,
        img,
        max_area: int = 480 * 832,
        frame_num: int = 81,
        chunk_size: int = 3,
        text_seq_len: int = 512,
    ):
        """Opt-in pre-warm. Run one dummy DiT forward at the same shape a
        subsequent generate() call will use, so CUDA kernels are autotuned,
        FSDP all-gathers happen, and Ulysses all-to-alls handshake — all
        outside the timed generate() window.

        Without this call, the first generate() pays a ~7s warmup tax in
        chunk 0 (CUDA lazy init, kernel autotuning, NCCL handshake). On
        8xH100 at 480*832/81 frames, calling prewarm() before the first
        generate() reduces generate()'s wall-clock by ~6.5s (~30%) with
        bit-identical output.

        Idempotent: subsequent calls on the same pipe are no-ops.
        Shape-keyed: if generate() is later invoked with a different shape,
        the autotuner will warm those kernels on demand in chunk 0 (no
        incorrect output, just the tax re-paid once).

        Args:
            img: PIL image or torch tensor — used only for its h/w to match
                generate()'s lat_h/lat_w derivation.
            max_area, frame_num, chunk_size: shape parameters; must match
                the subsequent generate() call to be effective.
            text_seq_len: T5 sequence length (defaults to config.text_len).

        Caller pattern:
            pipe = WanI2VCausal(...)
            pipe.prewarm(img, max_area=..., frame_num=...)
            # start your timer here
            video = pipe.generate(prompt, img, ...)
        """
        if self.infer_mode != "causal_fast":
            logging.info("prewarm is only supported for infer_mode='causal_fast'; skipping.")
            return
        if getattr(self, "_warmed", False):
            return

        cfg = self.config

        # Match generate()'s shape derivation exactly.
        F = frame_num
        h, w = (img.shape[1], img.shape[2]) if hasattr(img, 'shape') else (img.size[1], img.size[0])
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // cfg.vae_stride[1] //
            cfg.patch_size[1] * cfg.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // cfg.vae_stride[2] //
            cfg.patch_size[2] * cfg.patch_size[2])
        lat_f = (F - 1) // cfg.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))

        frame_seqlen = (lat_h * lat_w) // (cfg.patch_size[1] * cfg.patch_size[2])
        max_seq_len = chunk_size * frame_seqlen
        head_dim = cfg.dim // cfg.num_heads
        local_num_heads = cfg.num_heads // self.sp_size

        if self.local_attn_size > -1:
            kv_size = frame_seqlen * self.local_attn_size
        else:
            kv_size = frame_seqlen * lat_f

        transformer_dtype = self.pipe_dtype
        # generate() folds the VAE spatial stride into the Plücker channel
        # dim via rearrange 'f (h s1) (w s2) c -> (f h w) (c s1 s2)' with
        # s1=s2=vae_stride[1]=8, so control_dim=6 → 6 * 8 * 8 = 384.
        plucker_channels = 6 * cfg.vae_stride[1] * cfg.vae_stride[2]
        # T5 (umt5-xxl) hidden size; cross-attn projects t5_hidden → cfg.dim.
        t5_hidden = 4096

        warmup_self_kv = self._initialize_self_kv_cache(
            num_layers=cfg.num_layers,
            shape=[1, kv_size, local_num_heads, head_dim],
            dtype=transformer_dtype,
            device=self.device)
        warmup_cross_kv = self._initialize_crossattn_cache(
            num_layers=cfg.num_layers,
            shape=[1, text_seq_len, cfg.num_heads, head_dim],
            dtype=transformer_dtype,
            device=self.device)

        # `y` is concat([msk_4ch, vae_latent_16ch]) → 20 channels; combined
        # with latent's 16 ch at patch-embed concat, the DiT sees 36 ch in.
        dummy_latent = torch.zeros(
            16, chunk_size, lat_h, lat_w,
            device=self.device, dtype=torch.float32)
        dummy_y = torch.zeros(
            20, chunk_size, lat_h, lat_w,
            device=self.device, dtype=transformer_dtype)
        dummy_c2ws = torch.zeros(
            1, plucker_channels, chunk_size, lat_h, lat_w,
            device=self.device, dtype=self.param_dtype)
        dummy_context = torch.zeros(
            text_seq_len, t5_hidden,
            device=self.device, dtype=self.param_dtype)
        dummy_t = torch.tensor(
            [500.0], device=self.device, dtype=torch.float32)

        @contextmanager
        def _noop_no_sync():
            yield
        no_sync_model = getattr(self.model, 'no_sync', _noop_no_sync)

        if dist.is_initialized():
            device_synchronize(self.device)
            dist.barrier()
        t0 = time.perf_counter()

        with device_autocast(self.device, dtype=self.param_dtype), \
             torch.no_grad(), \
             no_sync_model():
            _ = self.model(
                x=[dummy_latent],
                t=dummy_t,
                context=[dummy_context],
                seq_len=max_seq_len,
                y=[dummy_y],
                dit_cond_dict={"c2ws_plucker_emb": (dummy_c2ws,)},
                kv_cache=warmup_self_kv,
                crossattn_cache=warmup_cross_kv,
                current_start=0,
                max_attention_size=kv_size,
                frame_seqlen=frame_seqlen,
            )

        if dist.is_initialized():
            device_synchronize(self.device)
            dist.barrier()

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logging.info(f"WanI2VCausal.prewarm: {dt_ms:.0f} ms")

        del (warmup_self_kv, warmup_cross_kv, dummy_latent, dummy_y,
             dummy_c2ws, dummy_context, dummy_t)
        device_empty_cache(self.device)
        self._warmed = True

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_causal, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward_causal, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            # MPS: streaming loader already placed params on target device/dtype.
            # Skip model.to() calls which would create a CPU copy and lose MPS placement.
            if self.device.type != "mps":
                if convert_model_dtype:
                    model.to(self.param_dtype)
                if not self.init_on_cpu:
                    model.to(self.device)

        return model

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, F, H, W]
        xt: the input noisy data with shape [B, C, F, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        # MPS does not support float64; use float32 on MPS, float64 elsewhere
        high_prec_dtype = torch.float32 if flow_pred.device.type == 'mps' else torch.float64
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.to(dtype=high_prec_dtype, device=flow_pred.device), [flow_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred

        return x0_pred.to(original_dtype)


    def generate(self,
                 input_prompt,
                 img,
                 action_path,
                 chunk_size=3,
                 max_area=480 * 832,
                 frame_num=81,
                 timesteps_index=[0, 250, 500, 750],
                 shift=5.0,
                 seed=-1,
                 offload_model=True,
                 max_sequence_length=512,
                 max_attention_size=None,
                 stage="full",
                 image_condition_file=None,
                 dump_image_condition=None,
                 latents_file=None,
                 output_latents_file=None,):
        r"""
        Generates video frames from input image and text prompt.

        Dispatches to the mode-specific implementation according to
        `self.infer_mode`:
            - "causal_fast": distilled few-step sampling (`_generate_causal_fast`)
            - "causal_pretrain": 40-step CFG sampling (`_generate_causal_pretrain`)
        """
        gen_fn = (self._generate_causal_fast
                  if self.infer_mode == "causal_fast"
                  else self._generate_causal_pretrain)

        # M3.5: In sequential mode, ensure required models are loaded before
        # generation. T5 is loaded lazily inside _get_text_context if needed.
        if self.sequential_load:
            # Load only the models needed for this stage
            if stage in ("encode-image", "decode", "full"):
                if self.vae is None:
                    self.load_vae()
            if stage in ("generate-latents", "full"):
                if self.model is None:
                    self.load_dit()

        return gen_fn(
            input_prompt,
            img,
            action_path,
            chunk_size=chunk_size,
            max_area=max_area,
            frame_num=frame_num,
            timesteps_index=timesteps_index,
            shift=shift,
            seed=seed,
            offload_model=offload_model,
            max_sequence_length=max_sequence_length,
            max_attention_size=max_attention_size,
            stage=stage,
            image_condition_file=image_condition_file,
            dump_image_condition=dump_image_condition,
            latents_file=latents_file,
            output_latents_file=output_latents_file)

    def _generate_causal_fast(self,
                              input_prompt,
                              img,
                              action_path,
                              chunk_size=3,
                              max_area=480 * 832,
                              frame_num=81,
                              timesteps_index=[0, 179, 358, 679],
                              shift=5.0,
                              seed=-1,
                              offload_model=True,
                              max_sequence_length=512,
                              max_attention_size=None,
                              stage="full",
                              image_condition_file=None,
                              dump_image_condition=None,
                              latents_file=None,
                              output_latents_file=None,):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """

        if input_prompt is not None and isinstance(input_prompt, str):
            batch_size = 1
        elif input_prompt is not None and isinstance(input_prompt, list):
            batch_size = len(input_prompt)
        else:
            batch_size = 1
        
        assert action_path is not None, "action_path is required"

        # M3.6: Stage validation. In generate-latents stage, VAE must not
        # be loaded; in decode stage, DiT must not be loaded.
        if stage == "generate-latents":
            assert image_condition_file is not None, \
                "--stage generate-latents requires --image_condition_file"
            if self.sequential_load:
                assert self.vae is None, "VAE must not be loaded in generate-latents stage"
        elif stage == "decode":
            assert latents_file is not None, \
                "--stage decode requires --latents_file"
            if self.sequential_load:
                assert self.model is None, "DiT must not be loaded in decode stage"

        c2ws = np.load(os.path.join(action_path, "poses.npy")) # opencv coordinate
        len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
        frame_num = ((frame_num - 1) // 4) * 4 + 1
        frame_num = min(frame_num, len_c2ws)
        c2ws = c2ws[:frame_num]

        # preprocess
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        # Reset per-generate state: cross-attn K/V cache will be freshly
        # initialized below; the first DiT forward must compute and store.
        self._cross_attn_initialized = False

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            lat_f,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, shift=shift)
        timesteps = self.scheduler.timesteps[timesteps_index]

        # preprocess
        # T5 cache: skip the encoder entirely if we've seen this exact prompt
        # before in this pipe instance. Bit-identical: cached tensor is the
        # same object returned by the prior call.
        cache_key = hashlib.sha256(input_prompt.encode('utf-8')).hexdigest()
        if cache_key in self._t5_cache:
            context = self._t5_cache[cache_key]
        elif stage in ("encode-image", "decode"):
            # M3.6: encode-image and decode stages don't need T5
            context = [torch.zeros(1, 1, 4096, device=self.device)]
        elif self.text_encoder is None and self.prompt_embeds_file is not None:
            # Load pre-computed prompt embedding, skip T5 entirely
            from wan.utils.prompt_embedding import load_prompt_embedding
            context_tensor, _ = load_prompt_embedding(
                self.prompt_embeds_file,
                expected_prompt=input_prompt,
                expected_hidden_dim=getattr(self.config, 'text_dim', 4096),
                max_text_len=getattr(self.config, 'text_len', 512),
            )
            context = [context_tensor.to(self.device)]
            self._t5_cache[cache_key] = context
        else:
            # M3.4: In sequential mode, load T5 on demand if not already loaded
            if self.sequential_load and self.text_encoder is None:
                self.load_text_encoder()
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
            self._t5_cache[cache_key] = context

        Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()

        # The provided intrinsics are for original image size (480p). We need to transform them according to the new image size (h, w).
        Ks = get_Ks_transformed(Ks,
                                height_org=480,
                                width_org=832,
                                height_resize=h,
                                width_resize=w,
                                height_final=h,
                                width_final=w)
        Ks = Ks[0]

        len_c2ws = len(c2ws)
        len_c2ws_ = int((len_c2ws - 1) // 4) + 1
        len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks = Ks.repeat(len(c2ws_infer), 1)

        c2ws_infer = c2ws_infer.to(self.device)
        Ks = Ks.to(self.device)
        wasd_action = None
        c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(h // lat_h),
            c2=int(w // lat_w),
        )
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...] # [b, f*h*w, c]
        c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
        if wasd_action is not None:
            wasd_action_tensor = wasd_action[:, None, None, :].repeat(1, h, w, 1) # [f, h, w, 3]
            wasd_action_tensor = rearrange(
                wasd_action_tensor,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            wasd_action_tensor = wasd_action_tensor[None, ...] # [b, f*h*w, c]
            wasd_action_tensor = rearrange(wasd_action_tensor, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
            c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_action_tensor], dim=1)

        # M3.6: In generate-latents stage, load image condition from cache
        # and skip VAE encode entirely.
        if stage == "generate-latents" and image_condition_file is not None:
            from wan.utils.staged_cache import load_image_condition, ImageConditionMetadata
            logging.info(f"Loading image condition from cache: {image_condition_file}")
            expected_meta = ImageConditionMetadata(
                aligned_frame_num=F,
                chunk_size=chunk_size,
                h=h, w=w,
                lat_f=lat_f, lat_h=lat_h, lat_w=lat_w,
                vae_stride=tuple(self.vae_stride),
                patch_size=tuple(self.patch_size),
            )
            y, loaded_meta = load_image_condition(image_condition_file, expected_meta)
            y = y.to(self.device)
            logging.info(f"Image condition loaded: shape={y.shape}, dtype={y.dtype}")
        else:
            y = self.vae.encode([
                torch.concat([
                    torch.nn.functional.interpolate(
                        img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, F - 1, h, w)
                ],
                             dim=1).to(self.device)
            ])[0]
            y = torch.concat([msk, y])

            # M3.6: In encode-image stage, save image condition and return
            if stage == "encode-image":
                if dump_image_condition:
                    from wan.utils.staged_cache import save_image_condition, ImageConditionMetadata, sha256_file
                    meta = ImageConditionMetadata(
                        source_image_sha256="",  # img is tensor, not file
                        requested_frame_num=frame_num,
                        aligned_frame_num=F,
                        chunk_size=chunk_size,
                        h=h, w=w,
                        lat_f=lat_f, lat_h=lat_h, lat_w=lat_w,
                        vae_stride=tuple(self.vae_stride),
                        patch_size=tuple(self.patch_size),
                        dtype=str(y.dtype),
                    )
                    save_image_condition(dump_image_condition, y, meta)
                    logging.info(f"Image condition saved to: {dump_image_condition}")

                # Unload VAE and return
                if self.sequential_load:
                    self.unload_vae()
                logging.info("encode-image stage complete")
                return None

        # M3.5: In sequential mode, unload VAE after image encode to free
        # ~0.5GB before DiT generation. VAE will be reloaded before decode.
        if self.sequential_load and stage != "generate-latents":
            logging.info("Sequential mode: unloading VAE after image encode")
            self.unload_vae()

        # M3.6: In decode stage, load latents from cache and skip DiT entirely
        if stage == "decode" and latents_file is not None:
            from wan.utils.staged_cache import load_generated_latents, GeneratedLatentsMetadata
            logging.info(f"Loading generated latents from cache: {latents_file}")
            # Match the metadata fields set during generate-latents stage
            expected_meta = GeneratedLatentsMetadata(
                checkpoint_id=str(getattr(self.config, 'fast_checkpoint', '')),
                aligned_frame_num=F,
                lat_f=lat_f, lat_h=lat_h, lat_w=lat_w,
                dtype=str(torch.float32),  # MPS uses FP32 for DiT
            )
            pred_latent_chunks, loaded_meta = load_generated_latents(latents_file, expected_meta)
            pred_latent_chunks = pred_latent_chunks.to(self.device)
            logging.info(f"Latents loaded: shape={pred_latent_chunks.shape}, dtype={pred_latent_chunks.dtype}")

            # Force memory cleanup before VAE decode (ensure DiT is not loaded)
            if self.sequential_load:
                if self.model is not None:
                    logging.warning("DiT model still loaded during decode stage, unloading...")
                    self.unload_dit()
                import gc
                gc.collect()
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
                    torch.mps.synchronize()
                logging.info(f"Memory after cleanup: MPS allocated={torch.mps.current_allocated_memory()/1e9:.2f}GB" if self.device.type == 'mps' else "")

            # Decode stage: load VAE, decode, unload VAE, return
            if self.sequential_load and self.vae is None:
                self.load_vae()
            videos = self.vae.decode([pred_latent_chunks])
            if self.sequential_load:
                self.unload_vae()
            logging.info("decode stage complete")
            return videos[0] if self.rank == 0 else None

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)

        # Initialize KV cache to all zeros
        model_args = self.model.config
        transformer_dtype = self.pipe_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1]// 4)
        if self.local_attn_size > -1:
            kv_size = frame_seqlen * self.local_attn_size
        else:
            kv_size = frame_seqlen * lat_f
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size
        self_kv_shape = [batch_size, kv_size, local_num_heads, head_dim]
        self_kv_cache = self._initialize_self_kv_cache(num_layers=model_args.num_layers,
                                                       shape=self_kv_shape,
                                                       dtype=transformer_dtype,
                                                       device=self.device)
        cross_kv_shape = [batch_size, max_sequence_length, model_args.num_heads, head_dim]
        cross_kv_cache = self._initialize_crossattn_cache(num_layers=model_args.num_layers,
                                                          shape=cross_kv_shape,
                                                          dtype=transformer_dtype,
                                                          device=self.device)
        # evaluation mode
        with (
                device_autocast(self.device, dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_model(),
        ):
            # sample videos
            latent = noise
            latents_chunk = latent.split(chunk_size, dim=1) # [c, f, h, w]
            condition_chunk = y.split(chunk_size, dim=1)
            c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []
            for chunk_id in tqdm(range(num_inference_chunk)):
                current_latent = latents_chunk[chunk_id]
                current_condition = condition_chunk[chunk_id]
                current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]

                dit_cond_dict = {
                    "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
                }

                kwargs = {
                    'context': [context[0]],
                    'seq_len': max_seq_len,
                    'y': [current_condition],
                    'dit_cond_dict': dit_cond_dict,
                    'kv_cache': self_kv_cache,
                    'crossattn_cache': cross_kv_cache,
                    'current_start': chunk_id * chunk_size * frame_seqlen,
                    'max_attention_size': kv_size if max_attention_size is None else max_attention_size,
                    'frame_seqlen': frame_seqlen,
                }

                if offload_model:
                    device_empty_cache(self.device)

                for timestep_idx in range(len(timesteps)):
                    latent_model_input = [current_latent.to(self.device)]
                    current_timestep = [timesteps[timestep_idx]]

                    timestep = torch.stack(current_timestep).to(self.device)

                    noise_pred = self.model(
                        x=latent_model_input, t=timestep,
                        cross_attn_first_call=not self._cross_attn_initialized,
                        **kwargs)[0]
                    self._cross_attn_initialized = True

                    if offload_model:
                        device_empty_cache(self.device)

                    x0 = self._convert_flow_pred_to_x0(
                        flow_pred=noise_pred,
                        xt=current_latent,
                        timestep=current_timestep[0],
                        scheduler=self.scheduler,
                    )

                    if timestep_idx < len(timesteps) - 1:
                        next_timestep = timesteps[timestep_idx + 1]
                        current_latent = self.scheduler.add_noise(x0, torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype), next_timestep)
                    else:
                        # note return x0
                        break

                pred_latent_chunks.append(x0)

                # Update kv cache
                context_timestep = [timesteps[-1] * 0.0]
                timestep = torch.stack(context_timestep).to(self.device)
                self.model(x=[x0], t=timestep,
                           cross_attn_first_call=False,
                           **kwargs)

            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)

            # M3.6: In generate-latents stage, save latents and return
            if stage == "generate-latents":
                if output_latents_file:
                    from wan.utils.staged_cache import save_generated_latents, GeneratedLatentsMetadata
                    meta = GeneratedLatentsMetadata(
                        checkpoint_id=str(getattr(self.config, 'fast_checkpoint', '')),
                        seed=seed,
                        requested_frame_num=frame_num,
                        aligned_frame_num=F,
                        chunk_size=chunk_size,
                        h=h, w=w,
                        lat_f=lat_f, lat_h=lat_h, lat_w=lat_w,
                        dtype=str(pred_latent_chunks.dtype),
                    )
                    save_generated_latents(output_latents_file, pred_latent_chunks, meta)
                    logging.info(f"Generated latents saved to: {output_latents_file}")

                # Unload DiT and return
                if hasattr(self.model, 'selfattn_cache'):
                    del self.model.selfattn_cache
                if hasattr(self.model, 'crossattn_cache'):
                    del self.model.crossattn_cache
                if self.sequential_load:
                    self.unload_dit()
                logging.info("generate-latents stage complete")
                return None

            if self.sequential_load:
                # M3.5: Fully unload DiT (not just .cpu()) to free ~3.4GB
                # before VAE decode. Also release KV caches.
                logging.info("Sequential mode: fully unloading DiT before VAE decode")
                if hasattr(self.model, 'selfattn_cache'):
                    del self.model.selfattn_cache
                if hasattr(self.model, 'crossattn_cache'):
                    del self.model.crossattn_cache
                self.unload_dit()
            elif offload_model:
                self.model.cpu()
                device_empty_cache(self.device)

            if self.rank == 0:
                # M3.5: Reload VAE if it was unloaded in sequential mode
                if self.sequential_load and self.vae is None:
                    logging.info("Sequential mode: reloading VAE for decode")
                    self.load_vae()
                videos = self.vae.decode([pred_latent_chunks])

        # del noise, latent, x0
        # del sample_scheduler
        if offload_model:
            gc.collect()
            device_synchronize(self.device)
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def _generate_causal_pretrain(self,
                                  input_prompt,
                                  img,
                                  action_path,
                                  chunk_size=3,
                                  max_area=480 * 832,
                                  frame_num=81,
                                  timesteps_index=None,
                                  shift=5.0,
                                  seed=-1,
                                  offload_model=True,
                                  max_sequence_length=512,
                                  max_attention_size=None,):
        r"""
        Generates video frames with the pretrained causal model using
        40-step CFG sampling per chunk. `timesteps_index` is unused in this
        mode (kept for signature compatibility with `generate`).
        """
        guide_scale = 5.0
        n_prompt = "画面突变，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

        if input_prompt is not None and isinstance(input_prompt, list):
            batch_size = len(input_prompt)
        else:
            batch_size = 1

        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy"))  # opencv coordinate
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = ((frame_num - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]

        # preprocess
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, lat_f, lat_h, lat_w,
            dtype=torch.float32, generator=seed_g, device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # 2. Prepare timesteps (scheduler object created once, state reset per chunk)
        sample_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)

        # preprocess text: cond + uncond
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context      = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt],     self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context      = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt],     torch.device('cpu'))
            context      = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # cam preparation (only if action_path is provided)
        c2ws_plucker_emb = None
        if action_path is not None:
            Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
            Ks = get_Ks_transformed(Ks,
                                    height_org=480, width_org=832,
                                    height_resize=h, width_resize=w,
                                    height_final=h, width_final=w)
            Ks = Ks[0]

            len_c2ws = len(c2ws)
            len_c2ws_ = int((len_c2ws - 1) // 4) + 1
            len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h), c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...]  # [b, f*h*w, c]
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb, 'b (f h w) c -> b c f h w',
                f=lat_f, h=lat_h, w=lat_w,
            ).to(self.param_dtype)

        y = self.vae.encode([
            torch.concat(
                [
                    torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                    torch.zeros(3, F - 1, h, w)
                ], 
                dim=1,
            ).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)

        model_args = self.model.config
        transformer_dtype = self.pipe_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1] // 4)
        kv_size = frame_seqlen * lat_f
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size
        self_kv_shape = [batch_size, kv_size, local_num_heads, head_dim]
        cross_kv_shape = [batch_size, max_sequence_length, model_args.num_heads, head_dim]

        # CFG requires separate caches for the cond / uncond streams.
        self_kv_cache_cond = self._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=self_kv_shape,
            dtype=transformer_dtype,
            device=self.device)
        self_kv_cache_uncond = self._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=self_kv_shape,
            dtype=transformer_dtype,
            device=self.device)
        cross_kv_cache_cond = self._initialize_crossattn_cache_pretrain(
            num_layers=model_args.num_layers,
            shape=cross_kv_shape,
            dtype=transformer_dtype,
            device=self.device)
        cross_kv_cache_uncond = self._initialize_crossattn_cache_pretrain(
            num_layers=model_args.num_layers,
            shape=cross_kv_shape,
            dtype=transformer_dtype,
            device=self.device)

        with (
                device_autocast(self.device, dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_model(),
        ):
            latent = noise
            latents_chunk = latent.split(chunk_size, dim=1)          # [c, f, h, w]
            condition_chunk = y.split(chunk_size, dim=1)
            c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []

            for chunk_id in tqdm(range(num_inference_chunk)):
                # Reset the multi-step scheduler state for each chunk
                sample_scheduler.set_timesteps(40, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps

                current_latent = latents_chunk[chunk_id]
                current_condition = condition_chunk[chunk_id]
                current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]
                dit_cond_dict = {
                    "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
                }

                common = {
                    'seq_len': max_seq_len,
                    'y': [current_condition],
                    'dit_cond_dict': dit_cond_dict,          # camera condition is kept for uncond as well
                    'current_start': chunk_id * chunk_size * frame_seqlen,
                    'max_attention_size': kv_size if max_attention_size is None else max_attention_size,
                }
                kwargs_cond = {
                    **common,
                    'context': [context[0]],
                    'kv_cache': self_kv_cache_cond,
                    'crossattn_cache': cross_kv_cache_cond,
                }
                kwargs_uncond = {
                    **common,
                    'context': context_null,
                    'kv_cache': self_kv_cache_uncond,
                    'crossattn_cache': cross_kv_cache_uncond,
                }

                if offload_model:
                    device_empty_cache(self.device)

                for timestep_idx in tqdm(range(len(timesteps)), desc=f"infer chunk {chunk_id}"):
                    latent_model_input = [current_latent.to(self.device)]
                    t = timesteps[timestep_idx]
                    timestep = torch.stack([t]).to(self.device)

                    noise_pred_cond = self.model(
                        x=latent_model_input, t=timestep, **kwargs_cond)[0]
                    noise_pred_uncond = self.model(
                        x=latent_model_input, t=timestep, **kwargs_uncond)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    if offload_model:
                        device_empty_cache(self.device)

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0), t,
                        current_latent.unsqueeze(0),
                        return_dict=False, generator=seed_g)[0]
                    current_latent = temp_x0.squeeze(0)

                    del latent_model_input, timestep

                pred_latent_chunks.append(current_latent)

                # Update both self KV caches with the clean latent (once for cond, once for uncond)
                timestep0 = torch.stack([timesteps[-1] * 0.0]).to(self.device)
                self.model(x=[current_latent], t=timestep0, **kwargs_cond)
                self.model(x=[current_latent], t=timestep0, **kwargs_uncond)

            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)

            if offload_model:
                self.model.cpu()
                device_empty_cache(self.device)

            if self.rank == 0:
                videos = self.vae.decode([pred_latent_chunks])

        if offload_model:
            gc.collect()
            device_synchronize(self.device)
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def _initialize_self_kv_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a Per-GPU KV cache for the SelfAttn.
        """
        self_kv_cache = []
        for _ in range(num_layers):
            self_kv_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
                'local_end_index': torch.tensor([0], dtype=torch.long, device=device)
            })

        return self_kv_cache


    def _initialize_crossattn_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a per-GPU cross-attention cache.
        """
        crossattn_cache = []
        for _ in range(num_layers):
            crossattn_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'is_init': torch.tensor(0, dtype=torch.int32, device=device),
            })

        return crossattn_cache

    def _initialize_crossattn_cache_pretrain(self, num_layers, shape, dtype, device):
        """
        Initialize a per-GPU cross-attention cache for the pretrained causal
        model, which expects `is_init` to be a plain Python bool.
        """
        crossattn_cache = []
        for _ in range(num_layers):
            crossattn_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'is_init': False,
            })

        return crossattn_cache