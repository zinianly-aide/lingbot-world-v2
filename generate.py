import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video, str2bool
from world_condition import (
    MiniCPMVPerceiver,
    WorldDescription,
    compose_world_prompt,
    load_world_condition,
    save_world_condition,
    save_world_prompt,
)


_I2V_EXAMPLE = {
    "prompt":
        "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder.",
    "image":
        "examples/04/image.jpg",
}

EXAMPLE_PROMPT = {
    "i2v-A14B": _I2V_EXAMPLE,
    "i2v-1.3B": _I2V_EXAMPLE,
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task.startswith("i2v"):
        assert args.image is not None, "Please specify the image path for i2v."

    if args.vlm_world_prompt or args.world_condition_file:
        if args.vlm_image is None:
            args.vlm_image = args.image
        if args.world_condition_file is None:
            assert args.vlm_image is not None, (
                "VLM world prompting requires --vlm_image (or the i2v --image)."
            )

    cfg = WAN_CONFIGS[args.task]

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--infer_mode",
        type=str,
        default="causal_fast",
        choices=["causal_fast", "causal_pretrain"],
        help="Inference mode: 'causal_fast' for the distilled few-step model, "
             "'causal_pretrain' for the pretrained causal model with 40-step CFG sampling.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4,
        help="The chunk size."),
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--assets_dir",
        type=str,
        default=None,
        help="Optional directory that holds shared T5 / VAE / tokenizer assets "
             "(used when the DiT checkpoint folder does not include them).")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="The camera path to generate the video from.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--local_attn_size",
        type=int,
        default=-1,
        help='The local size of kv cache during inference')
    parser.add_argument(
        "--sink_size",
        type=int,
        default=0,
        help='The sink size of kv cache during inference')
    parser.add_argument(
        "--max_attention_size",
        type=int,
        default=None,
        help="The size of kv cache during inference.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default='output',
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--vlm_world_prompt",
        action="store_true",
        default=False,
        help="Use MiniCPM-V to build an observation-only world prompt before UMT5.")
    parser.add_argument(
        "--vlm_backend",
        type=str,
        default="transformers",
        choices=["transformers", "mlx"],
        help="VLM perception backend. 'mlx' uses 4-bit quantised MiniCPM-V via mlx-vlm (Apple Silicon recommended).")
    parser.add_argument(
        "--vlm_model",
        type=str,
        default=None,
        help="MiniCPM-V checkpoint used only during world-prompt construction. "
             "Defaults to openbmb/MiniCPM-V-4.6 (transformers) or mlx-community/MiniCPM-V-4.6-4bit (mlx).")
    parser.add_argument(
        "--vlm_device",
        type=str,
        default="auto",
        help="VLM device ('auto', 'cpu', or a CUDA device); default is auto.")
    parser.add_argument(
        "--vlm_image",
        type=str,
        default=None,
        help="Image for VLM perception; defaults to --image.")
    parser.add_argument(
        "--world_condition_file",
        type=str,
        default=None,
        help="Cached world_condition.json; skips VLM loading when supplied.")
    parser.add_argument(
        "--dump_world_prompt",
        action="store_true",
        default=False,
        help="Save world_condition.json and world_prompt.txt under --save_dir.")

    args = parser.parse_args()
    _validate_args(args)

    return args


def _prepare_world_prompt(args, original_prompt):
    """Best-effort VLM stage.  Any failure returns the original prompt."""
    if not (args.vlm_world_prompt or args.world_condition_file):
        return original_prompt, None

    world = WorldDescription()
    error = None
    if args.world_condition_file:
        try:
            world = load_world_condition(args.world_condition_file)
            logging.info("Loaded cached world condition: %s", args.world_condition_file)
        except Exception as exc:
            error = f"cache load failed: {type(exc).__name__}: {exc}"
    else:
        # Auto-select default model per backend if user did not override.
        vlm_model = args.vlm_model or {
            "transformers": "openbmb/MiniCPM-V-4.6",
            "mlx": "mlx-community/MiniCPM-V-4.6-4bit",
        }.get(args.vlm_backend, "openbmb/MiniCPM-V-4.6")
        perceiver = MiniCPMVPerceiver(
            model_name=vlm_model,
            device=args.vlm_device,
            backend=args.vlm_backend,
        )
        try:
            image = Image.open(args.vlm_image).convert("RGB")
            result = perceiver.analyze(image, user_prompt=original_prompt)
            world, error = result.world, result.error
            if result.raw_text:
                logging.info("MiniCPM-V world observation received (%d chars).", len(result.raw_text))
        except Exception as exc:
            error = f"VLM stage failed: {type(exc).__name__}: {exc}"
        finally:
            perceiver.release()

    if error:
        logging.warning("World prompt conditioning unavailable (%s); using original prompt.", error)
        if args.dump_world_prompt:
            save_world_condition(world, os.path.join(args.save_dir, "world_condition.json"))
            save_world_prompt(original_prompt, os.path.join(args.save_dir, "world_prompt.txt"))
        return original_prompt, world

    composed = compose_world_prompt(world, original_prompt)
    if args.dump_world_prompt:
        save_world_condition(world, os.path.join(args.save_dir, "world_condition.json"))
        save_world_prompt(composed, os.path.join(args.save_dir, "world_prompt.txt"))
        logging.info("Saved world condition and prompt under %s.", args.save_dir)
    return composed, world


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def run_causal(args, cfg, img, device, rank, mode="causal_fast"):
    """Run inference with the distilled few-step model (infer_mode='causal_fast')."""
    logging.info(f"Creating WanI2VCausal pipeline (infer_mode={mode}).")
    wan_i2v = wan.WanI2VCausal(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        infer_mode=mode,
        assets_dir=args.assets_dir,
    )
    logging.info("Generating video ...")
    return wan_i2v.generate(
        args.prompt,
        img,
        action_path=args.action_path,
        chunk_size=args.chunk_size,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        seed=args.base_seed,
        offload_model=args.offload_model,
        max_attention_size=args.max_attention_size)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    cfg = WAN_CONFIGS[args.task]

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), "sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, "The number of ulysses_size should be equal to the world size."
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
        init_distributed_group()

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # Only rank zero loads MiniCPM-V. Broadcast the resulting text so all
    # workers feed exactly the same string into the unchanged UMT5 interface.
    original_prompt = args.prompt
    if args.vlm_world_prompt or args.world_condition_file:
        if dist.is_initialized():
            payload = [None]
            if rank == 0:
                payload[0] = _prepare_world_prompt(args, original_prompt)[0]
            dist.broadcast_object_list(payload, src=0)
            args.prompt = payload[0]
        else:
            args.prompt = _prepare_world_prompt(args, original_prompt)[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    video = run_causal(args, cfg, img, device, rank, mode=args.infer_mode)

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = '.mp4'
            args.save_file = f"lingbot-world-v2-{args.infer_mode}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix
            args.save_file = f'{args.save_dir}/{args.save_file}'

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))

    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
