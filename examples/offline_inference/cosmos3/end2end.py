# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

DEFAULT_NEGATIVE_PROMPT = "blurry, distorted, low quality"
TASK_DEFAULTS = {
    "t2i": {
        "height": 1024,
        "width": 1024,
        "num_frames": None,
        "num_inference_steps": 50,
        "guidance_scale": 7.0,
        "fps": 24,
        "output": "cosmos3_t2i.png",
    },
    "t2v": {
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 35,
        "guidance_scale": 4.0,
        "fps": 24,
        "output": "cosmos3_t2v.mp4",
    },
    "i2v": {
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 35,
        "guidance_scale": 4.0,
        "fps": 24,
        "output": "cosmos3_i2v.mp4",
    },
    "t2v_sound": {
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 35,
        "guidance_scale": 4.0,
        "fps": 24,
        "output": "cosmos3_t2v_sound.mp4",
    },
    "action_policy": {
        "height": 480,
        "width": 640,
        "num_frames": 17,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "fps": 24,
        "output": "cosmos3_action_policy.mp4",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosmos3 offline inference examples.")
    parser.add_argument(
        "--model",
        default=os.environ.get("COSMOS3_MODEL"),
        help="Local Diffusers-format Cosmos3 checkpoint. Defaults to COSMOS3_MODEL.",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_DEFAULTS),
        default="t2v",
        help="Cosmos3 example task to run.",
    )
    parser.add_argument(
        "--prompt",
        default="A small warehouse robot moves a blue box across a clean floor.",
        help="Text prompt.",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument("--image", default=None, help="Input image for i2v or action_policy.")
    parser.add_argument("--output", default=None, help="Output PNG or MP4 path. Default depends on --task.")
    parser.add_argument(
        "--action-output",
        default=None,
        help="Action JSON path for action_policy. Defaults to the video output stem plus _action.json.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=None, help="Output height. Default depends on --task.")
    parser.add_argument("--width", type=int, default=None, help="Output width. Default depends on --task.")
    parser.add_argument("--num-frames", type=int, default=None, help="Video frames. Default depends on --task.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Sampling steps. Default depends on --task.",
    )
    parser.add_argument("--guidance-scale", type=float, default=None, help="CFG scale. Default depends on --task.")
    parser.add_argument("--fps", type=int, default=None, help="Output video fps. Default depends on --task.")
    parser.add_argument(
        "--sound-duration",
        type=float,
        default=None,
        help="Audio duration in seconds for t2v_sound. Defaults to generated video duration.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=24000,
        help="Fallback sample rate used when muxing audio if the model does not return one.",
    )
    parser.add_argument(
        "--domain-name",
        default="bridge_orig_lerobot",
        help="Cosmos3 action embodiment name for action_policy.",
    )
    parser.add_argument("--domain-id", type=int, default=None, help="Cosmos3 action embodiment id.")
    parser.add_argument(
        "--raw-action-dim",
        type=int,
        default=2,
        help="Number of action dimensions to keep for action_policy.",
    )
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=16,
        help="Number of action steps for action_policy.",
    )
    parser.add_argument(
        "--cache-backend",
        type=str,
        default=None,
        choices=["cache_dit"],
        help="Cache backend for supported Cosmos3 generation paths.",
    )
    parser.add_argument("--enable-layerwise-offload", action="store_true", help="Enable layerwise offload.")
    parser.add_argument("--vae-use-slicing", action="store_true", help="Enable VAE slicing.")
    parser.add_argument("--vae-use-tiling", action="store_true", help="Enable VAE tiling.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--ulysses-degree", type=int, default=1, help="Ulysses sequence parallel degree.")
    parser.add_argument("--ring-degree", type=int, default=1, help="Ring sequence parallel degree.")
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2], help="CFG parallel size.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1, help="VAE patch parallel size.")
    parser.add_argument("--use-hsdp", action="store_true", help="Enable HSDP.")
    parser.add_argument("--hsdp-shard-size", type=int, default=1, help="HSDP shard size.")
    parser.add_argument("--hsdp-replicate-size", type=int, default=1, help="HSDP replicate size.")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8", "mxfp8", "int8", "gguf"],
        help="Transformer quantization method.",
    )
    return parser.parse_args()


def _cache_config(cache_backend: str | None) -> dict[str, Any] | None:
    if cache_backend != "cache_dit":
        return None
    return {
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "max_cached_steps": 20,
        "residual_diff_threshold": 0.24,
        "max_continuous_cached_steps": 3,
        "enable_taylorseer": False,
        "taylorseer_order": 1,
        "scm_steps_mask_policy": None,
        "scm_steps_policy": "dynamic",
    }


def _first_output(outputs: Any) -> Any:
    if isinstance(outputs, list):
        if not outputs:
            raise ValueError("No output generated.")
        return outputs[0]
    return outputs


def _inner_output(output: Any) -> Any:
    if isinstance(output, OmniRequestOutput) and output.is_pipeline_output and output.request_output is not None:
        return output.request_output
    return output


def _extract_images(outputs: Any) -> list[Any]:
    output = _inner_output(_first_output(outputs))
    if isinstance(output, OmniRequestOutput) and output.images:
        return output.images
    images = getattr(output, "images", None)
    if images:
        return images
    raise ValueError("No images found in output.")


def _extract_video_audio_action(outputs: Any) -> tuple[Any, Any | None, int | None, dict[str, Any]]:
    outer = _first_output(outputs)
    output = _inner_output(outer)
    audio = None
    audio_sample_rate = None
    action = {}

    for candidate in (outer, output):
        if isinstance(candidate, OmniRequestOutput):
            if candidate.multimodal_output:
                audio = audio or candidate.multimodal_output.get("audio")
                audio_sample_rate = audio_sample_rate or candidate.multimodal_output.get("audio_sample_rate")
            if candidate.custom_output:
                action.update(candidate.custom_output)

    videos = None
    if isinstance(output, OmniRequestOutput):
        if output.multimodal_output:
            videos = output.multimodal_output.get("video")
            audio = audio or output.multimodal_output.get("audio")
            audio_sample_rate = audio_sample_rate or output.multimodal_output.get("audio_sample_rate")
        if videos is None and output.images:
            videos = output.images
    else:
        videos = getattr(output, "images", None)
        mm = getattr(output, "multimodal_output", None)
        if mm:
            videos = videos or mm.get("video")
            audio = audio or mm.get("audio")
            audio_sample_rate = audio_sample_rate or mm.get("audio_sample_rate")

    if isinstance(videos, list) and len(videos) == 1:
        first = videos[0]
        if isinstance(first, tuple) and len(first) == 2:
            videos, audio = first
        elif isinstance(first, dict):
            audio = audio or first.get("audio")
            audio_sample_rate = audio_sample_rate or first.get("audio_sample_rate")
            videos = first.get("frames") or first.get("video")
        elif isinstance(first, list):
            videos = first

    if isinstance(videos, tuple) and len(videos) == 2:
        videos, audio = videos
    elif isinstance(videos, dict):
        audio = audio or videos.get("audio")
        audio_sample_rate = audio_sample_rate or videos.get("audio_sample_rate")
        videos = videos.get("frames") or videos.get("video")

    if videos is None:
        raise ValueError("No video frames found in output.")
    return videos, audio, audio_sample_rate, action


def _normalize_frame(frame: Any) -> Any:
    if isinstance(frame, torch.Tensor):
        frame_tensor = frame.detach().cpu()
        if frame_tensor.dim() == 4 and frame_tensor.shape[0] == 1:
            frame_tensor = frame_tensor[0]
        if frame_tensor.dim() == 3 and frame_tensor.shape[0] in (3, 4):
            frame_tensor = frame_tensor.permute(1, 2, 0)
        if frame_tensor.is_floating_point():
            frame_tensor = frame_tensor.clamp(-1, 1) * 0.5 + 0.5
        return frame_tensor.float().numpy()
    if isinstance(frame, np.ndarray):
        frame_array = frame
        if frame_array.ndim == 4 and frame_array.shape[0] == 1:
            frame_array = frame_array[0]
        if np.issubdtype(frame_array.dtype, np.integer):
            frame_array = frame_array.astype(np.float32) / 255.0
        return frame_array
    if isinstance(frame, PIL.Image.Image):
        return np.asarray(frame).astype(np.float32) / 255.0
    return frame


def _ensure_frame_list(video: Any) -> Any:
    if isinstance(video, list):
        if not video:
            return video
        first = video[0]
        if isinstance(first, np.ndarray):
            if first.ndim == 5:
                return list(first[0])
            if first.ndim == 4:
                return list(first)
            if first.ndim == 3:
                return video
        return video
    if isinstance(video, np.ndarray):
        if video.ndim == 5:
            return list(video[0])
        if video.ndim == 4:
            return list(video)
        if video.ndim == 3:
            return [video]
    return video


def _video_to_array(video: Any) -> Any:
    if isinstance(video, torch.Tensor):
        video_tensor = video.detach().cpu()
        if video_tensor.dim() == 5:
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        return video_tensor.float().numpy()
    if isinstance(video, np.ndarray):
        video_array = video
        if video_array.ndim == 5:
            video_array = video_array[0]
        if np.issubdtype(video_array.dtype, np.integer):
            video_array = video_array.astype(np.float32) / 255.0
        return video_array
    if isinstance(video, list):
        if not video:
            raise ValueError("No video frames found in output.")
        return [_normalize_frame(frame) for frame in video]
    return video


def _save_video(video: Any, output_path: Path, fps: int, audio: Any | None, audio_sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_array = _ensure_frame_list(_video_to_array(video))

    if audio is not None:
        from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

        frames_np = np.stack(video_array, axis=0) if isinstance(video_array, list) else np.asarray(video_array)
        if frames_np.ndim == 4 and frames_np.shape[-1] == 4:
            frames_np = frames_np[..., :3]
        frames_u8 = (np.clip(frames_np, 0.0, 1.0) * 255).round().clip(0, 255).astype("uint8")

        audio_np = audio
        if isinstance(audio_np, list):
            audio_np = audio_np[0] if audio_np else None
        if isinstance(audio_np, torch.Tensor):
            audio_np = audio_np.detach().cpu().float().numpy()
        if isinstance(audio_np, np.ndarray):
            audio_np = np.squeeze(audio_np).astype(np.float32)

        video_bytes = mux_video_audio_bytes(
            frames_u8,
            audio_np,
            fps=float(fps),
            audio_sample_rate=audio_sample_rate,
        )
        output_path.write_bytes(video_bytes)
        return

    try:
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise ImportError("diffusers is required for export_to_video.") from exc
    export_to_video(video_array, str(output_path), fps=fps)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _build_omni(args: argparse.Namespace) -> Omni:
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
    )
    kwargs: dict[str, Any] = {
        "model": args.model,
        "model_class_name": "Cosmos3OmniDiffusersPipeline",
        "enable_layerwise_offload": args.enable_layerwise_offload,
        "vae_use_slicing": args.vae_use_slicing,
        "vae_use_tiling": args.vae_use_tiling,
        "enforce_eager": args.enforce_eager,
        "parallel_config": parallel_config,
        "cache_backend": args.cache_backend,
        "cache_config": _cache_config(args.cache_backend),
    }
    if args.quantization is not None:
        kwargs["quantization"] = args.quantization
    return Omni(**kwargs)


def main() -> None:
    args = parse_args()
    if not args.model:
        raise ValueError("Set COSMOS3_MODEL or pass --model with a Cosmos3 Diffusers checkpoint path.")

    defaults = TASK_DEFAULTS[args.task]
    height = args.height or defaults["height"]
    width = args.width or defaults["width"]
    num_frames = args.num_frames if args.num_frames is not None else defaults["num_frames"]
    num_inference_steps = args.num_inference_steps or defaults["num_inference_steps"]
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else defaults["guidance_scale"]
    fps = args.fps or defaults["fps"]
    output_path = Path(args.output or defaults["output"])

    if args.task in {"i2v", "action_policy"} and args.image is None:
        raise ValueError(f"--image is required for {args.task}.")

    image = PIL.Image.open(args.image).convert("RGB") if args.image else None
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    omni = _build_omni(args)

    prompt: dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "modalities": ["image"] if args.task == "t2i" else ["video"],
    }
    if image is not None:
        prompt["multi_modal_data"] = {"image": image}
    if args.task == "t2v_sound":
        prompt["generate_sound"] = True

    extra_args: dict[str, Any] = {}
    if args.task == "t2v_sound":
        extra_args["generate_sound"] = True
        if args.sound_duration is not None:
            extra_args["sound_duration"] = args.sound_duration
    if args.task == "action_policy":
        extra_args.update(
            {
                "action_mode": "policy",
                "action_chunk_size": args.action_chunk_size,
                "raw_action_dim": args.raw_action_dim,
            }
        )
        if args.domain_id is not None:
            extra_args["domain_id"] = args.domain_id
        else:
            extra_args["domain_name"] = args.domain_name

    sampling = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        frame_rate=float(fps),
        extra_args=extra_args,
    )

    print("Cosmos3 generation configuration:")
    print(f"  Task: {args.task}")
    print(f"  Model: {args.model}")
    print(f"  Size: {width}x{height}")
    if num_frames is not None:
        print(f"  Frames: {num_frames}")
    print(f"  Steps: {num_inference_steps}")
    print(f"  Guidance scale: {guidance_scale}")

    start = time.perf_counter()
    outputs = omni.generate(prompt, sampling)
    elapsed = time.perf_counter() - start
    print(f"Total generation time: {elapsed:.4f} seconds")

    if args.task == "t2i":
        images = _extract_images(outputs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(output_path)
        print(f"Saved image to {output_path}")
        return

    video, audio, returned_sample_rate, action = _extract_video_audio_action(outputs)
    _save_video(
        video, output_path, fps=fps, audio=audio, audio_sample_rate=returned_sample_rate or args.audio_sample_rate
    )
    print(f"Saved video to {output_path}")

    if args.task == "action_policy":
        action_path = (
            Path(args.action_output) if args.action_output else output_path.with_name(f"{output_path.stem}_action.json")
        )
        action_path.parent.mkdir(parents=True, exist_ok=True)
        action_path.write_text(json.dumps(_jsonable(action), indent=2) + "\n", encoding="utf-8")
        print(f"Saved action metadata to {action_path}")


if __name__ == "__main__":
    main()
