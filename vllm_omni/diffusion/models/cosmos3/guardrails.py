# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 guardrail hooks for vllm-omni.

Text: Blocklist (keyword matching) + Qwen3Guard (0.6B LLM classifier)
Video: RetinaFace face blur

Enable via custom_pipeline_args or the test script:
    python test_cosmos3.py --model ... --guardrails
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import cv2
import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import GuardrailViolationError
from vllm_omni.diffusion.models.progress_bar import _is_rank_zero

logger = init_logger(__name__)

TextGuardrailFn = Callable[[str], None]
VideoGuardrailFn = Callable[[np.ndarray], np.ndarray]

_text_guardrail: TextGuardrailFn | None = None
_video_guardrail: VideoGuardrailFn | None = None
_initialized = False

GUARDRAIL_HF_REPO = "nvidia/Cosmos-Guardrail1"
GUARDRAIL_HF_REVISION = "d6d4bfa899a71454a700907664f3e88f503950cf"


def set_text_guardrail(fn: TextGuardrailFn) -> None:
    global _text_guardrail
    _text_guardrail = fn


def set_video_guardrail(fn: VideoGuardrailFn) -> None:
    global _video_guardrail
    _video_guardrail = fn


# ---------------------------------------------------------------------------
# Face pixelation utility
# ---------------------------------------------------------------------------
def _pixelate_face(face_img: np.ndarray, blocks: int = 5) -> np.ndarray:
    h, w = face_img.shape[:2]
    if h == 0 or w == 0:
        return face_img
    temp = cv2.resize(face_img, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Default guardrail builders
# ---------------------------------------------------------------------------
def _download_checkpoint() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(GUARDRAIL_HF_REPO, revision=GUARDRAIL_HF_REVISION)


def _move_tokenizer_output_to_device(tokenizer_output: object, device: str) -> object:
    if hasattr(tokenizer_output, "to"):
        return tokenizer_output.to(device)
    if isinstance(tokenizer_output, Mapping):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in tokenizer_output.items()}
    return tokenizer_output


def _qwen_input_length(input_ids: object) -> int:
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if isinstance(input_ids, list | tuple):
        if input_ids and isinstance(input_ids[0], list | tuple):
            return len(input_ids[0])
        return len(input_ids)
    raise TypeError(f"Qwen3Guard tokenizer returned unsupported input_ids type: {type(input_ids).__name__}")


def _generate_qwen_guardrail_response(prompt: str, tokenizer: Any, model: Any, device: str) -> str:
    conversations = [{"role": "user", "content": prompt}]
    model_inputs = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    model_inputs = _move_tokenizer_output_to_device(model_inputs, device)

    if isinstance(model_inputs, torch.Tensor):
        input_ids = model_inputs
        generate_kwargs = {}
        generate_args = (input_ids,)
    elif isinstance(model_inputs, Mapping):
        if "input_ids" not in model_inputs:
            raise TypeError("Qwen3Guard tokenizer output must include input_ids.")
        input_ids = model_inputs["input_ids"]
        generate_kwargs = dict(model_inputs)
        generate_args = ()
    else:
        input_ids = getattr(model_inputs, "input_ids", None)
        if input_ids is None:
            raise TypeError(
                "Qwen3Guard tokenizer must return a tensor or mapping with input_ids; "
                f"got {type(model_inputs).__name__}"
            )
        generate_kwargs = {"input_ids": input_ids}
        generate_args = ()

    input_length = _qwen_input_length(input_ids)
    with torch.no_grad():
        output_ids = model.generate(*generate_args, **generate_kwargs, max_new_tokens=128)
    return tokenizer.decode(
        output_ids[0][input_length:],
        skip_special_tokens=True,
    )


def _build_text_guardrail(offload_to_cpu: bool) -> TextGuardrailFn:
    checkers: list[Callable[[str], tuple[bool, str]]] = []

    # 1. Blocklist
    try:
        import nltk
        from better_profanity import profanity as profanity_filter

        ckpt_dir = _download_checkpoint()
        blocklist_dir = os.path.join(ckpt_dir, "blocklist")
        nltk.data.path.append(os.path.join(blocklist_dir, "nltk_data"))

        def _read_keywords(dirpath: str) -> list[str]:
            words: list[str] = []
            if not os.path.isdir(dirpath):
                return words
            for fname in sorted(os.listdir(dirpath)):
                fpath = os.path.join(dirpath, fname)
                if os.path.isfile(fpath):
                    with open(fpath) as f:
                        words.extend(line.strip() for line in f if line.strip())
            return words

        blocklist_words = _read_keywords(os.path.join(blocklist_dir, "custom"))
        whitelist_words = _read_keywords(os.path.join(blocklist_dir, "whitelist"))
        profanity_filter.load_censor_words(custom_words=blocklist_words, whitelist_words=whitelist_words)

        def _blocklist_check(prompt: str) -> tuple[bool, str]:
            if profanity_filter.contains_profanity(prompt):
                return False, "Blocked by keyword filter"
            return True, ""

        checkers.append(_blocklist_check)
        if _is_rank_zero():
            logger.info("Blocklist guardrail loaded (%d keywords)", len(blocklist_words))
    except ImportError:
        logger.warning("better-profanity or nltk not installed; skipping blocklist guardrail")

    # 2. Qwen3Guard
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "Qwen/Qwen3Guard-Gen-0.6B"
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_id)
        device = "cpu" if offload_to_cpu else "cuda"
        qwen_model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )

        def _qwen_check(prompt: str) -> tuple[bool, str]:
            response = _generate_qwen_guardrail_response(prompt, qwen_tokenizer, qwen_model, device)
            if "unsafe" in response.lower():
                return False, f"Qwen3Guard: {response.strip()}"
            return True, ""

        checkers.append(_qwen_check)
        if _is_rank_zero():
            logger.info("Qwen3Guard guardrail loaded")
    except ImportError:
        logger.warning("transformers not installed; skipping Qwen3Guard")

    def text_guardrail(prompt: str) -> None:
        for checker in checkers:
            is_safe, msg = checker(prompt)
            if not is_safe:
                raise GuardrailViolationError(f"Guardrail blocked prompt: {msg}")

    return text_guardrail


def _build_video_guardrail(offload_to_cpu: bool) -> VideoGuardrailFn:
    ckpt_dir = _download_checkpoint()
    face_blurrer: Callable[[np.ndarray], np.ndarray] | None = None

    # `offload_to_cpu` controls idle weight placement only; the forward pass
    # always runs on `compute_device` and weights are returned to CPU after.
    compute_device = "cuda"
    idle_device = "cpu" if offload_to_cpu else compute_device

    # Face blur: RetinaFace + pixelation
    try:
        from retinaface.data import cfg_re50
        from retinaface.layers.functions.prior_box import PriorBox
        from retinaface.models.retinaface import RetinaFace
        from retinaface.utils.nms.py_cpu_nms import py_cpu_nms

        face_ckpt = os.path.join(ckpt_dir, "face_blur_filter", "Resnet50_Final.pth")
        if not os.path.exists(face_ckpt):
            raise FileNotFoundError(face_ckpt)

        cfg = dict(cfg_re50)
        cfg["pretrain"] = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            retinaface_net = RetinaFace(cfg=cfg, phase="test")

        # Load weights (strip 'module.' prefix if present)
        pretrained_dict = torch.load(face_ckpt, map_location="cpu", weights_only=True)
        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        pretrained_dict = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in pretrained_dict.items()
        }
        retinaface_net.load_state_dict(pretrained_dict, strict=False)
        retinaface_net = retinaface_net.to(idle_device, dtype=torch.float32).eval()

        CONF_THRESH = 0.7
        NMS_THRESH = 0.4
        TOP_K = 5000
        KEEP_TOP_K = 750

        def _decode_batch(loc, priors, variances):
            batch_size = loc.size(0)
            p = priors.unsqueeze(0).expand(batch_size, -1, -1)
            boxes = torch.cat(
                (
                    p[:, :, :2] + loc[:, :, :2] * variances[0] * p[:, :, 2:],
                    p[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),
                ),
                dim=2,
            )
            boxes[:, :, :2] -= boxes[:, :, 2:] / 2
            boxes[:, :, 2:] += boxes[:, :, :2]
            return boxes

        def _face_blur(frames: np.ndarray) -> np.ndarray:
            nonlocal retinaface_net
            if offload_to_cpu:
                retinaface_net = retinaface_net.to(compute_device)

            prior_data = None
            scale = None
            result_frames = []

            try:
                for frame in frames:
                    frame_t = torch.from_numpy(frame).to(compute_device, dtype=torch.float32)
                    frame_t = frame_t.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                    frame_t = frame_t[:, [2, 1, 0], :, :]  # RGB → BGR
                    means = torch.tensor([104.0, 117.0, 123.0], device=compute_device, dtype=torch.float32).view(
                        1, 3, 1, 1
                    )
                    frame_t = frame_t - means

                    h, w = frame_t.shape[2], frame_t.shape[3]
                    if prior_data is None:
                        priorbox = PriorBox(cfg, image_size=(h, w))
                        prior_data = priorbox.forward().to(compute_device, dtype=torch.float32)
                    if scale is None:
                        scale = torch.tensor([w, h, w, h], device=compute_device, dtype=torch.float32)

                    with torch.no_grad():
                        loc, conf, _ = retinaface_net(frame_t)

                    boxes = _decode_batch(loc, prior_data, cfg["variance"])
                    boxes = (boxes * scale).squeeze(0).cpu().numpy()
                    scores = conf.squeeze(0)[:, 1].cpu().numpy()

                    # Filter by confidence
                    inds = np.where(scores > CONF_THRESH)[0]
                    boxes_f = boxes[inds]
                    scores_f = scores[inds]
                    order = scores_f.argsort()[::-1][:TOP_K]
                    boxes_f = boxes_f[order]
                    scores_f = scores_f[order]

                    # NMS
                    dets = np.hstack((boxes_f, scores_f[:, np.newaxis])).astype(np.float32)
                    keep = py_cpu_nms(dets, NMS_THRESH)
                    dets = dets[keep][:KEEP_TOP_K]

                    out_frame = frame.copy()
                    for det in dets:
                        x1, y1, x2, y2 = map(int, det[:4])
                        if x2 - x1 < 20 or y2 - y1 < 20:
                            continue
                        max_h, max_w = out_frame.shape[:2]
                        y1c, y2c = max(y1, 0), min(y2, max_h)
                        x1c, x2c = max(x1, 0), min(x2, max_w)
                        out_frame[y1c:y2c, x1c:x2c] = _pixelate_face(out_frame[y1c:y2c, x1c:x2c])

                    result_frames.append(out_frame)
            finally:
                if offload_to_cpu:
                    retinaface_net = retinaface_net.to("cpu")

            return np.array(result_frames)

        face_blurrer = _face_blur
        if _is_rank_zero():
            logger.info("Face blur filter loaded (RetinaFace Resnet50)")
    except (ImportError, FileNotFoundError) as e:
        logger.warning("Could not load face blur filter: %s", e)

    def video_guardrail(frames: np.ndarray) -> np.ndarray:
        if face_blurrer is not None:
            frames = face_blurrer(frames)
        return frames

    return video_guardrail


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def _init_default_guardrails(offload_to_cpu: bool = False) -> None:
    global _text_guardrail, _video_guardrail, _initialized
    if _initialized:
        return
    if _is_rank_zero():
        logger.info("Initializing Cosmos3 guardrails (offload_to_cpu=%s)...", offload_to_cpu)
    # Build into locals first so a partial failure doesn't leave the module
    # in a half-initialized state (one guardrail set, the other missing,
    # and `_initialized` still False so the next call retries from scratch).
    text_fn = _build_text_guardrail(offload_to_cpu)
    video_fn = _build_video_guardrail(offload_to_cpu)
    _text_guardrail = text_fn
    _video_guardrail = video_fn
    _initialized = True
    if _is_rank_zero():
        logger.info("Cosmos3 guardrails initialized.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ensure_initialized(od_config: Any) -> None:
    if not is_guardrails_enabled(od_config):
        return
    _init_default_guardrails(offload_to_cpu=get_offload_flag(od_config))


def check_text_safety(prompt: str) -> None:
    if _text_guardrail is not None:
        _text_guardrail(prompt)


def check_video_safety(video_tensor: torch.Tensor) -> torch.Tensor:
    if _video_guardrail is None:
        return video_tensor

    v = video_tensor.detach().cpu().float()
    if v.dim() == 5:
        v = v[0]
    v = v.clamp(-1, 1) * 0.5 + 0.5
    frames_np = (v.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)

    frames_np = _video_guardrail(frames_np)

    # Convert back to [-1, 1] to match the VAE output range.
    result = torch.from_numpy(frames_np.copy()).float() / 127.5 - 1.0
    result = result.permute(3, 0, 1, 2)
    if video_tensor.dim() == 5:
        result = result.unsqueeze(0)
    return result.to(video_tensor.device)


def is_guardrails_enabled(od_config: Any, sampling_params: Any = None) -> bool:
    """Resolve the active guardrail gate.

    Server-level ``od_config.model_config["guardrails"]`` decides whether the
    guardrail models are loaded at all (eager load at pipeline build time).
    When that is False, no per-request override can turn checks back on,
    because the singletons in this module are never populated.

    When the server gate is on, ``sampling_params.extra_args["guardrails"]``
    may override on a per-request basis: ``False`` skips the check for that
    request, anything else (or missing) keeps the default behavior.
    """
    cfg = getattr(od_config, "model_config", None) or {}
    if not bool(cfg.get("guardrails", True)):
        return False
    if sampling_params is not None:
        extra = getattr(sampling_params, "extra_args", None) or {}
        per_request = extra.get("guardrails")
        if per_request is not None:
            return bool(per_request)
    return True


def get_offload_flag(od_config: Any) -> bool:
    cfg = getattr(od_config, "model_config", None) or {}
    return bool(cfg.get("offload_guardrail_models", False))
