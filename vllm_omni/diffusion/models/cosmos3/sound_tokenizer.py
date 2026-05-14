# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 sound tokenizer integration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.progress_bar import _is_rank_zero

from .audio_tokenizer import Cosmos3AVAEAudioTokenizer

logger = init_logger(__name__)

DEFAULT_SOUND_SAMPLE_RATE = 48000
DEFAULT_SOUND_CHANNELS = 2
DEFAULT_SOUND_DIM = 64
DEFAULT_SOUND_HOP_SIZE = 1920
DEFAULT_SOUND_LATENT_FPS = DEFAULT_SOUND_SAMPLE_RATE / DEFAULT_SOUND_HOP_SIZE
SOUND_TOKENIZER_COMPONENT_NAME = "sound_tokenizer"
SOUND_TOKENIZER_CHECKPOINT_NAME = "model.safetensors"


def _pipeline_args(od_config: OmniDiffusionConfig) -> dict[str, Any]:
    return dict(getattr(od_config, "custom_pipeline_args", None) or {})


def _resolve_model_file(path: Any, model_root: str | None) -> str | None:
    if not path:
        return None
    path = str(path)
    if "://" in path or os.path.isabs(path) or os.path.exists(path) or not model_root:
        return path
    return str(Path(model_root) / path)


def get_sound_config_value(
    od_config: OmniDiffusionConfig,
    name: str,
    default: Any,
    aliases: tuple[str, ...] = (),
) -> Any:
    keys = (name, *aliases)
    for config in (
        _pipeline_args(od_config),
        getattr(od_config, "model_config", None),
        getattr(od_config, "tf_model_config", None),
    ):
        if config is None:
            continue
        for key in keys:
            if hasattr(config, "get"):
                value = config.get(key, None)
            else:
                value = getattr(config, key, None)
            if value is not None:
                return value
    return default


def get_sound_sample_rate(od_config: OmniDiffusionConfig) -> int:
    return int(
        get_sound_config_value(
            od_config,
            "sound_sample_rate",
            DEFAULT_SOUND_SAMPLE_RATE,
            ("sample_rate",),
        )
    )


def get_sound_channels(od_config: OmniDiffusionConfig) -> int:
    return int(
        get_sound_config_value(
            od_config,
            "sound_audio_channels",
            DEFAULT_SOUND_CHANNELS,
            ("audio_channels",),
        )
    )


def get_sound_dim(od_config: OmniDiffusionConfig | None) -> int:
    if od_config is None:
        return DEFAULT_SOUND_DIM
    return int(
        get_sound_config_value(
            od_config,
            "sound_dim",
            DEFAULT_SOUND_DIM,
            ("io_channels", "latent_ch"),
        )
    )


def get_sound_hop_size(od_config: OmniDiffusionConfig) -> int:
    return int(
        get_sound_config_value(
            od_config,
            "sound_hop_size",
            DEFAULT_SOUND_HOP_SIZE,
            ("hop_size",),
        )
    )


def get_sound_latent_fps(od_config: OmniDiffusionConfig | None) -> float:
    if od_config is None:
        return DEFAULT_SOUND_LATENT_FPS
    sample_rate = get_sound_sample_rate(od_config)
    hop_size = get_sound_hop_size(od_config)
    return float(get_sound_config_value(od_config, "sound_latent_fps", sample_rate / hop_size))


class Cosmos3SoundTokenizer:
    """Thin adapter around the local AVAE tokenizer implementation."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.sample_rate = int(getattr(tokenizer, "sample_rate", DEFAULT_SOUND_SAMPLE_RATE))
        self.audio_channels = int(getattr(tokenizer, "audio_channels", DEFAULT_SOUND_CHANNELS))
        self.latent_ch = int(getattr(tokenizer, "latent_ch", DEFAULT_SOUND_DIM))
        self.hop_size = int(getattr(tokenizer, "temporal_compression_factor", DEFAULT_SOUND_HOP_SIZE))

    @classmethod
    def from_config(cls, od_config: OmniDiffusionConfig) -> Cosmos3SoundTokenizer:
        args = _pipeline_args(od_config)
        model_path = getattr(od_config, "model", None)
        explicit_avae_path = (
            args.get("sound_tokenizer_path")
            or args.get("avae_path")
            or args.get("cosmos3_avae_path")
            or os.environ.get("COSMOS3_SOUND_TOKENIZER_PATH")
        )
        explicit_config_path = args.get("sound_tokenizer_config_path") or os.environ.get(
            "COSMOS3_SOUND_TOKENIZER_CONFIG_PATH"
        )

        model_root = str(model_path) if model_path and os.path.isdir(model_path) else None
        if model_root is None and model_path and not explicit_avae_path:
            from huggingface_hub import snapshot_download

            model_root = snapshot_download(
                repo_id=str(model_path),
                revision=getattr(od_config, "revision", None),
                allow_patterns=[
                    f"{SOUND_TOKENIZER_COMPONENT_NAME}/config.json",
                    f"{SOUND_TOKENIZER_COMPONENT_NAME}/{SOUND_TOKENIZER_CHECKPOINT_NAME}",
                ],
            )

        if explicit_avae_path:
            avae_path = _resolve_model_file(explicit_avae_path, model_root)
        else:
            tokenizer_dir = Path(model_root) / SOUND_TOKENIZER_COMPONENT_NAME if model_root else None
            candidate = tokenizer_dir / SOUND_TOKENIZER_CHECKPOINT_NAME if tokenizer_dir else None
            avae_path = str(candidate) if candidate and candidate.exists() else None

        if not avae_path:
            raise ValueError(
                "Cosmos3 sound generation was requested, but no AVAE sound "
                "tokenizer checkpoint was provided. Set "
                "custom_pipeline_args['sound_tokenizer_path'] or "
                "COSMOS3_SOUND_TOKENIZER_PATH, or include "
                "sound_tokenizer/model.safetensors under the model path."
            )

        sample_rate = get_sound_sample_rate(od_config)
        audio_channels = get_sound_channels(od_config)
        sound_dim = get_sound_dim(od_config)
        hop_size = get_sound_hop_size(od_config)

        config_path = _resolve_model_file(explicit_config_path, model_root)
        if config_path is None and model_root:
            candidate = Path(model_root) / SOUND_TOKENIZER_COMPONENT_NAME / "config.json"
            config_path = str(candidate) if candidate.exists() else None
        tokenizer = Cosmos3AVAEAudioTokenizer(
            checkpoint_path=str(avae_path),
            config_path=config_path,
            sample_rate=sample_rate,
            audio_channels=audio_channels,
            io_channels=sound_dim,
            hop_size=hop_size,
            normalize_latents=bool(args.get("sound_normalize_latents", True)),
            normalization_type=args.get("sound_normalization_type", "none"),
            tanh_input_scale=float(args.get("sound_tanh_input_scale", 1.5)),
            tanh_output_scale=float(args.get("sound_tanh_output_scale", 3.5)),
            tanh_clamp=float(args.get("sound_tanh_clamp", 0.995)),
            dtype=getattr(od_config, "dtype", torch.bfloat16),
            device=get_local_device(),
        )
        if _is_rank_zero():
            logger.info(
                "Loaded Cosmos3 AVAE sound tokenizer from %s (sr=%d, channels=%d, latent_ch=%d, hop=%d)",
                avae_path,
                sample_rate,
                audio_channels,
                sound_dim,
                hop_size,
            )
        return cls(tokenizer)

    def get_latent_num_samples(self, num_audio_samples: int) -> int:
        return int(self.tokenizer.get_latent_num_samples(num_audio_samples))

    def get_audio_num_samples(self, num_latent_samples: int) -> int:
        return int(self.tokenizer.get_audio_num_samples(num_latent_samples))

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode sound latents.

        Args:
            latents: ``[B, C, T]`` or ``[C, T]`` tensor.

        Returns:
            ``[B, audio_channels, N]`` tensor for batched input, or
            ``[audio_channels, N]`` for unbatched input.
        """
        squeeze = latents.ndim == 2
        if squeeze:
            latents = latents.unsqueeze(0)
        audio = self.tokenizer.decode(latents)
        audio = audio.clamp(-1.0, 1.0)
        return audio.squeeze(0) if squeeze else audio
