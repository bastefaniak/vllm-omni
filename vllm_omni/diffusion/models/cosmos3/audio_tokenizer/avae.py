# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local AVAE audio tokenizer used by Cosmos3 sound generation."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.models.progress_bar import _is_rank_zero

from .config import AttrDict
from .models import load_generator

logger = init_logger(__name__)


def _default_avae_config(
    *,
    sample_rate: int,
    audio_channels: int,
    io_channels: int,
    hop_size: int,
) -> AttrDict:
    return AttrDict(
        {
            "model_type": "autoencoder_v2",
            "sampling_rate": sample_rate,
            "stereo": audio_channels == 2,
            "use_wav_as_input": True,
            "normalize_volume": True,
            "hop_size": hop_size,
            "input_channels": 1,
            "enc_type": "spec_convnext",
            "enc_dim": 192,
            "enc_intermediate_dim": 768,
            "enc_num_layers": 12,
            "enc_num_blocks": 2,
            "enc_n_fft": 64,
            "enc_hop_length": 16,
            "enc_latent_dim": 128,
            "enc_c_mults": [1, 2, 4],
            "enc_strides": [4, 4, 8],
            "enc_identity_init": False,
            "enc_use_snake": True,
            "dec_type": "oobleck",
            "dec_dim": 320,
            "dec_c_mults": [1, 2, 4, 8, 16],
            "dec_strides": [2, 4, 4, 8, 8],
            "dec_use_snake": True,
            "dec_final_tanh": False,
            "dec_out_channels": audio_channels,
            "dec_anti_aliasing": False,
            "dec_use_nearest_upsample": False,
            "dec_use_tanh_at_final": False,
            "bottleneck_type": "vae",
            "bottleneck": {"type": "vae"},
            "activation": "snakebeta",
            "snake_logscale": True,
            "anti_aliasing": False,
            "use_cuda_kernel": False,
            "causal": False,
            "padding_mode": "zeros",
            "vocoder_input_dim": io_channels,
        }
    )


def _load_config(
    config_path: str | Path | None,
    *,
    sample_rate: int,
    audio_channels: int,
    io_channels: int,
    hop_size: int,
) -> AttrDict:
    if config_path:
        with open(config_path, encoding="utf-8") as f:
            return AttrDict(json.load(f))
    return _default_avae_config(
        sample_rate=sample_rate,
        audio_channels=audio_channels,
        io_channels=io_channels,
        hop_size=hop_size,
    )


def _load_checkpoint(path: str | Path, map_location: torch.device | str) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Loading AVAE .safetensors checkpoints requires safetensors.") from exc
        checkpoint = load_file(str(path), device=str(map_location))
    else:
        checkpoint = torch.load(path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"AVAE checkpoint must be a dict, got {type(checkpoint)!r}.")

    for key in ("generator", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        tensor_items = {key: value for key, value in checkpoint.items() if isinstance(value, torch.Tensor)}
        if not tensor_items:
            raise RuntimeError(f"No tensor state dict found in AVAE checkpoint keys: {list(checkpoint.keys())[:16]}")
        checkpoint = tensor_items

    return checkpoint


def _strip_prefixes(
    state_dict: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "generator.", "model.")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        candidates = [key]
        current = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if current.startswith(prefix):
                    current = current[len(prefix) :]
                    candidates.append(current)
                    changed = True
                    break
        selected = next((candidate for candidate in candidates if candidate in model_state), candidates[-1])
        normalized[selected] = value
    return normalized


class Cosmos3AVAEAudioTokenizer(nn.Module):
    """AVAE tokenizer/decoder for Cosmos3 audio latents."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        sample_rate: int = 48000,
        audio_channels: int = 2,
        io_channels: int = 64,
        hop_size: int = 1920,
        normalize_latents: bool = True,
        normalization_type: str = "none",
        tanh_input_scale: float = 1.5,
        tanh_output_scale: float = 3.5,
        tanh_clamp: float = 0.995,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.audio_channels = int(audio_channels)
        self.latent_ch = int(io_channels)
        self.hop_size = int(hop_size)
        self.dtype = dtype
        self.device = torch.device(device)
        self.normalize_volume = True

        if normalization_type == "none" and normalize_latents:
            normalization_type = "tanh"
        self.normalization_type = normalization_type
        self.tanh_input_scale = float(tanh_input_scale)
        self.tanh_output_scale = float(tanh_output_scale)
        self.tanh_clamp = float(tanh_clamp)

        config = _load_config(
            config_path,
            sample_rate=self.sample_rate,
            audio_channels=self.audio_channels,
            io_channels=self.latent_ch,
            hop_size=self.hop_size,
        )
        self.model = load_generator(config.model_type, config, self.device)
        state_dict = _strip_prefixes(
            _load_checkpoint(checkpoint_path, self.device),
            self.model.state_dict(),
        )
        matching_keys = set(state_dict).intersection(self.model.state_dict())
        if not matching_keys:
            raise RuntimeError("AVAE checkpoint did not contain any keys matching the local AVAE model.")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if _is_rank_zero():
            logger.info(
                "Loaded Cosmos3 AVAE checkpoint from %s; missing=%d unexpected=%d",
                checkpoint_path,
                len(missing),
                len(unexpected),
            )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        if hasattr(self.model, "remove_weight_norm"):
            self.model.remove_weight_norm()
        self.model.to(dtype=self.dtype)

    @property
    def temporal_compression_factor(self) -> int:
        return self.hop_size

    def get_latent_num_samples(self, num_audio_samples: int) -> int:
        return int(num_audio_samples) // self.temporal_compression_factor

    def get_audio_num_samples(self, num_latent_samples: int) -> int:
        return int(num_latent_samples) * self.temporal_compression_factor

    def _normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.normalization_type == "tanh":
            in_dtype = latent.dtype
            return (torch.tanh(latent.float() / self.tanh_input_scale) * self.tanh_output_scale).to(in_dtype)
        if self.normalization_type != "none":
            raise ValueError(f"Unsupported AVAE normalization_type={self.normalization_type!r}.")
        return latent

    def _denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.normalization_type == "tanh":
            in_dtype = latent.dtype
            latent = torch.clamp(
                latent.float() / self.tanh_output_scale,
                -self.tanh_clamp,
                self.tanh_clamp,
            )
            return (torch.atanh(latent) * self.tanh_input_scale).to(in_dtype)
        if self.normalization_type != "none":
            raise ValueError(f"Unsupported AVAE normalization_type={self.normalization_type!r}.")
        return latent

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, force_pad: bool = False) -> torch.Tensor:
        in_dtype = audio.dtype
        x = audio.to(self.device)
        if x.ndim != 3:
            raise ValueError(f"AVAE audio input must be [B, C, T], got {tuple(x.shape)}.")
        if x.shape[1] == 1 and self.audio_channels == 2:
            x = x.repeat(1, 2, 1)
        elif x.shape[1] > self.audio_channels:
            x = x[:, : self.audio_channels]
        if self.normalize_volume:
            x = x / (x.abs().amax(dim=(-2, -1), keepdim=True) + 1e-5) * 0.95
        if force_pad or not self.model.training:
            pad_amount = (self.hop_size - (x.shape[-1] % self.hop_size)) % self.hop_size
            if pad_amount:
                x = F.pad(x, (0, pad_amount), mode="constant", value=0)
        encoded = self.model.encode(x.to(self.dtype))
        latent = encoded["latent"] if isinstance(encoded, dict) else encoded
        return self._normalize_latent(latent).to(in_dtype)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        in_dtype = latent.dtype
        z = self._denormalize_latent(latent.to(self.device)).to(self.dtype)
        decoded = self.model.decode(z)
        if not isinstance(decoded, dict) or "decoder_out" not in decoded:
            raise RuntimeError("AVAE decoder did not return decoder_out.")
        audio = decoded["decoder_out"].clamp(-1.0, 1.0)
        return audio.to(in_dtype)
