# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME = "diffusion_pytorch_model.safetensors"


class _FakeAVAEAudioTokenizer:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.sample_rate = int(kwargs["sample_rate"])
        self.audio_channels = int(kwargs["audio_channels"])
        self.latent_ch = int(kwargs["io_channels"])
        self.temporal_compression_factor = int(kwargs["hop_size"])

    def get_latent_num_samples(self, num_audio_samples: int) -> int:
        return int(num_audio_samples) // self.temporal_compression_factor

    def get_audio_num_samples(self, num_latent_samples: int) -> int:
        return int(num_latent_samples) * self.temporal_compression_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.zeros(latents.shape[0], self.audio_channels, 8)


def test_from_config_loads_default_sound_tokenizer_component(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    model_dir = tmp_path / "model"
    tokenizer_dir = model_dir / "sound_tokenizer"
    tokenizer_dir.mkdir(parents=True)
    checkpoint_path = tokenizer_dir / DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME
    config_path = tokenizer_dir / "config.json"
    checkpoint_path.write_bytes(b"stub")
    config_path.write_text("{}", encoding="utf-8")

    created = {}

    class FakeAVAE(_FakeAVAEAudioTokenizer):
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(sound_tokenizer, "Cosmos3AVAEAudioTokenizer", FakeAVAE)
    monkeypatch.setattr(sound_tokenizer, "get_local_device", lambda: torch.device("cpu"))

    tokenizer = sound_tokenizer.Cosmos3SoundTokenizer.from_config(
        SimpleNamespace(
            model=str(model_dir),
            custom_pipeline_args={
                "sound_sample_rate": 32000,
                "sound_hop_size": 800,
                "sound_dim": 3,
            },
            dtype=torch.float32,
        )
    )

    assert created["checkpoint_path"] == str(checkpoint_path)
    assert created["config_path"] == str(config_path)
    assert tokenizer.sample_rate == 32000
    assert tokenizer.latent_ch == 3
    assert tokenizer.hop_size == 800


def test_from_config_downloads_default_sound_tokenizer_from_hf_repo(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    cache_dir = tmp_path / "hf"
    tokenizer_dir = cache_dir / "sound_tokenizer"
    tokenizer_dir.mkdir(parents=True)
    checkpoint_path = tokenizer_dir / DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME
    config_path = tokenizer_dir / "config.json"
    checkpoint_path.write_bytes(b"stub")
    config_path.write_text("{}", encoding="utf-8")

    calls = []

    def fake_snapshot_download(
        repo_id: str,
        *,
        revision: str | None,
        allow_patterns: list[str],
    ) -> str:
        calls.append((repo_id, revision, allow_patterns))
        return str(cache_dir)

    created = {}

    class FakeAVAE(_FakeAVAEAudioTokenizer):
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(sound_tokenizer, "Cosmos3AVAEAudioTokenizer", FakeAVAE)
    monkeypatch.setattr(sound_tokenizer, "get_local_device", lambda: torch.device("cpu"))

    tokenizer = sound_tokenizer.Cosmos3SoundTokenizer.from_config(
        SimpleNamespace(
            model="nvidia/cosmos3",
            revision="test-rev",
            custom_pipeline_args={
                "sound_sample_rate": 32000,
                "sound_hop_size": 800,
                "sound_dim": 3,
            },
            dtype=torch.float32,
        )
    )

    assert created["checkpoint_path"] == str(checkpoint_path)
    assert created["config_path"] == str(config_path)
    assert tokenizer.sample_rate == 32000
    assert tokenizer.latent_ch == 3
    assert calls == [
        (
            "nvidia/cosmos3",
            "test-rev",
            ["sound_tokenizer/config.json", f"sound_tokenizer/{DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME}"],
        )
    ]


def test_from_config_uses_diffusers_sound_tokenizer_checkpoint_name(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    model_dir = tmp_path / "model"
    tokenizer_dir = model_dir / "sound_tokenizer"
    tokenizer_dir.mkdir(parents=True)
    checkpoint_path = tokenizer_dir / DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME
    checkpoint_path.write_bytes(b"stub")
    (tokenizer_dir / "config.json").write_text("{}", encoding="utf-8")

    created = {}

    class FakeAVAE(_FakeAVAEAudioTokenizer):
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(sound_tokenizer, "Cosmos3AVAEAudioTokenizer", FakeAVAE)
    monkeypatch.setattr(sound_tokenizer, "get_local_device", lambda: torch.device("cpu"))

    sound_tokenizer.Cosmos3SoundTokenizer.from_config(
        SimpleNamespace(model=str(model_dir), custom_pipeline_args={}, dtype=torch.float32)
    )

    assert created["checkpoint_path"] == str(checkpoint_path)


def test_default_component_requires_sound_tokenizer_checkpoint(tmp_path) -> None:
    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    model_dir = tmp_path / "model"
    (model_dir / "sound_tokenizer").mkdir(parents=True)

    with pytest.raises(ValueError, match="no AVAE sound tokenizer checkpoint"):
        sound_tokenizer.Cosmos3SoundTokenizer.from_config(
            SimpleNamespace(model=str(model_dir), custom_pipeline_args={}, dtype=torch.float32)
        )


def test_default_component_rejects_legacy_sound_tokenizer_checkpoint_name(tmp_path) -> None:
    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    model_dir = tmp_path / "model"
    tokenizer_dir = model_dir / "sound_tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "model.safetensors").write_bytes(b"stub")

    with pytest.raises(ValueError, match=DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME):
        sound_tokenizer.Cosmos3SoundTokenizer.from_config(
            SimpleNamespace(model=str(model_dir), custom_pipeline_args={}, dtype=torch.float32)
        )


def test_from_config_uses_nested_normalization_config(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    model_dir = tmp_path / "model"
    tokenizer_dir = model_dir / "sound_tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME).write_bytes(b"stub")
    (tokenizer_dir / "config.json").write_text("{}", encoding="utf-8")

    created = {}

    class FakeAVAE(_FakeAVAEAudioTokenizer):
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(sound_tokenizer, "Cosmos3AVAEAudioTokenizer", FakeAVAE)
    monkeypatch.setattr(sound_tokenizer, "get_local_device", lambda: torch.device("cpu"))

    sound_tokenizer.Cosmos3SoundTokenizer.from_config(
        SimpleNamespace(
            model=str(model_dir),
            custom_pipeline_args={},
            model_config={
                "sound_tokenizer": {
                    "normalize_latents": False,
                    "normalization_type": "none",
                }
            },
            dtype=torch.float32,
        )
    )

    assert created["normalize_latents"] is False
    assert created["normalization_type"] == "none"


def test_from_config_custom_normalization_overrides_nested_config(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    model_dir = tmp_path / "model"
    tokenizer_dir = model_dir / "sound_tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME).write_bytes(b"stub")
    (tokenizer_dir / "config.json").write_text("{}", encoding="utf-8")

    created = {}

    class FakeAVAE(_FakeAVAEAudioTokenizer):
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(sound_tokenizer, "Cosmos3AVAEAudioTokenizer", FakeAVAE)
    monkeypatch.setattr(sound_tokenizer, "get_local_device", lambda: torch.device("cpu"))

    sound_tokenizer.Cosmos3SoundTokenizer.from_config(
        SimpleNamespace(
            model=str(model_dir),
            custom_pipeline_args={
                "sound_normalize_latents": True,
                "sound_normalization_type": "tanh",
                "sound_tanh_input_scale": 2.0,
            },
            model_config={
                "sound_tokenizer": {
                    "normalize_latents": False,
                    "normalization_type": "none",
                    "tanh_input_scale": 1.0,
                }
            },
            dtype=torch.float32,
        )
    )

    assert created["normalize_latents"] is True
    assert created["normalization_type"] == "tanh"
    assert created["tanh_input_scale"] == 2.0


def test_from_config_uses_component_config_architecture_values(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    model_dir = tmp_path / "model"
    tokenizer_dir = model_dir / "sound_tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME).write_bytes(b"stub")
    (tokenizer_dir / "config.json").write_text(
        ('{"sampling_rate": 48000, "dec_out_channels": 2, "vocoder_input_dim": 64, "hop_size": 1920}'),
        encoding="utf-8",
    )

    created = {}

    class FakeAVAE(_FakeAVAEAudioTokenizer):
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(sound_tokenizer, "Cosmos3AVAEAudioTokenizer", FakeAVAE)
    monkeypatch.setattr(sound_tokenizer, "get_local_device", lambda: torch.device("cpu"))

    tokenizer = sound_tokenizer.Cosmos3SoundTokenizer.from_config(
        SimpleNamespace(
            model=str(model_dir),
            custom_pipeline_args={},
            model_config={
                "sound_tokenizer": {
                    "sample_rate": 32000,
                    "audio_channels": 1,
                    "io_channels": 3,
                    "hop_size": 800,
                }
            },
            dtype=torch.float32,
        )
    )

    assert created["sample_rate"] == 48000
    assert created["audio_channels"] == 2
    assert created["io_channels"] == 64
    assert created["hop_size"] == 1920
    assert tokenizer.sample_rate == 48000
    assert tokenizer.latent_ch == 64
    assert tokenizer.hop_size == 1920


def test_from_config_rejects_custom_architecture_conflict_with_component_config(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    model_dir = tmp_path / "model"
    tokenizer_dir = model_dir / "sound_tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME).write_bytes(b"stub")
    (tokenizer_dir / "config.json").write_text(
        ('{"sampling_rate": 48000, "dec_out_channels": 2, "vocoder_input_dim": 64, "hop_size": 1920}'),
        encoding="utf-8",
    )

    class FakeAVAE(_FakeAVAEAudioTokenizer):
        pass

    monkeypatch.setattr(sound_tokenizer, "Cosmos3AVAEAudioTokenizer", FakeAVAE)
    monkeypatch.setattr(sound_tokenizer, "get_local_device", lambda: torch.device("cpu"))

    with pytest.raises(ValueError, match=r"sample_rate.*48000.*32000"):
        sound_tokenizer.Cosmos3SoundTokenizer.from_config(
            SimpleNamespace(
                model=str(model_dir),
                custom_pipeline_args={"sound_sample_rate": 32000},
                dtype=torch.float32,
            )
        )


def test_avae_uses_diffusers_decoder_state_dict_layout(tmp_path) -> None:
    from safetensors.torch import save_file

    from vllm_omni.diffusion.models.cosmos3.audio_tokenizer import avae

    config = {
        "sampling_rate": 8000,
        "hop_size": 2,
        "dec_dim": 4,
        "dec_c_mults": [1],
        "dec_strides": [2],
        "dec_out_channels": 1,
        "vocoder_input_dim": 2,
        "normalization_type": "none",
    }
    checkpoint_path = tmp_path / DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    decoder = avae.OobleckDecoder(
        channels=4,
        input_channels=2,
        audio_channels=1,
        upsampling_ratios=[2],
        channel_multiples=[1],
    )
    save_file({f"decoder.{key}": value for key, value in decoder.state_dict().items()}, str(checkpoint_path))

    tokenizer = avae.Cosmos3AVAEAudioTokenizer(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        dtype=torch.float32,
        device="cpu",
    )

    keys = set(tokenizer.state_dict())
    assert "decoder.conv1.weight_g" in keys
    assert "decoder.block.0.snake1.alpha" in keys
    assert "decoder.block.0.conv_t1.weight_g" in keys
    assert "decoder.block.0.res_unit1.conv1.weight_g" in keys
    assert "decoder.snake1.alpha" in keys
    assert "decoder.conv2.weight_g" in keys
    assert not any(key.startswith("decoder.layers.") for key in keys)
    assert not any(key.startswith("model.decoder.") for key in keys)
    assert tokenizer.decode(torch.zeros(1, 2, 3)).shape == (1, 1, 6)
    with pytest.raises(NotImplementedError, match="decoder-only"):
        tokenizer.encode(torch.zeros(1, 1, 6))
