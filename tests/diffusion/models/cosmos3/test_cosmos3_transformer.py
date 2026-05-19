# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_compute_mrope_position_ids_text_offsets_all_axes() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        compute_mrope_position_ids_text,
    )

    ids, next_offset = compute_mrope_position_ids_text(num_tokens=3, temporal_offset=5)

    assert ids.tolist() == [[5, 6, 7], [5, 6, 7], [5, 6, 7]]
    assert next_offset == 8


def test_compute_mrope_position_ids_vision_without_fps_modulation() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        compute_mrope_position_ids_vision,
    )

    ids, next_offset = compute_mrope_position_ids_vision(
        grid_t=2,
        grid_h=2,
        grid_w=3,
        temporal_offset=10,
        fps=None,
    )

    assert ids.shape == (3, 12)
    assert ids[0].tolist() == [10] * 6 + [11] * 6
    assert ids[1].tolist() == [0, 0, 0, 1, 1, 1] * 2
    assert ids[2].tolist() == [0, 1, 2, 0, 1, 2] * 2
    assert next_offset == 12


def test_compute_mrope_position_ids_vision_with_fps_modulation() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        compute_mrope_position_ids_vision,
    )

    ids, next_offset = compute_mrope_position_ids_vision(
        grid_t=2,
        grid_h=1,
        grid_w=1,
        temporal_offset=10,
        fps=12.0,
        base_fps=24.0,
        temporal_compression_factor=4,
    )

    torch.testing.assert_close(ids[0], torch.tensor([10.0, 12.0]))
    assert ids.dtype == torch.float32
    assert next_offset == 13


def test_compute_mrope_position_ids_sound_uses_sound_latent_fps() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        compute_mrope_position_ids_sound,
    )

    ids, next_offset = compute_mrope_position_ids_sound(
        grid_t=3,
        temporal_offset=10,
        sound_latent_fps=25.0,
        base_fps=24.0,
    )

    torch.testing.assert_close(ids[0], torch.tensor([10.0, 10.96, 11.92]))
    assert ids[1].tolist() == [0.0, 0.0, 0.0]
    assert ids[2].tolist() == [0.0, 0.0, 0.0]
    assert next_offset == 12


def test_compute_mrope_position_ids_action_uses_start_frame_offset() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        compute_mrope_position_ids_action,
    )

    ids, next_offset = compute_mrope_position_ids_action(
        grid_t=3,
        temporal_offset=10,
        action_fps=None,
        start_frame_offset=1,
    )

    assert ids.tolist() == [[11, 12, 13], [0, 0, 0], [0, 0, 0]]
    assert next_offset == 14


def test_compute_mrope_position_ids_action_keeps_video_base_temporal_compression() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        compute_mrope_position_ids_action,
    )

    ids, next_offset = compute_mrope_position_ids_action(
        grid_t=3,
        temporal_offset=10,
        action_fps=24.0,
        base_fps=24.0,
        base_temporal_compression_factor=4,
        start_frame_offset=0,
    )

    torch.testing.assert_close(ids[0], torch.tensor([10.0, 10.25, 10.5]))
    assert next_offset == 11


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("qk_norm_for_diffusion", False),
        ("qk_norm_for_text", False),
        ("position_embedding_type", "rotary"),
        ("unified_3d_mrope_reset_spatial_ids", False),
        ("joint_attn_implementation", "one_way"),
    ],
)
def test_validate_supported_config_rejects_unsupported_flags(key: str, value) -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    with pytest.raises(ValueError, match=f"{key}="):
        Cosmos3VFMTransformer._validate_supported_config({key: value})


def test_validate_supported_config_accepts_defaults() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    Cosmos3VFMTransformer._validate_supported_config({})
    Cosmos3VFMTransformer._validate_supported_config(None)


def test_cosmos3_hsdp_conditions_match_und_and_gen_blocks() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.language_model = nn.Module()
    model.language_model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])
    model.gen_layers = nn.ModuleList([nn.Linear(2, 2)])
    model.norm_moe_gen = nn.LayerNorm(2)

    conditions = model._hsdp_shard_conditions
    matched = [
        name for name, module in model.named_modules() if any(condition(name, module) for condition in conditions)
    ]

    assert matched == [
        "language_model.layers.0",
        "language_model.layers.1",
        "gen_layers.0",
    ]


def test_cosmos3_transformer_exposes_layerwise_offload_and_repeated_blocks() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    assert Cosmos3VFMTransformer._layerwise_offload_blocks_attrs == ["gen_layers"]
    assert Cosmos3VFMTransformer._repeated_blocks == ["Cosmos3GenDecoderLayer"]


def test_patchify_unpatchify_round_trip_crops_padding() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.latent_patch_size = 2
    model.latent_channel_size = 3

    latents = torch.arange(1 * 3 * 1 * 3 * 5, dtype=torch.float32).reshape(1, 3, 1, 3, 5)

    tokens = model.patchify(latents, t=1, h=3, w=5)
    restored = model.unpatchify(tokens, t=1, h=3, w=5)

    assert tokens.shape == (1, 6, 12)
    torch.testing.assert_close(restored, latents)


def _tiny_cosmos3_config(**overrides):
    config = {
        "hidden_size": 8,
        "num_hidden_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 16,
        "vocab_size": 32,
        "latent_patch_size": 1,
        "latent_channel": 2,
        "rope_scaling": {"mrope_section": [1, 1, 0]},
    }
    config.update(overrides)
    return config


def test_sound_modules_created_only_when_sound_config_present() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    tiny = _tiny_cosmos3_config()

    no_sound = Cosmos3VFMTransformer(SimpleNamespace(tf_model_config=tiny, dtype=torch.float32))
    explicit_disabled = Cosmos3VFMTransformer(
        SimpleNamespace(
            tf_model_config={**tiny, "sound_gen": False, "sound_dim": 3},
            dtype=torch.float32,
        )
    )
    with_sound = Cosmos3VFMTransformer(
        SimpleNamespace(
            tf_model_config={**tiny, "sound_gen": True, "sound_dim": 3},
            dtype=torch.float32,
        )
    )
    with_nested_sound_dim = Cosmos3VFMTransformer(
        SimpleNamespace(
            tf_model_config={**tiny, "sound_gen": True},
            model_config={"sound_tokenizer": {"io_channels": 5}},
            custom_pipeline_args={},
            dtype=torch.float32,
        )
    )

    assert no_sound.sound_gen is False
    assert not hasattr(no_sound, "sound2llm")
    assert explicit_disabled.sound_gen is False
    assert not hasattr(explicit_disabled, "sound2llm")
    assert with_sound.sound_gen is True
    assert with_sound.sound2llm.in_features == 3
    assert with_sound.llm2sound.out_features == 3
    assert tuple(with_sound.sound_modality_embed.shape) == (8,)
    assert with_nested_sound_dim.sound_dim == 5
    assert with_nested_sound_dim.sound2llm.in_features == 5


def test_action_modules_created_only_when_action_config_present() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    tiny = _tiny_cosmos3_config()

    no_action = Cosmos3VFMTransformer(SimpleNamespace(tf_model_config=tiny, dtype=torch.float32))
    explicit_disabled = Cosmos3VFMTransformer(
        SimpleNamespace(
            tf_model_config={**tiny, "action_gen": False, "max_action_dim": 6},
            dtype=torch.float32,
        )
    )
    with_action = Cosmos3VFMTransformer(
        SimpleNamespace(
            tf_model_config={**tiny, "action_gen": True, "max_action_dim": 6, "num_embodiment_domains": 9},
            dtype=torch.float32,
        )
    )

    assert no_action.action_gen is False
    assert not hasattr(no_action, "action2llm")
    assert explicit_disabled.action_gen is False
    assert not hasattr(explicit_disabled, "action2llm")
    assert with_action.action_gen is True
    assert with_action.action_dim == 6
    assert with_action.action2llm.num_domains == 9
    assert tuple(with_action.action_modality_embed.shape) == (8,)


def test_sound_latent_fps_derives_from_sound_tokenizer_config() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    tiny = _tiny_cosmos3_config(sound_gen=True, sound_dim=3)

    derived = Cosmos3VFMTransformer(
        SimpleNamespace(
            tf_model_config=tiny,
            model_config={"sound_tokenizer": {"sample_rate": 32000, "hop_size": 800}},
            custom_pipeline_args={},
            dtype=torch.float32,
        )
    )
    explicit = Cosmos3VFMTransformer(
        SimpleNamespace(
            tf_model_config=tiny,
            custom_pipeline_args={
                "sound_sample_rate": 32000,
                "sound_hop_size": 800,
                "sound_latent_fps": 12.5,
            },
            dtype=torch.float32,
        )
    )

    assert derived.sound_latent_fps == 40.0
    assert explicit.sound_latent_fps == 12.5


def test_pack_unpack_sound_round_trip_and_shape_validation() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.sound_dim = 3

    latents = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    tokens = model.pack_sound(latents)
    restored = model.unpack_sound(tokens)

    assert tokens.shape == (2, 4, 3)
    torch.testing.assert_close(restored, latents)
    with pytest.raises(ValueError, match="channel mismatch"):
        model.pack_sound(torch.zeros(1, 4, 2))


def test_pack_unpack_action_round_trip_and_shape_validation() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.action_dim = 3

    latents = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    tokens = model.pack_action(latents)
    restored = model.unpack_action(tokens)

    assert tokens.shape == (2, 4, 3)
    torch.testing.assert_close(restored, latents)
    with pytest.raises(ValueError, match="dimension mismatch"):
        model.pack_action(torch.zeros(1, 2, 4))


def test_forward_with_sound_returns_video_and_sound_predictions() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    model = Cosmos3VFMTransformer(
        SimpleNamespace(
            tf_model_config=_tiny_cosmos3_config(sound_gen=True, sound_dim=3, sound_latent_fps=24.0),
            dtype=torch.float32,
        )
    )

    video = torch.zeros(1, 2, 1, 2, 2)
    sound = torch.zeros(1, 3, 4)
    output = model(
        hidden_states=video,
        timestep=torch.tensor([1.0]),
        text_ids=torch.tensor([[1, 2]], dtype=torch.long),
        text_mask=torch.ones(1, 2, dtype=torch.long),
        video_shape=(1, 2, 2),
        fps=24.0,
        sound_latents=sound,
    )

    assert isinstance(output, tuple)
    video_pred, sound_pred = output
    assert video_pred.shape == video.shape
    assert sound_pred.shape == sound.shape


def test_forward_with_action_returns_video_and_action_predictions() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    model = Cosmos3VFMTransformer(
        SimpleNamespace(
            tf_model_config=_tiny_cosmos3_config(
                action_gen=True,
                max_action_dim=3,
                num_embodiment_domains=4,
            ),
            dtype=torch.float32,
        )
    )

    video = torch.zeros(1, 2, 1, 2, 2)
    action = torch.zeros(1, 5, 3)
    output = model(
        hidden_states=video,
        timestep=torch.tensor([1.0]),
        text_ids=torch.tensor([[1, 2]], dtype=torch.long),
        text_mask=torch.ones(1, 2, dtype=torch.long),
        video_shape=(1, 2, 2),
        fps=24.0,
        action_latents=action,
        action_domain_ids=torch.tensor([2]),
        action_noisy_mask=torch.ones(1, 5, 1),
    )

    assert isinstance(output, tuple)
    video_pred, action_pred = output
    assert video_pred.shape == video.shape
    assert action_pred.shape == action.shape


def test_forward_with_sound_ulysses_error_mentions_combined_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 as cosmos3_module

    model = cosmos3_module.Cosmos3VFMTransformer(
        SimpleNamespace(
            tf_model_config=_tiny_cosmos3_config(sound_gen=True, sound_dim=3),
            dtype=torch.float32,
        )
    )
    monkeypatch.setattr(cosmos3_module, "_get_ulysses_state", lambda: (2, 0, None))

    with pytest.raises(
        ValueError,
        match=r"GEN sequence length \(3 = video tokens 2 \+ sound tokens 1\).*combined media sequence",
    ):
        model(
            hidden_states=torch.zeros(1, 2, 1, 1, 2),
            timestep=torch.tensor([1.0]),
            text_ids=torch.tensor([[1, 2]], dtype=torch.long),
            text_mask=torch.ones(1, 2, dtype=torch.long),
            video_shape=(1, 1, 2),
            fps=24.0,
            sound_latents=torch.zeros(1, 3, 1),
        )


def test_reset_cache_clears_und_and_gen_cache() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.cached_kv = object()
    model.cached_freqs_gen = object()

    model.reset_cache()

    assert model.cached_kv is None
    assert model.cached_freqs_gen is None


def test_compute_rope_freqs_pads_text_and_offsets_vision_positions() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    class FakeRotary:
        def __init__(self) -> None:
            self.position_ids: list[torch.Tensor] = []

        def __call__(self, x, position_ids):
            del x
            self.position_ids.append(position_ids.detach().cpu())
            batch = position_ids.shape[1]
            seq = position_ids.shape[2]
            return torch.zeros(batch, seq, 4), torch.ones(batch, seq, 4)

    rotary = FakeRotary()
    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.language_model = SimpleNamespace(rotary_emb=rotary)
    model.temporal_modality_margin = 100
    model.base_fps = 24.0
    model.temporal_compression_factor = 4
    model.enable_fps_modulation = False

    freqs_und, freqs_gen = model._compute_rope_freqs(
        text_mask=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long),
        t=2,
        hp=1,
        wp=1,
        fps=24.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    text_pos, vision_pos = rotary.position_ids
    assert text_pos[:, 0, :].tolist() == [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert text_pos[:, 1, :].tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert vision_pos[0, 0].tolist() == [102, 103]
    assert vision_pos[0, 1].tolist() == [101, 102]
    assert freqs_und[0].shape == (2, 3, 1, 4)
    assert freqs_gen[0].shape == (2, 2, 1, 4)


def test_compute_rope_freqs_appends_sound_positions_after_vision() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    class FakeRotary:
        def __init__(self) -> None:
            self.position_ids: list[torch.Tensor] = []

        def __call__(self, x, position_ids):
            del x
            self.position_ids.append(position_ids.detach().cpu())
            batch = position_ids.shape[1]
            seq = position_ids.shape[2]
            return torch.zeros(batch, seq, 4), torch.ones(batch, seq, 4)

    rotary = FakeRotary()
    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.language_model = SimpleNamespace(rotary_emb=rotary)
    model.temporal_modality_margin = 100
    model.base_fps = 24.0
    model.temporal_compression_factor = 4
    model.enable_fps_modulation = True
    model.temporal_compression_factor_sound = 1
    model.sound_latent_fps = 25.0

    model._compute_rope_freqs(
        text_mask=torch.tensor([[1, 1]], dtype=torch.long),
        t=2,
        hp=1,
        wp=1,
        fps=24.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        t_sound=3,
    )

    _, gen_pos = rotary.position_ids
    assert gen_pos.shape == (3, 1, 5)
    torch.testing.assert_close(
        gen_pos[0, 0],
        torch.tensor([102.0, 103.0, 102.0, 102.96, 103.92]),
    )


def test_compute_rope_freqs_appends_action_positions_between_vision_and_sound() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    class FakeRotary:
        def __init__(self) -> None:
            self.position_ids: list[torch.Tensor] = []

        def __call__(self, x, position_ids):
            del x
            self.position_ids.append(position_ids.detach().cpu())
            batch = position_ids.shape[1]
            seq = position_ids.shape[2]
            return torch.zeros(batch, seq, 4), torch.ones(batch, seq, 4)

    rotary = FakeRotary()
    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.language_model = SimpleNamespace(rotary_emb=rotary)
    model.temporal_modality_margin = 100
    model.base_fps = 24.0
    model.temporal_compression_factor = 4
    model.enable_fps_modulation = False
    model.temporal_compression_factor_sound = 1
    model.sound_latent_fps = 25.0

    model._compute_rope_freqs(
        text_mask=torch.tensor([[1, 1]], dtype=torch.long),
        t=2,
        hp=1,
        wp=1,
        fps=24.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        t_action=2,
        action_start_frame_offset=1,
        t_sound=1,
    )

    _, gen_pos = rotary.position_ids
    assert gen_pos.shape == (3, 1, 5)
    assert gen_pos[0, 0].tolist() == [102, 103, 103, 104, 102]


def test_compute_rope_freqs_promotes_mixed_video_sound_position_dtypes() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    class FakeRotary:
        def __init__(self) -> None:
            self.position_ids: list[torch.Tensor] = []

        def __call__(self, x, position_ids):
            del x
            self.position_ids.append(position_ids.detach().cpu())
            batch = position_ids.shape[1]
            seq = position_ids.shape[2]
            return torch.zeros(batch, seq, 4), torch.ones(batch, seq, 4)

    rotary = FakeRotary()
    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.language_model = SimpleNamespace(rotary_emb=rotary)
    model.temporal_modality_margin = 100
    model.base_fps = 24.0
    model.temporal_compression_factor = 4
    model.enable_fps_modulation = True
    model.temporal_compression_factor_sound = 1
    model.sound_latent_fps = 25.0

    model._compute_rope_freqs(
        text_mask=torch.tensor([[1, 1]], dtype=torch.long),
        t=1,
        hp=1,
        wp=1,
        fps=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        t_sound=3,
    )

    _, gen_pos = rotary.position_ids
    assert gen_pos.dtype == torch.float32
    torch.testing.assert_close(
        gen_pos[0, 0],
        torch.tensor([102.0, 102.0, 102.96, 103.92]),
    )
