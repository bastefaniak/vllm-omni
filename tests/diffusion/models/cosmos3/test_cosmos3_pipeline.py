# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from tests.diffusion.models.cosmos3.conftest import (
    StubScheduler,
    make_sampling_params,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _ids(value: int) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.long)


def _mask() -> torch.Tensor:
    return torch.ones(1, 1, dtype=torch.long)


class TestRegistryIntegration:
    def test_pipeline_registered_and_exported(self) -> None:
        from vllm_omni.diffusion.cache.cache_dit_backend import CUSTOM_DIT_ENABLERS
        from vllm_omni.diffusion.models import cosmos3
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
            Cosmos3OmniDiffusersPipeline,
        )
        from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
        from vllm_omni.diffusion.registry import (
            _DIFFUSION_MODELS,
            _DIFFUSION_POST_PROCESS_FUNCS,
            _DIFFUSION_PRE_PROCESS_FUNCS,
        )

        assert issubclass(Cosmos3OmniDiffusersPipeline, nn.Module)
        assert issubclass(Cosmos3OmniDiffusersPipeline, ProgressBarMixin)
        assert Cosmos3OmniDiffusersPipeline.support_image_input is True
        assert _DIFFUSION_MODELS["Cosmos3OmniDiffusersPipeline"] == (
            "cosmos3",
            "pipeline_cosmos3",
            "Cosmos3OmniDiffusersPipeline",
        )
        assert _DIFFUSION_PRE_PROCESS_FUNCS["Cosmos3OmniDiffusersPipeline"] == "get_cosmos3_pre_process_func"
        assert _DIFFUSION_POST_PROCESS_FUNCS["Cosmos3OmniDiffusersPipeline"] == "get_cosmos3_post_process_func"
        assert "Cosmos3OmniDiffusersPipeline" in CUSTOM_DIT_ENABLERS
        assert hasattr(cosmos3, "Cosmos3OmniDiffusersPipeline")
        assert "Cosmos3OmniDiffusersPipeline" in cosmos3.__all__


class TestPreAndPostProcess:
    def test_preprocess_leaves_t2v_string_prompt_unchanged(self) -> None:
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_pre_process_func

        request = SimpleNamespace(
            prompts=["A robot walks through a warehouse."],
            sampling_params=SimpleNamespace(height=None, width=None),
        )

        result = get_cosmos3_pre_process_func(SimpleNamespace())(request)

        assert result is request
        assert result.prompts == ["A robot walks through a warehouse."]
        assert result.sampling_params.height is None
        assert result.sampling_params.width is None

    def test_preprocess_resizes_i2v_image_to_720p_aspect_and_stores_tensor(self) -> None:
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_pre_process_func

        request = SimpleNamespace(
            prompts=[
                {
                    "prompt": "A slow camera push.",
                    "multi_modal_data": {"image": Image.new("RGB", (320, 160), "red")},
                }
            ],
            sampling_params=SimpleNamespace(height=None, width=None),
        )

        result = get_cosmos3_pre_process_func(SimpleNamespace())(request)
        prompt = result.prompts[0]

        assert result.sampling_params.height == 672
        assert result.sampling_params.width == 1344
        preprocessed = prompt["additional_information"]["preprocessed_image"]
        assert isinstance(preprocessed, torch.Tensor)
        assert tuple(preprocessed.shape[-2:]) == (672, 1344)

    def test_preprocess_preserves_explicit_size_for_i2v(self) -> None:
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_pre_process_func

        request = SimpleNamespace(
            prompts=[
                {
                    "prompt": "A slow camera push.",
                    "multi_modal_data": {"image": Image.new("RGB", (320, 160), "red")},
                }
            ],
            sampling_params=SimpleNamespace(height=64, width=96),
        )

        result = get_cosmos3_pre_process_func(SimpleNamespace())(request)

        assert tuple(result.prompts[0]["additional_information"]["preprocessed_image"].shape[-2:]) == (64, 96)

    def test_postprocess_latent_passthrough_and_t2i_shape_validation(self) -> None:
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_post_process_func

        func = get_cosmos3_post_process_func(SimpleNamespace())
        video = torch.zeros(1, 3, 1, 4, 4)

        assert func(video, output_type="latent") is video

        images = func({"image": video})
        assert len(images) == 1
        assert images[0].size == (4, 4)

        video_result = func({"video": video})
        assert "video" in video_result

        sound_result = func(
            {
                "video": video,
                "audio": torch.ones(1, 2, 16),
                "audio_sample_rate": 48000,
            },
            sampling_params=SimpleNamespace(extra_args={"resolved_frame_rate": 12}),
        )
        assert "video" in sound_result
        assert sound_result["audio"].shape == (1, 2, 16)
        assert sound_result["audio_sample_rate"] == 48000
        assert sound_result["fps"] == 12

        with pytest.raises(ValueError, match="text-to-image postprocess expects"):
            func({"image": torch.zeros(1, 3, 2, 4, 4)})

        with pytest.raises(ValueError, match="both image and video"):
            func({"image": video, "video": video})

        with pytest.raises(ValueError, match="does not support audio output"):
            func({"image": video, "audio": torch.ones(1, 2, 16)})


class TestPipelineHelpers:
    def test_get_sp_param_prefers_extra_args_then_direct_attribute(self) -> None:
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
            Cosmos3OmniDiffusersPipeline,
        )

        sp = SimpleNamespace(extra_args={"flow_shift": 3.0}, flow_shift=2.0)
        assert Cosmos3OmniDiffusersPipeline._get_sp_param(sp, "flow_shift", 1.0) == 3.0

        sp = SimpleNamespace(extra_args={}, flow_shift=2.0)
        assert Cosmos3OmniDiffusersPipeline._get_sp_param(sp, "flow_shift", 1.0) == 2.0

        sp = SimpleNamespace(extra_args={})
        assert Cosmos3OmniDiffusersPipeline._get_sp_param(sp, "flow_shift", 1.0) == 1.0

    def test_apply_metadata_templates_adds_duration_and_resolution(self) -> None:
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
            Cosmos3OmniDiffusersPipeline,
        )

        prompt = Cosmos3OmniDiffusersPipeline._apply_metadata_templates(
            "A city street.",
            num_frames=48,
            frame_rate=24,
            height=720,
            width=1280,
        )

        assert prompt == (
            "A city street. The video is 2.0 seconds long and is of 24 FPS. This video is of 720x1280 resolution."
        )

    @pytest.mark.parametrize(
        "tokenized",
        [
            [1, 2],
            (1, 2),
            {"input_ids": [[1, 2]]},
            torch.tensor([1, 2]),
        ],
    )
    def test_normalize_token_ids_accepts_common_tokenizer_outputs(self, tokenized) -> None:
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
            Cosmos3OmniDiffusersPipeline,
        )

        assert Cosmos3OmniDiffusersPipeline._normalize_token_ids(tokenized) == [1, 2]

    def test_normalize_token_ids_rejects_unknown_or_non_integer_values(self) -> None:
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
            Cosmos3OmniDiffusersPipeline,
        )

        with pytest.raises(TypeError, match="must return token IDs"):
            Cosmos3OmniDiffusersPipeline._normalize_token_ids(object())

        with pytest.raises(TypeError, match="non-integer token"):
            Cosmos3OmniDiffusersPipeline._normalize_token_ids([object()])

    def test_tokenize_prompt_adds_generation_tokens_and_padding(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()

        class FakeTokenizer:
            eos_token_id = 99
            pad_token_id = 0

            def __init__(self) -> None:
                self.conversations = None

            def apply_chat_template(self, conversations, tokenize: bool, add_generation_prompt: bool):
                self.conversations = conversations
                assert tokenize is True
                assert add_generation_prompt is True
                return [10, 11]

            def convert_tokens_to_ids(self, token: str) -> int:
                assert token == "<|vision_start|>"
                return 88

        tokenizer = FakeTokenizer()
        pipeline.tokenizer = tokenizer

        input_ids, attention_mask = pipeline._tokenize_prompt(
            "hello",
            max_sequence_length=6,
            use_system_prompt=True,
            system_prompt="system",
        )

        assert input_ids.tolist() == [[10, 11, 99, 88, 0, 0]]
        assert attention_mask.tolist() == [[1, 1, 1, 1, 0, 0]]
        assert tokenizer.conversations == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ]

    def test_format_and_tokenize_uses_video_and_image_metadata_modes(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        captured: list[tuple[str, bool, str | None]] = []

        def fake_tokenize(text, max_sequence_length, use_system_prompt=False, system_prompt=None):
            del max_sequence_length
            captured.append((text, use_system_prompt, system_prompt))
            return _ids(len(captured)), _mask()

        pipeline._tokenize_prompt = fake_tokenize  # type: ignore[method-assign]

        pipeline._format_and_tokenize_prompts(
            "A robot",
            "bad",
            num_frames=48,
            frame_rate=24,
            height=720,
            width=1280,
            max_sequence_length=32,
            sp=SimpleNamespace(extra_args={"negative_metadata_mode": "inverse"}),
            use_system_prompt=True,
            is_t2i=False,
        )
        assert "The video is 2.0 seconds long" in captured[0][0]
        assert "This video is of 720x1280 resolution" in captured[0][0]
        assert "The video is not 2.0 seconds long" in captured[1][0]
        assert captured[0][1] is True

        captured.clear()
        pipeline._format_and_tokenize_prompts(
            "A robot",
            "bad",
            num_frames=1,
            frame_rate=24,
            height=1024,
            width=1024,
            max_sequence_length=32,
            sp=SimpleNamespace(extra_args={}),
            use_system_prompt=False,
            is_t2i=True,
        )
        assert "This image is of 1024x1024 resolution" in captured[0][0]
        assert "seconds long" not in captured[0][0]
        assert captured[1][0] == "bad"

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("transformer.vae2llm.weight", "transformer.vae2llm.weight"),
            ("model.embed_tokens.weight", "transformer.language_model.embed_tokens.weight"),
            ("model.norm.weight", "transformer.language_model.norm.weight"),
            ("model.norm_moe_gen.weight", "transformer.norm_moe_gen.weight"),
            (
                "model.layers.3.self_attn.q_proj.weight",
                "transformer.language_model.layers.3.self_attn.q_proj.weight",
            ),
            (
                "model.layers.3.self_attn.q_proj_moe_gen.weight",
                "transformer.gen_layers.3.cross_attention.q_proj.weight",
            ),
            (
                "model.layers.3.mlp_moe_gen.down_proj.weight",
                "transformer.gen_layers.3.mlp.down_proj.weight",
            ),
            ("sound2llm.weight", "transformer.sound2llm.weight"),
            ("llm2sound.bias", "transformer.llm2sound.bias"),
            ("sound_modality_embed", "transformer.sound_modality_embed"),
            ("sound_modality_embed.weight", "transformer.sound_modality_embed"),
            ("action2llm.fc.weight", "transformer.action2llm.fc.weight"),
            ("llm2action.bias.weight", "transformer.llm2action.bias.weight"),
            ("action_modality_embed", "transformer.action_modality_embed"),
            ("action_modality_embed.weight", "transformer.action_modality_embed"),
            ("action_pos_embed.weight", None),
            ("lm_head.weight", None),
            ("other.weight", None),
        ],
    )
    def test_remap_ckpt_key(self, key: str, expected: str | None) -> None:
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
            Cosmos3OmniDiffusersPipeline,
        )

        assert Cosmos3OmniDiffusersPipeline._remap_ckpt_key(key) == expected

    def test_prepare_latents_shape_uses_cosmos_temporal_and_spatial_factors(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()

        latents = pipeline._prepare_latents(
            height=16,
            width=24,
            num_frames=5,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )

        assert latents.shape == (1, 2, 2, 2, 3)
        assert latents.dtype == torch.float32

    def test_sound_request_detection_uses_prompt_and_extra_args(self) -> None:
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
            Cosmos3OmniDiffusersPipeline,
        )

        assert Cosmos3OmniDiffusersPipeline._is_sound_request(
            {"prompt": "x", "generate_sound": True},
            SimpleNamespace(extra_args={}),
        )
        assert Cosmos3OmniDiffusersPipeline._is_sound_request(
            {"prompt": "x"},
            SimpleNamespace(extra_args={"enable_sound_generation": "true"}),
        )
        assert not Cosmos3OmniDiffusersPipeline._is_sound_request(
            {"prompt": "x"},
            SimpleNamespace(extra_args={"generate_sound": False}),
        )

    def test_prepare_sound_latents_uses_lazy_tokenizer_and_duration(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, sound_gen=True, sound_dim=3)

        class FakeSoundTokenizer:
            sample_rate = 10
            latent_ch = 3

            def get_latent_num_samples(self, samples: int) -> int:
                assert samples == 20
                return 5

            def decode(self, latents: torch.Tensor) -> torch.Tensor:
                return torch.ones(latents.shape[0], 2, 7)

        pipeline._sound_tokenizer = FakeSoundTokenizer()

        target_samples, duration, sample_rate = pipeline._resolve_sound_target_samples(
            SimpleNamespace(extra_args={"sound_duration": 2.0}),
            num_frames=9,
            frame_rate=3.0,
        )
        latents, latent_frames = pipeline._prepare_sound_latents(
            target_samples,
            torch.Generator(device="cpu").manual_seed(0),
        )
        audio = pipeline._decode_sound_latents(torch.zeros(1, 3, 5), target_audio_samples=5)

        assert (target_samples, duration, sample_rate) == (20, 2.0, 10)
        assert latents.shape == (1, 3, 5)
        assert latent_frames == 5
        assert audio.shape == (1, 2, 5)

    def test_init_eagerly_loads_sound_tokenizer_when_transformer_supports_sound(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 as cosmos3_module
        from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class FakeVAE:
            config = SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=8)

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def to(self, device):
                self.device = device
                return self

        class FakeScheduler:
            config = SimpleNamespace(flow_shift=1.0)

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class FakeTransformer:
            sound_gen = True

        fake_sound_tokenizer = object()
        calls = []

        def fake_from_config(od_config):
            calls.append(od_config)
            return fake_sound_tokenizer

        monkeypatch.setattr(cosmos3_module, "AutoTokenizer", FakeTokenizer)
        monkeypatch.setattr(cosmos3_module, "DistributedAutoencoderKLWan", FakeVAE)
        monkeypatch.setattr(cosmos3_module, "UniPCMultistepScheduler", FakeScheduler)
        monkeypatch.setattr(cosmos3_module, "Cosmos3VFMTransformer", lambda *args, **kwargs: FakeTransformer())
        monkeypatch.setattr(sound_tokenizer.Cosmos3SoundTokenizer, "from_config", staticmethod(fake_from_config))
        monkeypatch.setattr(
            cosmos3_module.Cosmos3OmniDiffusersPipeline,
            "setup_diffusion_pipeline_profiler",
            lambda self, **kwargs: None,
        )

        od_config = SimpleNamespace(
            model=str(tmp_path),
            dtype=torch.float32,
            enable_cpu_offload=False,
            flow_shift=None,
            enable_diffusion_pipeline_profiler=False,
        )
        pipeline = cosmos3_module.Cosmos3OmniDiffusersPipeline(od_config=od_config)

        assert calls == [od_config]
        assert pipeline._sound_tokenizer is fake_sound_tokenizer
        source = pipeline.weights_sources[0]
        assert source.subfolder is None
        assert source.prefix == "transformer."
        assert source.allow_patterns_overrides == ["transformer/*.safetensors"]

    def test_prepare_latents_i2v_conditions_first_latent_frame(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()

        def fake_encode(image_tensor, num_frames, height, width):
            del image_tensor, num_frames, height, width
            return torch.full((1, 2, 2, 2, 3), 5.0)

        pipeline._encode_conditioning_video = fake_encode  # type: ignore[method-assign]

        latents, velocity_mask, image_latent = pipeline._prepare_latents_i2v(
            image_tensor=torch.zeros(1, 3, 16, 24),
            height=16,
            width=24,
            num_frames=5,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )

        assert latents.shape == (1, 2, 2, 2, 3)
        torch.testing.assert_close(latents[:, :, 0], torch.full((1, 2, 2, 3), 5.0))
        assert velocity_mask.tolist() == [[[[[0.0]], [[1.0]]]]]
        torch.testing.assert_close(image_latent, torch.full((1, 2, 1, 2, 3), 5.0))

    def test_prepare_action_latents_policy_uses_noise_and_raw_dim_mask(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(action_gen=True, action_dim=4)

        action, velocity_mask, clean, raw_dim = pipeline._prepare_action_latents(
            mode="policy",
            action_chunk_size=3,
            raw_action_dim=2,
            generator=torch.Generator(device="cpu").manual_seed(0),
            sp=SimpleNamespace(extra_args={}),
        )

        assert action.shape == (1, 3, 4)
        assert raw_dim == 2
        assert velocity_mask.tolist() == [[[1.0], [1.0], [1.0]]]
        torch.testing.assert_close(action[:, :, 2:], torch.zeros(1, 3, 2))
        torch.testing.assert_close(clean, torch.zeros(1, 3, 4))

    def test_prepare_action_latents_forward_dynamics_conditions_supplied_actions(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(action_gen=True, action_dim=4)

        action, velocity_mask, clean, raw_dim = pipeline._prepare_action_latents(
            mode="forward_dynamics",
            action_chunk_size=2,
            raw_action_dim=None,
            generator=torch.Generator(device="cpu").manual_seed(0),
            sp=SimpleNamespace(extra_args={"action": [[1.0, 2.0], [3.0, 4.0]]}),
        )

        assert raw_dim == 2
        assert velocity_mask.tolist() == [[[0.0], [0.0]]]
        torch.testing.assert_close(action, clean)
        torch.testing.assert_close(action[0, :, :2], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_set_flow_shift_rebuilds_only_when_target_changes(self, make_cosmos3_pipeline, monkeypatch) -> None:
        import vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 as cosmos3_module

        pipeline = make_cosmos3_pipeline()

        class FakeUniPCMultistepScheduler:
            calls: list[tuple[object, float]] = []

            @classmethod
            def from_config(cls, config, flow_shift: float):
                cls.calls.append((config, flow_shift))
                return StubScheduler([1], flow_shift=flow_shift)

        monkeypatch.setattr(cosmos3_module, "UniPCMultistepScheduler", FakeUniPCMultistepScheduler)
        original_scheduler = pipeline.scheduler

        pipeline._set_flow_shift(1.0)
        assert pipeline.scheduler is original_scheduler
        assert FakeUniPCMultistepScheduler.calls == []

        pipeline._set_flow_shift(3.0)
        assert pipeline.scheduler is not original_scheduler
        assert pipeline._current_flow_shift == 3.0
        assert FakeUniPCMultistepScheduler.calls == [(pipeline._base_scheduler_config, 3.0)]


class TestDiffuse:
    def test_diffuse_without_cfg_runs_one_cond_forward_per_step(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        latents = torch.zeros(1, 2, 2, 1, 1)

        result = pipeline.diffuse(
            latents=latents,
            timesteps=torch.tensor([7, 3]),
            cond_ids=_ids(2),
            cond_mask=_mask(),
            uncond_ids=_ids(1),
            uncond_mask=_mask(),
            guidance_scale=1.0,
            shared_kwargs={"video_shape": (2, 1, 1), "fps": 24.0},
        )

        assert pipeline.transformer.reset_calls == 1
        assert [call["token"] for call in pipeline.transformer.calls] == [2, 2]
        torch.testing.assert_close(result, torch.full_like(latents, 4.0))

    def test_diffuse_sequential_cfg_uses_separate_caches_and_interval_skip(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        latents = torch.zeros(1, 2, 1, 1, 1)

        result = pipeline.diffuse(
            latents=latents,
            timesteps=torch.tensor([900, 100]),
            cond_ids=_ids(2),
            cond_mask=_mask(),
            uncond_ids=_ids(1),
            uncond_mask=_mask(),
            guidance_scale=3.0,
            shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0},
            guidance_interval=(500.0, 1000.0),
        )

        assert [call["token"] for call in pipeline.transformer.calls] == [2, 1, 2]
        assert pipeline.transformer.calls[0]["cache_before"] is None
        assert pipeline.transformer.calls[1]["cache_before"] is None
        assert pipeline.transformer.calls[2]["cache_before"] is not None
        torch.testing.assert_close(result, torch.full_like(latents, 6.0))

    def test_diffuse_cfg_parallel_uses_scale_one_outside_guidance_interval(
        self,
        make_cosmos3_pipeline,
    ) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline._cfg_parallel_active = lambda: True  # type: ignore[method-assign]
        latents = torch.zeros(1, 2, 1, 1, 1)
        calls = []

        def fake_predict_noise_maybe_with_cfg(**kwargs):
            calls.append(kwargs)
            return torch.ones_like(latents)

        pipeline.predict_noise_maybe_with_cfg = fake_predict_noise_maybe_with_cfg  # type: ignore[method-assign]

        result = pipeline.diffuse(
            latents=latents,
            timesteps=torch.tensor([900, 100]),
            cond_ids=_ids(2),
            cond_mask=_mask(),
            uncond_ids=_ids(1),
            uncond_mask=_mask(),
            guidance_scale=4.0,
            shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0},
            guidance_interval=(500.0, 1000.0),
        )

        assert [call["true_cfg_scale"] for call in calls] == [4.0, 1.0]
        assert calls[0]["positive_kwargs"]["text_ids"].item() == 2
        assert calls[0]["negative_kwargs"]["text_ids"].item() == 1
        torch.testing.assert_close(result, torch.full_like(latents, 2.0))

    def test_diffuse_i2v_masks_conditioned_frame_and_reinjects_image_latent(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        latents = torch.zeros(1, 2, 2, 1, 1)
        velocity_mask = torch.tensor([[[[[0.0]], [[1.0]]]]])
        image_latent = torch.full((1, 2, 1, 1, 1), 7.0)

        result = pipeline.diffuse(
            latents=latents,
            timesteps=torch.tensor([7]),
            cond_ids=_ids(2),
            cond_mask=_mask(),
            uncond_ids=_ids(1),
            uncond_mask=_mask(),
            guidance_scale=1.0,
            shared_kwargs={"video_shape": (2, 1, 1), "fps": 24.0},
            velocity_mask=velocity_mask,
            image_latent=image_latent,
        )

        torch.testing.assert_close(result[:, :, 0:1], image_latent)
        torch.testing.assert_close(result[:, :, 1:2], torch.full((1, 2, 1, 1, 1), 2.0))

    def test_diffuse_with_sound_steps_video_and_sound_jointly(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, sound_gen=True, sound_dim=3)
        latents = torch.zeros(1, 2, 1, 1, 1)
        sound_latents = torch.zeros(1, 3, 2)

        video_result, sound_result = pipeline.diffuse(
            latents=latents,
            sound_latents=sound_latents,
            timesteps=torch.tensor([7, 3]),
            cond_ids=_ids(2),
            cond_mask=_mask(),
            uncond_ids=_ids(1),
            uncond_mask=_mask(),
            guidance_scale=1.0,
            shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0},
        )

        torch.testing.assert_close(video_result, torch.full_like(latents, 4.0))
        torch.testing.assert_close(sound_result, torch.full_like(sound_latents, 24.0))
        assert pipeline.scheduler.step_calls[0][0].shape == (1, latents.numel() + sound_latents.numel())

    def test_diffuse_with_action_steps_video_and_action_jointly(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, action_gen=True, action_dim=4)
        latents = torch.zeros(1, 2, 1, 1, 1)
        action_latents = torch.zeros(1, 3, 4)

        video_result, action_result = pipeline.diffuse(
            latents=latents,
            action_latents=action_latents,
            action_velocity_mask=torch.ones(1, 3, 1),
            action_condition_latents=torch.zeros(1, 3, 4),
            timesteps=torch.tensor([7, 3]),
            cond_ids=_ids(2),
            cond_mask=_mask(),
            uncond_ids=_ids(1),
            uncond_mask=_mask(),
            guidance_scale=1.0,
            shared_kwargs={
                "video_shape": (1, 1, 1),
                "fps": 24.0,
                "action_domain_ids": torch.tensor([0]),
                "action_noisy_mask": torch.ones(1, 3, 1),
            },
        )

        torch.testing.assert_close(video_result, torch.full_like(latents, 4.0))
        torch.testing.assert_close(action_result, torch.full_like(action_latents, 44.0))
        assert pipeline.scheduler.step_calls[0][0].shape == (1, latents.numel() + action_latents.numel())


class TestForwardRouting:
    def _install_forward_stubs(self, pipeline):
        captured: dict[str, object] = {"diffuse_calls": [], "prepare_calls": []}

        def fake_format(
            prompt,
            negative_prompt,
            num_frames,
            frame_rate,
            height,
            width,
            max_sequence_length,
            sp,
            use_system_prompt=False,
            is_t2i=False,
        ):
            captured["format"] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "height": height,
                "width": width,
                "max_sequence_length": max_sequence_length,
                "use_system_prompt": use_system_prompt,
                "is_t2i": is_t2i,
                "sp": sp,
            }
            return _ids(2), _mask(), _ids(1), _mask()

        def fake_prepare(height, width, num_frames, generator):
            captured["prepare_calls"].append((height, width, num_frames, generator.initial_seed()))
            return torch.zeros(1, 2, 1, 1, 1)

        def fake_set_flow_shift(target):
            captured.setdefault("flow_shifts", []).append(target)
            pipeline._current_flow_shift = target

        def fake_set_scheduler_timesteps(num_inference_steps):
            captured.setdefault("scheduler_steps", []).append(num_inference_steps)
            pipeline.scheduler.timesteps = torch.tensor([7])

        def fake_diffuse(**kwargs):
            captured["diffuse_calls"].append(kwargs)
            outputs = [kwargs["latents"] + len(captured["diffuse_calls"])]
            if kwargs.get("action_latents") is not None:
                outputs.append(kwargs["action_latents"] + 3.0)
            if kwargs.get("sound_latents") is not None:
                outputs.append(kwargs["sound_latents"] + 2.0)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        pipeline._format_and_tokenize_prompts = fake_format  # type: ignore[method-assign]
        pipeline._prepare_latents = fake_prepare  # type: ignore[method-assign]
        pipeline._set_flow_shift = fake_set_flow_shift  # type: ignore[method-assign]
        pipeline._set_scheduler_timesteps = fake_set_scheduler_timesteps  # type: ignore[method-assign]
        pipeline.diffuse = fake_diffuse  # type: ignore[method-assign]
        pipeline._decode_latents = lambda latents: latents  # type: ignore[method-assign]
        return captured

    def _install_sound_stubs(self, pipeline):
        sound_latents = torch.zeros(1, 3, 4)
        decoded_audio = torch.ones(1, 2, 20)

        def fake_resolve_sound_target_samples(sp, num_frames, frame_rate):
            del sp, num_frames, frame_rate
            return 20, 2.0, 10

        def fake_prepare_sound_latents(target_samples, generator):
            del target_samples, generator
            return sound_latents, 4

        pipeline._resolve_sound_target_samples = fake_resolve_sound_target_samples  # type: ignore[method-assign]
        pipeline._prepare_sound_latents = fake_prepare_sound_latents  # type: ignore[method-assign]
        pipeline._decode_sound_latents = lambda latents, target_samples: decoded_audio  # type: ignore[method-assign]
        return sound_latents, decoded_audio

    def test_forward_uses_t2i_defaults_and_generates_multiple_outputs(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        captured = self._install_forward_stubs(pipeline)
        req = SimpleNamespace(
            prompts=[{"prompt": "A painted robot", "modalities": ["image"]}],
            sampling_params=make_sampling_params(num_outputs_per_prompt=2),
        )

        output = pipeline.forward(req)

        assert captured["flow_shifts"] == [3.0]
        assert captured["scheduler_steps"] == [50, 50]
        assert captured["format"]["is_t2i"] is True
        assert captured["format"]["height"] == 1024
        assert captured["format"]["width"] == 1024
        assert captured["format"]["num_frames"] == 1
        assert len(captured["diffuse_calls"]) == 2
        assert captured["diffuse_calls"][0]["guidance_interval"] == (400.0, 1000.0)
        assert output.output["image"].shape[0] == 2

    def test_forward_uses_t2v_defaults_and_engine_flow_shift(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        captured = self._install_forward_stubs(pipeline)
        req = SimpleNamespace(
            prompts=[{"prompt": "A warehouse robot", "modalities": ["video"]}],
            sampling_params=make_sampling_params(),
        )

        pipeline.forward(req)

        assert captured["flow_shifts"] == [1.0]
        assert captured["scheduler_steps"] == [35]
        assert captured["format"]["is_t2i"] is False
        assert captured["format"]["height"] == 720
        assert captured["format"]["width"] == 1280
        assert captured["format"]["num_frames"] == 81
        assert captured["diffuse_calls"][0]["guidance_interval"] is None

    def test_forward_defaults_to_video_without_modalities(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        captured = self._install_forward_stubs(pipeline)
        req = SimpleNamespace(
            prompts=["A warehouse robot"],
            sampling_params=make_sampling_params(),
        )

        output = pipeline.forward(req)

        assert captured["format"]["is_t2i"] is False
        assert "video" in output.output

    def test_forward_selects_i2v_latents_for_image_conditioning(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        captured = self._install_forward_stubs(pipeline)
        image_tensor = torch.zeros(1, 3, 16, 16)
        velocity_mask = torch.tensor([[[[[0.0]], [[1.0]]]]])
        image_latent = torch.full((1, 2, 1, 1, 1), 5.0)

        def fake_prepare_i2v(image, height, width, num_frames, generator):
            captured["i2v_prepare"] = (image, height, width, num_frames, generator.initial_seed())
            return torch.zeros(1, 2, 2, 1, 1), velocity_mask, image_latent

        def fail_prepare(*args, **kwargs):
            del args, kwargs
            raise AssertionError("T2V latent preparation should not run for an I2V request")

        pipeline._prepare_latents = fail_prepare  # type: ignore[method-assign]
        pipeline._prepare_latents_i2v = fake_prepare_i2v  # type: ignore[method-assign]
        req = SimpleNamespace(
            prompts=[
                {
                    "prompt": "A robot starts moving.",
                    "modalities": ["video"],
                    "negative_prompt": "bad",
                    "additional_information": {"preprocessed_image": image_tensor},
                }
            ],
            sampling_params=make_sampling_params(height=16, width=16, num_frames=5),
        )

        pipeline.forward(req)

        prepared_image, prepared_height, prepared_width, prepared_frames, _ = captured["i2v_prepare"]
        assert prepared_image is image_tensor
        assert prepared_height == 16
        assert prepared_width == 16
        assert prepared_frames == 5
        diffuse_call = captured["diffuse_calls"][0]
        assert diffuse_call["velocity_mask"] is velocity_mask
        assert diffuse_call["image_latent"] is image_latent
        assert diffuse_call["shared_kwargs"]["noisy_frame_mask"] is velocity_mask

    def test_forward_policy_action_returns_custom_output(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, action_gen=True, action_dim=4)
        captured = self._install_forward_stubs(pipeline)
        image_tensor = torch.zeros(1, 3, 16, 16)
        req = SimpleNamespace(
            prompts=[
                {
                    "prompt": "Pick the block.",
                    "modalities": ["video"],
                    "additional_information": {"preprocessed_image": image_tensor},
                }
            ],
            sampling_params=make_sampling_params(
                height=16,
                width=16,
                extra_args={
                    "action_mode": "policy",
                    "action_chunk_size": 2,
                    "raw_action_dim": 2,
                    "domain_name": "bridge_orig_lerobot",
                },
            ),
        )

        output = pipeline.forward(req)

        diffuse_call = captured["diffuse_calls"][0]
        assert diffuse_call["action_latents"].shape == (1, 2, 4)
        assert diffuse_call["action_velocity_mask"].tolist() == [[[1.0], [1.0]]]
        assert diffuse_call["shared_kwargs"]["action_domain_ids"].tolist() == [7]
        assert diffuse_call["shared_kwargs"]["action_start_frame_offset"] == 1
        assert output.custom_output["action"].shape == (1, 2, 2)
        assert output.custom_output["raw_action_dim"] == 2
        assert output.custom_output["action_mode"] == "policy"
        assert output.custom_output["domain_id"] == 7

    def test_forward_action_defaults_to_reference_chunk_size(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, action_gen=True, action_dim=4)
        captured = self._install_forward_stubs(pipeline)
        req = SimpleNamespace(
            prompts=[
                {
                    "prompt": "Pick the block.",
                    "modalities": ["video"],
                    "additional_information": {"preprocessed_image": torch.zeros(1, 3, 16, 16)},
                }
            ],
            sampling_params=make_sampling_params(
                height=16,
                width=16,
                extra_args={
                    "action_mode": "policy",
                    "raw_action_dim": 2,
                    "domain_id": 0,
                },
            ),
        )

        pipeline.forward(req)

        assert captured["format"]["num_frames"] == 17
        assert captured["diffuse_calls"][0]["action_latents"].shape == (1, 16, 4)

    def test_forward_video_sound_decodes_and_returns_audio_payload(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, sound_gen=True, sound_dim=3)
        captured = self._install_forward_stubs(pipeline)
        sound_latents = torch.zeros(1, 3, 4)
        decoded_audio = torch.ones(1, 2, 20)

        def fake_resolve_sound_target_samples(sp, num_frames, frame_rate):
            del sp, num_frames, frame_rate
            return 20, 2.0, 10

        def fake_prepare_sound_latents(target_samples, generator):
            del target_samples, generator
            return sound_latents, 4

        pipeline._resolve_sound_target_samples = fake_resolve_sound_target_samples  # type: ignore[method-assign]
        pipeline._prepare_sound_latents = fake_prepare_sound_latents  # type: ignore[method-assign]
        pipeline._decode_sound_latents = lambda latents, target_samples: decoded_audio  # type: ignore[method-assign]

        req = SimpleNamespace(
            prompts=[{"prompt": "A robot", "modalities": ["video"], "generate_sound": True}],
            sampling_params=make_sampling_params(num_frames=9, frame_rate=3.0),
        )

        output = pipeline.forward(req)

        assert captured["diffuse_calls"][0]["sound_latents"] is sound_latents
        assert output.output["audio"] is decoded_audio
        assert output.output["audio_sample_rate"] == 10
        assert "video" in output.output

    def test_forward_decode_info_logs_only_on_rank_zero(
        self,
        make_cosmos3_pipeline,
        monkeypatch: pytest.MonkeyPatch,
        caplog,
    ) -> None:
        from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3 as cosmos3_pipeline

        monkeypatch.setattr(cosmos3_pipeline, "_is_rank_zero", lambda: True)
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, sound_gen=True, sound_dim=3)
        self._install_forward_stubs(pipeline)
        self._install_sound_stubs(pipeline)
        req = SimpleNamespace(
            prompts=[{"prompt": "A robot", "modalities": ["video"], "generate_sound": True}],
            sampling_params=make_sampling_params(num_frames=9, frame_rate=3.0),
        )

        target_logger = logging.getLogger(cosmos3_pipeline.logger.name)
        target_logger.addHandler(caplog.handler)
        prev_level = target_logger.level
        target_logger.setLevel(logging.INFO)
        try:
            pipeline.forward(req)
        finally:
            target_logger.removeHandler(caplog.handler)
            target_logger.setLevel(prev_level)

        messages = [record.getMessage() for record in caplog.records if record.name == cosmos3_pipeline.logger.name]
        assert "Decoding video..." in messages
        assert any(message.startswith("Video decoded in ") for message in messages)
        assert any(message.startswith("Total pipeline time: ") for message in messages)
        assert "Decoding sound..." in messages

    def test_forward_decode_info_logs_suppressed_on_nonzero_rank(
        self,
        make_cosmos3_pipeline,
        monkeypatch: pytest.MonkeyPatch,
        caplog,
    ) -> None:
        from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3 as cosmos3_pipeline

        monkeypatch.setattr(cosmos3_pipeline, "_is_rank_zero", lambda: False)
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, sound_gen=True, sound_dim=3)
        self._install_forward_stubs(pipeline)
        _, decoded_audio = self._install_sound_stubs(pipeline)
        req = SimpleNamespace(
            prompts=[{"prompt": "A robot", "modalities": ["video"], "generate_sound": True}],
            sampling_params=make_sampling_params(num_frames=9, frame_rate=3.0),
        )

        target_logger = logging.getLogger(cosmos3_pipeline.logger.name)
        target_logger.addHandler(caplog.handler)
        prev_level = target_logger.level
        target_logger.setLevel(logging.INFO)
        try:
            output = pipeline.forward(req)
        finally:
            target_logger.removeHandler(caplog.handler)
            target_logger.setLevel(prev_level)

        messages = [record.getMessage() for record in caplog.records if record.name == cosmos3_pipeline.logger.name]
        assert output.output["audio"] is decoded_audio
        assert not any(
            message == "Decoding video..."
            or message.startswith("Video decoded in ")
            or message.startswith("Total pipeline time: ")
            or message == "Decoding sound..."
            for message in messages
        )

    def test_forward_rejects_multiple_prompts(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        req = SimpleNamespace(
            prompts=["one", "two"],
            sampling_params=make_sampling_params(),
        )

        with pytest.raises(ValueError, match="currently supports a single prompt"):
            pipeline.forward(req)

    def test_forward_rejects_conflicting_modalities(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        req = SimpleNamespace(
            prompts=[{"prompt": "one", "modalities": ["image", "video"]}],
            sampling_params=make_sampling_params(),
        )

        with pytest.raises(ValueError, match="cannot request both image and video"):
            pipeline.forward(req)

    def test_forward_rejects_sound_for_text_to_image(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, sound_gen=True, sound_dim=3)
        req = SimpleNamespace(
            prompts=[{"prompt": "A robot", "modalities": ["image"], "generate_sound": True}],
            sampling_params=make_sampling_params(),
        )

        with pytest.raises(ValueError, match="only for video outputs"):
            pipeline.forward(req)

    def test_forward_rejects_action_without_action_modules(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        req = SimpleNamespace(
            prompts=[{"prompt": "A robot", "modalities": ["video"]}],
            sampling_params=make_sampling_params(extra_args={"action_mode": "policy", "raw_action_dim": 2}),
        )

        with pytest.raises(ValueError, match="without action modules"):
            pipeline.forward(req)

    def test_forward_rejects_action_without_explicit_domain(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, action_gen=True, action_dim=4)
        req = SimpleNamespace(
            prompts=[
                {
                    "prompt": "A robot",
                    "modalities": ["video"],
                    "additional_information": {"preprocessed_image": torch.zeros(1, 3, 16, 16)},
                }
            ],
            sampling_params=make_sampling_params(
                height=16,
                width=16,
                extra_args={"action_mode": "policy", "raw_action_dim": 2},
            ),
        )

        with pytest.raises(ValueError, match=r"domain_id.*domain_name"):
            pipeline.forward(req)

    def test_forward_rejects_action_with_sound(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(
            latent_channel_size=2,
            action_gen=True,
            action_dim=4,
            sound_gen=True,
            sound_dim=3,
        )
        req = SimpleNamespace(
            prompts=[{"prompt": "A robot", "modalities": ["video"], "generate_sound": True}],
            sampling_params=make_sampling_params(extra_args={"action_mode": "policy", "raw_action_dim": 2}),
        )

        with pytest.raises(ValueError, match=r"action\+sound"):
            pipeline.forward(req)
