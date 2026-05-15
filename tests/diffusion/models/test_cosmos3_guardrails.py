# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.tokenization_utils_base import BatchEncoding

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeTokenizer:
    def __init__(self, model_inputs):
        self.model_inputs = model_inputs
        self.decoded_ids: list[int] | None = None

    def apply_chat_template(self, conversations, *, tokenize: bool, return_tensors: str, add_generation_prompt: bool):
        assert conversations == [{"role": "user", "content": "a safe prompt"}]
        assert tokenize is True
        assert return_tensors == "pt"
        assert add_generation_prompt is True
        return self.model_inputs

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decoded_ids = token_ids.tolist()
        return "safe"


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        input_ids = args[0] if args else kwargs["input_ids"]
        return torch.cat([input_ids, torch.tensor([[99]], dtype=input_ids.dtype)], dim=-1)


def test_qwen_guardrail_generation_accepts_batch_encoding() -> None:
    from vllm_omni.diffusion.models.cosmos3.guardrails import _generate_qwen_guardrail_response

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    tokenizer = _FakeTokenizer(BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask}))
    model = _FakeModel()

    response = _generate_qwen_guardrail_response("a safe prompt", tokenizer, model, "cpu")

    assert response == "safe"
    assert tokenizer.decoded_ids == [99]
    args, kwargs = model.calls[0]
    assert args == ()
    assert torch.equal(kwargs["input_ids"], input_ids)
    assert torch.equal(kwargs["attention_mask"], attention_mask)
    assert kwargs["max_new_tokens"] == 128


def test_qwen_guardrail_generation_accepts_tensor_input_ids() -> None:
    from vllm_omni.diffusion.models.cosmos3.guardrails import _generate_qwen_guardrail_response

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    tokenizer = _FakeTokenizer(input_ids)
    model = _FakeModel()

    response = _generate_qwen_guardrail_response("a safe prompt", tokenizer, model, "cpu")

    assert response == "safe"
    assert tokenizer.decoded_ids == [99]
    args, kwargs = model.calls[0]
    assert len(args) == 1
    assert torch.equal(args[0], input_ids)
    assert kwargs == {"max_new_tokens": 128}


def test_siglip_feature_extraction_accepts_tensor() -> None:
    from vllm_omni.diffusion.models.cosmos3.guardrails import _extract_siglip_image_features

    features = torch.randn(1, 1152)

    assert _extract_siglip_image_features(features) is features


def test_siglip_feature_extraction_accepts_base_model_output_with_pooling() -> None:
    from vllm_omni.diffusion.models.cosmos3.guardrails import _extract_siglip_image_features

    last_hidden_state = torch.randn(1, 729, 1152)
    pooler_output = torch.randn(1, 1152)
    output = BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooler_output)

    assert _extract_siglip_image_features(output) is pooler_output


def test_siglip_feature_extraction_accepts_tuple_output() -> None:
    from vllm_omni.diffusion.models.cosmos3.guardrails import _extract_siglip_image_features

    last_hidden_state = torch.randn(1, 729, 1152)
    pooler_output = torch.randn(1, 1152)

    assert _extract_siglip_image_features((last_hidden_state, pooler_output)) is pooler_output


def test_siglip_feature_extraction_rejects_unpooled_features() -> None:
    from vllm_omni.diffusion.models.cosmos3.guardrails import _extract_siglip_image_features

    last_hidden_state = torch.randn(1, 729, 1152)

    with pytest.raises(TypeError, match="pooled features"):
        _extract_siglip_image_features(last_hidden_state)
