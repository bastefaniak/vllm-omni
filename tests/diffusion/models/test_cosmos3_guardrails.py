# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
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


@pytest.mark.parametrize("as_batch_encoding", [True, False])
def test_qwen_guardrail_generation_accepts_supported_tokenizer_outputs(as_batch_encoding: bool) -> None:
    from vllm_omni.diffusion.models.cosmos3.guardrails import _generate_qwen_guardrail_response

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    model_inputs = (
        BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask}) if as_batch_encoding else input_ids
    )
    tokenizer = _FakeTokenizer(model_inputs)
    model = _FakeModel()

    response = _generate_qwen_guardrail_response("a safe prompt", tokenizer, model, "cpu")

    assert response == "safe"
    assert tokenizer.decoded_ids == [99]
    args, kwargs = model.calls[0]
    if as_batch_encoding:
        assert args == ()
        assert torch.equal(kwargs["input_ids"], input_ids)
        assert torch.equal(kwargs["attention_mask"], attention_mask)
        assert kwargs["max_new_tokens"] == 128
    else:
        assert len(args) == 1
        assert torch.equal(args[0], input_ids)
        assert kwargs == {"max_new_tokens": 128}
