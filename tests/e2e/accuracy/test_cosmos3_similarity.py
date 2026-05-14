# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import pytest
import requests
import torch
from PIL import Image

from tests.e2e.accuracy.helpers import model_output_dir
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


MODEL_ENV_VAR = "VLLM_TEST_COSMOS3_MODEL"
MODEL_ID = "cosmos3"
PROMPT = "A small warehouse robot moves a blue box across a clean floor."
NEGATIVE_PROMPT = "blurry, distorted, low quality"
SEED = 42
WIDTH = 256
HEIGHT = 256
NUM_INFERENCE_STEPS = 2


def _model_name() -> str:
    model = os.environ.get(MODEL_ENV_VAR)
    if not model:
        pytest.skip(f"Set {MODEL_ENV_VAR} to run Cosmos3 full-model smoke tests.")
    return model


def _server_args() -> list[str]:
    return [
        "--num-gpus",
        "1",
        "--model-class-name",
        "Cosmos3OmniDiffusersPipeline",
        "--stage-init-timeout",
        "900",
        "--init-timeout",
        "1200",
    ]


def _image_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_cosmos3_t2i_serving_smoke(accuracy_artifact_root: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Cosmos3 full-model smoke tests require CUDA.")

    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)
    with OmniServer(model, _server_args(), use_omni=True) as server:
        response = requests.post(
            f"http://{server.host}:{server.port}/v1/images/generations",
            json={
                "model": server.model,
                "prompt": PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
                "size": f"{WIDTH}x{HEIGHT}",
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "guidance_scale": 1.0,
                "seed": SEED,
            },
            timeout=1800,
        )

    response.raise_for_status()
    payload = response.json()
    assert len(payload["data"]) == 1
    image = Image.open(io.BytesIO(base64.b64decode(payload["data"][0]["b64_json"]))).convert("RGB")
    image.save(output_dir / "cosmos3_t2i.png")
    assert image.size == (WIDTH, HEIGHT)


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_cosmos3_t2v_sync_serving_smoke(accuracy_artifact_root: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Cosmos3 full-model smoke tests require CUDA.")

    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)
    with OmniServer(model, _server_args(), use_omni=True) as server:
        response = requests.post(
            f"http://{server.host}:{server.port}/v1/videos/sync",
            data={
                "model": server.model,
                "prompt": PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
                "size": f"{WIDTH}x{HEIGHT}",
                "num_frames": "1",
                "fps": "1",
                "num_inference_steps": str(NUM_INFERENCE_STEPS),
                "guidance_scale": "1.0",
                "seed": str(SEED),
            },
            timeout=1800,
        )

    response.raise_for_status()
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.content
    (output_dir / "cosmos3_t2v.mp4").write_bytes(response.content)


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_cosmos3_i2v_sync_serving_smoke(accuracy_artifact_root: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Cosmos3 full-model smoke tests require CUDA.")

    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)
    reference = Image.new("RGB", (96, 64), color=(40, 80, 160))
    with OmniServer(model, _server_args(), use_omni=True) as server:
        response = requests.post(
            f"http://{server.host}:{server.port}/v1/videos/sync",
            data={
                "model": server.model,
                "prompt": "The blue rectangle moves slowly forward.",
                "negative_prompt": NEGATIVE_PROMPT,
                "image_reference": json.dumps({"image_url": _image_data_url(reference)}),
                "size": f"{WIDTH}x{HEIGHT}",
                "num_frames": "5",
                "fps": "1",
                "num_inference_steps": str(NUM_INFERENCE_STEPS),
                "guidance_scale": "1.0",
                "seed": str(SEED),
            },
            timeout=1800,
        )

    response.raise_for_status()
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.content
    (output_dir / "cosmos3_i2v.mp4").write_bytes(response.content)
