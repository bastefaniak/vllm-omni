# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm, unpack_diffusion_output_shm


def test_diffusion_output_dict_tensors_round_trip_through_shm() -> None:
    image = torch.arange(300_000, dtype=torch.float32)
    video = torch.arange(300_000, dtype=torch.float32) * 2
    output = DiffusionOutput(output={"image": image, "video": video, "metadata": {"keep": "inline"}})

    pack_diffusion_output_shm(output)

    assert output.output["image"]["__tensor_shm__"] is True
    assert output.output["video"]["__tensor_shm__"] is True
    assert output.output["metadata"] == {"keep": "inline"}

    unpack_diffusion_output_shm(output)

    torch.testing.assert_close(output.output["image"], image)
    torch.testing.assert_close(output.output["video"], video)
    assert output.output["metadata"] == {"keep": "inline"}
