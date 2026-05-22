# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def __getattr__(name: str):
    """Lazy import so the pipeline registry can load
    ``vllm_omni.diffusion.models.cosmos3.pipeline`` (which only needs
    ``COSMOS3_PIPELINE``) without dragging in diffusers, retina-face, and
    other heavy deps that ``pipeline_cosmos3`` / ``transformer_cosmos3``
    pull in at module-import time. Mirrors the pattern used in
    ``vllm_omni/model_executor/models/glm_image/__init__.py``.
    """
    if name in ("Cosmos3OmniDiffusersPipeline", "get_cosmos3_post_process_func", "get_cosmos3_pre_process_func"):
        from . import pipeline_cosmos3

        return getattr(pipeline_cosmos3, name)
    if name == "Cosmos3VFMTransformer":
        from .transformer_cosmos3 import Cosmos3VFMTransformer

        return Cosmos3VFMTransformer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Cosmos3OmniDiffusersPipeline",
    "Cosmos3VFMTransformer",
    "get_cosmos3_post_process_func",
    "get_cosmos3_pre_process_func",
]
