# Cosmos3-Edge

> Text-to-image, text-to-video, image-to-video

## Summary

- Vendor: NVIDIA
- Model: `nvidia/Cosmos3-Edge`
- Task: Text-to-image (T2I), text-to-video (T2V), image-to-video (I2V), action policy
- Mode: Online serving with the OpenAI-compatible image/video APIs, plus offline generation via the `Omni` API
- Maintainer: Community

## When to use this recipe

Use this recipe to deploy `nvidia/Cosmos3-Edge` for image and video generation.
A single pipeline class (`Cosmos3OmniDiffusersPipeline`) serves these modes; the
mode is selected per request:

- **T2I** — `POST /v1/images/generations` (or a prompt carrying `modalities=["image"]`).
- **T2V** — `POST /v1/videos/sync` with `num_frames > 1` and no reference image/video.
- **I2V** — `POST /v1/videos/sync` with a reference image (`input_reference` file
  upload, or `image_reference` JSON).
- **Action** — pass `extra_params={"action_mode": ...}` to drive Physical-AI tasks:
  - `forward_dynamics` — given a first frame or video **and** an action trajectory,
    roll out the resulting video. Synchronous: `POST /v1/videos/sync`.
  - `policy` — given a first frame or video and a language instruction,
    **predict** the action trajectory (and a rollout video). Use the async
    `POST /v1/videos` endpoint and read the predicted action from the top-level
    `action` field.
  - `inverse_dynamics` — given a video, **recover** the action trajectory. Use
    the async `POST /v1/videos` endpoint and read the recovered action from
    the top-level `action` field
    (`{data, shape, dtype, raw_action_dim, domain_id}`).

  Action requests also take `domain_name` (e.g. `av`, `bridge_orig_lerobot`,
  `droid_lerobot`, `agibotworld`, …; or a numeric `domain_id`), `raw_action_dim`,
  and `action_chunk_size` (must equal `num_frames` or `num_frames - 1`). For
  `forward_dynamics` also pass the `action` array. The dedicated policy checkpoint
  **`nvidia/Cosmos3-Edge-Policy-DROID`** is served the same way
  (`domain_name=droid_lerobot`).

- **DROID OpenPI policy server** — serve `nvidia/Cosmos3-Edge-Policy-DROID` and
  connect an OpenPI-compatible websocket client to `/v1/realtime/robot/openpi`.
  This path returns action chunks directly instead of an mp4.

  Action requests can use `input_reference` or `video_reference` for video input.
  `policy` and `forward_dynamics` can also use an image reference; `inverse_dynamics`
  requires a video reference.

## References

- Model card: <https://huggingface.co/nvidia/Cosmos3-Edge>
- Example prompts, inputs, and outputs will be provided later.
- Prompt upsampling (recommended for quality): the model expects JSON-upsampled
  structured prompts; see NVIDIA's `cosmos-framework` prompt-upsampling docs.
- Pipeline: [`vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py`](../../vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py)
<!-- - Smoke tests (canonical request formats): [`tests/e2e/accuracy/test_cosmos3_similarity.py`](../../tests/e2e/accuracy/test_cosmos3_similarity.py) -->

## Hardware Support

## GPU

### 1x H200 141GB / B300 (Online serving)

#### Environment

- OS: Ubuntu 22.04+
- Python: 3.12+
- Driver / runtime: NVIDIA CUDA environment
- vLLM version: match the repository requirements from your current checkout
- vLLM-Omni version or commit: use the commit you are deploying from

#### Command

Requires the `vllm-omni` package (or the `vllm/vllm-omni:cosmos3` container),
which provides the `vllm serve … --omni` entrypoint used below.

Safety guardrails are **on by default** (NVIDIA Open Model License). They load
the **gated** `nvidia/Cosmos-1.0-Guardrail` model, so to keep them on you must:

1. `pip install cosmos-guardrail`
2. Accept the license at <https://huggingface.co/nvidia/Cosmos-1.0-Guardrail>
3. Export a token with access: `export HF_TOKEN=hf_...`

Then launch the recommended server:

```bash
vllm serve nvidia/Cosmos3-Edge \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --init-timeout 1800
```

To run **without** guardrails (you are responsible for license compliance),
add `--no-guardrails` (no token/`cosmos-guardrail` needed). For extra GPUs use
`--ulysses-degree N` (context parallel) or `--tensor-parallel-size N`;
`--enable-layerwise-offload` reduces VRAM on smaller GPUs. The pipeline
auto-resolves from `model_index.json`; pass
`--model-class-name Cosmos3OmniDiffusersPipeline` to force it explicitly.

#### Examples

Example requests and reference outputs will be provided later.

#### Notes

- **Performance:** TBD. Checkpoint is not yet released
- **Memory:** TBD. Checkpoint is not yet released
- **Determinism:** identical seed reproduces identical output on the same
  hardware; outputs are not bit-identical across different GPU types.
- **Supported sizes:** 256p and 480p at 16:9, 4:3, 1:1, 3:4, and 9:16.
  Defaults: T2I 640×640, 50 steps, guidance 7.0, `flow_shift=3.0`; T2V/I2V
  832×480, 189 frames at 24 FPS, 35 steps, guidance 5.0,
  `flow_shift=3.0`.
- **Key flags / params:** `--no-guardrails` (server) or
  `extra_params={"guardrails":false}` (per request) toggles safety. The
  per-request flag only takes effect when the server was launched **with**
  guardrails enabled (it cannot re-enable them on a `--no-guardrails` server).
  `use_resolution_template` / `use_duration_template` are off by default and only
  needed when not using upsampled prompts that already encode resolution/duration.
- **DROID OpenPI observations:** include a string `prompt`, either
  `observation/image` or the three-view DROID camera keys
  (`observation/wrist_image_left`, `observation/exterior_image_1_left`,
  `observation/exterior_image_2_left`), plus `observation/gripper_position` and
  `observation/joint_position`. Optional extra params include `history_length`,
  `conditioning_fps`, `action_chunk_size`, `raw_action_dim`, `deterministic_seed`,
  and `session_id`.
- **Known limitations:**
  - Guardrails-on requires `cosmos-guardrail` **and** access to the gated
    `nvidia/Cosmos-1.0-Guardrail` repo (accept license + `HF_TOKEN`); otherwise
    the server fails at pipeline build with a gated-repo / safety-checker error.
  - A guardrail-blocked prompt currently returns HTTP 500
    (`"Guardrail blocked prompt"`).
  - Action `forward_dynamics`, `policy`, and `inverse_dynamics` are supported
    online. Use async `POST /v1/videos` when you need the predicted/recovered
    action payload under the top-level `action` field; sync `/v1/videos/sync`
    returns raw MP4 bytes and does not expose action metadata in the response body.

### 1x GPU (Offline generation)

#### Environment

- OS: Ubuntu 22.04+
- Python: 3.12+
- Driver / runtime: NVIDIA CUDA environment
- vLLM-Omni version or commit: use the commit you are deploying from

#### Examples

Offline generation examples will be provided later.

#### Notes

- A single `Cosmos3OmniDiffusersPipeline` serves every mode; the standard examples
  select it automatically from `model_index.json`. T2I is chosen by the
  `text_to_image` prompt builder (which marks `modalities=["image"]`); `text_to_video`
  defaults to T2V; `image_to_video` adds `multi_modal_data={"image": ...}` (I2V).
- Model-specific knobs (`flow_shift`, `max_sequence_length`, `guardrails`,
  `action_*`, ...) are declared
  once in `vllm_omni/model_extras/cosmos3.py` and forwarded through `--extra-body`;
  unknown keys for the model are dropped.
