# Cosmos3

Cosmos3 uses `Cosmos3OmniDiffusersPipeline` for text-to-image, text-to-video, image-to-video, video-with-sound, and action generation (policy, forward dynamics, inverse dynamics). Examples default to the `nvidia/Cosmos3-Nano` Hugging Face repo; override with `--model` or by exporting `COSMOS3_MODEL` to a local Diffusers-format checkpoint.

## Canonical inputs

Each modality has a JSON file under `inputs/` carrying the long-form prompt and the matching sampling parameters:

| Modality                          | Input file                                       |
| --------------------------------- | ------------------------------------------------ |
| Text-to-Image                     | `inputs/t2i.json`                                |
| Text-to-Video                     | `inputs/t2v.json`                                |
| Text-to-Video with sound          | `inputs/t2v_sound.json`                          |
| Image-to-Video                    | `inputs/i2v.json`                                |
| Action — policy (robot)           | `inputs/action_policy_robot.json`                |
| Action — policy (AV)              | `inputs/action_policy_av.json`                   |
| Action — forward dynamics (robot) | `inputs/action_forward_dynamics_robot.json`      |
| Action — forward dynamics (AV)    | `inputs/action_forward_dynamics_av.json`         |
| Action — forward dynamics (camera)| `inputs/action_forward_dynamics_camera.jsonl`    |
| Action — inverse dynamics (robot) | `inputs/action_inverse_dynamics_robot.json`      |
| Action — inverse dynamics (AV)    | `inputs/action_inverse_dynamics_av.json`         |

Pass any of these to `--input-json`. Recognized fields (`prompt`, `negative_prompt`, `vision_path`, `action_path`, `height`, `width`, `num_frames`, `num_inference_steps`, `guidance_scale`, `fps`, `seed`, `action_mode`, `action_chunk_size`, `raw_action_dim`, `domain_name`, `domain_id`, `generate_sound`, `sound_duration`) override the task defaults; explicit CLI flags still win over the JSON record.

`vision_path` and `action_path` may be local paths or `http(s)` URLs. Remote assets are downloaded to a cache directory (`COSMOS3_EXAMPLE_CACHE`, defaults to `$TMPDIR/cosmos3_examples`).

JSONL inputs (e.g. `action_forward_dynamics_camera.jsonl`) generate one output per record, with `_0`, `_1`, … appended to the output stem.

## Text-to-Image

```bash
python end2end.py --task t2i --input-json inputs/t2i.json --output cosmos3_t2i.png
```

## Text-to-Video

```bash
python end2end.py --task t2v --input-json inputs/t2v.json --output cosmos3_t2v.mp4
```

## Image-to-Video

The companion image (`robot_153.jpg`) is referenced by URL inside `inputs/i2v.json` and auto-cached on first run.

```bash
python end2end.py --task i2v --input-json inputs/i2v.json --output cosmos3_i2v.mp4
```

To use your own image, override the vision path:

```bash
python end2end.py --task i2v --input-json inputs/i2v.json --vision-path /path/to/image.jpg --prompt "..."
```

## Video With Sound

```bash
python end2end.py --task t2v_sound --input-json inputs/t2v_sound.json --output cosmos3_t2v_sound.mp4
```

The JSON sets `generate_sound: true` and `sound_duration: 3.4`; override on the command line with `--sound-duration` if needed.

## Action — Policy

Policy mode consumes an image plus a language instruction and returns a video together with the predicted action chunk. The bundled vision asset for these modes is a video clip (`bridge_0.mp4` / `av_vision_25_*.mp4`); end2end.py auto-extracts the first frame for image-input modes (see [Video assets for image-input action modes](#video-assets-for-image-input-action-modes)).

Robot (`bridge_orig_lerobot`, `raw_action_dim=10`, `action_chunk_size=16`):

```bash
python end2end.py --task action_policy --input-json inputs/action_policy_robot.json \
  --output cosmos3_action_policy_robot.mp4 \
  --action-output cosmos3_action_policy_robot_action.json
```

Autonomous vehicle (`raw_action_dim=9`, `action_chunk_size=60`, "Please go backward"):

```bash
python end2end.py --task action_policy --input-json inputs/action_policy_av.json \
  --output cosmos3_action_policy_av.mp4 \
  --action-output cosmos3_action_policy_av_action.json
```

## Action — Forward Dynamics

Forward dynamics consumes a vision input plus a chunk of action data and predicts the resulting video.
When the vision input is a video, the example uses the first `action_chunk_size + 1` frames to match
native Cosmos3 conditioning. `--action-path` (URL or local path) is required; the JSON points at the
cosmos-dependencies asset and gets cached locally on first run.

Robot:

```bash
python end2end.py --task action_forward_dynamics \
  --input-json inputs/action_forward_dynamics_robot.json \
  --output cosmos3_forward_dynamics_robot.mp4
```

Autonomous vehicle:

```bash
python end2end.py --task action_forward_dynamics \
  --input-json inputs/action_forward_dynamics_av.json \
  --output cosmos3_forward_dynamics_av.mp4
```

Camera-pose (JSONL with two scenes — `mountain` and `solar`):

```bash
python end2end.py --task action_forward_dynamics \
  --input-json inputs/action_forward_dynamics_camera.jsonl \
  --output cosmos3_forward_dynamics_camera.mp4
# Produces cosmos3_forward_dynamics_camera_0.mp4 and cosmos3_forward_dynamics_camera_1.mp4
```

## Action — Inverse Dynamics

Inverse dynamics consumes a video plus a language instruction and predicts the action chunk. Video input is fed through `multi_modal_data["video"]`. The action JSON is written to the `--action-output` path.

Robot:

```bash
python end2end.py --task action_inverse_dynamics \
  --input-json inputs/action_inverse_dynamics_robot.json \
  --output cosmos3_inverse_dynamics_robot.mp4 \
  --action-output cosmos3_inverse_dynamics_robot_action.json
```

Autonomous vehicle:

```bash
python end2end.py --task action_inverse_dynamics \
  --input-json inputs/action_inverse_dynamics_av.json \
  --output cosmos3_inverse_dynamics_av.mp4 \
  --action-output cosmos3_inverse_dynamics_av_action.json
```

## Video assets for action modes

`forward_dynamics` uses the first `action_chunk_size + 1` frames when `--vision-path` resolves to a
video file, matching the native Cosmos3 action loader. Still images are also accepted as a fallback.
`policy` uses a still image; when its `--vision-path` resolves to a video file, end2end.py extracts
the first frame automatically. Video frame loading requires `imageio` with the ffmpeg plugin:

```bash
pip install "imageio[ffmpeg]"
```

To bypass video loading/extraction, pass `--vision-path /path/to/still.jpg`.

## Common Options

- `--input-json PATH`: load any of the `inputs/*.json` or `inputs/*.jsonl` records; CLI flags still override individual fields.
- `--vision-path PATH_OR_URL`: image or video input (alias `--image` is kept for back-compat).
- `--action-path PATH_OR_URL`: action JSON for forward-dynamics.
- `--action-mode {forward_dynamics,inverse_dynamics,policy}`: override action_mode (otherwise derived from `--task`).
- `--generate-sound`: force-enable sound generation outside the `t2v_sound` task.
- `--enable-layerwise-offload`: use layerwise offload for memory-constrained runs.
- `--cache-backend cache_dit`: enable Cache-DiT where supported.
- `--cfg-parallel-size 2`, `--ulysses-degree`, `--tensor-parallel-size`, `--use-hsdp`: enable parallel execution options.
- `--height`, `--width`, `--num-frames`, `--num-inference-steps`, `--guidance-scale`, `--fps`: override JSON/task defaults.

Do not use model-level `--enable-cpu-offload` for Cosmos3. Use `--enable-layerwise-offload` instead.
