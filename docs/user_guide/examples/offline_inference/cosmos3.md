# Cosmos3

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/cosmos3>.


Cosmos3 uses `Cosmos3OmniDiffusersPipeline` for text-to-image, text-to-video, image-to-video, video-with-sound, and action generation. Examples default to the `nvidia/Cosmos3-Nano` Hugging Face repo, but you can override the checkpoint with `--model` or by exporting `COSMOS3_MODEL` to a local Diffusers-format checkpoint.

```bash
cd examples/offline_inference/cosmos3
```

## Text-to-Image

```bash
python end2end.py \
  --task t2i \
  --prompt "A small warehouse robot carrying a blue box, clean product photography" \
  --negative-prompt "blurry, distorted, low quality" \
  --output cosmos3_t2i.png
```

## Text-to-Video

```bash
python end2end.py \
  --task t2v \
  --prompt "A small warehouse robot moves a blue box across a clean floor." \
  --negative-prompt "blurry, distorted, low quality" \
  --output cosmos3_t2v.mp4
```

## Image-to-Video

Download an example image or provide your own image path.

```bash
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg

python end2end.py \
  --task i2v \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "blurry, distorted, low quality" \
  --output cosmos3_i2v.mp4
```

## Video With Sound

This path requires a sound-capable Cosmos3 checkpoint with `sound_gen` weights.

```bash
python end2end.py \
  --task t2v_sound \
  --prompt "A small warehouse robot rolls across the floor with soft motor sounds." \
  --negative-prompt "blurry, distorted, low quality" \
  --sound-duration 3.4 \
  --output cosmos3_t2v_sound.mp4
```

## Action Policy

This path requires an action-capable Cosmos3 checkpoint with `action_gen` weights. The example returns a video plus an action JSON payload. Pass either `--domain-name` or `--domain-id`.

```bash
python end2end.py \
  --task action_policy \
  --image cherry_blossom.jpg \
  --prompt "Predict the robot action for moving toward the target." \
  --domain-name bridge_orig_lerobot \
  --raw-action-dim 2 \
  --action-chunk-size 16 \
  --output cosmos3_action_policy.mp4 \
  --action-output cosmos3_action_policy_action.json
```

## Common Options

- `--enable-layerwise-offload`: use layerwise offload for memory-constrained runs.
- `--cache-backend cache_dit`: enable Cache-DiT where supported.
- `--cfg-parallel-size 2`, `--ulysses-degree`, `--tensor-parallel-size`, `--use-hsdp`: enable parallel execution options.
- `--height`, `--width`, `--num-frames`, `--num-inference-steps`, `--guidance-scale`, `--fps`: override task defaults.

Do not use model-level `--enable-cpu-offload` for Cosmos3. Use `--enable-layerwise-offload` instead.

## Example materials

??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/cosmos3/end2end.py"
    ``````
