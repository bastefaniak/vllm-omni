# Image-To-Video

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_video>.


This example demonstrates how to generate videos from images using Wan2.2 Image-to-Video models and Cosmos3 with vLLM-Omni's offline inference API.

## Supported Models

| Model | Default Resolution | Default Frames | Default Steps | Guidance |
|-------|--------------------|----------------|---------------|----------|
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | auto, 480p area | 81 | 50 | 5.0 |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | auto, 480p area | 81 | 50 | 5.0 |
| `$COSMOS3_MODEL` with `Cosmos3OmniDiffusersPipeline` | auto, 720p area | 81 | 35 | 4.0 |

## Local CLI Usage

Download the example image:

```bash
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg
```

### Wan2.2-I2V-A14B-Diffusers (MoE)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num-frames 48 \
  --guidance-scale 5.0 \
  --guidance-scale-high 6.0 \
  --num-inference-steps 40 \
  --boundary-ratio 0.875 \
  --flow-shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

### Wan2.2-TI2V-5B-Diffusers (Unified)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num-frames 48 \
  --guidance-scale 4.0 \
  --num-inference-steps 40 \
  --flow-shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

### Cosmos3

Cosmos3 uses one pipeline for text-to-image, text-to-video, and image-to-video. Set `COSMOS3_MODEL` to a local Diffusers-format Cosmos3 checkpoint or model reference, and select the pipeline explicitly.

```bash
export COSMOS3_MODEL=/path/to/cosmos3-diffusers

python image_to_video.py \
  --model "$COSMOS3_MODEL" \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "blurry, distorted, low quality" \
  --height 720 \
  --width 1280 \
  --num-frames 81 \
  --guidance-scale 4.0 \
  --num-inference-steps 35 \
  --fps 24 \
  --output cosmos3_i2v_output.mp4
```

For Cosmos3 I2V, the input image is resized and center-cropped by the pipeline. If `--height` and `--width` are omitted, this example chooses a 720p-area resolution from the input aspect ratio. Cosmos3 currently supports one prompt and one video per request, and model-level CPU offload is not supported; use `--enable-layerwise-offload` instead.

Key arguments:

- `--model`: Model ID (I2V-A14B for MoE, TI2V-5B for unified T2V+I2V).
- `--model-class-name`: explicit pipeline class. Use `Cosmos3OmniDiffusersPipeline` for Cosmos3 checkpoints.
- `--image`: Path to input image (required).
- `--prompt`: Text description of desired motion/animation.
- `--height/--width`: Output resolution (auto-calculated from image if not set). Dimensions should be multiples of 16.
- `--num-frames`: Number of frames (default is model-specific).
- `--guidance-scale` and `--guidance-scale-high`: CFG scale (applied to low/high-noise stages for MoE).
- `--negative-prompt`: Optional list of artifacts to suppress.
- `--boundary-ratio`: Boundary split ratio for two-stage MoE models.
- `--flow-shift`: Scheduler flow shift. Defaults are model-specific.
- `--sample-solver`: Wan2.2 sampling solver. Use `unipc` for the default multistep solver, or `euler` for Lightning/Distill checkpoints.
- `--num-inference-steps`: Number of denoising steps (default is model-specific).
- `--fps`: Frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--frame-rate`: Generation frame rate for models that use it. Defaults to `--fps`.
- `--output`: Path to save the generated video.
- `--vae-use-slicing`: Enable VAE slicing for memory optimization.
- `--vae-use-tiling`: Enable VAE tiling for memory optimization.
- `--cfg-parallel-size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion/parallelism/cfg_parallel.md).
- `--tensor-parallel-size`: tensor parallel size (effective for models that support TP, e.g. LTX2).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.
- `--use-hsdp`: Enable Hybrid Sharded Data Parallel to shard model weights across GPUs.
- `--hsdp-shard-size`: Number of GPUs to shard model weights across within each replica group. -1 (default) auto-calculates as world_size / replicate_size.
- `--hsdp-replicate-size`: Number of replica groups for HSDP. Each replica holds a full sharded copy. Default 1 means pure sharding (no replication).



> ℹ️ If you encounter OOM errors, try using `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage.

For Wan2.2 LightX2V-converted local Diffusers directories and related LoRA
assets, see the [LoRA guide](../../diffusion/lora.md#wan22-lightx2v-offline-assembly).

## Example materials

??? abstract "image_to_video.py"
    ``````py
    --8<-- "examples/offline_inference/image_to_video/image_to_video.py"
    ``````
