#!/bin/bash
# Cosmos3 image-to-video example using the sync video API.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8091}"
IMAGE_PATH="${IMAGE_PATH:-cherry_blossom.jpg}"
OUTPUT_PATH="${OUTPUT_PATH:-cosmos3_i2v.mp4}"

curl -sS -X POST "${BASE_URL}/v1/videos/sync" \
  -F "prompt=Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  -F "negative_prompt=blurry, distorted, low quality" \
  -F "input_reference=@${IMAGE_PATH}" \
  -F "size=1280x720" \
  -F "num_frames=81" \
  -F "fps=24" \
  -F "num_inference_steps=35" \
  -F "guidance_scale=4.0" \
  -F "seed=42" \
  -o "${OUTPUT_PATH}"

echo "Saved video to ${OUTPUT_PATH}"
