#!/bin/bash
# Cosmos3 video-with-sound example. Requires a sound-capable checkpoint.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8091}"
OUTPUT_PATH="${OUTPUT_PATH:-cosmos3_t2v_sound.mp4}"

curl -sS -X POST "${BASE_URL}/v1/videos/sync" \
  -F "prompt=A small warehouse robot rolls across the floor with soft motor sounds." \
  -F "negative_prompt=blurry, distorted, low quality" \
  -F "size=1280x720" \
  -F "num_frames=81" \
  -F "fps=24" \
  -F "num_inference_steps=35" \
  -F "guidance_scale=4.0" \
  -F "generate_sound=true" \
  -F "sound_duration=3.4" \
  -F "seed=42" \
  -o "${OUTPUT_PATH}"

echo "Saved video to ${OUTPUT_PATH}"
