#!/bin/bash
# Cosmos3 text-to-image example using the images API.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8091}"
OUTPUT_PATH="${OUTPUT_PATH:-cosmos3_t2i.png}"

curl -sS -X POST "${BASE_URL}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A small warehouse robot carrying a blue box, clean product photography",
    "size": "1024x1024",
    "n": 1,
    "num_inference_steps": 50,
    "guidance_scale": 7.0,
    "negative_prompt": "blurry, distorted, low quality",
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > "${OUTPUT_PATH}"

echo "Saved image to ${OUTPUT_PATH}"
