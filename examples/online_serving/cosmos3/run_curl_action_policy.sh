#!/bin/bash
# Cosmos3 action policy example. Requires an action-capable checkpoint.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8091}"
IMAGE_PATH="${IMAGE_PATH:-cherry_blossom.jpg}"
OUTPUT_PATH="${OUTPUT_PATH:-cosmos3_action_policy.mp4}"
ACTION_OUTPUT_PATH="${ACTION_OUTPUT_PATH:-cosmos3_action_policy_action.json}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

create_response=$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    -F "prompt=Predict the robot action for moving toward the target." \
    -F "input_reference=@${IMAGE_PATH}" \
    -F "size=640x480" \
    -F "num_frames=17" \
    -F "fps=24" \
    -F "num_inference_steps=30" \
    -F "guidance_scale=1.0" \
    -F 'extra_params={"action_mode":"policy","domain_name":"bridge_orig_lerobot","raw_action_dim":2,"action_chunk_size":16}' \
    -F "seed=42"
)

video_id="$(echo "${create_response}" | jq -r '.id')"
if [ -z "${video_id}" ] || [ "${video_id}" = "null" ]; then
  echo "Failed to create video job:"
  echo "${create_response}" | jq .
  exit 1
fi

echo "Created video job ${video_id}"
while true; do
  status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
  status="$(echo "${status_response}" | jq -r '.status')"

  case "${status}" in
    queued|in_progress)
      echo "Video job ${video_id} status: ${status}"
      sleep "${POLL_INTERVAL}"
      ;;
    completed)
      echo "${status_response}" | jq '.data[0].action' > "${ACTION_OUTPUT_PATH}"
      break
      ;;
    failed)
      echo "Video generation failed:"
      echo "${status_response}" | jq .
      exit 1
      ;;
    *)
      echo "Unexpected status response:"
      echo "${status_response}" | jq .
      exit 1
      ;;
  esac
done

curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content" -o "${OUTPUT_PATH}"

echo "Saved video to ${OUTPUT_PATH}"
echo "Saved action metadata to ${ACTION_OUTPUT_PATH}"
