#!/bin/bash
# Cosmos3 online serving startup script.

set -euo pipefail

MODEL="${MODEL:-${COSMOS3_MODEL:-}}"
PORT="${PORT:-8091}"
CACHE_BACKEND="${CACHE_BACKEND:-none}"
ENABLE_LAYERWISE_OFFLOAD="${ENABLE_LAYERWISE_OFFLOAD:-0}"
CFG_PARALLEL_SIZE="${CFG_PARALLEL_SIZE:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-1}"
USE_HSDP="${USE_HSDP:-0}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/}"

if [ -z "${MODEL}" ]; then
  echo "Set COSMOS3_MODEL or MODEL to a local Diffusers-format Cosmos3 checkpoint."
  exit 1
fi

args=(
  vllm serve "${MODEL}"
  --omni
  --port "${PORT}"
  --model-class-name Cosmos3OmniDiffusersPipeline
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}"
  --cfg-parallel-size "${CFG_PARALLEL_SIZE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)

if [ "${ULYSSES_DEGREE}" != "1" ]; then
  args+=(--usp "${ULYSSES_DEGREE}")
fi

if [ "${CACHE_BACKEND}" != "none" ]; then
  args+=(--cache-backend "${CACHE_BACKEND}")
fi

if [ "${ENABLE_LAYERWISE_OFFLOAD}" != "0" ]; then
  args+=(--enable-layerwise-offload)
fi

if [ "${USE_HSDP}" != "0" ]; then
  args+=(--use-hsdp)
fi

echo "Starting Cosmos3 server on port ${PORT}"
exec "${args[@]}"
