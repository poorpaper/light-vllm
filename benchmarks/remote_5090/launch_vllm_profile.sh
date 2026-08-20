#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:?model path is required}
KV_CACHE_BYTES=${2:?KV cache bytes are required}
TRACE_DIR=${3:?trace directory is required}
PORT=${4:-8000}
VLLM_BIN=${VLLM_BIN:-/root/autodl-tmp/conda-envs/vllm/bin/vllm}
PROFILE_DELAY=${PROFILE_DELAY:-12}
PROFILE_MAX_ITERATIONS=${PROFILE_MAX_ITERATIONS:-30}

mkdir -p "$TRACE_DIR"
export VLLM_USE_FLASHINFER_SAMPLER=0

arguments=(
  serve "$MODEL_PATH"
  --host 127.0.0.1
  --port "$PORT"
  --dtype bfloat16
  --max-model-len 4096
  --block-size 16
  --kv-cache-memory-bytes "$KV_CACHE_BYTES"
  --max-num-seqs 16
  --max-num-batched-tokens 512
  --enable-chunked-prefill
  --no-enable-prefix-caching
  --generation-config vllm
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TRACE_DIR\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true,\"delay_iterations\":$PROFILE_DELAY,\"max_iterations\":$PROFILE_MAX_ITERATIONS}"
)

if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
  arguments+=(--enforce-eager)
fi

exec "$VLLM_BIN" "${arguments[@]}"
