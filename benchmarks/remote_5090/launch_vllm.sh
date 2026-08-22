#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:?model path is required}
KV_CACHE_BYTES=${2:?KV cache bytes are required}
SPEC_MODE=${3:-none}
MAX_NUM_SEQS=${4:-16}
MAX_BATCHED_TOKENS=${5:-512}
PORT=${6:-8000}
VLLM_BIN=${VLLM_BIN:-/root/autodl-tmp/conda-envs/vllm/bin/vllm}

arguments=(
  serve "$MODEL_PATH"
  --host 127.0.0.1
  --port "$PORT"
  --dtype bfloat16
  --max-model-len 4096
  --block-size 16
  --kv-cache-memory-bytes "$KV_CACHE_BYTES"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --enable-chunked-prefill
  --no-enable-prefix-caching
  --generation-config vllm
)

if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
  arguments+=(--enforce-eager)
fi

case "$SPEC_MODE" in
  none)
    ;;
  ngram)
    arguments+=(
      --speculative-config
      '{"method":"ngram","num_speculative_tokens":7,"prompt_lookup_min":2,"prompt_lookup_max":5}'
    )
    ;;
  *)
    echo "unsupported speculative mode: $SPEC_MODE" >&2
    exit 2
    ;;
esac

# FlashInfer's sampler capability check rejects this RTX 5090 environment.
# Attention remains on vLLM's selected FlashAttention backend.
export VLLM_USE_FLASHINFER_SAMPLER=0
exec "$VLLM_BIN" "${arguments[@]}"
