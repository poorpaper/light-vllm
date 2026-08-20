#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "usage: $0 ROOT MODE OUTPUT_DIR [PORT] [ARRIVAL_MODE] [REQUEST_RATE]" >&2
  exit 2
fi

ROOT=$1
MODE=$2
OUTPUT_DIR=$3
PORT=${4:-8010}
ARRIVAL_MODE=${5:-burst}
REQUEST_RATE=${6:-1.0}

PYTHON=/root/autodl-tmp/conda-envs/vllm/bin/python
MODEL=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
WORKLOAD=/root/autodl-tmp/light-vllm-results/workloads/production-sharegpt-eos.json
REPLAY_REFERENCE=/root/autodl-tmp/light-vllm-results/preemption-latest-20260820/replay-reference.json
KV_CACHE_BYTES=234881024
LOG=${OUTPUT_DIR}/${MODE}-server.log

if [[ ! -d "$ROOT" || ! -d "$MODEL" || ! -f "$WORKLOAD" ]]; then
  echo "benchmark input is missing" >&2
  exit 2
fi
if [[ ! -f "$REPLAY_REFERENCE" ]]; then
  echo "replay reference is missing: $REPLAY_REFERENCE" >&2
  exit 2
fi
if [[ -e "$LOG" ]]; then
  echo "refusing to overwrite an existing mode: $MODE" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID"
    wait "$SERVER_PID" || true
  fi
}
trap cleanup EXIT

backend=light-vllm
model_args=()
case "$MODE" in
  light-strict)
    server=(
      "$PYTHON" -m light_vllm.entrypoints.http
      --host 127.0.0.1
      --port "$PORT"
      --architecture qwen2
      --loader safetensors
      --weights "$MODEL"
      --device cuda:0
      --dtype bfloat16
      --runtime engine
      --kv-reservation blocks
      --paged-attention-backend triton
      --max-num-sequences 16
      --max-num-scheduled-tokens 512
      --num-kv-blocks 256
      --kv-block-size 16
    )
    ;;
  light-rolling10)
    server=(
      "$PYTHON" -m light_vllm.entrypoints.http
      --host 127.0.0.1
      --port "$PORT"
      --architecture qwen2
      --loader safetensors
      --weights "$MODEL"
      --device cuda:0
      --dtype bfloat16
      --runtime engine
      --kv-reservation blocks
      --paged-attention-backend triton
      --max-num-sequences 16
      --max-num-scheduled-tokens 512
      --num-kv-blocks 256
      --kv-block-size 16
      --enable-nonpreemptive-sharing
      --nonpreemptive-guaranteed-sequences 1
      --nonpreemptive-watermark-ratio 0.1
    )
    ;;
  vllm-preempt16)
    backend=vllm
    model_args=(--model "$MODEL")
    server=(
      bash benchmarks/remote_5090/launch_vllm.sh
      "$MODEL" "$KV_CACHE_BYTES" none 16 512 "$PORT"
    )
    ;;
  vllm-nopreempt6)
    backend=vllm
    model_args=(--model "$MODEL")
    server=(
      bash benchmarks/remote_5090/launch_vllm.sh
      "$MODEL" "$KV_CACHE_BYTES" none 6 512 "$PORT"
    )
    ;;
  *)
    echo "unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

PYTHONPATH="$ROOT/src" nohup "${server[@]}" >"$LOG" 2>&1 &
SERVER_PID=$!

ready=false
for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/metrics" >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  tail -100 "$LOG" >&2
  exit 1
fi

run_case() {
  local suffix=$1
  local output=${OUTPUT_DIR}/${MODE}-${suffix}.json
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite an existing run: $output" >&2
    exit 2
  fi
  PYTHONPATH="$ROOT/src" "$PYTHON" benchmarks/remote_5090/serve_benchmark.py \
    --backend "$backend" \
    --base-url "http://127.0.0.1:${PORT}" \
    "${model_args[@]}" \
    --workload "$WORKLOAD" \
    --replay-lengths-from "$REPLAY_REFERENCE" \
    --output "$output" \
    --case "${MODE}-${suffix}" \
    --arrival-mode "$ARRIVAL_MODE" \
    --request-rate "$REQUEST_RATE" \
    --seed 20260820
}

run_case warmup
for run in 1 2 3; do
  run_case "r${run}"
done
