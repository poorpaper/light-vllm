#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 ROOT LABEL DETAIL RUN [OUTPUT_DIR] [PORT]" >&2
  exit 2
fi

ROOT=$1
LABEL=$2
DETAIL=$3
RUN=$4
OUTPUT_DIR=${5:-/root/autodl-tmp/light-vllm-results/cpu-fix-ab-20260820}
PORT=${6:-8010}

PYTHON=/root/autodl-tmp/conda-envs/vllm/bin/python
MODEL=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
WORKLOAD=/root/autodl-tmp/light-vllm-results/workloads/decode-steady.json
PREFIX=${LABEL}-${DETAIL}-r${RUN}
STAGES=${OUTPUT_DIR}/${PREFIX}-stages.json
RESULT=${OUTPUT_DIR}/${PREFIX}.json
WARMUP=${OUTPUT_DIR}/${PREFIX}-warmup.json
LOG=${OUTPUT_DIR}/${PREFIX}.log

if [[ ! -d "$ROOT" || ! -f "$WORKLOAD" || ! -d "$MODEL" ]]; then
  echo "benchmark input is missing" >&2
  exit 2
fi
if [[ -e "$STAGES" || -e "$RESULT" || -e "$WARMUP" || -e "$LOG" ]]; then
  echo "refusing to overwrite an existing run: $PREFIX" >&2
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

PYTHONPATH="$ROOT/src" \
LIGHT_VLLM_CPU_PROFILE_OUTPUT="$STAGES" \
LIGHT_VLLM_CPU_PROFILE_DETAIL="$DETAIL" \
nohup "$PYTHON" benchmarks/remote_5090/profile_light_cpu_stages.py \
  --host 127.0.0.1 \
  --port "$PORT" \
  --architecture qwen2 \
  --loader safetensors \
  --weights "$MODEL" \
  --device cuda:0 \
  --dtype bfloat16 \
  --runtime engine \
  --kv-reservation blocks \
  --paged-attention-backend triton \
  --max-num-sequences 16 \
  --max-num-scheduled-tokens 512 \
  --num-kv-blocks 2048 \
  --kv-block-size 16 \
  >"$LOG" 2>&1 &
SERVER_PID=$!

ready=false
for _ in $(seq 1 180); do
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
  tail -80 "$LOG" >&2
  exit 1
fi

PYTHONPATH="$ROOT/src" "$PYTHON" benchmarks/remote_5090/serve_benchmark.py \
  --backend light-vllm \
  --base-url "http://127.0.0.1:${PORT}" \
  --workload "$WORKLOAD" \
  --output "$WARMUP" \
  --case "${PREFIX}-warmup" \
  --arrival-mode burst

kill -USR1 "$SERVER_PID"
sleep 0.2

PYTHONPATH="$ROOT/src" "$PYTHON" benchmarks/remote_5090/serve_benchmark.py \
  --backend light-vllm \
  --base-url "http://127.0.0.1:${PORT}" \
  --workload "$WORKLOAD" \
  --output "$RESULT" \
  --case "$PREFIX" \
  --arrival-mode burst

kill -USR2 "$SERVER_PID"
for _ in $(seq 1 100); do
  [[ -s "$STAGES" ]] && break
  sleep 0.1
done
[[ -s "$STAGES" ]]
