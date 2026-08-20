#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 ROOT MODE CASE OUTPUT_DIR [PORT] [REQUEST_RATE]" >&2
  exit 2
fi

ROOT=$1
MODE=$2
CASE=$3
OUTPUT_DIR=$4
PORT=${5:-8010}
REQUEST_RATE=${6:-}

PYTHON=${PYTHON:-/root/autodl-tmp/conda-envs/vllm/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct}
WORKLOAD_DIR=${WORKLOAD_DIR:-/root/autodl-tmp/light-vllm-results/workloads}
REPLAY_REFERENCE=${REPLAY_REFERENCE:-/root/autodl-tmp/light-vllm-results/preemption-latest-20260820/replay-reference.json}
PROFILE_DETAIL=${PROFILE_DETAIL:-full}
RUNS=${RUNS:-3}
SERVER_PREFIX=${MODE}-${CASE}
LOG=${OUTPUT_DIR}/${SERVER_PREFIX}-server.log

configure_case() {
  local benchmark_case=$1
  case "$benchmark_case" in
  fixed)
    WORKLOAD=${WORKLOAD_DIR}/decode-steady.json
    ARRIVAL_MODE=burst
    RATE=1
    KV_TOKENS=32768
    WARMUP_LIMIT=16
    USE_REPLAY=0
    ;;
  normal-low)
    WORKLOAD=${WORKLOAD_DIR}/production-sharegpt-eos.json
    ARRIVAL_MODE=poisson
    RATE=2
    KV_TOKENS=32768
    WARMUP_LIMIT=8
    USE_REPLAY=1
    ;;
  normal-loaded)
    WORKLOAD=${WORKLOAD_DIR}/production-sharegpt-eos.json
    ARRIVAL_MODE=poisson
    RATE=${REQUEST_RATE:-8}
    KV_TOKENS=32768
    WARMUP_LIMIT=8
    USE_REPLAY=1
    ;;
  pressure)
    WORKLOAD=${WORKLOAD_DIR}/production-sharegpt-eos.json
    ARRIVAL_MODE=poisson
    RATE=${REQUEST_RATE:-8}
    KV_TOKENS=4096
    WARMUP_LIMIT=8
    USE_REPLAY=1
    ;;
  *)
    echo "unsupported case: $benchmark_case" >&2
    exit 2
    ;;
  esac
}

if [[ "$CASE" == runtime ]]; then
  BENCHMARK_CASES=(fixed normal-low normal-loaded)
  PROFILE_CASE=normal-loaded
  configure_case fixed
else
  BENCHMARK_CASES=("$CASE")
  PROFILE_CASE=$CASE
  configure_case "$CASE"
fi

for validation_case in "${BENCHMARK_CASES[@]}"; do
  configure_case "$validation_case"
  if [[ ! -d "$ROOT" || ! -d "$MODEL" || ! -f "$WORKLOAD" ]]; then
    echo "benchmark input is missing for case: $validation_case" >&2
    exit 2
  fi
  if [[ "$USE_REPLAY" == 1 && ! -f "$REPLAY_REFERENCE" ]]; then
    echo "replay reference is missing: $REPLAY_REFERENCE" >&2
    exit 2
  fi
done
configure_case "${BENCHMARK_CASES[0]}"
if [[ -e "$LOG" ]]; then
  echo "refusing to overwrite an existing case: $SERVER_PREFIX" >&2
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

MAX_SEQS=16
KV_BLOCKS=$((KV_TOKENS / 16))
KV_BYTES=$((KV_TOKENS * 57344))
BACKEND=light-vllm
MODEL_ARGS=()
PROFILE_PREFIX=${OUTPUT_DIR}/${MODE}-${PROFILE_CASE}-profile
SERVER_ENV=(PYTHONPATH=${ROOT}/src)
if [[ "$PROFILE_DETAIL" == off ]]; then
  LIGHT_ENTRY=("$PYTHON" -m light_vllm.entrypoints.http)
else
  LIGHT_ENTRY=("$PYTHON" benchmarks/remote_5090/profile_light_cpu_stages.py)
fi
LIGHT_COMMON=(
  "${LIGHT_ENTRY[@]}"
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
  --max-num-sequences "$MAX_SEQS"
  --max-num-scheduled-tokens 512
  --num-kv-blocks "$KV_BLOCKS"
  --kv-block-size 16
  --max-pending-requests off
  --ttft-kv-cache-watermark off
)

case "$MODE" in
  light-strict)
    SERVER=("${LIGHT_COMMON[@]}")
    if [[ "$PROFILE_DETAIL" != off ]]; then
      SERVER_ENV+=(
        LIGHT_VLLM_CPU_PROFILE_OUTPUT=${PROFILE_PREFIX}-stages.json
        LIGHT_VLLM_CPU_PROFILE_DETAIL=$PROFILE_DETAIL
      )
    fi
    ;;
  light-optimistic)
    SERVER=(
      "${LIGHT_COMMON[@]}"
      --enable-self-resubmit
      --max-self-resubmits 2
      --self-resubmit-strict-fallback-rolled-back-tokens 4096
      --self-resubmit-initial-extra-blocks 1
      --self-resubmit-kv-admission-watermark 0.9
    )
    if [[ "$PROFILE_DETAIL" != off ]]; then
      SERVER_ENV+=(
        LIGHT_VLLM_CPU_PROFILE_OUTPUT=${PROFILE_PREFIX}-stages.json
        LIGHT_VLLM_CPU_PROFILE_DETAIL=$PROFILE_DETAIL
      )
    fi
    ;;
  vllm-eager|vllm-default|vllm-nopreempt4)
    BACKEND=vllm
    MODEL_ARGS=(--model "$MODEL")
    if [[ "$MODE" == vllm-nopreempt4 ]]; then
      MAX_SEQS=4
    fi
    SERVER=(
      bash benchmarks/remote_5090/launch_vllm.sh
      "$MODEL" "$KV_BYTES" none "$MAX_SEQS" 512 "$PORT"
    )
    SERVER_ENV=(
      PYTHONPATH=${ROOT}/benchmarks/remote_5090/vllm_profile_hook:${ROOT}/src
      VLLM_CPU_PROFILE_OUTPUT=${PROFILE_PREFIX}.{pid}.json
    )
    if [[ "$MODE" == vllm-eager ]]; then
      SERVER_ENV+=(VLLM_ENFORCE_EAGER=1)
    fi
    ;;
  *)
    echo "unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

nohup env "${SERVER_ENV[@]}" "${SERVER[@]}" >"$LOG" 2>&1 &
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
  local limit=${2:-}
  local output=${OUTPUT_DIR}/${PREFIX}-${suffix}.json
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite an existing run: $output" >&2
    exit 2
  fi
  local arguments=(
    "$PYTHON" benchmarks/remote_5090/serve_benchmark.py
    --backend "$BACKEND"
    --base-url "http://127.0.0.1:${PORT}"
    "${MODEL_ARGS[@]}"
    --workload "$WORKLOAD"
    --output "$output"
    --case "${PREFIX}-${suffix}"
    --arrival-mode "$ARRIVAL_MODE"
    --request-rate "$RATE"
    --seed 20260821
  )
  if [[ "$USE_REPLAY" == 1 ]]; then
    arguments+=(--replay-lengths-from "$REPLAY_REFERENCE")
  fi
  if [[ -n "$limit" ]]; then
    arguments+=(--limit "$limit")
  fi
  PYTHONPATH="$ROOT/src" "${arguments[@]}"
}

for benchmark_case in "${BENCHMARK_CASES[@]}"; do
  configure_case "$benchmark_case"
  PREFIX=${MODE}-${benchmark_case}
  run_case warmup "$WARMUP_LIMIT"
  for run in $(seq 1 "$RUNS"); do
    run_case "r${run}"
  done
done

if [[ "$PROFILE_DETAIL" != off ]]; then
  if [[ "$BACKEND" == light-vllm ]]; then
    kill -USR1 "$SERVER_PID"
  else
    for pid_file in "${PROFILE_PREFIX}."*.pid; do
      [[ -f "$pid_file" ]] || continue
      profile_pid=$(<"$pid_file")
      kill -0 "$profile_pid" 2>/dev/null && kill -USR1 "$profile_pid"
    done
  fi
  run_case profile
  if [[ "$BACKEND" == light-vllm ]]; then
    kill -USR2 "$SERVER_PID"
  else
    for pid_file in "${PROFILE_PREFIX}."*.pid; do
      [[ -f "$pid_file" ]] || continue
      profile_pid=$(<"$pid_file")
      kill -0 "$profile_pid" 2>/dev/null && kill -USR2 "$profile_pid"
    done
  fi
  sleep 1
fi

BENCHMARK_MODE="$MODE" \
BENCHMARK_CASE="$CASE" \
BENCHMARK_MODEL="$MODEL" \
BENCHMARK_KV_TOKENS="$KV_TOKENS" \
BENCHMARK_MAX_SEQS="$MAX_SEQS" \
"$PYTHON" - <<'PY' >"${OUTPUT_DIR}/${SERVER_PREFIX}-environment.json"
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import torch

def command(*args):
    return subprocess.check_output(args, text=True).strip()

print(json.dumps({
    "git_sha": command("git", "rev-parse", "HEAD"),
    "git_status": command("git", "status", "--short"),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": command(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ),
    "mode": os.environ.get("BENCHMARK_MODE"),
    "case": os.environ.get("BENCHMARK_CASE"),
    "kv_tokens": int(os.environ["BENCHMARK_KV_TOKENS"]),
    "max_num_sequences": int(os.environ["BENCHMARK_MAX_SEQS"]),
    "model": os.environ["BENCHMARK_MODEL"],
    "model_config_sha256": hashlib.sha256(
        (Path(os.environ["BENCHMARK_MODEL"]) / "config.json").read_bytes()
    ).hexdigest(),
}, indent=2))
PY
