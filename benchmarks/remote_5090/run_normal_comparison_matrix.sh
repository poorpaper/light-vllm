#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "usage: $0 ROOT OUTPUT_ROOT" >&2
  exit 2
fi

ROOT=$1
OUTPUT_ROOT=$2
PORT=${PORT:-8050}

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite an existing comparison directory: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

run_one() {
  local rate=$1
  local mode=$2
  local output=${OUTPUT_ROOT}/rate${rate}
  mkdir -p "$output"
  PROFILE_DETAIL=off \
  RUNS=3 \
  WARMUP_LIMIT_OVERRIDE=64 \
  HARNESS_ROOT="$ROOT" \
  bash "$ROOT/benchmarks/remote_5090/run_runtime_gap_case.sh" \
    "$ROOT" "$mode" normal-loaded "$output" "$PORT" "$rate" \
    >"${OUTPUT_ROOT}/rate${rate}-${mode}.log" 2>&1
  PORT=$((PORT + 1))
}

# 改变各 rate 的启动顺序，避免温度与频率漂移总偏向同一后端。
run_one 2 light-strict
run_one 2 vllm-eager
run_one 2 vllm-default
run_one 4 vllm-eager
run_one 4 light-strict
run_one 4 vllm-default
run_one 6 light-strict
run_one 6 vllm-default
run_one 6 vllm-eager
run_one 8 vllm-default
run_one 8 vllm-eager
run_one 8 light-strict
