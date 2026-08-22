#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 BASELINE_ROOT CANDIDATE_ROOT OUTPUT_ROOT CASE [REQUEST_RATE]" >&2
  exit 2
fi

BASELINE_ROOT=$1
CANDIDATE_ROOT=$2
OUTPUT_ROOT=$3
CASE=$4
REQUEST_RATE=${5:-8}
PORT=${PORT:-8030}
HARNESS_ROOT=${HARNESS_ROOT:-$CANDIDATE_ROOT}

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite an existing A/B directory: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

run_one() {
  local label=$1
  local root=$2
  local round=$3
  local output=${OUTPUT_ROOT}/${label}-r${round}
  local engine_process=${BASELINE_ENGINE_PROCESS:-0}
  if [[ "$label" == candidate ]]; then
    engine_process=${CANDIDATE_ENGINE_PROCESS:-0}
  fi
  ENGINE_PROCESS="$engine_process" \
  PROFILE_DETAIL=off \
  RUNS=1 \
  HARNESS_ROOT="$HARNESS_ROOT" \
  bash "$HARNESS_ROOT/benchmarks/remote_5090/run_runtime_gap_case.sh" \
    "$root" light-strict "$CASE" "$output" "$PORT" "$REQUEST_RATE" \
    >"${OUTPUT_ROOT}/${label}-r${round}.log" 2>&1
  PORT=$((PORT + 1))
}

# 交替启动顺序，避免把温度或频率漂移固定算到同一版本。
run_one baseline "$BASELINE_ROOT" 1
run_one candidate "$CANDIDATE_ROOT" 1
run_one candidate "$CANDIDATE_ROOT" 2
run_one baseline "$BASELINE_ROOT" 2
run_one baseline "$BASELINE_ROOT" 3
run_one candidate "$CANDIDATE_ROOT" 3
