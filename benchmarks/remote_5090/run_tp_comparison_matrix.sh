#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 ROOT OUTPUT_DIR [PORT]" >&2
  exit 2
fi

ROOT=$1
OUTPUT_DIR=$2
PORT=${3:-8010}
HARNESS_ROOT=${HARNESS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-/root/autodl-tmp/conda-envs/vllm/bin/python}
RUNS=${RUNS:-3}

if [[ ! -d "$ROOT" ]]; then
  echo "repository root does not exist: $ROOT" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to overwrite comparison output: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

run_configuration() {
  local label=$1
  local mode=$2
  local tensor_parallel_size=$3
  local engine_process=$4
  local output=${OUTPUT_DIR}/${label}

  mkdir -p "$output"
  echo "running ${label}: mode=${mode}, tp=${tensor_parallel_size}"
  PROFILE_DETAIL=off \
  ENGINE_PROCESS="$engine_process" \
  TENSOR_PARALLEL_SIZE="$tensor_parallel_size" \
  WARMUP_LIMIT_OVERRIDE=64 \
  NCCL_DEBUG=INFO \
  RUNS="$RUNS" \
  PYTHON="$PYTHON" \
  HARNESS_ROOT="$HARNESS_ROOT" \
    bash "$HARNESS_ROOT/benchmarks/remote_5090/run_runtime_gap_case.sh" \
      "$ROOT" "$mode" runtime "$output" "$PORT"
}

# light-vLLM TP=1/2 都让 Engine 使用同一条 ExecutionLane，避免 scaling
# 指标同时混入独立 Engine 进程与 cooperative-inline 的拓扑差异。
run_configuration light-tp1 light-strict 1 0
run_configuration vllm-eager-tp1 vllm-eager 1 0
run_configuration light-tp2 light-strict 2 0
run_configuration vllm-eager-tp2 vllm-eager 2 0

echo "tensor-parallel comparison completed: $OUTPUT_DIR"
