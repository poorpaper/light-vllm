#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 ROOT MODEL OUTPUT_DIR [PORT]" >&2
  exit 2
fi

ROOT=$1
MODEL=$2
OUTPUT_DIR=$3
PORT=${4:-8020}
PYTHON=${PYTHON:-/root/autodl-tmp/conda-envs/vllm/bin/python}
LOG=${OUTPUT_DIR}/server.log

if [[ ! -d "$ROOT" || ! -d "$MODEL" ]]; then
  echo "repository or model directory does not exist" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to overwrite failure-test output: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
cd "$ROOT"

leader=""
request_pid=""
cleanup() {
  if [[ -n "$request_pid" ]] && kill -0 "$request_pid" 2>/dev/null; then
    kill -TERM "$request_pid" 2>/dev/null || true
    wait "$request_pid" 2>/dev/null || true
  fi
  if [[ -n "$leader" ]]; then
    kill -TERM -- "-$leader" 2>/dev/null || true
    wait "$leader" 2>/dev/null || true
  fi
}
trap cleanup EXIT

nohup setsid env \
  PYTHONPATH=${ROOT}/src \
  NCCL_DEBUG=INFO \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  TORCH_NCCL_ENABLE_MONITORING=1 \
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=15 \
  TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=1000 \
  "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc-per-node 2 \
    --module light_vllm.entrypoints.http \
    --host 127.0.0.1 \
    --port "$PORT" \
    --architecture qwen2 \
    --loader safetensors \
    --weights "$MODEL" \
    --device cuda \
    --dtype bfloat16 \
    --runtime engine \
    --kv-reservation blocks \
    --paged-attention-backend triton \
    --max-num-sequences 8 \
    --max-num-scheduled-tokens 128 \
    --num-kv-blocks 256 \
    --kv-block-size 16 \
    --distributed-timeout-seconds 10 \
    --tensor-parallel-size 2 >"$LOG" 2>&1 &
leader=$!

ready=false
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/metrics" >/dev/null; then
    ready=true
    break
  fi
  kill -0 "$leader" 2>/dev/null || break
  sleep 1
done
if [[ "$ready" != true ]]; then
  tail -100 "$LOG" >&2
  exit 1
fi

mapfile -t children < <(pgrep -P "$leader")
victim=""
for pid in "${children[@]}"; do
  if tr '\0' '\n' <"/proc/${pid}/environ" | grep -qx 'LOCAL_RANK=1'; then
    victim=$pid
    break
  fi
done
if [[ -z "$victim" ]]; then
  echo "rank 1 process was not found" >&2
  exit 1
fi

# 保持长请求在模型 step 中运行，再杀 Rank；这覆盖 idle worker 收割之外的
# collective 在途故障。请求结果只用于确认它确实已经发出。
curl -sS -X POST "http://127.0.0.1:${PORT}/generate" \
  -H 'Content-Type: application/json' \
  --data '{"input_ids":[1,2,3,4,5,6,7,8],"max_new_tokens":2048}' \
  >"${OUTPUT_DIR}/active-request.json" 2>"${OUTPUT_DIR}/active-request.stderr" &
request_pid=$!
active=false
for _ in $(seq 1 100); do
  kill -0 "$request_pid" 2>/dev/null || break
  utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -nr | head -1)
  if (( utilization > 0 )); then
    active=true
    break
  fi
  sleep 0.1
done
if [[ "$active" != true ]]; then
  echo "request did not enter GPU execution" >&2
  exit 1
fi

started_at=$(date +%s%N)
kill -TERM "$victim"
timed_out=false
deadline=$((SECONDS + 30))
while true; do
  mapfile -t remaining < <(
    pgrep -f "light_vllm.entrypoints.http.*--port ${PORT}" || true
  )
  if ! kill -0 "$leader" 2>/dev/null && [[ ${#remaining[@]} -eq 0 ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    timed_out=true
    kill -KILL -- "-$leader" 2>/dev/null || true
    if [[ ${#remaining[@]} -ne 0 ]]; then
      kill -KILL "${remaining[@]}" 2>/dev/null || true
    fi
    break
  fi
  sleep 0.1
done
set +e
wait "$leader"
exit_code=$?
set -e
finished_at=$(date +%s%N)
leader=""
for _ in $(seq 1 20); do
  kill -0 "$request_pid" 2>/dev/null || break
  sleep 0.1
done
if kill -0 "$request_pid" 2>/dev/null; then
  kill -TERM "$request_pid" 2>/dev/null || true
fi
wait "$request_pid" 2>/dev/null || true
request_pid=""

elapsed_ms=$(((finished_at - started_at) / 1000000))
for _ in $(seq 1 100); do
  mapfile -t remaining < <(
    pgrep -f "light_vllm.entrypoints.http.*--port ${PORT}" || true
  )
  [[ ${#remaining[@]} -eq 0 ]] && break
  sleep 0.1
done
port_released=false
if ! curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/metrics" >/dev/null 2>&1; then
  port_released=true
fi
gpu_memory_released=false
for _ in $(seq 1 100); do
  mapfile -t gpu_memory < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  if (( ${gpu_memory[0]} < 64 && ${gpu_memory[1]} < 64 )); then
    gpu_memory_released=true
    break
  fi
  sleep 0.1
done

"$PYTHON" - \
  "$OUTPUT_DIR/result.json" \
  "$victim" \
  "$exit_code" \
  "$elapsed_ms" \
  "$timed_out" \
  "$port_released" \
  "$gpu_memory_released" \
  "${remaining[*]}" \
  "${gpu_memory[*]}" <<'PY'
import json
import sys
from pathlib import Path

(
    path,
    victim,
    exit_code,
    elapsed_ms,
    timed_out,
    port_released,
    gpu_memory_released,
    remaining,
    gpu_memory,
) = sys.argv[1:]
payload = {
    "killed_local_rank": 1,
    "victim_pid": int(victim),
    "torchrun_exit_code": int(exit_code),
    "exit_elapsed_ms": int(elapsed_ms),
    "exit_timed_out": timed_out == "true",
    "port_released": port_released == "true",
    "gpu_memory_released": gpu_memory_released == "true",
    "remaining_server_pids": remaining.split(),
    "gpu_memory_used_mib": [int(value) for value in gpu_memory.split()],
}
Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

if [[
  "$exit_code" -eq 0
  || "$timed_out" == true
  || "$port_released" != true
  || "$gpu_memory_released" != true
  || ${#remaining[@]} -ne 0
]]; then
  exit 1
fi
