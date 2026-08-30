#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 REPO MODEL OUTPUT_DIR [PORT]" >&2
  exit 2
fi

REPO=$(realpath "$1")
MODEL=$(realpath "$2")
OUTPUT_DIR=$3
PORT=${4:-8020}
PYTHON=${PYTHON:-python}
VLLM_PYTHON=${VLLM_PYTHON:-$PYTHON}
VLLM_BIN=${VLLM_BIN:-vllm}
CUDA_DEVICE=${CUDA_DEVICE:-0}
NUM_KV_BLOCKS=${NUM_KV_BLOCKS:-4096}
VLLM_KV_CACHE_BYTES=${VLLM_KV_CACHE_BYTES:-805306368}
MAX_NUM_SEQUENCES=${MAX_NUM_SEQUENCES:-64}
MAX_NUM_SCHEDULED_TOKENS=${MAX_NUM_SCHEDULED_TOKENS:-32768}

if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "repository is not a Git checkout: $REPO" >&2
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "model directory does not exist: $MODEL" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to overwrite result directory: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --short)" ]]; then
  echo "benchmark checkout must be clean so the candidate SHA is reproducible" >&2
  exit 2
fi
if nvidia-smi -i "$CUDA_DEVICE" --query-compute-apps=pid \
  --format=csv,noheader,nounits | grep -Eq '^[0-9]+$'; then
  echo "GPU already has a compute process; refusing to disturb an existing experiment" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"/{logs,raw/light-vllm,raw/vllm,workloads}
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
COMMANDS=$OUTPUT_DIR/commands.txt
SERVER_PID=""

record_command() {
  printf '%q ' "$@" >>"$COMMANDS"
  printf '\n' >>"$COMMANDS"
}

process_group_is_running() {
  ps -eo pgid=,stat= 2>/dev/null |
    awk -v target="$1" '$1 == target && $2 !~ /^Z/ { found = 1 } END { exit !found }'
}

cleanup_server() {
  local pid=$SERVER_PID
  [[ -z "$pid" ]] && return
  if process_group_is_running "$pid"; then
    # setsid 让服务及其子进程属于本次实验自己的进程组，只回收这一组进程。
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
      process_group_is_running "$pid" || break
      sleep 1
    done
    if process_group_is_running "$pid"; then
      kill -KILL -- "-$pid" 2>/dev/null || true
      for _ in $(seq 1 30); do
        process_group_is_running "$pid" || break
        sleep 1
      done
    fi
  fi
  if process_group_is_running "$pid"; then
    SERVER_PID=""
    echo "server process group did not exit: $pid" >&2
    return 1
  fi
  wait "$pid" 2>/dev/null || true
  SERVER_PID=""
}
trap cleanup_server EXIT

wait_for_gpu_idle() {
  for _ in $(seq 1 120); do
    if ! nvidia-smi -i "$CUDA_DEVICE" --query-compute-apps=pid \
      --format=csv,noheader,nounits | grep -Eq '^[0-9]+$'; then
      return
    fi
    sleep 1
  done
  echo "GPU processes from the previous server did not exit" >&2
  return 1
}

WORKLOAD_COMMAND=(
  "$PYTHON" "$REPO/benchmarks/remote_5090/make_workloads.py"
  --tokenizer "$MODEL"
  --output-dir "$OUTPUT_DIR/workloads"
)
record_command env "PYTHONPATH=$REPO/src" "${WORKLOAD_COMMAND[@]}"
PYTHONPATH="$REPO/src" "${WORKLOAD_COMMAND[@]}"

export REPO MODEL OUTPUT_DIR PYTHON VLLM_PYTHON VLLM_BIN CUDA_DEVICE
export NUM_KV_BLOCKS VLLM_KV_CACHE_BYTES MAX_NUM_SEQUENCES MAX_NUM_SCHEDULED_TOKENS
"$PYTHON" - <<'PY' >"$OUTPUT_DIR/environment.json"
import importlib.metadata
import json
import os
import platform
import subprocess


def command(*args: str, allow_failure: bool = False) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        if allow_failure:
            return None
        raise


try:
    import torch
except ImportError:
    torch = None

vllm_python = os.environ["VLLM_PYTHON"]
vllm_probe = (
    "import importlib.metadata,json,vllm; "
    "d=importlib.metadata.distribution('vllm'); "
    "print(json.dumps({'version': d.version, 'path': vllm.__file__, "
    "'direct_url': d.read_text('direct_url.json')}))"
)
vllm_metadata = command(
    vllm_python,
    "-c",
    vllm_probe,
    allow_failure=True,
)
print(json.dumps({
    "candidate_sha": command("git", "-C", os.environ["REPO"], "rev-parse", "HEAD"),
    "candidate_status": command("git", "-C", os.environ["REPO"], "status", "--short"),
    "candidate_remote": command("git", "-C", os.environ["REPO"], "remote", "get-url", "origin", allow_failure=True),
    "model": os.environ["MODEL"],
    "model_files_sha256": command(
        os.environ["PYTHON"],
        "-c",
        "import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); print('\\n'.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(p).as_posix()}' for f in sorted(p.rglob('*')) if f.is_file()))",
        os.environ["MODEL"],
    ),
    "python": platform.python_version(),
    "torch": None if torch is None else torch.__version__,
    "torch_cuda": None if torch is None else torch.version.cuda,
    "vllm": None if vllm_metadata is None else json.loads(vllm_metadata),
    "vllm_binary_version": command(os.environ["VLLM_BIN"], "--version", allow_failure=True),
    "gpu": command("nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"),
    "nvcc": command("nvcc", "--version", allow_failure=True),
    "platform": platform.platform(),
    "config": {
        "cuda_visible_device": os.environ["CUDA_DEVICE"],
        "num_kv_blocks": int(os.environ["NUM_KV_BLOCKS"]),
        "kv_block_size": 16,
        "vllm_kv_cache_bytes": int(os.environ["VLLM_KV_CACHE_BYTES"]),
        "max_num_sequences": int(os.environ["MAX_NUM_SEQUENCES"]),
        "max_num_scheduled_tokens": int(os.environ["MAX_NUM_SCHEDULED_TOKENS"]),
        "prefix_cache": False,
        "dtype": "float16",
        "decoding": "greedy, ignore EOS",
        "startup_order": ["light-vllm", "vllm", "vllm", "light-vllm", "light-vllm", "vllm"],
        "measurement_runs_per_start": 2,
    },
}, indent=2))
PY

wait_until_ready() {
  local backend=$1
  local endpoint=/health
  [[ "$backend" == light-vllm ]] && endpoint=/metrics
  for _ in $(seq 1 900); do
    if curl -fsS "http://127.0.0.1:${PORT}${endpoint}" >/dev/null; then
      return
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  echo "$backend server did not become ready" >&2
  return 1
}

start_server() {
  local backend=$1
  local session=$2
  local log=$OUTPUT_DIR/logs/${session}-${backend}.log
  local command=()
  if [[ "$backend" == light-vllm ]]; then
    command=(
      env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE" "PYTHONPATH=$REPO/src"
      "$PYTHON" -m light_vllm.entrypoints.http
      --host 127.0.0.1 --port "$PORT"
      --architecture qwen2.5 --loader safetensors
      --weights "$MODEL" --tokenizer "$MODEL" --served-model-name "$MODEL"
      --request-timeout-seconds 1800
      --quantization auto --quantization-backend cuda
      --device cuda:0 --dtype float16
      --runtime engine --engine-process
      --kv-reservation blocks --paged-attention-backend triton
      --num-kv-blocks "$NUM_KV_BLOCKS" --kv-block-size 16
      --max-num-sequences "$MAX_NUM_SEQUENCES"
      --max-num-scheduled-tokens "$MAX_NUM_SCHEDULED_TOKENS"
      --max-pending-requests off --ttft-kv-cache-watermark off
    )
  else
    command=(
      env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE" VLLM_USE_FLASHINFER_SAMPLER=0
      "$VLLM_BIN" serve "$MODEL"
      --host 127.0.0.1 --port "$PORT"
      --dtype float16 --max-model-len 4096 --block-size 16
      --kv-cache-memory-bytes "$VLLM_KV_CACHE_BYTES"
      --max-num-seqs "$MAX_NUM_SEQUENCES"
      --max-num-batched-tokens "$MAX_NUM_SCHEDULED_TOKENS"
      --enable-chunked-prefill --no-enable-prefix-caching
      --generation-config vllm --enforce-eager
    )
  fi
  record_command "${command[@]}"
  setsid "${command[@]}" >"$log" 2>&1 &
  SERVER_PID=$!
  if ! wait_until_ready "$backend"; then
    tail -100 "$log" >&2
    return 1
  fi
}

run_workload() {
  local backend=$1
  local session=$2
  local workload_path=$3
  local suffix=$4
  local name
  name=$(basename "$workload_path" .json)
  local output=$OUTPUT_DIR/raw/$backend/${session}-${name}-${suffix}.json
  local command=(
    "$PYTHON" "$REPO/benchmarks/remote_5090/serve_benchmark.py"
    --backend "$backend" --base-url "http://127.0.0.1:$PORT"
    --workload "$workload_path" --output "$output"
    --case "${session}-${name}-${suffix}" --arrival-mode burst --seed 20260830
  )
  [[ "$backend" == vllm ]] && command+=(--model "$MODEL")
  record_command env "PYTHONPATH=$REPO/src" "${command[@]}"
  PYTHONPATH="$REPO/src" "${command[@]}"
  "$PYTHON" - "$output" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
if (
    summary["failed_requests"]
    or summary["successful_requests"] != summary["requests"]
    or summary["output_limit_hits"] != summary["requests"]
):
    raise SystemExit(f"incomplete benchmark run: {path}")
PY
}

run_session() {
  local backend=$1
  local session=$2
  local first=$OUTPUT_DIR/workloads/baseline.json
  local second=$OUTPUT_DIR/workloads/decode-steady.json
  if (( 10#${session:1:2} % 2 == 0 )); then
    first=$OUTPUT_DIR/workloads/decode-steady.json
    second=$OUTPUT_DIR/workloads/baseline.json
  fi

  start_server "$backend" "$session"
  for workload in "$first" "$second"; do
    # warmup 使用完整 burst，覆盖正式轮次的并发 shape 和内存分配路径。
    run_workload "$backend" "$session" "$workload" warmup
    run_workload "$backend" "$session" "$workload" r1
    run_workload "$backend" "$session" "$workload" r2
  done
  cleanup_server
  wait_for_gpu_idle
}

# 三组反转顺序消除总是先跑某个实现造成的温度和时序偏差；每次都重新启动服务。
BACKENDS=(light-vllm vllm vllm light-vllm light-vllm vllm)
for index in "${!BACKENDS[@]}"; do
  session=$(printf 's%02d' "$((index + 1))")
  run_session "${BACKENDS[$index]}" "$session"
done

record_command "$PYTHON" "$REPO/benchmarks/remote_5090/analyze_awq_vllm_comparison.py" "$OUTPUT_DIR"
"$PYTHON" "$REPO/benchmarks/remote_5090/analyze_awq_vllm_comparison.py" "$OUTPUT_DIR"
