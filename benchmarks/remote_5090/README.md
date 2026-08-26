# RTX 5090 comparison harness

This directory contains the reproducible serving harness used to compare
light-vllm and vLLM on one or two RTX 5090 GPUs. Both adapters receive the same
pre-tokenized prompt IDs and the same open-loop arrival trace.

The harness deliberately lives outside `src/`: it may depend on benchmark-only
packages and must not change runtime behavior.

## Files

- `make_workloads.py`: creates deterministic baseline, mixed scheduling,
  decode-heavy preemption-pressure, and repetition-heavy speculative workloads
  from one tokenizer snapshot.
- `serve_benchmark.py`: sends a workload to either streaming HTTP API and writes
  per-request timings plus raw Prometheus snapshots.
- `compare_outputs.py`: checks greedy token equality between two result files.
- `profile_model_step.py`: profiles the direct light-vllm model-step boundary
  without HTTP or scheduler time.
- `analyze_results.py`: aggregates repeated runs, exports a CSV, and renders the
  decode breakdown, baseline, scheduling-pressure, production-trace,
  production-burst, and speculative-decoding figures. The decode breakdown is
  emitted when `profiling_breakdown.json` exists next to the manifest.
- `launch_vllm.sh`: starts the matched vLLM server, including the RTX 5090
  sampler workaround, optional eager mode, and an optional seven-token n-gram
  proposer.
- `launch_vllm_profile.sh`: captures a delayed vLLM Torch profiler trace for a
  bounded number of active steps.
- `run_tp_comparison_matrix.sh`: runs matched light-vllm/vLLM TP=1/2 serving
  cases. It keeps the workload, KV token capacity, scheduling limits, warmup,
  and repeat count fixed across configurations.
- `analyze_tp_comparison.py`: validates request and output-token work, reports
  per-run medians, and renders throughput, latency, scaling, and memory charts.
- `profile_tp_primitives.py`: measures NCCL tensor collectives separately from
  the Gloo control channel used by the TP executor.
- `profile_tp_runtime.py`: records per-Rank model, command-channel and Engine
  wall-clock stages; set `TP_RUNTIME_PROFILE_OUTPUT='/path/rank-{rank}.json'`
  when invoking `run_runtime_gap_case.sh` to enable it.
- `render_tp_control_comparison.py`: renders the checked-in TP control-path
  summary as a PNG and SVG with per-run points and median bars.
- `test_tp_failure_exit.sh`: kills one active Rank and verifies bounded process
  exit, port release, and GPU-memory cleanup.

Typical remote usage:

```bash
python benchmarks/remote_5090/make_workloads.py \
  --tokenizer /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct \
  --output-dir /root/autodl-tmp/light-vllm-results/workloads

python benchmarks/remote_5090/serve_benchmark.py \
  --backend light-vllm \
  --base-url http://127.0.0.1:8000 \
  --workload /root/autodl-tmp/light-vllm-results/workloads/baseline.json \
  --output /root/autodl-tmp/light-vllm-results/raw/light-baseline.json \
  --arrival-mode poisson --request-rate 2 --seed 20260819
```

Use `eos_token_id=null` for light-vllm and `ignore_eos=true` for vLLM so every
request performs exactly the configured amount of decode work.

The fixed-output workloads above are stress tests, not production traffic. For
the production-like comparison, `make_production_workload.py` samples first-turn
ShareGPT pairs, applies the model chat template, and uses the observed assistant
response token count to filter and record the sampled distribution. The
resulting requests have a real long/short mix. When testing an intentionally
conservative common server limit, generate the workload with
`--declared-max-output-tokens 512`, first capture one canonical `--respect-eos`
result, and pass that result through `--replay-lengths-from`.
The client then ends each stream at its observed production length while both
schedulers still see the same larger upper bound. Raw EOS-terminated BF16 runs
are not used for performance ranking because small batch-shape differences can
change the generated path and therefore the amount of work.

Report steady traffic and overload separately. The steady case uses a fixed
Poisson trace; the burst case replays the same prompts and output lengths with
all arrival offsets set to zero. `output_limit_hits` counts requests that truly
reach the declared server limit, while `replay_target_hits` verifies that every
request completed its canonical replay length.
