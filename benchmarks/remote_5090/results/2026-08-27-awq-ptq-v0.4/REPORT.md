# AWQ W4A16 PTQ 与推理验收

日期：2026-08-27

设备：NVIDIA GeForce RTX 5090

模型：Qwen2.5-0.5B-Instruct

精度：权重 INT4、激活 FP16、group size 128

这份报告只证明当前模型、硬件和 workload 下的结果，不代表 light-vllm 普遍快于或等同于 vLLM。

## 1. PTQ 与 checkpoint

离线 PTQ 使用 16 个校准样本，每个样本最多 256 token，共 4096 个校准 token；逐层执行 activation-aware
scale search 与 clip search，各 20 个搜索步，最终量化 168 个线性层。导出的 checkpoint 使用 AutoAWQ
兼容 `quantization_config`，权重常驻格式为 `qweight + qzeros + scales`，并逐字节保留原 checkpoint 的 tokenizer、
chat template 和 generation 配置文件。

| 项目 | Dense | AWQ W4A16 |
| --- | ---: | ---: |
| checkpoint 目录大小 | 954 MB | 449 MB |
| 模型权重显存 | 950.17 MiB | 454.61 MiB |
| 显存比例 | 100% | 47.85% |
| 下一 token | `576` (` The`) | `576` (` The`) |
| logits cosine similarity | - | 0.946514 |
| logits MAE / max error | - | 0.967963 / 5.711426 |
| WikiText-2 切片 PPL | 24.8709 | 27.4884（+10.52%） |

logits 对照只覆盖一个短 prompt。质量基线额外使用 PyTorch examples 保存的 WikiText-2 test 文本，固定取
8×256 input token、2040 个 next-token 目标；原文件 SHA256 为
`d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0`，实际 token 流 SHA256 为
`6d74ca2b362ae54b097ef5d49f951b2b3cdadf4e8baabce4b03259dd74444a8f`。这能作为可重复的第一阶段质量基线，
但样本仍小，不能替代完整任务评测。

## 2. CUDA AWQ GEMM 微基准

矩阵形状固定为 `K=N=3584`，两边都使用同一份 AutoAWQ packed tensor、FP16 输入和 CUDA 同步计时。
light-vllm 对照的是 vLLM legacy `awq_gemm` CUDA kernel，不是 vLLM 0.26 端到端默认选择的 Marlin。

| 输入 token 数 | light-vllm 中位数 | vLLM legacy AWQ 中位数 | light / vLLM |
| ---: | ---: | ---: | ---: |
| 1 | 0.025424 ms | 0.025504 ms | 99.69% |
| 8 | 0.025040 ms | 0.025600 ms | 97.81% |
| 64 | 0.033168 ms | 0.033632 ms | 98.62% |
| 256 | 0.122400 ms | 0.125168 ms | 97.79% |

两边峰值临时显存和最大绝对误差一致。此结果证明当前 kernel 没有相对被参考实现产生可见退化，不代表它与
Marlin 的所有 shape 等价。

## 3. OpenAI 服务端到端对照

共同条件：同一 AWQ checkpoint、单张 RTX 5090、FP16、eager、相同 token workload。light-vllm 使用独立
Engine 进程、Triton Paged Attention、AWQ CUDA backend、4096 个 KV block、prefix cache、64 个最大序列和
32768 的单轮 token budget。vLLM 0.26 自动识别 checkpoint 为 AutoAWQ 并选择 Marlin。

| workload | 系统 | 输出吞吐 | TTFT P50 | TPOT P50 |
| --- | --- | ---: | ---: | ---: |
| burst 64 请求，256→64 | light-vllm | 5014.15 tok/s | 110.36 ms | 10.727 ms/token |
| burst 64 请求，256→64 | vLLM 0.26 | 4835.25 tok/s | 167.01 ms | 10.645 ms/token |
| burst 16 请求，16→512 | light-vllm | 1880.93 tok/s | 57.66 ms | 8.397 ms/token |
| burst 16 请求，16→512 | vLLM 0.26 | 1529.49 tok/s | 102.74 ms | 10.276 ms/token |

256→64 的正式结果使用六次 light-vllm 和六次 vLLM 测量的中位数；vLLM 分别在最终候选前后各运行三次，降低
单向执行顺序对结论的影响。这个 workload 下 light-vllm 输出吞吐高 3.70%，TTFT P50 低 33.92%，TPOT P50 高
0.77%；因此可以写成“达到同一量级并在该 workload 下略高”，不能外推为所有负载全面超过 vLLM。解码型负载沿用
此前匹配结果，light-vllm 吞吐高 22.98%。`light-baseline.json` 是未匹配 prefix cache 条件的诊断数据，不作为
正式横向结论。最终候选与后置 vLLM 基线的首轮 64 个请求逐 token 对照全部一致。

vLLM 在这台 SM 12.0 机器和 nvcc 12.8 组合下需要设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，否则其 FlashInfer
sampler 初始化会先于模型 benchmark 失败；该开关不改变 AWQ checkpoint 或 Marlin 选择。

### 3.1 首次执行与 GC 长尾

直接 model-step profile 显示，未预热的 32-token AWQ step 自身 CUDA 仅 2.001 ms，但 host 等待达到
229.859 ms，其中首次 `aten::mm` 占 171.523 ms；一次真实 prefill/decode 预热后，同 shape 的 host 时间降为
6.943 ms，CUDA 为 1.985 ms。因此最终实现只在单 Rank CUDA 服务 ready 前复用现有 `ModelExecutor` 做一次有界
prefill 和一次 decode，不在模型或 AWQ kernel 中维护 shape 特判。

连续服务 profile 又捕获到两个独立进程的 full GC 长停顿：CUDA Engine 子进程 Gen2 GC 为 129.038 ms，HTTP
父进程 Gen2 GC 为 140.149 ms；对应请求的相邻执行间隙或客户端 ITL 分别放大到 131.32 ms 和 162.26 ms，而模型
step 仍处于正常范围。原实验同时评估了启动期 heap freeze，但它会改变解释器级全局 GC 状态，最终量化分支不启用
该策略；原始 JSON 仅作为诊断证据保留，不能据此把 GC 优化收益归因给当前实现。

组合实验前六次普通服务运行的吞吐中位数为 4877.29 tok/s，最坏单请求 ITL 为 197.88 ms；同时启用预热与 GC
实验策略后六次为 5014.15 tok/s 和 56.25 ms，分别改善 2.81% 和 71.57%。由于当前分支只保留预热，不能把这组组合
实验当作当前实现的完整性能结论；后续应单独复测预热收益。

## 4. 已覆盖的行为

- PTQ 校准、逐层 scale/clip search、INT4 pack 和兼容 checkpoint 导出。
- light-vllm 自动识别 AWQ，也可以通过 CLI 显式要求 `awq` 或拒绝量化。
- Torch correctness backend 与 CUDA backend；CUDA op 在模型准备阶段一次选择，生成热路径不做后端分支。
- OpenAI `/v1/completions` 非流式和 `/v1/chat/completions` 流式请求。
- vLLM 0.26 可以直接加载导出的 checkpoint。
- 固定 WikiText-2 token 切片的 Dense/AWQ next-token NLL 与 PPL 对照。
- CUDA 服务 ready 前的真实执行链预热；全局 GC 堆冻结只保留诊断数据，不进入量化实现。
- CPU 契约测试覆盖 TP=2 packed tensor 切片；双卡 CUDA 端到端验收需要两张 GPU 都空闲后补跑。

## 5. 复现入口

```bash
light-vllm-quantize-awq \
  --model /models/Qwen2.5-0.5B-Instruct \
  --output /models/Qwen2.5-0.5B-Instruct-AWQ \
  --calibration-data calibration.jsonl \
  --max-calibration-samples 16 \
  --calibration-sequence-length 256 \
  --group-size 128 \
  --scale-search-steps 20 \
  --clip-search-steps 20 \
  --seed 17 \
  --device cuda \
  --dtype float16

light-vllm-serve \
  --architecture qwen2.5 \
  --loader safetensors \
  --weights /models/Qwen2.5-0.5B-Instruct-AWQ \
  --tokenizer /models/Qwen2.5-0.5B-Instruct-AWQ \
  --quantization auto \
  --quantization-backend cuda \
  --device cuda \
  --dtype float16 \
  --runtime engine \
  --engine-process \
  --kv-reservation blocks \
  --paged-attention-backend triton
```

单算子入口为 `benchmarks/remote_5090/benchmark_awq_kernels.py`，dense/AWQ 模型对照入口为
`benchmarks/remote_5090/validate_awq_checkpoint.py`。仓库纳入 `summary.json`、`performance-optimization-20260827.json`、
两份紧凑 CUDA 结果与本报告；工作机继续保留逐请求 JSON、Prometheus 前后快照和阶段 profile，便于复核原始条件
而不是只看摘要表。
