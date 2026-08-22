# light-vllm

一个以可维护性为第一约束的轻量 LLM 推理框架草稿。

我们的目标不是在第一天复刻 vLLM 的全部能力，而是先建立一组稳定、清晰、可组合的边界：

- **轻框架**：核心只保留注册、加载、执行三个必要动作。
- **热插拔**：模型和加载器通过 `Catalog` 注册；扩展能力不修改核心分发逻辑。
- **高扩展**：依赖接口和组合，不让功能矩阵演化成跨模块的 `if-else`。
- **高可读性**：配置、模型、权重加载、运行生命周期各自只有一个职责。
- **可验证**：每个扩展点都有小而直接的契约测试。

> 这是一个独立的实验项目，目前不是 vLLM 的兼容替代品。

## 当前最小闭环

```text
ModelSpec
   │
   ├── loader name ──────> Loader registry ──> ModelLoader
   └── architecture ─────> Model registry ──> ModelFactory
                                               │
                                               v
                                        torch.nn.Module
                                               │
HTTP / future RPC ──> EngineClient
                          ├── InProcessEngineClient
                          │       └── reference GenerationService ──> ModelRunner.forward
                          ├── EngineCore
                          │       ├── TokenBudgetScheduler
                          │       └── LocalModelExecutor ──> ModelWorker
                          └── future ProcessEngineClient

EngineClient.stream ──> GenerationEvent
EngineClient.generate ──> collect the same stream ──> GenerateResult
```

首版包含：

- `TinyCausalLM`：`Embedding -> Linear` 的最简单 causal-LM forward。
- `Qwen2ForCausalLM`：原生 Qwen2/Qwen2.5 full-attention 推理，连续与分页 KV 共用同一个模型实现。
- `InitModelLoader`：只初始化模型，适合测试和结构验证。
- `StateDictModelLoader`：从本地 PyTorch state dict 加载权重。
- `SafetensorsModelLoader`：读取 HF/ModelScope 兼容的本地配置、单文件或分片权重。
- `ModelRunner`：统一执行入口；新模型加载成功后原子替换旧模型。
- `Catalog` / `Registry`：显式扩展点，避免在核心路径增加类型判断。
- `ReferenceGenerationService`：以 token event stream 为唯一路径的最小生成参考实现。
- `InProcessEngineClient`：把同步 reference 实现适配为稳定的异步 serving 端口。
- `EngineCore`：按 `schedule -> execute -> update` 驱动异步请求与事件流。
- `TokenBudgetScheduler`：统一规划 prompt、chunked prefill 与 decode，并可组合短请求资源池和 self-resubmit。
- `PagedKVCacheManager`：管理逻辑 block 的预留、提交、回滚和释放。
- `LocalModelExecutor` / `LocalModelWorker`：把本地执行拓扑、模型版本和具体计算能力分开。
- `LocalModelWorker`：固定当前模型版本和请求生命周期，并组合 Step / Decode Handler。
- `PagedStepHandler`：消费 block table，使用全局物理页与可替换的 PyTorch/Triton Paged Attention。
- `ContiguousStepHandler`：保留请求级连续 K/V 的无分页正确性基线。
- `StandardDecodeHandler`：处理普通 prefill 和单 token decode。
- `SpeculativeDecodeHandler`：把 chain/trie 草稿树交给同一个目标模型并行验证，验收后压实命中路径 KV，不新增 Worker。
- `PerformanceObserver`：在统一 Engine 边界记录 TTFT、可见 token 间隔、step 延迟、队列与 KV 使用率。
- `PredictiveTTFTAdmission`：依次用队列、KV 水位和真实 step 延迟预测做动态早拒，支持请求级 TTFT SLO。
- `GreedySampler`：独立于 Executor 的贪心采样策略。
- FastAPI adapter：生成路由只依赖 `EngineClient`，`/metrics` 只依赖独立的性能快照读取端口。

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -e ".[dev]"
python examples/minimal_forward.py
python -m pytest
```

最小调用：

```python
import torch

from light_vllm import ForwardBatch, ModelSpec, create_runner

runner = create_runner()
runner.load(
    ModelSpec(
        architecture="tiny-causal-lm",
        loader="init",
        model_args={"vocab_size": 128, "hidden_size": 32},
    )
)

output = runner.open_session().forward(ForwardBatch(input_ids=torch.tensor([[1, 2, 3]])))
print(output.logits.shape)  # torch.Size([1, 3, 128])
```

## Qwen2 / Qwen2.5 快照

先用 Hugging Face 或 ModelScope 的官方工具把快照下载到本地，再把同一个目录交给 loader：

```python
from pathlib import Path

import torch

from light_vllm import ModelSpec, create_runner

runner = create_runner()
runner.load(
    ModelSpec(
        architecture="qwen2.5",
        loader="safetensors",
        weights=Path("D:/models/Qwen2.5-3B-Instruct"),
        device="cuda",
        dtype=torch.bfloat16,
    )
)
```

`qwen2.5` 是显式注册名；它与 `qwen2` 复用同一个 factory，因为官方 Qwen2.5 checkpoint 仍声明
`model_type: qwen2`，模型尺寸由快照的 `config.json` 决定。当前已用官方 Qwen2.5-3B-Instruct 配置验证
36 层、16 个 query head、2 个 KV head 和 BF16 目标构造。

当前支持 Qwen2/Qwen2.5 的 full attention、default RoPE、GQA、tied embedding 和 safetensors 分片。
sliding-window、RoPE scaling、量化权重和 tokenizer 尚未实现；因此这是明确的 Qwen 子集支持，不是“大多数 HF
模型都可直接运行”。安装 `.[validation]` 后，测试会用 Transformers 官方 Qwen2 实现对照同权重 logits；
Transformers 不参与实际推理。

## HTTP 服务

HTTP 是可选 adapter，不会成为核心运行时依赖：

```bash
python -m pip install -e ".[serve]"
light-vllm-serve \
  --architecture tiny-attention-causal-lm \
  --runtime engine \
  --kv-reservation blocks \
  --max-num-sequences 8 \
  --max-num-scheduled-tokens 2048 \
  --model-args '{"vocab_size": 128, "hidden_size": 32, "num_heads": 4}'
```

`--runtime` 支持两条清晰路径：

- `reference`：一次执行一个完整请求，作为最清楚的语义基线。
- `engine`：token-budget Scheduler、统一 Worker、连续或分页 Step Handler 和增量模型执行。

Engine 路径不区分 prefill/decode 模式：Scheduler 只返回每请求本轮 token 数，长 prompt 自然拆成
chunk；追上全部已知 token 后才采样输出。

`--kv-reservation blocks` 装配逻辑 block manager 与物理 `PagedStepHandler`。模型声明自己的 K/V layer/head
规格，Handler 以 `[block, offset, kv_head, head_size]` 布局创建页池。默认 PyTorch backend 直接逐页完成
attention，适合 CPU correctness；可选 Triton backend 直接按 block table 读取分页 K/V，在一个 kernel 内完成
QK、在线 softmax 和 PV。

CUDA Linux/WSL 环境可安装可选依赖并显式启用 Triton；该 extra 使用 Torch 2.6 或更高版本：

```bash
python -m pip install -e ".[serve,triton]"
light-vllm-serve \
  --architecture tiny-attention-causal-lm \
  --runtime engine \
  --device cuda \
  --dtype float16 \
  --paged-attention-backend triton \
  --model-args '{"vocab_size": 128, "hidden_size": 32, "num_heads": 4}'
```

首版 Triton kernel 支持 FP16/BF16、MHA/GQA、padded batch、树形 ancestor visibility 和不超过 256 的 head size。
K/V 写入与 attention 读取分成同一 CUDA stream 上的两个顺序步骤，避免 query 读取尚未写完的 KV。线性
prefill、decode、共享 prefix 和投机多 query 已在 RTX 5090 完成数值对照；树形 padded parity 测试已加入，
仍需在 CUDA 环境完成验收。长上下文性能和跨显卡调优也仍待完成，默认 backend 因此保持为 `torch`。

增加 `--enable-prefix-caching` 后，Engine 会按 token 内容复用已经算完的完整 prompt 页。共享页只读，
每个请求继续使用自己的可写尾页；模型重新加载后 cache epoch 改变，旧页索引会自动清空。

增加 `--num-speculative-tokens 3` 后，Engine 会把这 3 个位置作为单轮草稿节点预算，并用目标模型一次验证。
默认 `--speculative-proposer chain` 保持线性 N-Gram 行为；选择 `trie` 后，会把最长重复后缀的多个历史续写
按频次、最近位置和 token ID 确定性裁剪为草稿树。`--speculative-ngram-min/max` 控制匹配长度，
`--speculative-max-depth` 和 `--speculative-max-branching` 控制树形上限。找不到重复片段时自动退化为普通
单 token 解码。两种 proposer 共用连续/分页 KV、树形 attention 和验收压实流程。

`--kv-reservation unbounded` 装配无 block manager 与 `ContiguousStepHandler`，不限制逻辑 KV 容量。它保留
无 Paged Attention 的请求级连续 tensor 路径，主要用于测试和结果对照，不是生产容量保护机制。两种 Handler
都只读取模型的 `ModelKVCacheSpec`，装配层不重复填写 K/V 形状。

### 非抢占调度与 TTFT 保护

分页 KV 默认使用严格非抢占准入：请求进入 running 前领取覆盖其最大可提交长度的 completion claim，之后不会
因为其他请求占满 KV 而被挑作 victim。可以同时启用短请求资源池与 TTFT 早拒：

```bash
light-vllm-serve \
  --architecture tiny-attention-causal-lm \
  --runtime engine \
  --kv-reservation blocks \
  --num-kv-blocks 128 \
  --kv-block-size 16 \
  --max-num-sequences 8 \
  --max-num-scheduled-tokens 256 \
  --short-request-max-effective-prompt-tokens 32 \
  --short-request-max-total-tokens 64 \
  --short-request-reserved-scheduled-tokens 32 \
  --short-request-reserved-kv-token-slots 63 \
  --short-request-reserved-sequences 1 \
  --regular-request-aging-steps 8 \
  --max-tolerable-ttft-seconds 1.5 \
  --max-pending-requests 128 \
  --ttft-kv-cache-watermark 0.9
```

短请求按 prefix 命中后的有效 prompt 和最大总长度分类；scheduled token、KV slot 和 sequence 都有独立预留。
首次 token 可见后，请求回到通用 round-robin。常规请求等待达到 aging 阈值后可以借用 KV 水位，避免饥饿。
TTFT 预测使用 `新 prompt + waiting pending + running pending` 的全局当前工作量：prefill 贡献尚未计算的 prompt，
普通 decode 通常贡献当前待算的 1 个 token，所有请求再统一求和；未来 `max_new_tokens` 不会提前展开。延迟表按
每个 step 实际进入模型 forward 的 token 数更新，默认取 p90 并构造单调包络，可用
`--ttft-prediction-quantile` 调整；默认积累 100 个 step 后才启用预测。队列上限和 KV 水位不依赖预测器冷启动，
会始终生效。请求也可以用 JSON 字段 `max_tolerable_ttft_seconds` 覆盖全局 SLO。确定性容量不足返回 422，
当前负载不满足准入条件返回可重试的 429。
压测 Scheduler 本身时可把 `--max-pending-requests` 或 `--ttft-kv-cache-watermark` 设为 `off`，避免把早拒收益
误算成调度收益；生产默认仍分别是 128 和 0.9。

实验性 `--enable-self-resubmit` 改用乐观准入：每个常规请求先领取覆盖 `prompt + 1 block` 的小额 claim，
新请求合计最多使用全局 90% KV，余下 10% 留给已经运行的 decode 增长。decode 需要新页但空间不足时，
撞墙者只释放自己的 KV 并重新排队，不回滚第三方，也不重复输出已经可见的 token。该模式会强制开启 prefix cache，
因此重算可以找回已提交的完整 prompt 页；生成阶段的 KV 仍需重算。它默认关闭且仅支持 `blocks`。可用
`--max-self-resubmits` 和 `--self-resubmit-strict-fallback-rolled-back-tokens` 控制何时恢复严格 completion claim，
从而给活锁一个有界退出路径。`--self-resubmit-initial-extra-blocks` 和
`--self-resubmit-kv-admission-watermark` 可以调整上述两个乐观准入参数。

当前没有 tokenizer，因此接口直接接收 token IDs。普通生成返回一个 JSON：

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_ids":[1,2,3],"max_new_tokens":4}'
```

流式生成把同一生成事件编码为 SSE：

```bash
curl -N -X POST http://127.0.0.1:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"input_ids":[1,2,3],"max_new_tokens":4}'
```

另有 `GET /healthz` 与 `GET /readyz`。模型在应用 lifespan 中加载完成后，readiness 才返回成功。

## 性能指标

`engine` runtime 额外提供 Prometheus `GET /metrics`：

```bash
curl http://127.0.0.1:8000/metrics
```

它包含 TTFT、可见 token 间隔、真实完成的 step 延迟、waiting/running 请求、短请求首 token lane、两种 token
backlog、KV cache 使用率/claim、self-resubmit 回滚进度、准入拒绝和 token 吞吐，也包含投机尝试、命中节点、
验证产出 token、草稿根/分支、最大深度和 KV 搬运计数。平均验证产出长度可用
`rate(light_vllm_speculative_verified_tokens_total[5m]) / rate(light_vllm_speculation_attempts_total[5m])`
计算。`pending_tokens` 只统计当前
已知输入；`max_remaining_tokens` 还包含最大输出预算，适合保守扩缩容，二者不会混用。
指标来自同一套 Engine/Scheduler/KV 事实，与 `qwen2`、`qwen2.5` 或具体模型尺寸无关；`reference`
runtime 没有 Scheduler 和固定 KV 容量，因此不伪造这些指标。

可直接导入的 Grafana dashboard、Prometheus 抓取配置和 Kubernetes HPA 示例见
[`examples/monitoring`](examples/monitoring/README.md)。HPA 推荐消费
`light_vllm_waiting_max_remaining_tokens`；TTFT 预测应使用 pending 输入和 step 延迟，不得拿最大输出预算冒充
首 token 前工作量。

HTTP adapter 当前最多接受 4096 个输入 token，`max_new_tokens` 也最多为 4096。进程内
reference engine 仍然一次只执行一个完整请求；并发请求在 event loop 中等待准入，不占用推理线程。
每个被接纳的请求使用一个专属单线程执行器，保证同步 stream 的创建、推进和关闭都发生在同一线程。
流式客户端断开时会关闭 engine stream，并在当前同步 token step 安全结束后释放执行槽位。

## 扩展方式

第三方能力只需实现契约并注册：

```python
catalog.models.register("my-model", my_model_factory)
catalog.loaders.register("my-format", my_loader)
```

已有名称默认不可覆盖；需要有意识地替换时才传 `replace=True`。架构图与运行时序见
[`docs/design.md`](docs/design.md)，更细的边界规则见
[`docs/architecture.md`](docs/architecture.md)。

## 当前非目标

当前 Engine Core 已有 token budget、chunked prefill、逻辑 block reserve/commit/rollback、独立 Greedy
Sampler、原生 Qwen2 子集、整页 prefix cache、chain/trie 树形投机解码、PyTorch Paged Attention correctness backend
和可选的首版 Triton fused attention。Tokenizer、文本 prompt、随机 sampling、经过长上下文调优和跨显卡验收的
生产级 attention kernel、第三方 victim preemption、分布式执行和 OpenAI-compatible API 仍是后续能力。多进程实现将新增
`EngineClient` / Worker 拓扑，而不改 HTTP。
