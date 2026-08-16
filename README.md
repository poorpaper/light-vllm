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
- `TokenBudgetScheduler`：统一规划 prompt、chunked prefill 与 decode 的 token 数。
- `PagedKVCacheManager`：管理逻辑 block 的预留、提交、回滚和释放。
- `LocalModelExecutor` / `LocalModelWorker`：把本地执行拓扑、模型版本和具体计算能力分开。
- `LocalModelWorker`：固定当前模型版本和请求生命周期，并组合 Step / Decode Handler。
- `PagedStepHandler`：消费 block table，使用全局物理页与可替换的 PyTorch/Triton Paged Attention。
- `ContiguousStepHandler`：保留请求级连续 K/V 的无分页正确性基线。
- `StandardDecodeHandler`：处理普通 prefill 和单 token decode。
- `NGramSpeculativeDecodeHandler`：从当前请求历史提出候选，由同一个目标模型一次验证，不新增 Worker。
- `GreedySampler`：独立于 Executor 的贪心采样策略。
- FastAPI adapter：协议外层的 JSON/SSE 接口，只依赖 `EngineClient`。

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
        architecture="qwen2",
        loader="safetensors",
        weights=Path("D:/models/Qwen2.5-0.5B-Instruct"),
        device="cuda",
        dtype=torch.bfloat16,
    )
)
```

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
  --max-num-scheduled-tokens 256 \
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

首版 Triton kernel 支持 FP16/BF16、MHA/GQA、padded batch 和不超过 256 的 head size。K/V 写入与 attention
读取分成同一 CUDA stream 上的两个顺序步骤，避免 prefill 读取尚未写完的数据。RTX 5090 上的 FP16/BF16
prefill、decode、共享 prefix 和投机多 query 数值对照已经通过；长上下文性能和跨显卡调优仍待验收，默认
backend 因此保持为 `torch`。

增加 `--enable-prefix-caching` 后，Engine 会按 token 内容复用已经算完的完整 prompt 页。共享页只读，
每个请求继续使用自己的可写尾页；模型重新加载后 cache epoch 改变，旧页索引会自动清空。

增加 `--num-speculative-tokens 3` 后，Engine 会从当前请求的重复 token 片段提出最多 3 个候选，再用目标模型
一次验证。`--speculative-ngram-min` 和 `--speculative-ngram-max` 控制匹配长度；找不到重复片段时自动退化为
普通单 token 解码。该能力同时支持连续和分页 KV。

`--kv-reservation unbounded` 装配无 block manager 与 `ContiguousStepHandler`，不限制逻辑 KV 容量。它保留
无 Paged Attention 的请求级连续 tensor 路径，主要用于测试和结果对照，不是生产容量保护机制。两种 Handler
都只读取模型的 `ModelKVCacheSpec`，装配层不重复填写 K/V 形状。

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
Sampler、原生 Qwen2 子集、整页 prefix cache、简单 n-gram 投机解码、PyTorch Paged Attention correctness backend
和可选的首版 Triton fused attention。Tokenizer、文本 prompt、随机 sampling、经过长上下文调优和跨显卡验收的
生产级 attention kernel、preemption、分布式执行和 OpenAI-compatible API 仍是后续能力。多进程实现将新增
`EngineClient` / Worker 拓扑，而不改 HTTP。
