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
                          │       └── LocalModelExecutor
                          └── future ProcessEngineClient

EngineClient.stream ──> GenerationEvent
EngineClient.generate ──> collect the same stream ──> GenerateResult
```

首版包含：

- `TinyCausalLM`：`Embedding -> Linear` 的最简单 causal-LM forward。
- `InitModelLoader`：只初始化模型，适合测试和结构验证。
- `StateDictModelLoader`：从本地 PyTorch state dict 加载权重。
- `ModelRunner`：统一执行入口；新模型加载成功后原子替换旧模型。
- `Catalog` / `Registry`：显式扩展点，避免在核心路径增加类型判断。
- `ReferenceGenerationService`：以 token event stream 为唯一路径的最小生成参考实现。
- `InProcessEngineClient`：把同步 reference 实现适配为稳定的异步 serving 端口。
- `EngineCore`：按 `schedule -> execute -> update` 驱动异步请求与事件流。
- `TokenBudgetScheduler`：统一规划 prompt、chunked prefill 与 decode 的 token 数。
- `PagedKVCacheManager`：管理逻辑 block 的预留、提交、回滚和释放。
- `LocalModelExecutor`：使用连续 K/V tensor 执行本地模型计算。
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

output = runner.forward(ForwardBatch(input_ids=torch.tensor([[1, 2, 3]])))
print(output.logits.shape)  # torch.Size([1, 3, 128])
```

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
  --kv-num-heads 4 \
  --kv-head-size 8 \
  --model-args '{"vocab_size": 128, "hidden_size": 32, "num_heads": 4}'
```

`--runtime` 支持两条清晰路径：

- `reference`：一次执行一个完整请求，作为最清楚的语义基线。
- `engine`：token-budget Scheduler、逻辑 KV blocks 和增量模型执行。

Engine 路径不区分 prefill/decode 模式：Scheduler 只返回每请求本轮 token 数，长 prompt 自然拆成
chunk；追上全部已知 token 后才采样输出。当前物理 K/V 仍是连续 tensor，尚未实现 Paged Attention。

`--kv-reservation blocks` 使用逻辑 block 容量管理；`--kv-reservation unbounded` 不分配 block，也不限制
逻辑 KV 容量，用于在相同连续物理缓存上隔离 block 管理的实验影响。后者不是生产容量保护机制。

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

当前 Engine Core 已有 token budget、chunked prefill、逻辑 block reserve/commit/rollback 和独立 Greedy
Sampler，但物理执行仍使用连续 K/V tensor。Tokenizer、文本 prompt、随机 sampling、Paged Attention、
prefix caching、preemption、投机解码、分布式执行和 OpenAI-compatible API 仍是后续能力。多进程实现将
新增 `EngineClient` 实现，而不改 HTTP。
