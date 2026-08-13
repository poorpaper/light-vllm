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
                          ├── FullSequenceBatchEngine
                          │       ├── StaticBatchScheduler
                          │       └── ContinuousBatchScheduler
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
- `FullSequenceBatchEngine`：按 `schedule -> execute -> update` 驱动全序列重算批量生成。
- `StaticBatchScheduler`：静态批处理基线，当前批次清空后才接纳下一批请求。
- `ContinuousBatchScheduler`：每轮模型执行后补入等待请求。
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
  --architecture tiny-causal-lm \
  --batching continuous \
  --max-batch-size 8 \
  --model-args '{"vocab_size": 128, "hidden_size": 32}'
```

`--batching` 支持三种可直接对比的模式：

- `reference`：一次执行一个完整请求，作为最清楚的语义基线。
- `raw`：静态 iteration batching，批次未清空时不补位。
- `continuous`：iteration-level continuous batching，每轮结束后补位。

raw 与 continuous 共用同一个 `FullSequenceBatchEngine` 和 `GreedyFullSequenceBatchExecutor`，只替换 Scheduler，
因此可以用同一模型、请求集和 batch size 公平对比。当前批量执行会右侧补齐不同长度序列，并通过
`ForwardBatch.sequence_lengths` 标记每行有效长度。

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

当前 batching 是 P1 的可读参考实现，每轮仍会重新计算每个请求的完整 token 序列；它证明调度、批次、
取消和事件流边界，不代表已经具备 vLLM 的高性能内存路径。Tokenizer、文本 prompt、sampling、Paged
KV Cache、prefill/decode 拆分、定制 attention kernel、分布式执行和 OpenAI-compatible API 仍是后续
能力。多进程实现将新增 `EngineClient` 实现，而不改 HTTP。性能 Guardian 也只会在指标、token budget
和安全更新点稳定后，以有界控制面的形式加入。

未来 chunked prefill 不会扩张 `FullSequenceBatchEngine` 的“一请求每轮一个 token”契约。它应使用独立的
token-budget Scheduler 输出每个请求本轮的计算量，并由新的 KV-cache Engine Core 执行 mixed
prefill/decode batch；当前 full-sequence 路径继续作为 static/continuous 的可对比基线。
