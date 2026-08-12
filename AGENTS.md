# AGENTS.md

本文件适用于整个 light-vllm 仓库。开始修改代码前，先阅读本文件和
[`docs/design.md`](docs/design.md)。当核心边界发生变化时，同步更新这两份文档。

## 项目目标

light-vllm 是一个以可维护性为第一约束的轻量 LLM 推理运行时。长期目标是：

- 轻框架：核心保持小而稳定。
- 热插拔：模型、权重加载器和后续执行后端通过明确扩展点接入。
- 高扩展：新增能力优先增加实现并注册，不修改核心分发逻辑。
- 高可读性：配置、路由、加载、模型计算和运行生命周期职责分离。
- 少边界 case：避免由功能矩阵产生的跨模块 `if-elif-else` 条件树。

仓库远端为 `https://github.com/poorpaper/light-vllm`；当前工作站的主本地副本位于
`D:\light-vllm`。

## 当前阶段

当前完成 P0 model path，并增加了一条最小的 request/serving 参考纵切：

- `ModelSpec` 描述待加载模型。
- `Catalog` / `Registry` 选择 model factory 与 loader。
- `ModelLoader` 构造模型、加载权重并准备推理。
- `ModelRunner` 管理当前模型并提供统一 `forward`。
- `TinyCausalLM` 提供 `Embedding → Linear → logits` 的最小参考实现。
- `GenerateRequest` / `GenerateResult` / `GenerationEvent` 定义协议无关的生成边界。
- `GreedyTokenExecutor` 负责准备 PyTorch 输入、检查模型输出并选择最高分 token。
- `ReferenceGenerationService` 只编排 token、停止条件和同步事件流。
- `InProcessEngineClient` 把同步参考实现适配到异步 serving 端口。
- `ExecutionBatch` / `BatchTokenExecutor` 定义与单请求执行器分离的批量执行边界。
- `FullSequenceBatchEngine` 按 `schedule → execute → update` 驱动全序列重算基线。
- `StaticBatchScheduler` 保留静态批处理基线；`ContinuousBatchScheduler` 每轮补入等待请求。
- FastAPI adapter 提供 JSON 与 SSE 两种 HTTP 表达，但只依赖 `EngineClient`。
- 稳定接口按功能域放在各自的 `api.py`；具体实现放在同域的其他模块。

这条纵切与 iteration batching 用于验证分层，不代表完整 P2-P5 已完成。KV cache、Paged Attention、
prefill/decode 拆分、分布式执行、tokenizer、sampling 和生产级 serving 尚未实现。
不要让这些未来能力提前污染当前契约。

## 代码地图

| 文件 | 职责 |
| --- | --- |
| `src/light_vllm/models/api.py` | 模型配置、张量契约、factory 与 forward 接口 |
| `src/light_vllm/loaders/api.py` | 权重加载器接口 |
| `src/light_vllm/generation/api.py` | 生成数据契约与同步生成接口 |
| `src/light_vllm/execution/api.py` | token 执行接口与执行异常 |
| `src/light_vllm/scheduler/api.py` | iteration scheduler 稳定接口与批次选择结果 |
| `src/light_vllm/scheduler/sequence_batching.py` | static 与 continuous 序列批处理策略 |
| `src/light_vllm/engine/api.py` | serving 使用的异步 Engine 接口 |
| `src/light_vllm/registry.py` | 通用的名字到组件映射 |
| `src/light_vllm/catalog.py` | 持有 model/loader 两类扩展点 |
| `src/light_vllm/bootstrap.py` | 内置组件的唯一装配位置 |
| `src/light_vllm/loaders/torch.py` | 基础 PyTorch 模型加载器 |
| `src/light_vllm/models/tiny.py` | 最小参考模型 |
| `src/light_vllm/runner.py` | 模型生命周期与统一 forward |
| `src/light_vllm/execution/local.py` | 本地 PyTorch 输入准备、输出检查与贪心选 token |
| `src/light_vllm/generation/reference.py` | 协议无关、可直接对比的参考生成服务 |
| `src/light_vllm/engine/in_process.py` | 同步生成到异步 Engine 的进程内适配器 |
| `src/light_vllm/engine/full_sequence.py` | 全序列重算 batch Engine、请求状态与事件分发 |
| `src/light_vllm/serving/http.py` | FastAPI JSON/SSE adapter |
| `src/light_vllm/entrypoints/http.py` | runtime、engine client 与 HTTP 的装配入口 |
| `tests/test_minimal_forward.py` | 当前核心契约的行为测试 |
| `tests/test_generation.py` | 生成事件流与收集语义测试 |
| `tests/test_engine_client.py` | sync-to-async、收集与取消清理测试 |
| `tests/test_scheduler.py` | raw/continuous 调度策略测试 |
| `tests/test_full_sequence_engine.py` | 全序列 iteration loop、补位、取消与语义对比测试 |
| `tests/test_http_serving.py` | HTTP adapter 与端到端测试 |

## 必须保持的架构不变量

1. `ModelRunner` 不得按 architecture、loader 名称或具体模型类型写功能分支。
2. 模型和 loader 必须经 `Catalog` 中的注册表解析。
3. 候选模型应在生命周期锁外完整构造；成功后才替换当前模型。
4. 替换 `_model` 和递增 `_generation` 必须属于同一个临界区。
5. 候选模型加载失败时，当前模型与 generation 必须保持不变。
6. `forward` 只在锁内复制当前模型引用；实际 `model(batch)` 不得长期持有生命周期锁。
7. 所有模型接受 `ForwardBatch`，返回 `ModelOutput`。
8. loader 负责把模型移动到目标 device/dtype 并切换到 `eval()`。
9. 注册同名组件默认报错；只有调用方显式传入 `replace=True` 才能覆盖。
10. `ReferenceGenerationService` 只依赖 `TokenExecutor`；不得直接依赖 runner、torch、具体模型或 device。
11. transport adapter 只依赖异步 `EngineClient`；不得直接依赖 generation service、runner、torch
    或具体模型。
12. `stream` 是每个边界的唯一执行路径；同步/异步 `generate` 必须收集各自的同一事件流，不能复制
    token 生成循环。
13. HTTP/RPC schema、状态码和 wire format 不得进入核心生成契约。
14. 流在取消、关闭和异常时必须释放执行资源；首事件后的错误由 adapter 编码到流中。
15. `InProcessEngineClient` 只做 sync-to-async 适配，不承担 scheduling；未来进程 client 必须保持
    `EngineClient` 语义。
16. `InProcessEngineClient` 必须在进入 worker thread 前异步串行化请求；等待者不得在线程池中阻塞于
    reference service 的同步执行锁。
17. 内部代码必须从所属功能域的 `api.py` 导入稳定接口；需要具体实现时直接导入对应实现模块。
18. `FullSequenceBatchEngine` 只编排请求状态、Scheduler、批量执行与事件；不得包含具体模型或 transport 逻辑。
19. static 与 continuous batching 必须共用同一个 Engine 和 `BatchTokenExecutor`，只替换 Scheduler 策略。
20. 请求在模型迭代中取消时必须立即退出后续调度；已开始的同步执行到达安全边界后，其结果必须丢弃。
21. `ForwardBatch.sequence_lengths` 表示右侧补齐前的有效长度；批量执行器只能从有效位置选择 token。

## Runner 中锁的准确含义

当前锁是生命周期锁，只保护 `_model` 与 `_generation` 的一致性，并为并发 reload/forward
提供清晰的 happens-before 边界。`forward` 取得本地模型引用后立即释放锁，因此旧模型即使随后被
替换，本次 forward 仍持有旧对象的有效引用。

这个锁不负责：

- 串行化所有模型推理；
- 保证某个 `torch.nn.Module` 可被任意线程并发调用；
- 管理 CUDA stream、显存回收或 KV cache 迁移；
- 提供跨进程同步。

当前使用 `RLock` 只是实现选择，架构并不依赖可重入语义。若修改这一处，可以换成普通
`Lock`，但必须继续满足上面的临界区和失败回滚测试。

`ReferenceGenerationService` 还有一个独立的执行锁。它只为当前无 scheduler 的参考实现串行化完整生成，
并在 stream 关闭时释放；它不是 runner 生命周期锁，也不是未来调度器或 CUDA stream 管理器。

`InProcessEngineClient` 在 event loop 中还有一个异步准入锁。它保持相同的单请求执行语义，但确保并发
等待者不占用 worker thread。被接纳的 stream 使用专属单线程执行器，同步 stream 的创建、`next()` 和
`close()` 固定在同一线程；取消会先等待正在执行的同步步骤到达安全边界，再关闭 iterator。慢客户端
仍会占用当前 reference 执行槽；这是无 scheduler 基线的明确限制。

## Registry 与 API 的边界

`Registry[T]` 是显式路由表：将稳定名字映射到某一类组件。当前有两张表：

- `catalog.models`：architecture 名称 → `ModelFactory`；
- `catalog.loaders`：loader 名称 → `ModelLoader`。

Registry 不是完整依赖注入容器，也不负责自动发现 Python package entry point。自动插件发现如需
加入，应作为 Catalog 之上的独立装配能力实现。

稳定 API 按功能域就近定义：

- `models/api.py`：`ModelSpec`、`ForwardBatch`、`ModelOutput`、`ModelFactory`、`ModelForwarder`；
- `loaders/api.py`：`ModelLoader`；
- `execution/api.py`：`TokenExecutor`、`BatchTokenExecutor`、`ExecutionBatch` 以及执行异常；
- `scheduler/api.py`：iteration scheduler 与批次选择结果；
- `generation/api.py`：生成请求、结果、事件和 `GenerationService`；
- `engine/api.py`：serving 到 engine 的异步 `EngineClient`。

`api.py` 描述该功能域对外承诺的数据和行为，具体实现放在同域的其他模块。当前
`InProcessEngineClient` 包装同步 `GenerationService`；`FullSequenceBatchEngine` 直接实现同一异步端口。
未来 IPC client 也应保持该端口，不能要求 HTTP/RPC adapter 理解进程拓扑。

`Protocol` 是 Python 中表达行为接口的方式，主要服务于静态类型和可读性，并不会自动做完整运行时校验。必要的边界校验应放在数据
契约或真正消费该契约的位置，避免散落在具体实现中。

## 新增扩展的方式

新增模型：

1. 实现统一接收 `ModelSpec` 的 factory。
2. 返回接受 `ForwardBatch`、输出 `ModelOutput` 的 `nn.Module`。
3. 在 `bootstrap.py` 注册内置模型，或由外部 plugin 注册。
4. 增加测试，证明无需修改 `ModelRunner` 即可运行。

新增权重格式：

1. 实现 `ModelLoader.load(spec, factory)`。
2. 在 loader 内完成构造、权重加载、device/dtype 移动和 `eval()`。
3. 注册新 loader 名称；不要在 runner 中增加格式判断。
4. 覆盖成功加载、无效配置和加载失败不影响当前模型的测试。

新增 serving 协议：

1. 只消费 `EngineClient`，把协议输入转换成 `GenerateRequest`。
2. 把 `GenerateResult` 或 `GenerationEvent` 编码成协议自己的 wire format。
3. 在 adapter 内映射领域错误、取消和流结束语义。
4. 在 composition root 装配 adapter；不要修改生成服务或 `ModelRunner`。

## 开发与验证

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,serve]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
```

提交前至少执行 pytest、Ruff lint 和 Ruff format check。示例和核心测试应默认可在 CPU 上运行。
除非新依赖能明显简化稳定边界或提供必要能力，否则不要增加运行时依赖。

## 下一步建议

优先继续 P2 memory path：在现有 iteration scheduler 与执行批次之后增加 KV cache 接口、block allocator
和显存预算，再拆分 prefill/decode。当前 `TokenExecutor` 继续服务无 scheduler 的单请求参考路径；
`BatchTokenExecutor` 服务 static/continuous 的全序列重算基线，不应直接塞入 KV block 管理。token budget、
KV 分配和 preemption 应由 Scheduler/Engine 在安全点拥有，不能进入 `InProcessEngineClient`、reference
service 或 HTTP/RPC adapter。性能 Guardian 只能在指标、Scheduler 参数所有权和安全更新点明确后加入。
