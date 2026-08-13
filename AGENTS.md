# AGENTS.md

本文件适用于整个 light-vllm 仓库。开始修改代码前，先阅读本文件和
[`docs/design.md`](docs/design.md)。核心边界变化时必须同步更新两者。

## 项目目标

light-vllm 是一个以可维护性为第一约束的轻量 LLM 推理运行时：

- 核心保持小而稳定；
- 模型、loader、采样器和执行后端通过明确扩展点接入；
- 新能力优先增加实现并装配，不在核心热路径扩大条件树；
- 配置、调度、模型计算、生成语义和协议转换职责分离；
- 参考 vLLM、SGLang 等成熟框架已经验证的优秀边界，不为“不同”而重新发明概念。

仓库远端为 `https://github.com/poorpaper/light-vllm`，主本地副本位于 `D:\light-vllm`。

## 当前架构

- `ModelRunner` 通过 `Catalog` 中的 model/loader 注册表加载并原子替换模型。
- `ReferenceGenerationService` 保留无调度、全序列重算的同步正确性基线。
- `EngineCore` 按 `schedule → execute → update` 驱动异步请求和事件流。
- `TokenBudgetScheduler` 用统一 token budget 调度 prompt、chunked prefill 和 decode。
- KV manager 管理逻辑 reservation；`UnboundedKVCacheManager` 不限制容量或产生位置，
  `PagedKVCacheManager` 额外按容量分配 block table。
- `LocalModelExecutor` 拥有本地模型计算与连续 K/V tensor；它不决定谁运行或分配多少资源。
- `ExecutionOutput` 允许一个请求返回零到多个确认 token，为 chunked prefill 和投机解码保留正确语义。
- `Sampler` 独立于 Executor；当前只有 `GreedySampler`。
- FastAPI adapter 只依赖 `EngineClient`，不知道 scheduler、runner、torch 或 KV cache。
- 一级包按 `modeling`、`runtime`、`serving` 收敛；稳定契约位于对应子领域的 `interfaces.py`。

当前尚未实现物理 Paged Attention、prefix caching、preemption、投机解码、分布式执行、tokenizer 和
生产级 serving。逻辑 block manager 不是物理分页算子的伪实现；执行侧暂时仍使用连续 tensor。

## 代码地图

| 文件 | 职责 |
| --- | --- |
| `src/light_vllm/modeling/models/interfaces.py` | 模型配置、forward 与 K/V 张量契约 |
| `src/light_vllm/modeling/loaders/interfaces.py` | 权重加载器契约 |
| `src/light_vllm/modeling/runner.py` | 模型生命周期与统一 forward |
| `src/light_vllm/modeling/catalog.py` | model/loader 扩展点集合 |
| `src/light_vllm/runtime/generation/interfaces.py` | 生成请求、结果和事件 |
| `src/light_vllm/runtime/generation/reference.py` | 同步正确性基线 |
| `src/light_vllm/runtime/sampling.py` | Sampler 契约与贪心实现 |
| `src/light_vllm/runtime/kv_cache.py` | 逻辑 block manager 与连续 K/V tensor 存储 |
| `src/light_vllm/runtime/scheduler/interfaces.py` | `SchedulerOutput` 等稳定调度契约 |
| `src/light_vllm/runtime/scheduler/token_budget.py` | FCFS token-budget Scheduler |
| `src/light_vllm/runtime/execution/interfaces.py` | `ExecutionBatch`、`ExecutionOutput` 与 Executor 契约 |
| `src/light_vllm/runtime/execution/local.py` | 本地模型执行与采样 |
| `src/light_vllm/runtime/engine/core.py` | 请求状态、迭代循环、事件与安全取消 |
| `src/light_vllm/runtime/engine/in_process.py` | 同步 reference 到异步 Engine 的适配器 |
| `src/light_vllm/runtime/engine/interfaces.py` | serving 使用的异步 `EngineClient` |
| `src/light_vllm/serving/http.py` | FastAPI JSON/SSE adapter |
| `src/light_vllm/entrypoints/http.py` | 具体组件的装配入口 |

## 必须保持的架构不变量

1. `ModelRunner` 不得按 architecture、loader 或具体模型类型写功能分支。
2. 模型和 loader 必须经 `Catalog` 注册表解析；同名注册默认报错。
3. 候选模型在生命周期锁外完整构造；成功后才在同一临界区替换模型并递增 generation。
4. 加载失败不得改变当前模型或 generation。
5. `forward` 只在锁内复制当前模型引用，实际模型计算不得长期持有生命周期锁。
6. 所有模型接受 `ForwardBatch`，返回 `ModelOutput`；loader 负责 device、dtype 与 `eval()`。
7. `ReferenceGenerationService` 只依赖 `TokenExecutor`，不得依赖 runner、torch 或具体模型。
8. transport adapter 只依赖 `EngineClient`；HTTP/RPC schema 和 wire format 不得进入核心契约。
9. `stream` 是同步与异步生成的唯一执行路径；`generate` 只收集同一事件流。
10. 流在取消、关闭和异常时必须释放资源；首事件后的错误由 adapter 编码到流中。
11. `InProcessEngineClient` 只做 sync-to-async 适配，不承担 scheduling。
12. `EngineCore` 只编排请求、Scheduler、Executor 和事件，不依赖 torch、具体模型或 transport。
13. Scheduler 决定谁能运行、运行多少，并同步拥有逻辑 KV 分配；Executor 只执行已可行批次。
14. `SchedulerOutput` 以事实描述每请求 token 数和可选 block table，不使用 prefill/decode 模式枚举；
    非分页后端不得伪造 block ID。
15. `ExecutionOutput` 可以返回零到多个确认 token；Engine 必须逐个应用 EOS 和长度停止条件。
16. 逻辑 KV reservation 每轮必须以 commit 或 remove 结束；只提交实际计算成功的 token。
17. 模型执行失败、输出校验失败或请求取消时，不得把本轮 token 写入 Engine 状态。
18. 取消请求必须立即退出后续调度；已开始执行的同步步骤到达安全边界后，其结果必须丢弃。
19. Scheduler/Engine Core 管理 KV reservation、分页后端的 block pool、未来 prefix cache 和 preemption；
    Executor/Worker 管理 tensor、可选 block table 消费与未来 Paged Attention kernel。
20. `Sampler` 是独立策略；greedy、top-k、top-p 不得通过新增 Executor 表达。
21. 投机解码未来由 proposer、target verify 与 acceptance sampler 组成，不新增模式专用 Executor。
22. 不为尚未实现的 attention、memory 或 prefix routing 创建空包。
23. 内部代码从所属功能域的 `interfaces.py` 导入稳定契约；需要实现时直接导入实现模块。
24. 不维护未发布架构的历史兼容别名、空 facade 或旧路径。

## 锁与资源的准确含义

`ModelRunner` 的锁只保护 `_model` 与 `_generation` 的一致性，不串行化推理、不管理 CUDA stream 或 KV。

`ReferenceGenerationService` 的执行锁只串行化同步参考生成。`InProcessEngineClient` 的异步准入锁确保等待
reference 的请求不占用 worker thread，并让同步 iterator 的创建、`next()` 和 `close()` 固定在同一线程。

`EngineCore` 的异步锁保护请求状态、Scheduler 状态和 driver 生命周期。模型执行发生在锁外；执行前通过
Executor lease 固定物理资源，取消只标记释放，tensor 等 lease 退出后再销毁。

## 新增扩展的方式

新增模型或权重格式：实现相应 factory/loader，注册到 `Catalog`，不要修改 `ModelRunner` 分发逻辑。

新增采样策略：实现 `Sampler` 并在 composition root 装配，不修改 Scheduler 或 Engine。

新增执行拓扑：实现 `ModelExecutor`，保持 `ExecutionBatch → ExecutionOutput` 语义；本地、CUDA、多进程是
合理的 Executor 差异，greedy、KV 模式、prefill/decode 不是。

新增 serving 协议：只消费 `EngineClient`，在 adapter 内转换请求、结果、错误和 wire format。

## 开发与验证

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
git diff --check
```

核心测试默认可在 CPU 上运行。除非新依赖能明显简化稳定边界或提供必要能力，否则不要增加运行时依赖。

## 下一步

下一阶段实现执行侧真正的 block table 与 Paged Attention，使逻辑 block ID 映射到物理分页 K/V；之后再加
prefix caching 和 preemption。投机解码应等物理 KV 具备预留/提交/回滚能力后，从 `TokenProposer` 和
`AcceptanceSampler` 开始，不改变 EngineClient、generation 事件或 HTTP adapter。
