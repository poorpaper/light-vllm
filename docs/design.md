# light-vllm 架构设计（v0.8）

这份文档记录当前已经落地的设计。更细的职责说明见 [architecture.md](architecture.md)。

> **稳定核心只编排事实型契约；策略和后端通过组合接入。**

## 1. 设计目标

| 目标 | 当前落实方式 |
| --- | --- |
| 轻量 | `EngineCore → Scheduler → Executor` 单向调用链 |
| 高可读性 | 请求、调度、执行、采样、模型和协议各有唯一职责 |
| 高扩展性 | 模型/loader 注册；Sampler 与 Executor 通过组合替换 |
| 少模式分支 | 不用 prefill/decode/greedy/KV 专用 Executor |
| 成熟实践 | 采用统一 token budget、Scheduler/KV 协作和 Worker 物理缓存边界 |

## 2. 包结构

```text
src/light_vllm/
├── modeling/
│   ├── attention/
│   ├── models/
│   ├── loaders/
│   ├── runner.py
│   ├── catalog.py
│   └── registry.py
├── runtime/
│   ├── generation/
│   ├── scheduler/
│   ├── execution/
│   ├── engine/
│   ├── sampling.py
│   └── kv_cache.py
├── serving/
├── entrypoints/
└── bootstrap.py
```

一级包只表示大的所有权边界。`sampling.py`、`kv_cache.py` 目前职责单一，因此保留为叶子文件；只有真实
实现增长到多个清晰组件时才拆目录。

## 3. 当前组件图

```mermaid
flowchart TB
    HTTP["FastAPI adapter"] --> Client["EngineClient"]

    Client --> Core["EngineCore"]
    Client --> Bridge["InProcessEngineClient"]
    Bridge --> Reference["ReferenceGenerationService"]

    Core --> Scheduler["TokenBudgetScheduler"]
    Scheduler --> LogicalKV["KVCacheManager<br/>reservation / logical blocks"]
    Core --> Executor["LocalModelExecutor"]
    Executor --> Worker["ModelWorker"]
    Worker --> Contiguous["ContiguousModelWorker<br/>request-level tensors"]
    Worker --> Paged["PagedModelWorker<br/>global physical pages"]
    Paged --> Attention["TorchPagedAttention<br/>online softmax"]
    Worker --> Sampler["GreedySampler"]
    Reference --> TokenExecutor["LocalTokenExecutor"]
    TokenExecutor --> Sampler

    Worker --> Runner["ModelRunner"]
    TokenExecutor --> Runner
    Runner --> Catalog["Catalog"]
    Catalog --> Models["Model Registry"]
    Catalog --> Loaders["Loader Registry"]
```

`UnboundedKVCacheManager` 只维护 reservation/commit 生命周期，不限制容量或产生位置；
`PagedKVCacheManager` 分配固定大小逻辑 block。composition root 同时选择匹配的逻辑 manager 和 Worker：

- `unbounded → UnboundedKVCacheManager + ContiguousModelWorker`；
- `blocks → PagedKVCacheManager + PagedModelWorker`。

`LocalModelExecutor`、Scheduler 和 Engine 不包含 KV 模式判断。当前 `TorchPagedAttention` 是直接读取物理页的
PyTorch correctness backend；生产级 CUDA/Triton kernel 可以实现同一 attention 契约。

## 4. 稳定契约

| 契约 | 含义 |
| --- | --- |
| `GenerateRequest` / events | 协议无关的用户生成语义 |
| `SchedulerOutput` | 本轮每请求 token 数、computed 位置和可选逻辑 block table |
| `ExecutionBatch` | Engine 从请求状态切出的本轮真实 token |
| `ExecutionOutput` | 每请求完成的计算量与零到多个确认 token |
| `ModelExecutor` | 执行已可行批次并管理执行期物理资源 |
| `ModelWorker` | 一个设备 rank 内的模型计算、物理 KV 与采样单元 |
| `Sampler` | 从二维 `[batch, vocabulary]` logits 选择 token |
| `ForwardBatch` / `ModelOutput` | token、绝对 position 与模型输出的统一张量边界 |
| `ModelKVCacheSpec` | 模型声明的逐 attention 层 K/V 形状 |
| `AttentionContext` | 模型调用连续或分页 attention 后端的稳定边界 |
| `EngineClient` | serving 使用的异步生成端口 |

`SchedulerOutput` 和 `ExecutionOutput` 是未来扩展的关键：前者不包含模式名，后者不限制一次只能输出一个
token。chunked prefill、普通 decode 和未来投机验证因此能共用同一循环。

## 5. 一次迭代

```mermaid
sequenceDiagram
    participant E as EngineCore
    participant S as Scheduler
    participant K as Logical KV Manager
    participant X as ModelExecutor
    participant W as ModelWorker
    participant M as ModelRunner
    participant P as Sampler

    E->>S: schedule()
    S->>K: reserve(request, K)
    K-->>S: optional block_ids
    S-->>E: SchedulerOutput
    E->>E: 切出 input_token_ids
    E->>X: execute(ExecutionBatch)
    X->>W: execute(ExecutionBatch)
    W->>M: forward(ForwardBatch + AttentionContext)
    M-->>W: logits + optional KV updates
    W->>P: sample(last valid logits)
    P-->>W: token IDs
    W-->>X: ExecutionOutput
    X-->>E: ExecutionOutput
    E->>S: complete(computed, new tokens)
    S->>K: commit(computed)
    E->>E: 更新状态并发送事件
```

执行失败时 Engine 移除本轮请求；Scheduler 释放 reservation 与全部逻辑 block，Worker 的物理资源经
lease 在安全边界释放。执行中的请求取消时，block ID 也延迟到该边界后归还，防止物理页被过早复用。
不会出现逻辑状态已经前进但模型 K/V 没有成功写入的半提交状态。

## 6. KV reserve / commit / rollback

无 block 基线使用 `UnboundedKVCacheManager`，其 reservation 返回 `block_ids=None`，执行成功后仍按同一
commit 语义推进 token 计数。逻辑分页实现使用固定 `block_size`：

```text
committed=1, reserve=4, block_size=2
需要覆盖 5 token → 临时持有 3 blocks

若只 commit=2：
committed 变为 3 → 只需 2 blocks → 自动释放尾部 1 block
```

当前普通执行总是完整提交本轮输入；部分提交语义为未来投机验证准备。取消或失败使用 `remove()` 释放
整个请求，无需另外维护 rollback API。

分页 Worker 持有每层 `[block, offset, kv_head, head_size]` 的全局 K/V tensor。它把请求逻辑位置映射为
`block_id * block_size + offset`，原位写入本轮 K/V，并按 block table 逐页完成 causal attention。不同长度
请求会组成一个 padded forward batch，`sequence_lengths` 屏蔽 padding，`positions` 始终保存请求内绝对位置。
模型只声明 `ModelKVCacheSpec` 并调用 `AttentionContext`，不依赖具体 page layout。当前每个活动物理页归一个请求
独占；未来 prefix sharing 必须显式区分只读共享前缀与可写尾页，不能仅允许 block ID 别名。

## 7. Sampler

`GreedySampler` 已从本地 Executor 中抽离。两个执行路径显式接收 `Sampler`：

```python
sampler = GreedySampler()
reference_executor = LocalTokenExecutor(runner, sampler)
worker = PagedModelWorker(runner, sampler, paged_cache_config)
model_executor = LocalModelExecutor(worker)
```

新增 top-k/top-p 时实现新的 Sampler 并装配。投机解码的 acceptance sampler 不等同于普通 Sampler，
未来会作为 speculative decoder 的内部组件与 proposer、target verify 组合。

## 8. Reference 与 Engine Core

Reference 是同步、无调度、每轮全序列重算的语义基线；`InProcessEngineClient` 仅把它适配到异步端口。

Engine Core 是后续性能能力唯一继续生长的路径。旧的 `FullSequenceBatchEngine`、`KVCacheBatchEngine`、
`GreedyFullSequenceBatchExecutor`、`GreedyContiguousKVCacheExecutor` 和 `KVCacheBatchTokenExecutor` 已删除，
不保留兼容 alias。

## 9. 代码映射

| 角色 | 文件 |
| --- | --- |
| Attention 契约 | `src/light_vllm/modeling/attention/interfaces.py` |
| 模型契约 | `src/light_vllm/modeling/models/interfaces.py` |
| loader 契约 | `src/light_vllm/modeling/loaders/interfaces.py` |
| 模型生命周期 | `src/light_vllm/modeling/runner.py` |
| 生成契约 | `src/light_vllm/runtime/generation/interfaces.py` |
| reference 生成 | `src/light_vllm/runtime/generation/reference.py` |
| 采样 | `src/light_vllm/runtime/sampling.py` |
| 逻辑 KV / 连续物理基线 | `src/light_vllm/runtime/kv_cache.py` |
| 调度契约 | `src/light_vllm/runtime/scheduler/interfaces.py` |
| token-budget 调度 | `src/light_vllm/runtime/scheduler/token_budget.py` |
| 执行契约 | `src/light_vllm/runtime/execution/interfaces.py` |
| 本地 Executor | `src/light_vllm/runtime/execution/local.py` |
| 本地 Worker | `src/light_vllm/runtime/execution/worker.py` |
| 物理分页 KV | `src/light_vllm/runtime/execution/paged_cache.py` |
| Paged Attention | `src/light_vllm/runtime/execution/paged_attention.py` |
| Engine Core | `src/light_vllm/runtime/engine/core.py` |
| EngineClient | `src/light_vllm/runtime/engine/interfaces.py` |
| HTTP adapter | `src/light_vllm/serving/http.py` |
| 装配入口 | `src/light_vllm/entrypoints/http.py` |

## 10. 当前完成度

```mermaid
flowchart LR
    Model["Model path<br/>完成"] --> Serving["Reference serving<br/>完成"]
    Serving --> Core["Token-budget Engine Core<br/>完成"]
    Core --> Logical["逻辑 block reserve/commit/rollback<br/>完成"]
    Logical --> Sampling["独立 Greedy Sampler<br/>完成"]
    Sampling --> Paged["物理 Paged Attention<br/>PyTorch correctness 完成"]
    Paged --> Kernel["CUDA/Triton kernel<br/>下一步"]
    Kernel --> Prefix["Prefix cache / preemption"]
    Prefix --> Spec["Speculative decoding"]
```

当前“完成”指契约、CPU 参考实现和行为测试完成，不代表已经具有生产吞吐。

## 11. 验证要求

提交前至少运行：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
git diff --check
```

测试必须覆盖 token budget、chunked prefill、多 token 输出、逻辑 block 回滚、非连续物理页、block table
别名拒绝、跨页 prefill/decode、GQA、Sampler 替换、执行失败和取消资源释放。核心 CPU 测试不得依赖可选 GPU
环境。

边界命名与职责参考 [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)；分页布局与
按需读取原则参考 [PagedAttention 论文](https://arxiv.org/abs/2309.06180)。本项目保留这些成熟边界，但以
可读的 PyTorch correctness backend 先固定行为，再替换优化 kernel。
