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

- `ModelRunner` 通过 `Catalog` 中的 model/loader 注册表加载并原子替换模型；`open_session()` 固定模型对象和
  generation，一个生成请求不得跨 session。
- 原生 `Qwen2ForCausalLM` 支持 Qwen2/Qwen2.5 的 full-attention、default-RoPE 配置；HF 与 ModelScope 下载的
  兼容目录共用 `SafetensorsModelLoader`，不进入 Runner 或 Worker 分支。
- `ReferenceGenerationService` 保留无调度、全序列重算的同步正确性基线。
- `EngineCore` 按 `schedule → execute → update` 驱动异步请求和事件流。
- `TokenBudgetScheduler` 用统一 token budget 调度 prompt、chunked prefill 和 decode；可选短请求策略同时预留
  scheduled token、KV token slot 和 sequence，首 token 后回到通用 round-robin，常规请求用真实 waiting step aging。
- KV manager 管理逻辑 reservation；`UnboundedKVCacheManager` 不限制容量或产生位置，
  `PagedKVCacheManager` 额外按容量分配 block table。严格准入使用 completion claim；可选 self-resubmit 只允许
  常规请求 best-effort 准入，撞墙者释放自己的 KV 并带完整 token 历史回到 PREFILL。
- 可选 prefix cache 由 `PagedKVCacheManager` 管理：只复用已提交的完整 prompt 页，缓存页使用哈希链、
  引用计数和 LRU；共享前缀只读，各请求尾页独占。
- composition root 通过 `kv_reservation=blocks|unbounded` 同时选择匹配的逻辑 manager 和
  `ModelStepHandler`；该选择不得进入 Engine、Executor 或 Worker 热路径。
- `LocalModelExecutor` 只把执行端口委托给一个 `LocalModelWorker`。Worker 固定当前模型版本和请求生命周期；
  `ContiguousStepHandler` / `PagedStepHandler` 分别负责连续与分页 KV 的输入准备、物理缓存和模型 forward。
- `StandardDecodeHandler` 负责普通单 token 解码；`NGramSpeculativeDecodeHandler` 组合历史候选、目标验证和贪心验收，
  两者替换同一 Handler，不新增模式专用 Worker。
- 固定页数或 CUDA 空闲显存策略在模型加载后解析成同一个分页容量对象，同时供逻辑 manager 与物理页池使用。
- 可缓存模型只通过 `AttentionContext` 执行 attention，不内置 dense/paged fallback。reference 与连续缓存使用
  `TorchDenseAttention`；分页缓存默认使用逐页读取 K/V 的 `TorchPagedAttention`，也可装配直接读取 block table、
  融合 QK/在线 softmax/PV 的 `TritonPagedAttention`。
- `RequestOutput` 分开表达本轮输入计算量、零到多个确认输出，以及已经写入 KV 的输出前缀。
- `EngineCapabilities` 汇总模型上限、KV 容量和 Scheduler 上限；`CapacityAdmission` 只拒绝确定性不可满足的请求。
- 可选 `PredictiveTTFTAdmission` 用 `prompt + waiting pending + running pending` 和真实 step 延迟滑窗做动态早拒；
  它是独立控制组件，不属于只读 `PerformanceObserver`。HTTP 分别把容量拒绝和 SLO 过载表达为 422/429。
- `Sampler` 独立于 Executor；当前只有 `GreedySampler`。
- `PerformanceObserver` 在 Engine 已生效的生命周期边界记录 TTFT、可见 token 间隔、step 延迟与请求结果，只读取
  Scheduler/KV 不可变快照；Prometheus、Grafana 和 HPA 不进入推理热路径。
- FastAPI 生成路由只依赖 `EngineClient`；`/metrics` 只依赖独立的 `PerformanceMetricsReader`，两者都不知道
  scheduler、runner、torch 或具体模型。
- 一级包按 `modeling`、`runtime`、`serving` 收敛；稳定契约位于对应子领域的 `interfaces.py`。

当前尚未实现第三方 victim preemption、分布式执行、tokenizer、sliding-window/rope-scaling Qwen 配置和生产级 serving。
PyTorch Paged Attention 是物理分页正确性基线；首版 Triton backend 已在 RTX 5090 上完成 FP16/BF16 数值对照，
但尚未完成长上下文性能与跨显卡验收，仍不代表生产吞吐。当前也不宣称支持大多数 Transformers 模型。

## 代码地图

| 文件 | 职责 |
| --- | --- |
| `src/light_vllm/modeling/attention/interfaces.py` | 模型 KV 规格与后端 attention 契约 |
| `src/light_vllm/modeling/models/interfaces.py` | 模型配置、forward 与 K/V 张量契约 |
| `src/light_vllm/modeling/loaders/interfaces.py` | 权重加载器契约 |
| `src/light_vllm/modeling/loaders/safetensors.py` | HF/ModelScope 兼容的本地分片快照加载 |
| `src/light_vllm/modeling/models/qwen2.py` | 原生 Qwen2/Qwen2.5 推理模型 |
| `src/light_vllm/modeling/runner.py` | 模型生命周期与固定 `ModelSession` |
| `src/light_vllm/modeling/catalog.py` | model/loader 扩展点集合 |
| `src/light_vllm/runtime/generation/interfaces.py` | 生成请求、结果和事件 |
| `src/light_vllm/runtime/generation/reference.py` | 同步正确性基线 |
| `src/light_vllm/runtime/sampling.py` | Sampler 契约与贪心实现 |
| `src/light_vllm/runtime/kv_cache.py` | 逻辑 block manager 与连续 K/V tensor 存储 |
| `src/light_vllm/runtime/scheduler/interfaces.py` | `SchedulerOutput` 等稳定调度契约 |
| `src/light_vllm/runtime/scheduler/token_budget.py` | 分层 token-budget Scheduler、短请求资源池与可选 self-resubmit |
| `src/light_vllm/runtime/execution/interfaces.py` | `ExecutionBatch`、`ExecutionOutput` 与 Executor 契约 |
| `src/light_vllm/runtime/execution/local.py` | 本地 Executor 与 reference token 执行 |
| `src/light_vllm/runtime/execution/worker.py` | 本地 Worker、Step Handler 与普通 Decode Handler |
| `src/light_vllm/runtime/execution/dense_attention.py` | reference/连续缓存共用的 dense attention 上下文 |
| `src/light_vllm/runtime/execution/paged_cache.py` | 分页 Step Handler 拥有的物理 K/V tensor |
| `src/light_vllm/runtime/execution/paged_attention.py` | Paged metadata 与 PyTorch correctness backend |
| `src/light_vllm/runtime/execution/triton_paged_attention.py` | 可选 Triton fused Paged Attention backend |
| `src/light_vllm/runtime/engine/admission.py` | 确定性容量准入、step 延迟预测与 TTFT 早拒 |
| `src/light_vllm/runtime/engine/core.py` | 请求状态、迭代循环、事件与安全取消 |
| `src/light_vllm/runtime/engine/in_process.py` | 同步 reference 到异步 Engine 的适配器 |
| `src/light_vllm/runtime/engine/interfaces.py` | serving 使用的异步 `EngineClient` |
| `src/light_vllm/runtime/observability/interfaces.py` | 性能快照、读取端口与观察者契约 |
| `src/light_vllm/runtime/observability/performance.py` | 进程内 TTFT/ITL、step、token 与请求结果聚合 |
| `src/light_vllm/runtime/observability/dispatch.py` | observer 故障隔离与组合分发 |
| `src/light_vllm/serving/http.py` | FastAPI JSON/SSE adapter |
| `src/light_vllm/serving/prometheus.py` | 性能快照到 Prometheus 文本格式的转换 |
| `src/light_vllm/entrypoints/http.py` | 具体组件的装配入口 |

## 必须保持的架构不变量

1. `ModelRunner` 不得按 architecture、loader 或具体模型类型写功能分支。
2. 模型和 loader 必须经 `Catalog` 注册表解析；同名注册默认报错。
3. 候选模型在生命周期锁外完整构造；成功后才在同一临界区替换模型并递增 generation。
4. 加载失败不得改变当前模型或 generation。
5. `open_session()` 只在锁内复制模型引用和 generation；实际计算不持有生命周期锁。一个请求始终使用同一
   session，reload 后旧 session 继续引用旧模型。
6. 所有模型接受 `ForwardBatch`，返回只包含 logits 的 `ModelOutput`；K/V 读写由 `AttentionContext` 和 Step Handler
   完成，loader 负责 device、dtype 与 `eval()`。
7. `ReferenceGenerationService` 只依赖 `TokenExecutor`，不得依赖 runner、torch 或具体模型。
8. transport 的生成路由只依赖 `EngineClient`；监控路由可以额外依赖独立只读指标端口。HTTP/RPC schema、
   Prometheus 格式和 wire format 均不得进入核心契约。
9. `stream` 是同步与异步生成的唯一执行路径；`generate` 只收集同一事件流。
10. 流在取消、关闭和异常时必须释放资源；首事件后的错误由 adapter 编码到流中。
11. `InProcessEngineClient` 只做 sync-to-async 适配，不承担 scheduling。
12. `EngineCore` 只编排请求、Scheduler、Executor 和事件，不依赖 torch、具体模型或 transport。
13. Scheduler 决定谁能运行、运行多少，并同步拥有逻辑 KV 分配；Executor 只执行已可行批次。
14. `SchedulerOutput` 以事实描述 computed、scheduled、lookahead、最大输出数和可选 block table，不使用
    prefill/decode 模式枚举；非分页后端不得伪造 block ID。
15. `RequestOutput` 可以返回零到多个确认 token，并只用 `num_cached_output_tokens` 表达已经写入 KV 的连续
    输出前缀；Engine 必须逐个应用 EOS 和长度停止条件。
16. 逻辑 KV reservation 每轮必须以 commit 或 remove 结束；只提交实际算完的输入和可见的已缓存输出前缀。
17. 模型执行失败、输出校验失败或请求取消时，不得把本轮 token 写入 Engine 状态。
18. 取消请求必须立即退出后续调度；已开始执行的同步步骤到达安全边界后，其结果必须丢弃。
19. Scheduler/Engine Core 管理 KV reservation、逻辑 block ID、prefix cache、self-resubmit 和未来 victim preemption；
    Worker/Step Handler 管理 tensor、物理页池、block table 消费与 Paged Attention kernel。
20. `Sampler` 是独立策略；greedy、top-k、top-p 不得通过新增 Executor 表达。
21. 投机解码由 proposer、target verify 与 acceptance sampler 组成，不新增模式专用 Executor 或 Worker。
22. 不为尚未实现的 attention、memory 或 prefix routing 创建空包。
23. 内部代码从所属功能域的 `interfaces.py` 导入稳定契约；需要实现时直接导入实现模块。
24. 不维护未发布架构的历史兼容别名、空 facade 或旧路径。
25. `ForwardBatch.positions` 表示请求内绝对位置；连续与分页 Step Handler 都必须显式生成，模型不得从批次形态猜测。
26. 使用外部 KV 的模型必须声明 `ModelKVCacheSpec`；连续和分页 Step Handler 都从该规格初始化物理缓存，不再接受
    第二份层数、KV head 数或 head size 配置。
27. `blocks` 必须装配 `PagedKVCacheManager + PagedStepHandler`，`unbounded` 必须装配
    `UnboundedKVCacheManager + ContiguousStepHandler`；两者共用 `LocalModelWorker`，其他组件不得按 KV 模式分支。
28. Paged Attention 实现必须通过 `PagedAttentionBackend` 创建同一个 `AttentionContext`，并直接按 block table
    读取物理页；不得以拼接完整历史 tensor 冒充分页实现。
29. block table 必须精确覆盖本轮 `computed + query + lookahead reservation` 所需物理页，不得携带未预留尾页
    或在单请求内重复页；跨请求只能在相同逻辑位置共享双方都声明为只读的完整前缀页，可写尾页必须独占。
30. `LocalModelWorker` 初始化时固定一个 `ModelSession`。reload 后活动请求继续使用旧 session，Worker 拒绝新请求；活动
    请求清空后才可按新 generation 重建物理缓存。
31. 固定页数或显存发现策略必须解析成一个共享容量事实；逻辑 block manager 和物理页池不得各自配置容量。
32. `CapacityAdmission` 只判断请求在空闲引擎上是否必然不可满足；可选 `TTFTAdmission` 可基于负载动态拒绝，
    但必须独立于只读 Observer。等待、回滚和公平性属于 Scheduler，不进入 HTTP 或 Executor。
33. 模型层负责生成 Q/K/V、RoPE、norm 和 MLP；可缓存模型必须把 KV 读写及实际 attention 计算交给
    `AttentionContext`，不得保留模型内 dense fallback，也不得绑定 HF FlashAttention 或物理 page layout。
34. Hugging Face 与 ModelScope 只是 checkpoint 来源；兼容快照先落到本地目录，再由同一个 loader 校验配置、
    分片和权重，不能复制两套 Qwen 执行实现。
35. 严格分页准入必须保持 `已占用唯一页 + completion claims <= 总页数`；reserve 把 claim 转成页，trim 把尾页
    还原为 claim，已严格接纳请求不得在后续 decode 中失去容量保证。
36. 短请求预留必须同时受 token、KV 和 sequence 三个边界约束；`running <= max_num_sequences`，不得提前给超过
    admission slot 数量的请求批量发 completion claim。短请求产生首 token 后必须回到通用池。
37. Engine 必须把同一个已完成 `StepObservation` 显式交给 TTFT 控制器和 Observer；预测器失败应 fail-open，
    Observer 失败不得改变控制状态。
38. self-resubmit 只能回滚撞墙者自己，不得挑选第三方 victim；Engine 保留已经可见的 token 历史，重算不得
    重复发出旧 token，也不得让回滚凭空获得 aging。
39. self-resubmit 必须有 strict fallback：达到次数或累计重算阈值后，下一次准入领取 completion claim；
    整轮均无进展时最早回滚者也必须进入严格恢复路径。
35. `PerformanceObserver` 只能接收请求生命周期、完成的 step 和 Scheduler/KV 不可变事实；它不得执行 I/O、修改
    运行时状态或按 architecture/模型尺寸分支。安全组合器必须在 observer 首次失败后停用它，且不得在 Engine
    热路径同步写日志或让指标故障改变推理结果。
    Prometheus/Grafana/HPA 表达必须留在控制面 adapter。

## 锁与资源的准确含义

`ModelRunner` 的锁只保护 `_model` 与 `_generation` 的一致性。`ModelSession` 以强引用固定模型；锁不串行化推理、
不管理 CUDA stream 或 KV。

`ReferenceGenerationService` 的执行锁只串行化同步参考生成。`InProcessEngineClient` 的异步准入锁确保等待
reference 的请求不占用 worker thread，并让同步 iterator 的创建、`next()` 和 `close()` 固定在同一线程。

`EngineCore` 的异步锁保护请求状态、Scheduler 状态和 driver 生命周期。模型执行发生在锁外；执行前通过
Executor lease 固定物理资源，取消只标记释放，tensor 等 lease 退出后再销毁。正在执行的分页请求被取消时，
Scheduler 延迟归还其 block IDs，直到该同步执行步骤越过安全边界。

`TTFTAdmission` 是控制组件：在 Engine 锁内读取一次 Scheduler 快照做准入，在 step 完成后消费真实延迟；
`PerformanceObserver` 只记录同一事实。两者不得互相调用，任一旁路故障也不得持有模型执行锁或执行 I/O。

`InMemoryPerformanceObserver` 只有独立短临界区，记录单调时钟与计数；CPU step 用墙钟，CUDA step 用执行层 event
等待实际设备完成。它不持有 Engine 锁做 I/O，也不是未来性能 Guardian。Guardian 如需自动调参，必须通过单独
控制端口提交有界决策，不能反向拿 observer 修改内部状态。

## 新增扩展的方式

新增模型或权重格式：实现相应 factory/loader，注册到 `Catalog`，不要修改 `ModelRunner` 分发逻辑。

新增采样策略：实现 `Sampler` 并在 composition root 装配，不修改 Scheduler 或 Engine。

新增执行拓扑：实现 `ModelExecutor`，保持 `ExecutionBatch → ExecutionOutput` 语义；本地、CUDA、多进程是
合理的 Executor 差异，greedy、KV 模式、prefill/decode 不是。

新增本地 KV 布局或 attention 后端：实现 `ModelStepHandler`，创建相应 `AttentionContext`，分页 kernel 再通过
`PagedAttentionBackend` 组合；模型保持唯一调用入口，并继续只声明 `ModelKVCacheSpec`。

新增普通或投机解码流程：实现 `DecodeHandler`，组合 proposer、target verify 与 acceptance sampler；复用同一
`LocalModelWorker` 和 Step Handler，不修改 Scheduler、Engine 或模型分发。

新增 serving 协议：只消费 `EngineClient`，在 adapter 内转换请求、结果、错误和 wire format。

新增指标后端：实现 `PerformanceMetricsReader` 的表达适配器，消费不可变快照；不要把 Prometheus SDK、Grafana
或 HPA 逻辑放进 Engine、Scheduler、Worker 或模型。

## 开发与验证

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
git diff --check
```

核心测试默认可在 CPU 上运行。除非新依赖能明显简化稳定边界或提供必要能力，否则不要增加运行时依赖。

## 下一步

下一阶段针对长上下文把 Triton backend 改成分段计算与归并，并补充跨显卡性能验收；之后继续实现
Scheduler-owned victim preemption。两项能力都不得改变 EngineClient、generation 事件或 HTTP adapter。
