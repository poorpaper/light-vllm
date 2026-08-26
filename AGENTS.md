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
  兼容目录共用 `SafetensorsModelLoader`，不进入 Runner 或 Worker 分支。loader 在权重就绪后调用可选的模型自有
  `prepare_for_inference()` hook；Qwen 用它准备打包权重和 RoPE table。
- 模型通过显式 `TensorParallelContext` 组合列并行、行并行、词表并行和集合通信；并行层声明 checkpoint 切片，
  `SafetensorsModelLoader` 通用地读取本 Rank 权重，不按 Qwen 参数名分支。TP=1 使用同一套模型 forward。
- `ReferenceGenerationService` 保留无调度、全序列重算的同步正确性基线。
- `EngineCore` 按 `schedule → execute → update` 驱动异步请求和事件流；每次至多保留一个不可变
  `PreparedStep` 在模型侧执行。上一轮结果、执行结束状态和下一轮计划在同一个锁区原子推进，并只发布一次稳定
  Scheduler 快照。默认同步 Executor 固定在 Engine 私有的单在途 `ExecutionLane`；独立 Engine 进程使用协作式
  inline 执行，在状态推进前接收已到达控制任务，并在下一步模型执行前交付已发布事件，不引入墙钟 sleep。
- `TokenBudgetScheduler` 用统一 token budget 调度 prompt、chunked prefill 和 decode；可选短请求策略同时预留
  scheduled token、KV token slot 和 sequence，首 token 后回到通用 round-robin，常规请求用真实 waiting step aging。
- KV manager 管理逻辑 reservation；`UnboundedKVCacheManager` 不限制容量或产生位置，
  `PagedKVCacheManager` 额外按容量分配 block table。严格准入使用 completion claim；可选 self-resubmit 让常规
  请求先领取 `prompt + 1 block` 的初始 claim，并只允许新准入使用全局 90% KV。撞墙者释放自己的 KV、保留完整
  token 历史并回到 PREFILL；该模式强制启用 prefix cache，以找回已提交的完整 prompt 页。
- 可选 prefix cache 由 `PagedKVCacheManager` 管理：只复用已提交的完整 prompt 页，缓存页使用哈希链、
  引用计数和 LRU；共享前缀只读，各请求尾页独占。
- composition root 通过 `kv_reservation=blocks|unbounded` 同时选择匹配的逻辑 manager 和
  `ModelStepHandler`；该选择不得进入 Engine、Executor 或 Worker 热路径。
- `LocalModelExecutor` 只把执行端口委托给一个 `LocalModelWorker`。Worker 固定当前模型版本和请求生命周期；
  `ContiguousStepHandler` / `PagedStepHandler` 分别负责连续与分页 KV 的输入准备、物理缓存和模型 forward。
- 单机 TP 由 `TensorParallelModelExecutor` 表达：Rank 0 独占现有 Engine、Scheduler 和 serving，按顺序广播原有
  `ExecutionBatch`；每个 Rank 复用同一套 `LocalModelWorker` 和物理 KV。NCCL 只承载 device tensor collective，
  可替换命令通道承载 Worker 控制；单机 POSIX 默认使用 Unix socket，Gloo 只负责启动期协调、容量事实与显式回退。
  多主机 `auto` 自动保留 Gloo，避免把单机 IPC 假装成跨节点传输。各 Rank 的 KV 规划统一取最小页数。
- `StandardDecodeHandler` 负责普通单 token 解码；`SpeculativeDecodeHandler` 组合 `DraftProposer`、目标验证和
  `AcceptanceSampler`。`NGramChainProposer` 与 `NGramTrieProposer` 共用同一树形执行流程，不新增模式专用 Worker。
- `DraftTree` 只表达候选父子关系；`QueryLayout` 把正式输入与草稿树降为一次 model step 的位置和可见性事实。
  Dense/Paged 后端只允许 query 读取已提交前缀、祖先和自身；验收后 Step Handler 在返回前压实命中路径 KV。
- Step Handler 把不同请求的 query 拼成一维 token 流；`ForwardBatch.query_start_loc` 保存请求边界，Q/K/V、
  slot mapping 和模型 hidden states 都不包含 padding 行。Step Handler 在 H2D 前验证由语义布局生成的绝对位置，
  `ForwardBatch` 用显式信任标记避免模型在 CUDA 热路径重复检查。`QueryLayout` 仍只表达单请求内部语义。
- 固定页数或 CUDA 空闲显存策略在模型加载后解析成同一个分页容量对象，同时供逻辑 manager 与物理页池使用。
- 可缓存模型只通过 `AttentionContext` 执行 attention，不内置 dense/paged fallback。reference 与连续缓存使用
  `TorchDenseAttention`；分页缓存默认使用逐页读取 K/V 的 `TorchPagedAttention`，也可装配直接读取 block table、
  融合 QK/在线 softmax/PV 的 `TritonPagedAttention`。普通线性 query 直接使用因果位置关系，不物化树形
  visibility tensor；只有非线性草稿树使用显式可见性矩阵。
- `RequestOutput` 分开表达本轮输入计算量、零到多个确认输出，以及已经写入 KV 的输出前缀。
- `ExecutionOutput` 额外报告实际进入模型 forward 的 token 数；未产出的投机 lookahead 只保留为调度预留，
  不进入 step 延迟样本。
- Decode Handler 通过 `ModelStepRequest` 精确声明要消费 logits 的 query 行，Step Handler 把选择传入
  `ForwardBatch`；普通生成只投影每个请求的最后有效行，纯 prefill 不执行 vocabulary head，投机验证只投影
  正式输入最后一行和草稿节点行。`ModelStepOutput` 保留连续 logits 及其请求边界，普通采样不得先按请求切开再
  `stack` 回同一矩阵。
- `ExecutionRequest` 的完整 `context_token_ids` 快照只为草稿 proposer 物化；普通执行只携带本轮
  `input_token_ids`，不得在每个 decode step 复制和校验完整历史。
- `EngineCapabilities` 汇总模型上限、KV 容量、Scheduler 上限和 TP size；`CapacityAdmission` 只拒绝确定性不可满足的请求。
- `PredictiveTTFTAdmission` 先检查 pending 请求数与 KV 水位，再用
  `prompt + waiting pending + running pending` 的全局当前工作量和真实 step 延迟做动态早拒；prefill 贡献剩余
  prompt，普通 decode 通常贡献当前 1 个 token，不提前展开未来输出预算。预测器冷启动时 fail-open，但前两级
  门控仍生效；请求可以覆盖全局 TTFT SLO。它是独立控制组件，不属于只读 `PerformanceObserver`。HTTP 分别把
  容量拒绝和当前负载过载表达为 422/429。
- `Sampler` 独立于 Executor；`ConfigurableSampler` 消费不可变 `SamplingParams` 和逐请求 `SamplingMetadata`，支持
  greedy、temperature、top-k、top-p 与 seed。Engine 只为未指定 seed 的随机请求解析一次私有 seed；greedy 不生成
  随机 seed，随机序列不受动态 batching 行顺序影响。
- `PerformanceObserver` 在 Engine 已生效的生命周期边界记录 TTFT、可见 token 间隔、step 延迟与请求结果，只读取
  Scheduler/KV 不可变快照；Prometheus、Grafana、HPA 和 KEDA 不进入推理热路径。
- FastAPI token 路由只依赖 `EngineClient`；OpenAI 文本路由额外依赖 serving 自己的 `TextProcessor`，把本地
  tokenizer、chat template、增量解码、stop、usage 和 wire format 留在 adapter。`/metrics` 只依赖独立的
  `PerformanceMetricsReader`；这些路由都不知道 scheduler、runner、torch 或具体模型。
- 一级包按 `modeling`、`runtime`、`serving` 收敛；稳定契约位于对应子领域的 `interfaces.py`。

当前尚未实现第三方 victim preemption、sliding-window/rope-scaling Qwen 配置、量化、多节点通信和 MaaS 控制面。
单机 TP/NCCL 已完成双 RTX 5090 上的 TP=1/2 短序列逐 token 对照、显存、性能、取消与 Rank 故障退出验收；
该机器无 CUDA P2P/NVLink，TP=2 降低单卡显存但不产生吞吐加速。BF16 长生成已观察到跨 TP size 和同一 TP=1
重复运行的轨迹分叉；现象与数值路径差异的自回归放大相符，但尚未采集分叉点 logits，根因仍待量化，也未完成
逐步 logits 容差和质量验收。Docker/Kubernetes TP=2 仍需在有容器运行权限的双卡宿主机上完成性能 A/B。
OpenAI v0.2 首版不支持 tools、多 choice、logprobs 或批量 prompt，随机 sampling 不与投机解码组合。
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
| `src/light_vllm/modeling/tensor_parallel.py` | 显式 TP 上下文、并行层与 checkpoint 切片描述 |
| `src/light_vllm/modeling/runner.py` | 模型生命周期与固定 `ModelSession` |
| `src/light_vllm/modeling/catalog.py` | model/loader 扩展点集合 |
| `src/light_vllm/runtime/generation/interfaces.py` | 生成请求、结果和事件 |
| `src/light_vllm/runtime/generation/reference.py` | 同步正确性基线 |
| `src/light_vllm/runtime/sampling.py` | 逐请求采样参数、元数据与 greedy/top-k/top-p 实现 |
| `src/light_vllm/runtime/kv_cache.py` | 逻辑 block manager 与连续 K/V tensor 存储 |
| `src/light_vllm/runtime/scheduler/interfaces.py` | `SchedulerOutput` 等稳定调度契约 |
| `src/light_vllm/runtime/scheduler/token_budget.py` | 分层 token-budget Scheduler、短请求资源池与可选 self-resubmit |
| `src/light_vllm/runtime/execution/interfaces.py` | 执行、model-step、草稿树与投机观测契约 |
| `src/light_vllm/runtime/execution/layout.py` | 从父链推导线性/树形 query 的位置和可见性 |
| `src/light_vllm/runtime/execution/local.py` | 本地 Executor 与 reference token 执行 |
| `src/light_vllm/runtime/execution/distributed.py` | torchrun 进程组、TP Executor 与 Rank Worker 循环 |
| `src/light_vllm/runtime/execution/worker.py` | 本地 Worker、Step Handler 与普通 Decode Handler |
| `src/light_vllm/runtime/execution/dense_attention.py` | reference/连续缓存共用的 dense attention 上下文 |
| `src/light_vllm/runtime/execution/paged_cache.py` | 分页 Step Handler 拥有的物理 K/V tensor |
| `src/light_vllm/runtime/execution/paged_attention.py` | Paged metadata 与 PyTorch correctness backend |
| `src/light_vllm/runtime/execution/triton_paged_attention.py` | 可选 Triton fused Paged Attention backend |
| `src/light_vllm/runtime/engine/admission.py` | 确定性容量准入、step 延迟预测与 TTFT 早拒 |
| `src/light_vllm/runtime/engine/core.py` | 请求状态、迭代循环、事件与安全取消 |
| `src/light_vllm/runtime/engine/execution_lane.py` | 单在途同步 Executor 的常驻线程边界 |
| `src/light_vllm/runtime/engine/in_process.py` | 同步 reference 到异步 Engine 的适配器 |
| `src/light_vllm/runtime/engine/interfaces.py` | serving 使用的异步 `EngineClient` |
| `src/light_vllm/runtime/observability/interfaces.py` | 性能快照、读取端口与观察者契约 |
| `src/light_vllm/runtime/observability/performance.py` | 进程内 TTFT/ITL、step、token 与请求结果聚合 |
| `src/light_vllm/runtime/observability/dispatch.py` | observer 故障隔离与组合分发 |
| `src/light_vllm/serving/http.py` | FastAPI JSON/SSE adapter |
| `src/light_vllm/serving/interfaces.py` | 文本编码与增量解码契约 |
| `src/light_vllm/serving/text.py` | 本地 Hugging Face tokenizer adapter |
| `src/light_vllm/serving/streams.py` | serving adapter 共用的 Engine stream 关闭边界 |
| `src/light_vllm/serving/openai.py` | OpenAI Completion/Chat、stop、usage 与错误映射 |
| `src/light_vllm/serving/prometheus.py` | 性能快照到 Prometheus 文本格式的转换 |
| `src/light_vllm/entrypoints/http.py` | 具体组件的装配入口 |

## 必须保持的架构不变量

1. `ModelRunner` 不得按 architecture、loader 或具体模型类型写功能分支。
2. 模型和 loader 必须经 `Catalog` 注册表解析；同名注册默认报错。
3. 候选模型在生命周期锁外完整构造，包括 loader 调用的可选 post-load inference preparation；成功后才在同一
   临界区替换模型并递增 generation。
4. 加载失败不得改变当前模型或 generation。
5. `open_session()` 只在锁内复制模型引用和 generation；实际计算不持有生命周期锁。一个请求始终使用同一
   session，reload 后旧 session 继续引用旧模型。
6. 所有模型接受 `ForwardBatch`，返回只包含其中明确请求 query 行 logits 的 `ModelOutput`；未指定行选择时返回
   全部有效 query。K/V 读写由 `AttentionContext` 和 Step Handler 完成，loader 负责 device、dtype 与 `eval()`。
7. `ReferenceGenerationService` 只依赖 `TokenExecutor`，不得依赖 runner、torch 或具体模型。
8. transport 的运行时生成依赖只允许 `EngineClient`；文本协议可以额外依赖 serving 域的 `TextProcessor`，监控路由
   可以额外依赖独立只读指标端口。HTTP/RPC schema、tokenizer、Prometheus 格式和 wire format 均不得进入核心契约。
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
20. `Sampler` 是独立策略；greedy、top-k、top-p 不得通过新增 Executor 表达。普通 Decode Handler 始终传递同一个
    逐请求采样契约，由 Sampler 决定 greedy 或随机路径；固定 seed 的随机序列不得受动态 batching 的行顺序、其他
    请求进入或取消影响，Sampler 不得持有需要按请求清理的可变 RNG 状态。
21. 投机解码由 proposer、target verify 与 acceptance sampler 组成，不新增模式专用 Executor 或 Worker。
    proposer 只返回 `DraftTree`；Decode Handler 将其降为 `QueryLayout`，并在返回前调用 `compact()` 压实命中路径。
    Engine、Scheduler、Executor、Worker 和模型不得理解具体树策略。
22. 不为尚未实现的 attention、memory 或 prefix routing 创建空包。
23. 内部代码从所属功能域的 `interfaces.py` 导入稳定契约；需要实现时直接导入实现模块。
24. 不维护未发布架构的历史兼容别名、空 facade 或旧路径。
25. `ForwardBatch.positions` 表示请求内绝对位置；连续与分页 Step Handler 都必须显式生成，模型不得从批次形态猜测。
    `input_ids`、`positions` 和 attention Q/K/V 必须使用 token-major 一维布局；`query_start_loc` 必须从 0 开始、
    严格递增并以总 query token 数结束。只有 Step Handler 在 CPU 侧按模型上限验证过位置后，才可设置
    `positions_are_validated=True`；其他调用者仍由 `ForwardBatch` 和模型检查。不得在模型热路径重新引入
    `[batch, max_query_width]` padding。
    `ModelStepOutput.logits_start_loc` 允许空请求切片，但必须覆盖连续 logits 的全部行。
26. 使用外部 KV 的模型必须声明 `ModelKVCacheSpec`；连续和分页 Step Handler 都从该规格初始化物理缓存，不再接受
    第二份层数、KV head 数或 head size 配置。
27. `blocks` 必须装配 `PagedKVCacheManager + PagedStepHandler`，`unbounded` 必须装配
    `UnboundedKVCacheManager + ContiguousStepHandler`；两者共用 `LocalModelWorker`，其他组件不得按 KV 模式分支。
28. Paged Attention 实现必须通过 `PagedAttentionBackend` 创建同一个 `AttentionContext`，并直接按 block table
    读取物理页；不得以拼接完整历史 tensor 冒充分页实现。
29. block table 必须精确覆盖 `computed + num_reserved_query_tokens` 所需物理页；后者包含本轮正式输入和全部
    speculative slot reservation。不得携带未预留尾页
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
37. `ExecutionOutput` 必须报告实际进入模型 forward 的 token 数；Engine 用它构造同一个已完成
    `StepObservation` 并显式交给 TTFT 控制器和 Observer。预测器失败应 fail-open，Observer 失败不得改变控制状态。
38. self-resubmit 只能回滚撞墙者自己，不得挑选第三方 victim；Engine 保留已经可见的 token 历史，重算不得
    重复发出旧 token，也不得让回滚凭空获得 aging。该策略必须强制开启 prefix cache，至少复用已经提交的完整
    prompt 页。
39. self-resubmit 必须有 strict fallback：达到次数或累计回滚进度阈值后，下一次准入领取 completion claim；
    整轮均无进展时最早回滚者也必须进入严格恢复路径。
40. 乐观 self-resubmit 准入默认只承诺 `prompt + 1 block`，并把新准入限制在全局 KV 容量的 90%；初始 claim 必须
    与 completion claim 一起计入容量账本。该 10% 是只限制新准入的全局 decode 余量，不是每个请求的滚动 claim；
    已经 running 的请求可以继续使用这部分余量，真正撞墙时再执行 self-resubmit。
41. `PerformanceObserver` 只能接收请求生命周期、完成的 step 和 Scheduler/KV 不可变事实；它不得执行 I/O、修改
    运行时状态或按 architecture/模型尺寸分支。安全组合器必须在 observer 首次失败后停用它，且不得在 Engine
    热路径同步写日志或让指标故障改变推理结果。
    Prometheus/Grafana/HPA/KEDA 表达必须留在控制面 adapter。
42. TP 必须通过 `TensorParallelContext` 和 `TensorCollectives` 显式注入模型；不得新增进程级可变 parallel
    singleton，也不得让模型直接初始化或销毁 `torch.distributed`。
43. 并行层必须声明完整 checkpoint 到本 Rank 参数的连续切片；loader 只消费该描述，不得按 architecture 或参数名
    维护切分条件树。TP=1 与 TP>1 必须共用同一模型 forward。
44. `TensorParallelModelExecutor` 只改变执行拓扑。Rank 0 独占 Engine、Scheduler、逻辑 KV 和 serving；其他 Rank
    只执行顺序一致的 Worker 命令。HTTP、Engine 和 Scheduler 不得感知 rank 或 NCCL。
45. 每个 Rank 持有本地参数与物理 KV；同一请求在各 Rank 使用相同逻辑 block ID。自动容量规划必须采用所有 Rank
    都能满足的最小页数，不得让 Rank 0 暴露更大的逻辑容量。
46. 模型 tensor collective 与控制通信必须分组。NCCL 只接收当前 Rank CUDA device 上的 tensor；Worker 命令通过
    可替换命令通道传输，单机 POSIX 默认使用 Unix socket，Gloo 保留启动期协调、容量事实和跨节点回退。所有 Rank
    必须以相同顺序执行 initialize、请求生命周期、model step 和 shutdown；任一通道故障后整组状态不得继续复用。
47. Qwen query heads 和 MLP 中间维按 TP 切分；KV heads 足够时切分，不足时按完整 head 复制。不得把一个 attention
    head 切到两个 Rank，也不得让 attention backend 理解复制策略。

## 锁与资源的准确含义

`ModelRunner` 的锁只保护 `_model` 与 `_generation` 的一致性。`ModelSession` 以强引用固定模型；锁不串行化推理、
不管理 CUDA stream 或 KV。

`ReferenceGenerationService` 的执行锁只串行化同步参考生成。`InProcessEngineClient` 的异步准入锁确保等待
reference 的请求不占用 worker thread，并让同步 iterator 的创建、`next()` 和 `close()` 固定在同一线程。

`EngineCore` 的异步锁保护请求状态、Scheduler 状态和 driver 生命周期。模型执行发生在锁外；执行前通过
Executor lease 固定物理资源，取消只标记释放，tensor 等 lease 退出后再销毁。正在执行的分页请求被取消时，
Scheduler 延迟归还其 block IDs，直到该同步执行步骤越过安全边界。锁外只传递 `PreparedStep` 与
`CompletedStep` 事实；lease 释放后，Engine 在一个锁区内完成上一轮提交并准备下一轮，不允许 Executor 或
Observer 反向修改请求和 Scheduler 状态。`ExecutionLane` 只跨线程传递 `ExecutionBatch` 与
`ExecutionOutput`，不得提交 Scheduler 状态或提前释放 lease；Engine 关闭时先等待 driver 越过安全边界，再回收
lane 的常驻线程。独立 Engine 进程可以省略 lane，但必须在同步模型步骤之间通过事件循环检查点给 IPC command
pump 和请求 stream 公平执行机会；检查点只能推进已经 ready 的任务，不得用固定时长 sleep 调节吞吐或 TTFT。

TP 模式仍保留 Engine 的单在途执行边界：只有 Execution Lane 线程执行分布式 collective。请求加入和取消先更新
Rank 0 本地 Worker，再排队为控制命令，在下一个 model step 或 shutdown 前按序送达其他 Rank。取消不会在一次
collective 中途释放远端物理 KV；Engine 越过原有 lease 安全边界后才处理对应 free。

`TTFTAdmission` 是控制组件：在 Engine 锁内读取一次 Scheduler 快照做准入，在 step 完成后消费真实延迟；
`PerformanceObserver` 只记录同一事实。投机细节通过独立 `SpeculationObserver` 端口上报，包括候选/命中节点、
验证产出 token、树形状和 KV 搬运；其首次失败后必须停用。
这些旁路不得互相调用、持有模型执行锁、执行 I/O 或改变生成结果。

`InMemoryPerformanceObserver` 只有独立短临界区，记录单调时钟与计数；CPU step 用墙钟，CUDA step 用执行层 event
等待实际设备完成。它不持有 Engine 锁做 I/O，也不是未来性能 Guardian。Guardian 如需自动调参，必须通过单独
控制端口提交有界决策，不能反向拿 observer 修改内部状态。

## 新增扩展的方式

新增模型或权重格式：实现相应 factory/loader，注册到 `Catalog`，不要修改 `ModelRunner` 分发逻辑。

新增采样策略：实现 `Sampler` 并在 composition root 装配，不修改 Scheduler 或 Engine。

新增执行拓扑：实现 `ModelExecutor`，保持 `ExecutionBatch → ExecutionOutput` 语义；本地、CUDA、多进程是
合理的 Executor 差异，greedy、KV 模式、prefill/decode 不是。

新增 tensor parallel 模型：组合 `modeling/tensor_parallel.py` 的通用并行层，或让新并行层公开
`checkpoint_shards`；不得修改 safetensors loader 增加模型专用参数名。新增 collective 后端只实现
`TensorCollectives` 并在 composition root 注入。

新增本地 KV 布局或 attention 后端：实现 `ModelStepHandler`，创建相应 `AttentionContext`，分页 kernel 再通过
`PagedAttentionBackend` 组合；模型保持唯一调用入口，并继续只声明 `ModelKVCacheSpec`。

新增投机候选策略：实现 `DraftProposer` 并返回有界 `DraftTree`，在 composition root 注入通用
`SpeculativeDecodeHandler`；只有验收语义变化时才新增 `AcceptanceSampler`。复用同一 Worker、Step Handler 和
`QueryLayout`，不修改 Scheduler、Engine 或模型分发。

新增 serving 协议：运行时只消费 `EngineClient`；文本协议通过 `TextProcessor` 扩展点接入 tokenizer，并在 adapter
内转换请求、结果、stop、usage、错误和 wire format。

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

先补齐有容器运行权限的双 GPU Docker/Kubernetes A/B，并在更大模型或具备 P2P 的拓扑上补充“单卡无法加载、
双卡可加载”和通信收益边界。之后按路线图进入量化、显存治理和长上下文，再推进多机通信、Prefill/Decode 分离
与 MaaS 控制面。
