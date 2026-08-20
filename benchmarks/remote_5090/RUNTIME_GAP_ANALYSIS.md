# light-vllm 与 vLLM 0.26 运行时差距审计

## 结论

当前代码没有一个“关闭非抢占后等同 vLLM”的模式。

- 未开启 `--enable-self-resubmit` 时，light-vllm 是 **strict completion claim**：请求准入前先承诺从 prompt 到
  `max_new_tokens` 最坏上界所需的全部逻辑 KV 容量。后续 decode 不会挑选 victim，也不会因容量压力回滚。
- 开启 `--enable-self-resubmit` 时，light-vllm 是 **optimistic self-resubmit**：只给 prompt 和少量额外页，当前请求自己
  撞墙后释放自己的 KV、把 `num_computed_tokens` 归零并重新排队。
- vLLM 0.26 默认既不做 completion claim，也不是“撞墙者只回滚自己”。它按当前 step 增量分配 KV；不足时从 running
  队列尾部或按优先级选择 victim，释放 victim 的 KV、把 victim 的 computed 归零并插回 waiting 队首。

因此只有在 **KV 足够、两边都没有 preemption/self-resubmit、TTFT/prefix/speculation 都关闭** 时，两者的外在效果才近似：
running-first、token budget、chunked prefill、每个普通 decode 请求每步一个 token。即便此时调度政策被中和，执行拓扑和
模型热路径仍明显不同，不能称为“同一套调度器”。

已有固定 `B=16, W=1` 实验已经排除了“剩余差距主要来自 kernel 计算”：light-vllm 的 kernel 总时长反而略短，服务
step 却慢约 3.38 ms。确认的主要损失位于 CPU 编排以及 kernel 之间的提交空洞，而不是 attention kernel 本体。

## 1. 时间究竟花在哪里

数据来自
[`steady_decode_time_breakdown.json`](results/2026-08-20-profiled-qwen25-coder-7b/steady_decode_time_breakdown.json)
和
[`cpu_fix_ab_summary.json`](results/2026-08-20-profiled-qwen25-coder-7b/cpu_fix_ab_summary.json)。模型为
Qwen2.5-Coder-7B-Instruct/BF16，RTX 5090；vLLM 使用 eager，固定 16 个请求、16-token prompt、每请求生成 512 token。

| 指标 | light-vllm | vLLM eager | Light - vLLM |
| --- | ---: | ---: | ---: |
| trace kernel sum / step | 10.592 ms | 10.742 ms | **-0.150 ms** |
| service effective step | 14.359 ms | 10.983 ms | **+3.376 ms** |
| attention kernel | 0.485 ms | 0.432 ms | +0.053 ms |
| GEMM kernel | 9.764 ms | 9.804 ms | -0.040 ms |
| kernel launches / step | 324 | 383 | -59 |

这组形状是纯 decode `W=1`，不存在 padded query 宽度或二次方 visibility 的放大。因此它能回答一个很窄但很关键的
问题：**vLLM 更快不是因为它的 kernel 总量更少，也不是因为 light 发了更多 CUDA launch。**

### 已经确认的 CPU 差距

基线每步都会在 `EngineCore._build_execution_batch_locked()` 中复制完整 token history，并在
`ExecutionRequest.__post_init__()` 再扫描、tuple 化和校验。修复后普通 decode 的 `context_token_ids` 为 `None`，只有
投机 proposer 才物化完整 history：

- `src/light_vllm/runtime/engine/core.py:338-341`
- `src/light_vllm/runtime/execution/interfaces.py:206-217`

三轮配对实验中位数：

| 指标 | 修复前 | 修复后 | 变化 |
| --- | ---: | ---: | ---: |
| build batch / step | 0.841 ms | 0.119 ms | -0.722 ms |
| executor 外部间隙 / step | 2.600 ms | 1.645 ms | -0.954 ms |
| service step | 14.495 ms | 13.658 ms | -0.837 ms |
| 吞吐 | 1103.8 tok/s | 1171.5 tok/s | +6.13% |
| executor CUDA event | 11.831 ms | 11.946 ms | 基本不变 |

这证明第一段差距确实是 CPU history copy/validation；修复没有让 GPU 计算变快，却直接缩短了服务 step。

### 尚待当前分支实测拆分的剩余差距

CPU history 修复后仍有约 1.65 ms/step 的 executor 外部间隙。当前循环严格串行：

1. 锁内 `schedule → stats → build batch → acquire`；
2. `await run_in_executor(executor.execute)`；
3. 等同步执行完整返回后 `validate → release → apply`；
4. 下一轮才重新进入 schedule。

代码见 `src/light_vllm/runtime/engine/core.py:264-329`。这里没有下一步 CPU prepare 与当前步 GPU execute 的
double buffer。vLLM 的 batch-queue 执行拓扑把 future 等待、采样和下一轮调度放在连续流水中；因此即使双方 policy
不触发回滚，driver topology 也不相同。

本分支已修复过时 profiler（旧脚本只包裹 `asyncio.to_thread`，而当前 Engine 使用 `run_in_executor`），并新增逐 step
记录：batch size、query width、实际模型 token、executor wall/thread CPU、CUDA event、相邻 executor gap，以及 full
模式下每个嵌套 CPU stage 的时间戳。正常 Poisson 负载会据此按 decode-only、mixed 和 prefill 宽度分桶。

## 2. 调度路径对照

### light-vllm strict

`TokenBudgetScheduler._try_admit()` 在没有 self-resubmit policy 时令 `guarantee_completion=True`
（`token_budget.py:423-439`）。`PagedKVCacheManager.try_add_request()` 把
`max_num_committed_tokens` 换算成 `completion_block_limit`，并要求
`已占用页 + 新 completion claims <= 总页数`（`kv_cache.py:470-503`）。

后果：

- 好处：已准入请求不会因 KV 压力被抢占或重算；
- 代价：按用户声明的最坏输出上界准入，实际短输出也会暂占逻辑 claim，降低可运行并发；
- 与 vLLM 的差异：vLLM 对当前 step 做增量分配，不提前预留整个 completion。

### light-vllm optimistic self-resubmit

开启 self-resubmit 后，常规请求 `guarantee_completion=False`，只领取 prompt 加
`initial_extra_blocks`；reserve 失败时 `_resubmit()` 释放撞墙请求自己的全部 KV、累计回滚量、把
`num_computed_tokens=0` 并 append 到 waiting（`token_budget.py:485-512`）。

这不是“关闭非抢占后采用 vLLM 调度”，而是 light 自己的 best-effort admission + self rollback。它不挑选第三方 victim；
重排是 append；达到回滚次数/进度阈值后又回到 strict completion claim。

### vLLM 0.26

vLLM 先调度 running 请求并按本轮 token 数调用 `allocate_slots()`。失败时，FCFS 从 running 尾部 pop victim；priority
模式选择最低优先级 victim，直到当前请求可以分配或只剩自身。实现见 vLLM 0.26 官方源码
[`scheduler.py:519-566`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/sched/scheduler.py#L519-L566)。
`_preempt_request()` 释放 victim blocks、设 `num_computed_tokens=0` 并 prepend 到 waiting，见
[`scheduler.py:1115-1135`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/sched/scheduler.py#L1115-L1135)。

### 哪里近似，哪里不一致

| 维度 | Light strict | Light self-resubmit | vLLM 0.26 |
| --- | --- | --- | --- |
| KV admission | 最坏 completion 全量 claim | prompt + 少量页 | 当前 step 增量分配 |
| 容量不足对象 | 不应发生；否则报不变量错误 | 撞墙者自己 | running 尾部/低优先级 victim |
| 回滚后位置 | 不适用 | waiting 尾部 | waiting 队首 |
| prefix recovery | optimistic 强制开启，回收完整 prompt 页 | 完整 prompt 页可命中 | 取决于 prefix cache 配置 |
| 运行顺序 | admitted round-robin | admitted round-robin + strict fallback | running-first FCFS/priority |
| 无压力时 | 外在效果近似 vLLM | 外在效果近似 vLLM | continuous batching |

所以“关闭 policy 影响”的正确实验不是把 Light 某个开关称为 vLLM 模式，而是给足 KV，并要求两边观测到
`preemption=0`、`self_resubmit=0`、拒绝为 0。此时测到的差距才可以归到模型执行、CPU 编排和 driver topology。

## 3. 图片中问题的当前代码复核

| 图片判断 | 结论 | 当前代码证据与边界 |
| --- | --- | --- |
| padded `[B,W]` 让短 decode 跟长 prefill 一起跑满宽度 | **属实** | `worker.py:223-265` 仍构造 padded dense batch；会影响 mixed step，不解释固定 `W=1`。 |
| 每步构造 `[B,W,W]` visibility | **原问题属实，当前分支已修线性路径** | `paged_attention.py:158-174` 是原始 Python 构造；`triton_paged_attention.py:268-272` 现在只有非线性 tree 才调用，普通 causal chain 在 kernel 内解析判断。 |
| lm_head 对全部 padded query 做 vocab projection 后再切片 | **原问题属实，当前分支已修** | `ModelStepRequest.logit_query_indices` 精确声明所需行；`qwen2.py:512-513` 在 lm_head 前 gather。普通 decode 只选每请求最后行，prefill-only 为空，spec 选正式尾行+draft。 |
| strict completion claim 过度保守 | **属实，但属于政策权衡，不是 kernel bug** | `token_budget.py:425` 和 `kv_cache.py:470-503`。它解释容量压力下的并发/准入，不解释 KV 充足、固定 B16 的每步慢。 |
| 单 driver 严格串行、无 CPU/GPU overlap | **属实** | `engine/core.py:264-329`。这是固定 `W=1` 剩余 CPU gap 的首要架构候选。 |
| 每步 CUDA event `synchronize()` 是 1.65 ms gap 的全部来源 | **不成立** | sampler 的 `.tolist()` 已先等待 logits；历史 A/B 中 executor wall 与 CUDA event 只差约 0.04 ms，而 1.65 ms 是 executor 调用之间。timer 不能解释 executor 外 gap。 |
| Python tensor/list/逐元素校验是热路径开销 | **部分属实** | metadata、slot mapping、visibility 和 block validation 都在每步执行；但固定 W1 需逐项 profile，不能把所有 residual 都算给 tensor 构造。 |
| `.tolist()` 会同步 GPU | **属实** | `sampling.py:28`；但它的 wall time包含等待前面所有已入队 kernel，不能等同于 sampling kernel 自身耗时。 |
| Light CUDA launch 太多 | **被固定 W1 trace 反证** | Light 324 次/step，vLLM eager 383 次/step；问题是 launch/API 间空洞和流水，而不是次数更多。 |
| attention kernel 是主因 | **被固定 W1 trace 反证** | attention 只差 0.053 ms/step，整步差 3.376 ms。mixed/prefill 仍需单独测。 |
| 缺 CUDA Graph 是 eager 对比差距 | **不成立** | 匹配的 vLLM eager 对照也关闭 graph。vLLM default 会另列，不能把 graph 收益归因给 scheduler。 |
| self-resubmit 丢失已输出 token | **不属实** | Engine 的可见 token history 不会清空或重复发送；归零的是 KV computed 进度，代价是重算。 |

## 4. 当前分支修复与实验门槛

分支：`codex/runtime-gap-profile-and-fix`。

已实现：

1. 修正 profiler 的 `run_in_executor` 埋点，并输出逐 step 形状/CPU/CUDA/gap；
2. 普通线性 Triton attention 跳过 `[B,W,W]` visibility；tree 路径保持显式矩阵；
3. last-token/selected-row logits，在 vocabulary head 前裁剪；
4. CUDA positions 范围检查改为异步 assertion，避免热路径 host sync；
5. CPU 全量 pytest、ruff、format、`git diff --check` 已通过。

GPU 实验必须满足：

- 同一模型快照、BF16、block size 16、`max_num_sequences=16`、token budget 512；
- raw runtime 对比双方 prefix/TTFT/speculation 全关；
- vLLM eager 与 default 分开，不能把 CUDA Graph 收益算到调度；
- 正常 ShareGPT replay 使用 Poisson 2 rps 与一个两边都无回滚的 loaded rate；
- policy 压力组使用 Poisson 而非 burst，单列 Light strict、Light optimistic、vLLM preempt16 和 vLLM 低并发零抢占内控；
- 每个模式 warmup 后 3 轮；逐轮保留 JSON、Prometheus before/after、server log、profile、argv 和 git SHA；
- CUDA 数值测试必须同时覆盖 linear 和 tree visibility，输出请求数/长度必须一致。

在这些数据完成前，可以确认“固定 decode 的剩余差距主要不是 kernel 执行时间”，但不能把正常 mixed workload 的全部
差距提前归因给 CPU，也不能声称当前三个修复已经缩小了端到端差距。
