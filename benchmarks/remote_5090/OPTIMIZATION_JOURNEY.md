# light-vLLM 5090 性能与调度实验全记录

这份报告是对 2026-08-19 至 2026-08-22 全部实验的二次整理。它不只保留最终数字，也记录每一步为什么做、怎么做、哪里判断错了，以及错误是如何被下一轮数据纠正的。

最终运行时代码是 `codex/driver-control-fairness@192d050`，最后一轮 GPU A/B 与原始数据索引记录于 `30cde15`。Light 使用 strict completion claim，已接纳请求不被第三方抢占；prefix cache、TTFT admission、self-resubmit 和 speculative decoding 在主性能对比中全部关闭。

## 最终结论

正常 ShareGPT 回放下，Light 已经不再存在两个数量级的 TTFT 差距。8 req/s 时，Light 吞吐达到 vLLM eager 的 95.5%，TPOT P50 对应的生成速度达到 90.8%；TTFT P50、P90、P95 都低于本轮 vLLM eager。

| 到达率 | 系统 | 吞吐 | TTFT P50 / P90 / P95 | TPOT P50 / P90 / P95 |
| ---: | --- | ---: | ---: | ---: |
| 2 req/s | Light strict | 202.41 tok/s | 22.93 / 29.56 / 37.23 ms | 10.658 / 10.914 / 10.928 ms |
| 2 req/s | vLLM eager | 202.44 tok/s | 33.00 / 40.11 / 48.38 ms | 9.911 / 10.491 / 10.783 ms |
| 8 req/s | Light strict | 624.34 tok/s | 26.29 / 39.25 / 45.00 ms | 11.864 / 12.204 / 12.325 ms |
| 8 req/s | vLLM eager | 653.74 tok/s | 38.34 / 47.34 / 54.69 ms | 10.768 / 11.162 / 11.387 ms |

![最终吞吐](results/2026-08-22-complete-report/analysis/01_final_throughput.png)

![最终延迟分位数](results/2026-08-22-complete-report/analysis/02_final_latency_percentiles.png)

两组结论并不冲突。TTFT 衡量请求何时拿到第一个 token；TPOT 衡量后续 token 的生成间隔。最后一轮公平性修复让新请求和首 token 不再饿死在 Engine 子进程的事件循环里，因此 TTFT 下降，但没有缩短模型每一步的执行时间。8 req/s 的持续吞吐仍受 TPOT 限制。

2 req/s 的吞吐由到达率决定，只能用于低负载延迟与回归检查。8 req/s 才能反映当前持续供给下的执行效率。

这组结果也不能证明“非抢占整体优于抢占”。32K KV 足以让双方完成正常负载，Light 没有 self-resubmit，vLLM 也没有 preemption。主表比较的是同容量事件为零时的执行效率；非抢占的取舍要看单独的 4K KV 压力实验。

## 统计与实验口径

除局部诊断外，主性能对比使用同一环境：

- NVIDIA GeForce RTX 5090，显存 32607 MiB；
- Qwen2.5-Coder-7B-Instruct，BF16，greedy；
- block size 16，`max_num_sequences=16`，单步 token budget 512；
- 32768 KV tokens；
- 固定 seed `20260821` 的 ShareGPT 首轮回放，Poisson 到达，每轮 64 个请求；
- 每轮共 5804 个 prompt tokens、7514 个 output tokens；输出长度 P50 为 41，只有 10/64 达到 512；
- vLLM eager 是执行效率主基准，default/graph 只在局部实验中作上限参照。

最终 Light 和 vLLM 各独立运行 3 轮。TTFT、TPOT 先在每轮 64 个请求内用线性插值计算 P50、P90、P95，再对三轮同一指标取中位数；吞吐直接取三轮中位数。原始逐请求数据和重算脚本见文末。

历史实验并非都能塞进同一张图。4K 容量压力、固定 B16/W1、投机解码和正常 ShareGPT 回放回答的是不同问题。本报告只在环境与工作量一致的实验内横向比较，单次诊断图留在对应阶段，不把不同 workload 的数字拼成一条伪优化曲线。

## 路线总览

| 阶段 | 当时的问题 | 处理结果 |
| --- | --- | --- |
| strict 最坏长度预留 | 并发低，吞吐 213 对 787 | 证明零回滚和流连续性，不能证明吞吐优势 |
| 真实输出长度回放 | 固定 512 token 夸大容量压力 | 改成真实长度分布，512 只保留给边界测试 |
| 旧 rolling 10% | 想减少最坏长度高估，却出现大量暂停 | 找到 claim 边界制造的假容量墙 |
| eager refresh | 有空闲页时也因旧 claim 暂停 | 先续领再判断，吞吐明显恢复，但仍落后 vLLM |
| TTFT 1.25 秒准入 | 过载时新请求首 token 太晚 | 能减轻已接纳请求等待，但会拒绝请求，也不是硬 deadline |
| 当前 optimistic | `prompt + 1 block` 提前准入 | TTFT 下降；撞墙后全量重算导致吞吐和 token gap 恶化 |
| Chain / Trie 投机验证 | 一次 target forward 只能产一个 token | 两者都能加速，但谁更快取决于候选分支分布 |
| CUDA 与 fused kernel | Attention、KV write、launch 数看起来落后 | device 时间下降；matched trace 排除了“全部差距都在 kernel” |
| CPU history | 普通 decode 每步复制、扫描完整历史 | batch build 降 85.9%，吞吐提高 6.13% |
| selected logits / linear visibility | 无用 logits 和树形 mask 放大 mixed step | 吞吐提高 5.3%，但 padded 主布局仍在 |
| Packed Query | mixed batch 中 81.9% 的位置是 padding | 8 req/s TTFT P95 从约 1.53 秒降到 46.5 ms |
| Sampler / staging / timer | profile 中还有很多可疑小项 | 只保留有真实重复复制的 packed logits；其余无收益方案回退 |
| Driver 状态机与私有 lane | 相邻模型步骤间有 1 ms 级控制空档 | 派发下降，端到端收益较小，说明还需继续找边界问题 |
| 进程隔离与 cooperative inline | HTTP 与 Engine 需要解耦，但逐步线程派发拖慢 TPOT | inline 恢复吞吐，却让 Engine 事件循环饥饿，TTFT 反而上升 |
| schedule/prepare ahead | 想把 CPU 调度藏到 GPU 执行后面 | 普通 decode 存在当前 token 依赖，两个候选方案都回退 |
| 最终 Qwen hot path | TPOT 仍未到 vLLM 的 90% | 吞吐到 95.3%，TPOT 速度到 90.9%，随后冻结 kernel |
| TTFT 生命周期 trace | hot path 后 TTFT 仍约 78 ms | 62.6% 来自子事件循环等待，不是 Scheduler 计算 |
| 公平检查点 | ready task 连续错过完整模型 step | TTFT P95 降到 45.0 ms，吞吐与 TPOT不回退 |

## 1. Strict completion claim：连续性不是免费的

**发现的问题。** 最初每个请求都按 `prompt + max_new_tokens` 领取完整 completion claim。4K KV、prompt 128、最大输出 512 时，一次最多接纳：

```text
floor(4096 / (128 + 512)) = 6 requests
```

这不是 GPU 只能处理 6 条，也不是请求一定输出 512；它只是为用户声明的最坏情况预留容量。

**计划与实现。** 保持账本不变量：

```text
已使用的唯一页 + 所有 completion claims <= 总页数
```

请求一旦接纳，后续 decode 不挑第三方 victim，也不会因别的请求占满 KV 而丢掉自己的进度。

**实现中暴露的问题。** 最大输出长度通常只是上限。按上限锁容量会压低实际并发，把等待集中到 admission 队列。最初固定生成 512 token 又把这个副作用放大成了一个不符合生产分布的结果。

| 系统 | 吞吐 | P99 E2E | 容量事件 |
| --- | ---: | ---: | ---: |
| Light strict | 213.12 tok/s | 38.437 s | 0 回滚 |
| vLLM | 786.81 tok/s | 10.251 s | 13 次 preemption |

![最初的 strict 压力实验](results/2026-08-19-qwen25-coder-7b/figures/02_scheduling_pressure.png)

213 对 787 不能说明非抢占更好。它只证明已接纳请求没有回滚。换成真实输出长度的 burst 后，strict 最坏 token gap 为 0.361 秒，vLLM 为 3.360 秒；但吞吐仍是 395.18 对 786.15 tok/s。连续性改善和总效率下降同时成立。

## 2. 真实长度回放：先修实验，再谈调度

**发现的问题。** 所有请求都生成到 512 token，会把“最大长度”误当成“真实长度”，使 strict 的最坏预留几乎每次都显得正确，也让任何容量策略长期处于极端压力。

**计划与实现。** 服务端仍接收 `max_new_tokens=512`，客户端按固定参考输出长度结束。64 条 ShareGPT 请求共输出 7514 token，P50 长度 41，10 条触及 512。相同请求顺序、到达时间和目标长度在 Light 与 vLLM 之间复用。

**实现中遇到的问题。** BF16 动态 batch 会改变舍入路径，Light 与 vLLM 不保证逐 token ID 相同。因此正式验收检查请求数、错误数、输出长度和 replay 目标，不把“长度一致”写成“文本逐字一致”。固定 512 仍保留给容量上界和长稳态 profile，但不再代表正常生产结论。

## 3. 旧 rolling 10%：正确方向先被错误状态机拖垮

**发现的问题。** strict 为未必发生的最长输出占住太多页。旧 rolling 方案尝试让 running request 保留已有 KV，只追加“最坏剩余长度的 10%”；额度用完后再续领，失败则保留 KV、暂停等待。

这里的 10% 不是把 512 改成 51，也不是 10 ms 自动释放。实验中 `self-resubmit=false`，已生成 token 和对应 KV 都保留。

**计划与实现。** 请求跨过当前 claim 边界时扩大 claim，真实容量不足才等待。目标是不回滚，同时减少一次性最坏预留。

**实现中遇到的问题。** 第一版把“当前 claim 用完”当成“全局容量用完”，请求要等 KV release epoch 才重试，即使还有空闲页也会暂停。这是假的容量墙。

| workload | strict | rolling 10% 旧实现 |
| --- | ---: | ---: |
| Poisson 4 req/s 吞吐 | 317.14 tok/s | 258.37 tok/s |
| Poisson 4 req/s TTFT P95 | 5.353 s | 5.687 s |
| 64-request burst 吞吐 | 395.18 tok/s | 231.99 tok/s |
| 64-request burst 最坏 token gap | 0.361 s | 5.872 s |

rolling 比 strict 还慢，不是 10% 天生太小，而是续领状态机让请求频繁失去运行资格。

### eager refresh

修复顺序改为：先按最新 `known input + 10% worst remaining` 原子扩大 claim，再重试本轮 reservation；只有真实空闲容量不足才暂停。

| 指标 | 旧 rolling | eager refresh | 变化 |
| --- | ---: | ---: | ---: |
| steady 吞吐 | 258.37 | 327.94 tok/s | +26.9% |
| steady TTFT P95 | 5.687 | 3.308 s | -41.8% |
| steady 暂停 | 108.00 | 26.67 次 | -75.3% |
| burst 吞吐 | 231.99 | 343.87 tok/s | +48.2% |
| burst TTFT P95 | 14.974 | 11.226 s | -25.0% |

![rolling claim eager refresh](results/2026-08-20-profiled-qwen25-coder-7b/figures/07_rolling_claim_refresh.png)

三轮 A/B 都完成 64/64 请求和 7514 个输出 token，回滚仍为 0。它修掉了假暂停，但没有追平 vLLM。旧 rolling 后来没有进入主线。

## 4. TTFT 早拒：它保护服务，不会凭空增加算力

**发现的问题。** 容量紧张时，请求进入队列后可能等很久才做首次 prefill。既然 vLLM burst 的实测 TTFT P50 约为 1.25 秒，我们尝试在预测等待超过该值时提前返回 429。

`--max-tolerable-ttft-seconds 1.25` 的单位一直是秒；1.25 是对照数据，不是 vLLM 自身的配置。

**计划与实现。** `PredictiveTTFTAdmission` 根据当前 prompt、waiting/running pending tokens 与历史 step latency 估算首 token 等待。容量必然不满足仍返回 422，动态过载才返回 429。

**实现中遇到的问题。** 第一版只有 3 个 warm-up step 样本，拿低并发延迟外推 burst，每轮只接纳 3/64。正式实验将最少样本改为 100，并按同一 arrival mode 预热。即使如此，token-only predictor 仍不知道请求要等多久才能拿到 completion claim，所以 1.25 秒不是硬 SLO。

| 8 req/s 模式 | 成功请求 | 交付 token | goodput | 已接纳 TTFT P95 | 控制事件 |
| --- | ---: | ---: | ---: | ---: | --- |
| Light strict + 1.25s | 37/64 | 68.4% | 303.22 tok/s | 5.44 s | 27 个 429 |
| Light optimistic + 1.25s | 60/64 | 91.3% | 428.44 tok/s | 1.79 s | 4 个 429，2 次 resubmit |
| vLLM default，无早拒 | 64/64 | 100% | 603.88 tok/s | 1.33 s | 22 次 preemption |

![1.25 秒 TTFT 准入](results/2026-08-21-ttft125-comparison/comparison.png)

早拒减少的是已接受工作量，不是每步执行时间。报告必须同时给接纳率、429、交付 token 和 goodput；只报已接纳请求 TTFT 会掩盖代价。

## 5. 当前 optimistic：不是 rolling 10%

后续代码审计确认，当前主线没有旧 rolling claim。现在的 self-resubmit 策略是：

1. 常规请求首次只领取 `prompt + 1 block`；
2. 新乐观请求最多占用全局 90% KV，余下 10% 给已经 running 的 decode 增长；
3. running request 不持有“剩余长度 10%”的滚动 claim；
4. 真正申请新页失败时，只释放撞墙者自己的 KV并重新排队；
5. prefix cache 找回已提交的完整 prompt 页；已经发出的 token 不重复发送，但生成部分 KV 要重算；
6. 达到 resubmit 次数或累计回滚阈值后，下一次转回 strict completion claim。

**计划。** 用更小的初始承诺提前 admission，同时保留全局 decode 余量；真实撞墙才付重算代价。

**实现中遇到的问题。** self-resubmit 是全量释放该请求的 KV，`num_computed_tokens` 回到 0。已有 token 仍保留在 Engine 状态，但生成 KV 要从 prompt 重新建立。它把 strict 的 admission 等待搬到了生成中途，还会产生秒级 token gap。

| 4K KV，Poisson 8 req/s | 吞吐 | TTFT P95 | 最大 token gap P95 | 控制事件 |
| --- | ---: | ---: | ---: | --- |
| Light strict | 380.86 tok/s | 9.545 s | 0.642 s | 0 rollback |
| Light optimistic | 334.91 tok/s | 5.318 s | 3.256 s | 5 resubmit，回滚 1936 tokens |
| vLLM no-preempt，max seqs 6 | 494.04 tok/s | 5.225 s | 31 ms | 0 preemption |
| vLLM default，max seqs 16 | 606.49 tok/s | 1.285 s | 389 ms | 22 preemptions |

![当前 optimistic 与 vLLM](results/2026-08-21-optimistic-preemption-comparison/comparison.png)

optimistic 提前了首 token，但这批持续压力流量中吞吐低于 strict，流连续性也更差。当前最稳妥的产品语义仍是 strict；若继续优化 optimistic，重点应是保留已提交 KV 的局部等待或部分回滚，而不是继续调一个百分比。

## 6. Chain 与 Trie：并行验证有效，赢家取决于 workload

投机解码是独立支线。Chain-7 和 Trie-7 都把多个候选节点交给一次 target forward 验证；Trie 允许根分叉，适合第一条候选链容易走错、备选分支又经常命中的输入。

我们保留两组实验，因为它们恰好阻止了错误的泛化结论。

### 重复代码负载：Chain 更快

| 模式 | 吞吐 | TTFT P95 | TPOT P50 | 提议节点/次 | 接受 draft/次 | 验证产出/次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no spec | 251.89 tok/s | 19.974 s | 47.490 ms | — | — | 1.0 |
| Chain-7 | 516.77 tok/s | 10.488 s | 17.625 ms | 6.45 | 5.40 | 6.40 |
| Trie-7 | 468.54 tok/s | 11.311 s | 20.504 ms | 4.59 | 3.70 | 4.70 |

这组数据直接回答“Chain 接受 5.4、Trie 接受 3.7，为什么还能说 Trie 更好”：不能。这里 Chain 的平均接受数、验证产出和端到端吞吐都更高，Chain 就是更合适。

### 分支救援负载：Trie 更快

| 模式 | 吞吐 | TTFT P95 | TPOT P50 | 提议节点/次 | 接受 draft/次 | 验证产出/次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no spec | 433.25 tok/s | 0.904 s | 29.945 ms | — | — | 1.0 |
| Chain-7 | 513.34 tok/s | 0.897 s | 21.335 ms | 7.0 | 3.5 | 4.5 |
| Trie-7 | 710.92 tok/s | 0.641 s | 10.346 ms | 4.5 | 4.0 | 5.0 |

![投机验证的 workload 依赖](results/2026-08-22-complete-report/analysis/03_speculation_workload_dependence.png)

这批输入专门包含容易误导第一条链的分支前缀。Trie 用 4.5 个候选槽得到 5.0 个可见 token，Chain 用 7 个槽只得到 4.5 个，因此 Trie 更快。它证明“Trie 的分支结构在这类流量上有价值”，不证明自然流量里 Trie 普遍胜过 Chain。

两组正式实验都在旧执行版本上完成。它们足以记录功能路线和 workload 边界，但不能拿绝对吞吐与最终 Packed/Driver 版本横比。若未来要发布当前版本的投机性能主张，应重新设计代表性语料分层，而不是只重跑这组 rescue workload。

## 7. CUDA 排查：kernel 有损失，但不是全部差距

**发现的问题。** 早期 direct step wall 为 13.640 ms，CUDA total 11.861 ms，每步 549 次 launch。Attention、KV write 和大量小 kernel 都值得优化。

**计划与实现。** 依次完成 QKV 与 gate/up 权重打包、fused add + RMSNorm、SiLU、RoPE、每步复用 paged KV mapping，并调整 Triton paged attention 的分块。

优化后 direct step wall 降到 11.259 ms，CUDA total 降到 10.507 ms，launch 从 549 降到 354。这个收益是真实的。

**实现中遇到的问题。** 服务仍明显落后，说明“launch 多”不能直接解释端到端差距。随后固定 `B=16, W=1, context≈256`，双方 greedy，vLLM eager，统计 30 个稳态 decode step：

| GPU 类别 | Light | vLLM eager |
| --- | ---: | ---: |
| GEMM | 9.764 ms | 9.804 ms |
| Attention | 0.485 ms | 0.432 ms |
| KV write | 0.110 ms | 0.052 ms |
| 所有 kernels | 10.592 ms | 10.742 ms |
| launches / step | 324.0 | 383.1 |

![固定 decode 的 GPU 与服务时间](results/2026-08-20-profiled-qwen25-coder-7b/figures/08_steady_decode_time_breakdown.png)

Light 的 Attention 和 KV write 各慢约 0.05 ms，但 kernel 总和反而短 0.15 ms，launch 也更少；服务 step 却是 14.359 对 10.983 ms，相差 3.376 ms。“损失基本都在 kernel”到这里被推翻，下一步转向 CPU 和执行边界。

## 8. CPU history：第一处闭合的 CPU 热点

**发现的问题。** 相邻 Executor 调用之间有 2.637 ms 空档，其中 batch build 0.857 ms。普通 decode 每步都在复制和扫描完整 token history：

```text
Engine: tuple(state.token_ids)
  -> ExecutionRequest: 再 tuple 化
  -> 扫描全部 token
  -> 切 scheduled slice 与本轮 input 比较
```

16 条请求的 history 校验平均 0.735 ms/step，从生成 Q1 的 0.283 ms 墙钟增长到 Q4 的 1.194 ms。普通 decode 不消费这份 history，只有 draft proposer 需要。

**计划与实现。** `ExecutionRequest.context_token_ids` 改为可选。`num_lookahead_tokens=0` 时只传本轮 input；确实要提议草稿时才复制只读 history。没有把 Engine 的可变 list 直接暴露给执行线程。

**实现中遇到的问题。** 直接传 list view 虽然更省复制，却会破坏不可变批次与线程安全，因此没有采用。修复必须保留执行快照语义。

| 指标 | 修复前 | 修复后 | 变化 |
| --- | ---: | ---: | ---: |
| 吞吐 | 1103.8 | 1171.5 tok/s | +6.13% |
| Batch build | 0.841 | 0.119 ms | -85.9% |
| Executor 间空档 | 2.600 | 1.645 ms | -0.954 ms |
| CUDA event | 11.831 | 11.946 ms | 基本不变 |

![CPU history 修复](results/2026-08-20-profiled-qwen25-coder-7b/figures/10_cpu_history_fix_ab.png)

CUDA 时间不变，CPU 空档和吞吐同时改善，这一项因果关系闭合。

## 9. mixed 热路径窄修：删掉浪费，还没碰到主体

**发现的问题。** CPU history 修完后，正常 8 req/s 的 Light 仍只有 500.76 tok/s，vLLM eager 为 652.90；TTFT P95 是 1519.8 对 54.1 ms。profile 显示 mixed step 有 81.9% 的 token position 是 padding。

旧 Worker 把不同 query 长度补成 `[B,W]`。一个长 prefill 和多个单 token decode 混在一起时，所有 decode 行也按 W 运行。树形 visibility 和全宽 lm_head 又放大了浪费。

**计划与实现。** 第一轮只做低风险删除：普通线性 query 不再构造 tree visibility，lm_head 只投影 Decode Handler 声明的行。

**实现中遇到的问题。** 第一版 linear visibility 把 `QUERY_WIDTH` 设为 Triton `constexpr`，生产流量每遇到新宽度就编译新 kernel，出现 300～500 ms 冷 step。改成 runtime program axis 后才消除宽度特化。固定 W1 micro 完全看不到这个问题。

窄修使 8 req/s 吞吐提高 5.3%，TTFT 仍约 1.52 秒。说明 visibility 和 logits 的浪费是真的，但 `[B,W]` 主布局才是主体。

## 10. Packed Query：消掉两个数量级 TTFT 的主因

**计划。** 不维护 padded/packed 两套模型路径，统一成 token-major 契约：

- `ForwardBatch.input_ids/positions` 为 `[total_query_tokens]`；
- `query_start_loc=[0, len(q0), len(q0)+len(q1), ...]` 保存请求边界；
- Q/K/V、hidden states、slot mapping 都只包含真实 token；
- Triton grid 按 `total_query_tokens × heads` 启动；
- logits 只投影真正消费的全局 token 行；
- tree speculation 继续使用 `QueryLayout`，只把 visibility 紧凑拼接，不另建 padded 模型路径。

**实现流程。** 先改 `ForwardBatch` 契约和 Worker 输入，再贯通 Qwen、Dense/Paged Attention、KV slot mapping 与 Triton grid，最后补齐 linear/tree、prefix sharing、KV compact 和生成回归。旧 padded 路径没有作为长期 fallback 留在热路径。

**实现中遇到的问题。** 物理布局变化跨越模型、Attention、KV 与 logits 索引，最危险的是“数值对了但请求边界切错”。因此 `query_start_loc` 必须从 0 开始、严格递增并以总 token 数结束；selected logits 和 `logits_start_loc` 也单独做契约测试。

| req/s | 原版 TTFT P95 | Packed TTFT P95 | eager TTFT P95 | Packed / eager 吞吐 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 132.8 ms | 36.3 ms | 41.4 ms | 100.0% |
| 4 | 281.8 ms | 41.9 ms | 46.0 ms | 99.6% |
| 6 | 557.8 ms | 51.0 ms | 48.3 ms | 92.7% |
| 8 | 1532.9 ms | 46.5 ms | 53.8 ms | 91.2% |

![Packed Query TTFT](results/2026-08-21-packed-query-ttft/analysis/ttft_p95.png)

![Packed Query step 分解](results/2026-08-21-packed-query-ttft/analysis/step_breakdown.png)

8 req/s 的 57 个 mixed step 中，P50 从 63.06 降到 13.93 ms。旧布局执行 56,155 个位置，真实 token 只有 6,250；Packed 避免了 49,905 个 padding 位置。decode W1 基本不变，11.30→11.22 ms。

这组数据闭合了秒级 TTFT 根因：不是 strict 长期挡住请求，而是 mixed GPU step 太慢，首 token 消化速度低于到达速度。

同阶段还做了 `reserved_sequences=off/1/2/4/8` 消融。设为 8 会把一半 sequence slot 划给短池，TTFT 恶化到 2520.6 ms，吞吐降到 485.4 tok/s；默认值保持 1。没有数据支持把它改成 8。

## 11. Sampler、staging、timer 与伪 overlap

Packed 后剩余差距约 8%，我们逐项验证了 profile 中看起来可疑的部分。

**Sampler。** profile 表面显示约 0.62 ms/step，直接 argmax 只有 23.4 微秒；`.tolist()` 大部分时间在等待此前 GPU 工作。旧路径还会 split logits 再 stack，微基准为 86.2 微秒。packed logits 契约删除了真实重复复制，但 normal-8 吞吐只提高约 0.11%。

**Metadata staging。** 5 个小 tensor 的固定 B16/W1 传输从 83.3 降到 43.7 微秒，端到端没有稳定收益，还增加约 90 行 buffer 生命周期代码，已回退。pinned staging 也只有个位数微秒，没有达到“每步至少 0.5 ms 再引入”的门槛。

**CUDA event timer。** 关闭每步 `synchronize()` 后，固定 B16/W1 吞吐 1202.2→1204.6 tok/s，只差 0.20%；TPOT 13.176→13.187 ms。Sampler 本来就要等 logits，timer 不是 1.65 ms Executor 外 gap 的来源。

**伪 overlap。** 第一版只把 observer/stats 发布移到 Executor 提交之后。trace 证明两段没有重叠，交替 A/B 也无收益，因此回退。把函数改名成 `_prepare/_run/_finish` 不会自动形成流水。

这一阶段的重要产出不是某个大幅加速，而是删掉了四条错误方向，避免为了几十微秒引入长期复杂度。

## 12. Driver 状态机与私有 ExecutionLane

**发现的问题。** Packed 后，相同工作量中 Light 比 eager 多 1.271 秒，其中约 1.251 秒落在执行边界之间。这个数字只能定位 gap，不能说某个 Python 函数独占了 98.4%，因为双方 profile 边界并不完全同构。

**计划。** 保持单 in-flight 和 KV 安全边界，减少状态切换与通用线程池派发：

```text
completed output
  -> validate / apply / release
  -> schedule / build / acquire
  -> one immutable prepared step
```

`_PreparedStep` 与 `_CompletedStep` 让上一轮提交和下一轮准备在同一锁区原子推进；常驻私有 `ExecutionLane` 让同一个 Executor 固定在单线程上，不再每 token 提交进程级线程池。

**实现中遇到的问题。** 普通 decode 的 N+1 输入依赖本轮采样 token，KV commit 和取消状态也要先确定。不能为了“双缓冲”让同一请求在输出 apply 前再次调度。最终改动没有增加第二批 in-flight。

状态机单独 A/B 的吞吐只提高 0.29%，接近噪声；TTFT P95 降 7.15%。私有 lane 把 dispatch 从 0.328 降到 0.064 ms/step，热路径 gap 从 1.110 降到 0.808 ms/step，端到端吞吐仅提高 0.14%。GPU step 波动盖过了这段小收益。

![Driver 独立 A/B](results/2026-08-21-driver-step-pipeline/analysis/figures/03_isolated_driver_ab.png)

![Driver gap 拆解](results/2026-08-21-driver-step-pipeline/analysis/figures/04_driver_gap_breakdown.png)

当时最终 2/4/6/8 req/s 矩阵如下。它属于 `2faf265`，用于说明 Packed 与第一轮 Driver 修改后的负载曲线，不与 8 月 22 日最终分支冒充同一代码版本。

| req/s | Light strict 吞吐 | vLLM eager 吞吐 | Light / eager | Light TTFT P95 | eager TTFT P95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 202.38 tok/s | 202.43 tok/s | 100.0% | 41.77 ms | 44.31 ms |
| 4 | 398.86 tok/s | 400.15 tok/s | 99.7% | 44.33 ms | 52.25 ms |
| 6 | 522.21 tok/s | 560.78 tok/s | 93.1% | 48.71 ms | 50.89 ms |
| 8 | 603.27 tok/s | 653.40 tok/s | 92.3% | 53.89 ms | 55.73 ms |

## 13. 进程隔离：vLLM 多进程快，不是因为“多进程天然快”

**发现的问题。** HTTP、SSE、控制面和 GPU Engine 都在同一服务进程时，职责与故障边界混在一起。vLLM、SGLang 的成熟实现都把前端与执行引擎解耦，但关键在长期驻留、异步队列和稳定批次，不是简单多开一个进程。

**计划与实现。** `ProcessEngineClient` 让子进程独占 Engine、Scheduler、Worker 和 CUDA context；父进程只处理 HTTP。父子通过明确的 request/cancel/close 命令和 token/error/finish 事件通信，reader thread 将 Pipe 事件安全投递回父 asyncio loop。异常退出、取消、队列关闭和子进程回收都补了测试。

**实现中遇到的问题。** 第一组正式数据把旧的进程内 `ExecutionLane` 基线与“独立 Engine 进程 + cooperative inline”候选放在一起。8 req/s 三轮中位数从 596.42 tok/s、TTFT P95 53.14 ms、TPOT P50 13.269 ms，变成 613.69 tok/s、87.24 ms、12.292 ms。它说明新拓扑换回了持续生成性能，却不能把收益单独归因给“多进程”或“inline”。

后续在同一 `00ebbc2` 上只切换 process/inline 边界，结论仍一致：process inline 为 622.87 tok/s、TTFT P95 82.83 ms、TPOT P50 11.843 ms；关闭 process inline、恢复 `ExecutionLane` 后是 604.58 tok/s、42.95 ms、12.994 ms。多进程没有消除串行，只把问题改成了“模型 step 占住子事件循环”。

### IPC batching 与 response writer

我们还试过合并 token 事件、批量 Pipe write 和 grouped response。正式 response-writer A/B 中，baseline/candidate 三轮中位吞吐约 621.45/618.91 tok/s，TTFT P95 78.98/83.17 ms。没有收益，说明 IPC 序列化不是主要瓶颈。相关候选没有保留在产品路径。

## 14. schedule-ahead 与 prepare-ahead：两次真实但失败的 overlap

**计划。** 尝试在 GPU 执行 step N 时，让 CPU 提前 schedule/build step N+1。为了不破坏现有边界，候选都限制单设备执行，并给 provisional batch、lease 与取消建立显式生命周期。

**实现中遇到的问题。** 普通 decode 的下一输入依赖 step N 的 sampled token；新请求到达、EOS、取消、KV commit 也会改变下一批。提前准备要么只覆盖不依赖当前输出的请求子集，要么在完成后大量失效。额外状态、校验和锁竞争超过了可隐藏的 CPU 工作。

| 8 req/s 正式 A/B | baseline | candidate | 结果 |
| --- | ---: | ---: | --- |
| schedule-ahead 吞吐 | 597.05 | 593.07 tok/s | -0.7% |
| schedule-ahead TTFT P95 | 53.75 | 65.81 ms | 恶化 22% |
| prepare-ahead 吞吐 | 598.39 | 590.11 tok/s | -1.4% |
| prepare-ahead TTFT P95 | 53.35 | 67.12 ms | 恶化 26% |

两条候选都回退。真正的多步在途需要独立请求集、GPU-side sampled-token relay 或可证明安全的 provisional KV，不适合作为一次小重构塞进当前单在途 Engine。

## 15. 最终 Qwen hot path：到 90% 停止线后冻结 kernel

**发现的问题。** Packed 与 Driver 修复后，TPOT 生成速度仍略低于 vLLM。我们只处理 profile 中能指向重复工作的项，不继续泛化改 kernel。

**实现。** `00ebbc2` 缓存固定权重的转置 view，使用 `mm/addmm` 避免每 step 重建 view；位置范围在 CPU Step Handler 验证后用信任标记跳过 CUDA 热路径重复同步校验；input 与 attention metadata 复用单步 pinned staging；分页 KV 的 K/V 合并写入；GQA 同一 KV head 下的 query heads 共用 K/V 读取。

**实现中遇到的问题。** staging buffer 必须等待上一轮 H2D 完成后才能复用，但不能等待后续模型计算；跳过校验也必须以 CPU 侧完成同等验证为前提。所有优化都保持统一 Packed 契约，没有增加 Qwen 专用 Executor 分支。

| 8 req/s | 优化前 | `00ebbc2` | vLLM eager |
| --- | ---: | ---: | ---: |
| Throughput | 615.74 | 623.38 | 653.93 tok/s |
| TPOT P50 | 12.174 | 11.848 | 10.767 ms |
| TTFT P95 | 86.62 | 78.40 | 52.22 ms |

吞吐达到 vLLM 的 95.3%，TPOT 速度达到 90.9%。用户将 90% 设为可接受停止线，后续排查冻结模型、attention 和 kernel，只看 CPU、调度与 Driver。

## 16. 约 78 ms TTFT：CPU 没在算很多，而是在等 GPU

**发现的问题。** hot path 后 TPOT 已接近目标，TTFT P95 仍约 78 ms。我们给 HTTP、父子 Pipe、命令队列、Engine 注册、首次 schedule、Executor、token publish 与 SSE 增加纯内存时间戳；进程退出时才一次写盘。

64 个请求的完整 trace：

| 归并项 | 均值 | 占客户端 TTFT |
| --- | ---: | ---: |
| 子事件循环等待完整模型 step | 38.58 ms | 62.6% |
| 首 token 前实际 Executor 调用 | 14.49 ms | 23.5% |
| Engine/Scheduler bookkeeping | 0.89 ms | 1.4% |
| HTTP、IPC、dispatch、SSE | 7.63 ms | 12.4% |

最关键的三段是：child recv→command dequeue 15.62 ms、dequeue→stream task 11.55 ms、token publish→consumer wake 11.40 ms。Scheduler 本身不是大头：注册到首次 schedule 约 0.40 ms，schedule 到 execute 约 0.06 ms。

**实现中遇到的问题。** “CPU 时间”容易被误解成 Python 做了 38 ms 计算。实际是同步 Executor 占住同一个 asyncio 线程，CPU 控制任务要等 GPU/模型 step 返回。trace 加点后的 TTFT P95 79.13 ms，与无 trace 的 78.40 ms 接近，诊断开销没有改变结论。

将 process inline 关闭、恢复 ExecutionLane 可把 TTFT P95 从 82.83 降到 42.95 ms，但吞吐从 622.87 降到 604.58 tok/s，TPOT P50 从 11.84 升到 12.99 ms。直接换回线程只是用持续生成性能换 TTFT，不是最终修复。

## 17. 公平检查点：最后一处 TTFT 根因

**计划。** 保留 cooperative inline 的低 TPOT，不引入固定 sleep，也不恢复每 step 线程派发。只在两个语义边界给已经 ready 的控制任务公平运行机会：

1. 状态推进前，用两个无墙钟等待的 event-loop turn 完成 `reader → command pump → request stream` 交接；
2. `apply + schedule` 后、下一模型 step 前，让已经发布的首 token 被 stream 消费。

状态机仍是单 in-flight；`apply + schedule` 原子边界、KV lease、取消和失败提交语义都不变。

**实现中遇到的问题。** 一个裸 `await asyncio.sleep(0)` 不保证多级 ready task 链全部推进。IPC reader 先唤醒 command pump，command pump 再创建 request stream，至少需要明确覆盖这两级交接。固定睡 1～10 ms 虽容易见效，却会把延迟写死到每个 token，不能接受。

两项回归测试分别在修改前稳定失败：首 token 必须在下一模型 step 前可见；两级控制交接的新请求必须进入下一可调度 batch。修改后通过。

| 8 req/s | baseline `00ebbc2` | candidate `192d050` | vLLM eager |
| --- | ---: | ---: | ---: |
| TTFT P95 | 81.52 | **45.00** | 54.69 ms |
| TPOT P50 | 11.848 | **11.864** | 10.768 ms |
| TPOT P95 | 12.370 | **12.325** | 11.387 ms |
| Throughput | 624.01 | **624.34** | 653.74 tok/s |

TTFT P95 下降 44.8%，吞吐和 TPOT 没有实质回退。2 req/s 的 TTFT P95 也从 70.01 降到 37.23 ms。这个结果闭合了最终 TTFT 根因：被修复的是 Engine 子进程事件循环饥饿，不是 Scheduler 算法，也不是 kernel。

## 现在能说什么

### 非抢占

strict 的可靠价值是：已接纳请求不会因为第三方容量竞争丢掉 KV，压力下 token 流通常更连续。它把不确定性前移到 admission，因此最坏长度高估会降低并发、TTFT 和吞吐。正常 32K 实验没有容量事件，只能说明 strict 不再妨碍执行效率，不能证明非抢占胜过 vLLM 的抢占。

当前 optimistic 也没有形成更好的普遍折中。`prompt + 1 block + 90% watermark` 提前 admission，撞墙后的全量重算却让吞吐和 token gap 变差。下一步若继续调度研究，应先设计保留已提交 KV 的等待或部分回滚语义，并证明不会形成所有 running 请求互相占页的死锁。

### 执行性能

两个数量级的 TTFT 差距来自 mixed padding，已经由 Packed Query 消除；最终约 78 ms 的尾部来自 cooperative inline 下的事件循环饥饿，已经由公平检查点消除。

剩余高负载吞吐差距约 4.5%，TPOT 生成速度差距约 9.2%。最终 profile 边界并不完全同构，不能把差值全部算给某一类 kernel 或某一段 Python。考虑到 90% 停止线已经达到，继续追赶需要 CUDA Graph、更成熟的 fused kernel 或 GPU-side 控制 relay，成本明显高于本轮收益。

### Trie

Trie 不是“平均接受数更低但仍天然更好”。重复代码负载里 Chain 更快；分支救援负载里 Trie 更快。正确的产品方向是根据真实流量的候选分支分布选择 proposer，或建立在线但有界的策略选择，而不是把一组挑选 workload 的 1.64× 写成通用结论。

## 不应再重复的做法

- 不用固定 512-token 输出代表正常生产分布；它只用于容量边界和长稳态 profile。
- 不只报已接纳请求 TTFT。早拒实验必须同时给接纳率、429、交付 token 和 goodput。
- 不把 kernel duration、CUDA event、CPU wall 和服务 step 混加。
- 不用固定 W1 micro 推断 mixed prefill/decode 的生产表现。
- 不因为代码看起来能 overlap 就保留；trace 与交替 A/B 没收益就回退。
- 不把一次局部优化扩大成系统结论。eager refresh 修的是假暂停，Packed 修的是 padding，公平检查点修的是事件循环饥饿。

## 关键提交

| 提交 | 内容 |
| --- | --- |
| `6ff6f8e` | strict 非抢占 completion claim |
| `1ba494e` | 当前 optimistic admission、TTFT 控制与部分 fused kernel |
| `46901d4` | selected logits、线性 visibility 等窄修复 |
| `c1f18dc` | 统一 token-major Packed Query |
| `39d3031` | 保留 packed model-step logits |
| `578b8f6` | 合并 Driver 状态推进 |
| `a9b8114` | 常驻私有 ExecutionLane |
| `b8237ae` | Engine 独立进程与 ProcessEngineClient |
| `e8dcbf8` | 子进程 cooperative inline 执行 |
| `00ebbc2` | Qwen、Paged Attention 与 staging 最终 hot path |
| `192d050` | inline step 前后的事件循环公平检查点 |
| `30cde15` | 最终 Driver 对比报告与数据索引 |

schedule-ahead、prepare-ahead、metadata packing、伪 overlap、IPC response batching 和旧 rolling 10% 都是历史实验代码或已回退候选。本报告保留它们，是为了说明路线如何被数据淘汰，不是建议恢复。

## 数据与复现入口

- 本报告统一重算脚本：[`analyze_complete_journey.py`](analyze_complete_journey.py)
- 最终 P50/P90/P95、CSV 与三张主图：[`results/2026-08-22-complete-report/analysis/`](results/2026-08-22-complete-report/analysis/)
- strict、rolling、TTFT、CUDA、CPU、投机原始数据：[`results/2026-08-20-profiled-qwen25-coder-7b/`](results/2026-08-20-profiled-qwen25-coder-7b/)
- 当前 optimistic 与 vLLM 抢占：[`results/2026-08-21-optimistic-preemption-comparison/`](results/2026-08-21-optimistic-preemption-comparison/)
- 1.25 秒 TTFT gate：[`results/2026-08-21-ttft125-comparison/`](results/2026-08-21-ttft125-comparison/)
- Packed Query：[`results/2026-08-21-packed-query-ttft/report.md`](results/2026-08-21-packed-query-ttft/report.md)
- Sampler / staging：[`results/2026-08-21-driver-sampler-staging/report.md`](results/2026-08-21-driver-sampler-staging/report.md)
- 第一轮 Driver pipeline：[`results/2026-08-21-driver-step-pipeline/analysis/report.md`](results/2026-08-21-driver-step-pipeline/analysis/report.md)
- 最终 hot path：[`results/2026-08-22-final90-hotpath/analysis/report.md`](results/2026-08-22-final90-hotpath/analysis/report.md)
- TTFT 生命周期拆解：[`results/2026-08-22-ttft-cpu-scheduling-breakdown/REPORT.md`](results/2026-08-22-ttft-cpu-scheduling-breakdown/REPORT.md)
- 公平检查点 A/B：[`results/2026-08-22-driver-control-fairness/REPORT.md`](results/2026-08-22-driver-control-fairness/REPORT.md)

统一统计可以在仓库根目录复现：

```powershell
.\.venv\Scripts\python.exe benchmarks/remote_5090/analyze_complete_journey.py `
  --output benchmarks/remote_5090/results/2026-08-22-complete-report/analysis
```

本次二次整理没有再运行 GPU 实验。最终主表已有同机三轮逐请求数据；Trie/Chain 也已有常规与 rescue 两类重复实验。为了整理报告再跑一轮，不会改变 workload 依赖这个结论。
