# Driver step pipeline：根因、改动与 5090 实验

## 一句话结论

Light 的通用线程池派发确实是可消除损失，但不是全部差距。单在途常驻执行 lane 把下一轮派发从
`0.328` 降到 `0.064 ms/step`；最终正常 8 req/s 下 Light strict
达到 `603.27 tok/s`，为 vLLM eager 的 `92.3%`，TTFT P95 为
`53.89 ms`（eager `55.73 ms`）。

## 两个隔离改动

| 改动 | 吞吐变化 | TTFT P95 变化 | 判断 |
| --- | ---: | ---: | --- |
| 合并上一轮提交与下一轮准备 | +0.29% | -7.15% | 收益接近噪声 |
| 私有常驻 ExecutionLane | +0.14% | -5.08% | 派发边界下降，保留 |

两组都是独立 A/B，各自按 `A/B/B/A/A/B` 交替运行。不能把两组绝对吞吐简单相减，因为 GPU 频率会漂移。

## 剩余 Driver 时间

连续热路径（相邻执行边界总 gap `<5 ms`）的 `driver_control` 进一步拆成：输出处理
均值 `0.092`、转入原子推进 P50 `0.013`、原子 apply/schedule/build
`0.575`、进入下一次执行 `0.014 ms/step`。其中 apply、schedule 和 build 依赖
上一 token 的 CPU 结果，不能靠再次移动锁或再加一个线程安全隐藏。

若要继续接近 vLLM/SGLang，需要单独设计 GPU-side sampled-token relay、持久 batch/metadata 和
有界异步调度，让 CPU 准备 N+1 时不等待 N 的 token 回传。那是新的执行契约，不应伪装成本轮小修。

## 实验边界

- Qwen2.5-Coder-7B-Instruct、BF16、RTX 5090、32K KV tokens、block 16、
  max sequences 16、token budget 512。
- ShareGPT replay，Poisson 2/4/6/8 req/s；prefix cache、TTFT admission、speculation 关闭。
- 每点 64 请求 warmup 后 3 轮正式测试；吞吐取三轮中位数，TTFT 合并 192 个成功请求。
- Light strict 保持 completion claim，全部正式轮次要求零拒绝、零 self-resubmit。
- vLLM default 只作为 graph 性能上限；主归因基准是 vLLM eager。
- 36 个正式结果均为 64/64 请求成功，且每轮都生成相同的 7514 个输出 token；这是等工作量对照。
  BF16 动态批次会改变舍入路径，所以不同运行之间不保证生成 token ID 逐项完全相同，本文不作这项声明。

## 代码版本

- 原始基线：`cd4fd2ca19f6eb980d38247590976a15003635a1`
- 合并状态推进：`578b8f69b2db276f5bed4fc2b8f47090e238d5f0`
- 私有执行 lane：`a9b81148cab470de4a57da717caa4a9fb166b136`
- 最终实验与分析：`2a1e741`

## 图

- `figures/01_ttft_distribution.png`
- `figures/02_throughput_vs_eager.png`
- `figures/03_isolated_driver_ab.png`
- `figures/04_driver_gap_breakdown.png`
