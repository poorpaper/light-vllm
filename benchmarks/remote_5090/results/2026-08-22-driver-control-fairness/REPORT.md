# Driver 控制路径公平检查点实验

## 结论

在不修改模型、attention、CUDA kernel 和 Scheduler 语义的前提下，协作式 inline driver 的显式公平检查点把
8 req/s 的 TTFT P95 从 81.52 ms 降到 45.00 ms，下降 44.8%；吞吐从 624.01 tok/s 变为
624.34 tok/s，TPOT P50 从 11.848 ms 变为 11.864 ms，两者都没有实质回退。

与同机新跑的 vLLM eager 相比，候选版吞吐达到 95.5%，TPOT 生成速度达到 90.8%，TTFT P95 为
vLLM 的 82.3%。这说明本轮被修复的主要是 Engine 子进程事件循环饥饿，而不是 kernel 执行速度。

## 代码与环境

- baseline：`00ebbc2f78be20bf96dde968e31867dd55b6de2b`
- candidate：`192d05061e2222116078fb05fb46e73d75b7b375`
- 分支：`codex/driver-control-fairness`
- 模型：Qwen2.5-Coder-7B-Instruct，BF16
- GPU：NVIDIA GeForce RTX 5090，32607 MiB
- KV：32768 tokens，block size 16
- max sequences：16
- step token budget：512
- workload：固定 seed `20260821` 的 ShareGPT 首轮回放，Poisson 到达，每轮 64 请求
- Light：strict completion claim，非抢占，prefix cache、TTFT admission、speculation 均关闭
- vLLM：eager，prefix cache、speculation 关闭

每个 Light 版本交替启动三次，每次先预热再正式运行一次。vLLM eager 在同一机器重新启动后连续正式运行三次。
表格数值均为三轮汇总指标的中位数。

## 正式结果

### 8 req/s

| 指标 | baseline | candidate | vLLM eager | candidate 变化 |
| --- | ---: | ---: | ---: | ---: |
| TTFT P95 | 81.52 ms | **45.00 ms** | 54.69 ms | **-44.8%** |
| TPOT P50 | 11.848 ms | **11.864 ms** | 10.768 ms | +0.13% |
| TPOT P95 | 12.370 ms | **12.325 ms** | 11.387 ms | -0.36% |
| Throughput | 624.01 tok/s | **624.34 tok/s** | 653.74 tok/s | +0.05% |

合并三轮 192 个请求后，baseline/candidate 的 TTFT P95 分别为 83.37/45.24 ms，与逐轮中位数结论一致。

### 2 req/s

| 指标 | baseline | candidate | vLLM eager | candidate 变化 |
| --- | ---: | ---: | ---: | ---: |
| TTFT P95 | 70.01 ms | **37.23 ms** | 48.38 ms | **-46.8%** |
| TPOT P50 | 10.671 ms | **10.658 ms** | 9.911 ms | -0.12% |
| TPOT P95 | 11.114 ms | **10.928 ms** | 10.783 ms | -1.67% |
| Throughput | 202.24 tok/s | **202.41 tok/s** | 202.44 tok/s | 到达率受限 |

2 req/s 吞吐受请求到达率限制，只用于检查低负载延迟和回归，不代表引擎最大吞吐。

## 根因与修改

旧实现让同步 Executor 直接占用 Engine 子进程的 asyncio 事件循环，并仅在模型步骤返回后执行一次裸
`sleep(0)`。IPC reader、command pump、新请求 stream 和 token stream 可能连续错过事件循环调度，等待一个或多个
完整模型步骤。

候选版保持单在途状态机和原子的 `apply + schedule` 边界：

1. 状态推进前，用两个无墙钟等待的事件循环 turn 让 `reader → command pump → request stream` 完成控制交接；
2. `apply + schedule` 后、下一次模型执行前，再给已存在的 stream 一轮机会消费刚发布的 token；
3. 不增加固定 sleep，不把 Executor 切到线程，不改变 KV lease、取消或失败提交语义。

两项回归测试分别验证首 token 必须在下一步模型执行前可见，以及两级控制交接的新请求必须进入下一个可调度
batch。测试修改前稳定失败，修改后通过。

## 正确性与验证

- 8 req/s 和 2 req/s 的六个 Light 正式轮次均为 64/64 成功；
- 每轮均输出 7514 tokens，64/64 请求达到固定 replay 目标长度；
- baseline 自身不同轮次的 BF16 动态 batch 输出也不保证逐 token bitwise 相同，因此不把跨轮 token 内容一致性
  作为本次性能修复的正确性证据；输出长度、请求数、错误数和生成边界完全一致；
- 本地全量 pytest 通过；ruff check、format check 和 `git diff --check` 通过；
- 远端 Engine/Process 定向测试通过。

## 数据

- `driver-control-fairness-ab-192d050-20260822.tar.gz`：8 req/s Light 交替 A/B 全部 JSON、metrics 和日志；
- `driver-control-fairness-low-ab-192d050-20260822.tar.gz`：2 req/s Light 交替 A/B；
- `driver-control-fairness-vllm-eager-20260822.tar.gz`：同机新跑的 8 req/s vLLM eager；
- `analysis/summary.json`：三轮中位数和相对比例；
- `analysis/comparison.png`：2/8 req/s 对比图；
- `code/light-vllm-driver-control-fairness-192d050.bundle`：可离线恢复实验代码的 Git bundle。

实验机按要求保持开机。本分支和实验结果均未推送 GitHub。
