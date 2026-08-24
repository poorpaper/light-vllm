# 生产部署骨架性能 A/B

## 结论

在本次 RTX 5090、Qwen2.5-Coder-7B-Instruct BF16 条件下，没有观察到生产部署骨架带来的明显原生性能劣化。
production-like 三轮中位数的吞吐下降 0.41%，TPOT P50 增加 0.77%，TTFT P95 增加 2.68 ms；
固定 decode 场景的吞吐下降 0.03%，TPOT P50 增加 0.003 ms。两组都通过预设门槛：

- 吞吐或 TPOT 回退不超过 3%；
- TTFT P95 绝对增量不超过 5 ms。

这只证明非 Docker 原生模式没有明显回归。测试节点没有 Docker、Podman、containerd 或其他可用容器运行时，
所以当前不能宣称 Docker 模式也没有性能损耗。

## 对照条件

| 项目 | Baseline | Candidate |
| --- | --- | --- |
| Commit | `6fb26c3` | `79322eb` |
| `src/` | 与 Candidate 完全相同 | 与 Baseline 完全相同 |
| GPU | RTX 5090 32607 MiB | 相同 |
| Driver | 595.71.05 | 相同 |
| Python / Torch / CUDA | 3.12.13 / 2.11.0+cu130 / 13.0 | 相同 |
| 模型 | Qwen2.5-Coder-7B-Instruct BF16 | 相同 |
| Runtime | Engine process + Triton Paged Attention | 相同 |
| Scheduler | 16 sequences / 512 tokens per step | 相同 |
| KV | 32768 tokens / block size 16 | 相同 |

Candidate 只增加 Dockerfile、Compose、systemd、Kubernetes 和部署文档，两个提交之间的 `src/` diff 为空。

## Production-like ShareGPT 回放

每轮 64 个请求、8 req/s、固定到达序列，并从同一个参考结果回放每个请求的输出长度。每侧先 warmup，
再正式运行三轮；表中是三轮中位数。

| 指标 | Baseline | Candidate | 变化 |
| --- | ---: | ---: | ---: |
| 成功请求 | 64/64 | 64/64 | 相同 |
| 输出 token | 7514 | 7514 | 相同 |
| 吞吐 | 625.246 tok/s | 622.679 tok/s | -0.41% |
| TTFT P50 | 27.519 ms | 28.657 ms | +1.138 ms |
| TTFT P95 | 44.113 ms | 46.788 ms | +2.675 ms |
| TPOT P50 | 11.814 ms | 11.905 ms | +0.77% |
| TPOT P95 | 12.225 ms | 12.370 ms | +1.19% |
| E2E P95 | 6.101 s | 6.161 s | +1.00% |

为降低顺序运行偏差，在 Candidate 后又补跑一轮 Baseline：吞吐 624.713 tok/s、TTFT P95 45.092 ms、
TPOT P50 11.835 ms，仍落在两侧的相邻区间。结合运行时代码完全相同，0.41% 的吞吐差应视为本次测量噪声，
不能归因为部署文件。

## 固定 decode 场景

16 个请求同时到达，每个请求固定生成 512 tokens；表中是 warmup 后的一轮正式结果。

| 指标 | Baseline | Candidate | 变化 |
| --- | ---: | ---: | ---: |
| 成功请求 | 16/16 | 16/16 | 相同 |
| 输出 token | 8192 | 8192 | 相同 |
| 吞吐 | 1333.777 tok/s | 1333.359 tok/s | -0.03% |
| TTFT P95 | 59.411 ms | 60.049 ms | +0.637 ms |
| TPOT P50 | 11.893 ms | 11.896 ms | +0.003 ms |

## 输出一致性边界

独立单请求生成 128 tokens 时，Baseline 与 Candidate 的 token 序列完全一致。

并发回放不能作为 token-exact 证明：production-like 三轮分别有 26、24、22 个请求的序列不同，固定 decode
也有 9/16 不同。但这不是 Candidate 独有现象：Baseline 自己的两轮 production-like 有 28/64 不同，Baseline
自己的 fixed warmup 与正式轮也有 9/16 不同。证据说明当前 BF16 并发执行本身存在输出漂移；动态 batching
时序和数值路径是合理推断，但本次测试没有进一步定位其内部根因。

因此本报告只把固定输出长度用于匹配性能工作量，并把单请求结果用于 token-exact 正确性检查；不把并发的
“输出长度相同”写成“token 完全相同”。

## 原始证据

- `baseline/`、`candidate/`：production-like warmup、三轮原始请求结果、Prometheus 前后快照、服务日志和环境清单；
- `baseline-reverse/`：Candidate 之后补跑的 Baseline；
- `fixed-baseline/`、`fixed-candidate/`：固定 decode 结果；
- `isolated-baseline.json`、`isolated-candidate.json`：串行 128-token token-exact 对照；
- `summary.json`：本报告中的机器可读汇总。
