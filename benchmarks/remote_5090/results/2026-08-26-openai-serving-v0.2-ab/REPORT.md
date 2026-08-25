# OpenAI Serving v0.2 原生性能 A/B

## 结论

最终提交 `9afada2` 在 RTX 5090、Qwen2.5-Coder-7B-Instruct BF16 上通过原生性能门槛，
没有观察到相对 `main@fc285b9` 的可归因性能损失。

为降低先后运行造成的漂移，最终统计用 Candidate 前后的两组 Baseline 包围 Candidate：
production-like 使用 6 轮有效 Baseline 的中位数，固定 decode 也使用 6 轮 Baseline 的中位数；
Candidate 均使用 warmup 后 3 轮中位数。

| 负载 | 指标 | Baseline | Candidate | 变化 |
| --- | --- | ---: | ---: | ---: |
| Production-like | 吞吐 | 605.871 tok/s | 606.673 tok/s | +0.13% |
| Production-like | TTFT P95 | 50.487 ms | 50.565 ms | +0.078 ms |
| Production-like | TPOT P50 | 12.606 ms | 12.577 ms | -0.23% |
| Production-like | TPOT P95 | 13.316 ms | 13.280 ms | -0.27% |
| 固定 decode | 吞吐 | 1270.388 tok/s | 1277.556 tok/s | +0.56% |
| 固定 decode | TTFT P95 | 72.639 ms | 75.968 ms | +3.329 ms |
| 固定 decode | TPOT P50 | 12.483 ms | 12.392 ms | -0.73% |
| 固定 decode | TPOT P95 | 12.492 ms | 12.404 ms | -0.70% |

两组都通过预设门槛：吞吐或 TPOT 回退不超过 3%，TTFT P95 绝对增量不超过 5 ms。
表中的小幅正负变化落在本次同机反向复跑的波动范围内，不能写成性能提升。

## 发现并修掉的回归

简化后的首个候选提交 `f7e37b1` 虽未超过 3% 总门槛，但出现了稳定的 decode 差异：

| 负载 | 吞吐变化 | TTFT P95 变化 | TPOT P50 变化 | TPOT P95 变化 |
| --- | ---: | ---: | ---: | ---: |
| Production-like | -0.49% | -0.22 ms | +1.55% | +1.81% |
| 固定 decode | -1.08% | +5.21 ms | +0.97% | +0.97% |

原因是 `ConfigurableSampler` 即使面对全 greedy 批次，也逐行执行 `argmax().item()`；在 CUDA 上这会把一次
批量结果读取拆成多次主机同步。最终修复保留 Worker 到 Sampler 的单一调用边界，只在 Sampler 内识别同质
greedy 批次，并恢复一次 `logits.argmax(dim=-1).tolist()`。没有把采样模式条件树放回 Worker。

## 对照条件

| 项目 | Baseline | Candidate |
| --- | --- | --- |
| Commit | `fc285b9` | `9afada2` |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB | 相同 |
| Driver / 功耗上限 | 580.95.05 / 575 W | 相同 |
| Python / Torch / Triton | 3.12.13 / 2.8.0+cu128 / 3.4.0 | 相同 |
| 模型 | Qwen2.5-Coder-7B-Instruct BF16 | 相同 |
| Runtime | 独立 Engine 进程 + Triton Paged Attention | 相同 |
| Scheduler | 16 sequences / 512 scheduled tokens | 相同 |
| KV | 2048 blocks × 16 tokens = 32768 tokens | 相同 |

Candidate 启用了本地 tokenizer 和 served model 配置。A/B 请求走双方共有的 `/generate/stream` token 接口，
因此本报告验证的是新增 OpenAI adapter 和采样元数据接入后，既有 Engine greedy 热路径没有回归；它不把文本
tokenization 和 detokenization 的协议开销混入 GPU 执行对照。

## 负载与统计

- Production-like：64 个 ShareGPT 首轮请求，8 req/s Poisson 到达；prompt 长度 30～509 tokens；从同一个
  reference 结果回放每个请求的输出长度，每轮 7514 个输出 tokens。
- 固定 decode：16 个请求同时到达；每个 prompt 16 tokens，每个请求固定生成 512 tokens；每轮共 8192 个输出
  tokens。
- 每个进程先 warmup，再运行正式轮次。Baseline 的 production `r1`、`r2` 仍含首次 Triton shape 编译，未纳入
  排名；`r4`、`r5` 因误并行启动、互相争用同一服务而无效，也未纳入排名，但原始文件保留用于审计。
- 最终 production Baseline 使用 `baseline/r3,r6,r7` 与 `baseline-reverse/r1,r2,r3`；固定 decode Baseline 使用
  两侧各 `r1,r2,r3`。Candidate 使用 `candidate-fixed/r1,r2,r3`。

## 证据边界

本报告只证明这台机器上的非 Docker、单卡、greedy Engine 热路径未出现可测回归。它不证明：

- Docker/Kubernetes 模式没有额外开销；测试机没有可用 Docker 运行时；
- 随机采样与 greedy 等速；随机采样本来就需要过滤、softmax 和 multinomial；
- 其他 GPU、模型、上下文长度或更高并发也有相同结果。

原始 JSON 和 Prometheus 前后快照位于 `baseline/`、`candidate/`、`candidate-fixed/` 和
`baseline-reverse/`；机器可读汇总见 `summary.json`。
