# Light-vLLM 90% 热路径优化实验

主结论基于 Poisson 生产长度负载，每个实现独立启动并运行 3 次，取各轮指标中位数。

## 8 req/s

| 指标 | Light optimized | vLLM eager | 相对结果 |
| --- | ---: | ---: | ---: |
| Throughput | 623.38 tok/s | 653.93 tok/s | 95.3% |
| TPOT P50 | 11.848 ms | 10.767 ms | 90.9% speed |
| TTFT P95 | 78.40 ms | 52.22 ms | 1.50x |

## 2 req/s

| 指标 | Light optimized | vLLM eager | 相对结果 |
| --- | ---: | ---: | ---: |
| Throughput | 202.25 tok/s | 202.44 tok/s | 99.9% |
| TPOT P50 | 10.692 ms | 9.911 ms | 92.7% speed |
| TTFT P95 | 70.29 ms | 48.38 ms | 1.45x |

## 执行边界诊断

| 边界 | Light optimized | vLLM eager |
| --- | ---: | ---: |
| 设备/模型执行 | 10.913 ms/step (CUDA event) | 8.860 ms/step (execute_model wall) |
| 步间空档 | 0.579 ms/step | 0.210 ms/step |
| 端到端 step span | 11.532 ms/step | 10.533 ms/step |

设备/模型执行边界并不完全相同，不能把两列之差直接解释成某个 kernel 的差值；它们只用于定位剩余差距仍同时包含模型执行与 Driver 空档。

## 结论边界

Light strict 保持 completion claim 非抢占语义；所有正式请求均成功。该结论只适用于记录的模型、RTX 5090、vLLM eager 配置和本次负载。
继续追赶最后约 9% TPOT 预计需要 CUDA Graph、更多融合 kernel 或真正的多步在途流水，改动与验证成本显著高于本轮保留的局部热路径优化，因此在 90% 停止线收敛。
