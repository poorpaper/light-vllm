# TP 控制路径优化验收

本轮只测裸机 TP，不涉及 Docker。目标是验证旧实现的每步 Gloo Python object collective 是否构成真实瓶颈，
并在不改变 Engine、Scheduler、模型分片和 NCCL tensor collective 的前提下替换控制传输。

## 结论

- 旧路径每个 step 的 Gloo 命令广播与状态同步合计约 `1.007 ms`；不是截图中的猜测，而是 Rank 0 分层计时结果。
- 单机 Unix socket 命令通道将同一开销降至约 `0.170 ms`，减少 `83.1%`，每步节省约 `0.837 ms`。
- 截图把完整词表 AllGather 当作 light-vllm 特有问题并不成立：当前 vLLM 的词表并行层在采样前同样 gather logits。
  本机 primitive 测量中该 AllGather 约 `0.138 ms`，不是这轮 1 ms 级差距的主因，因此没有提前引入分布式采样。
- 固定 `16 x 512` decode 的 3 轮中位吞吐从 `1125.48` 提升到 `1209.68 tok/s`，提升 `7.48%`。
- 同条件 vLLM eager 为 `1302.81 tok/s`；light-vllm 达到其 `92.85%`，差距 `7.15%`。
- TP=1 用相反运行顺序各做一组 `3+3`，合计每个版本 6 轮；吞吐中位数为 `1194.14` 与
  `1194.96 tok/s`，候选版本 `+0.07%`，未观察到单卡回退。

## 条件与边界

- 机器：两张 RTX 5090 32 GiB；GPU 间为 `SYS` 拓扑，无 CUDA P2P。
- 软件：Python 3.12、PyTorch 2.11.0+cu130、vLLM 0.26.0。
- 模型：Qwen2.5-Coder-7B-Instruct，BF16。
- light-vllm 使用 Triton paged attention；vLLM 使用 eager，关闭 CUDA Graph。
- 每个配置先 warmup，再运行 3 轮；表中使用正式轮次中位数。
- 该结论只覆盖此模型、负载和无 P2P 双卡拓扑，不外推到跨节点或其他 batch/context 分布。

| 实现 | 吞吐中位数 | TPOT P50 中位数 |
| --- | ---: | ---: |
| light-vllm Gloo 控制 | 1125.48 tok/s | 14.034 ms/token |
| light-vllm Unix socket 控制 | 1209.68 tok/s | 13.101 ms/token |
| vLLM eager | 1302.81 tok/s | 12.161 ms/token |

## 正确性与故障

- TP=1/2 对 4 个 prompt、每个 8 个输出 token 逐 token 比较，无差异。
- 活动请求中终止 Rank 1 后，torchrun 以非零状态退出；16.519 秒内端口释放，两张卡显存归零，无残留进程。
- 显式 `gloo` 回退与默认 `auto`/socket 使用同一 Executor 契约；`auto` 检测到多主机时保留 Gloo，为后续跨节点
  执行留出边界。

精确逐轮数字见 [summary.json](summary.json)。远端原始 JSON、日志和 GPU 采样保留在
`/root/autodl-tmp/tp-control-perf-20260826/`；本地只保留这份小摘要，避免占用本机磁盘。

架构边界参考了 vLLM 的
[MultiprocExecutor](https://github.com/vllm-project/vllm/blob/main/vllm/v1/executor/multiproc_executor.py) 与
[共享内存广播队列](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/shm_broadcast.py)，
以及 SGLang 的
[共享内存广播实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/shm_broadcast.py)。
light-vllm 当前命令小、单在途，先用更短且易审计的 Unix socket 实现同一“tensor collective 与控制传输分离”
边界；`_CommandChannel` 保留以后替换共享内存或跨节点 RPC 的扩展点。vLLM 的
[词表并行实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py)
也明确记录了采样阶段 gather logits 的语义。
