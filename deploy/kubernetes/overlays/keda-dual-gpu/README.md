# 单机双 GPU 的 KEDA 扩缩容

这个 overlay 用两个独立副本验证最小横向扩缩容路径：

```text
请求 → Service → Pod A / GPU 0
              └→ Pod B / GPU 1

Pod /metrics → Prometheus → KEDA → HPA → Deployment
```

它不是 tensor parallel。模型必须能装进一张 GPU；每个 Pod 都加载完整模型并拥有独立 Engine、Scheduler
和 KV cache。Kubernetes NVIDIA Device Plugin 会把两个 `nvidia.com/gpu: 1` 请求分配给不同 GPU。

## 前置条件

- 单节点 Kubernetes，节点上有两张可调度且型号接近的 NVIDIA GPU；
- NVIDIA Container Toolkit 和 Kubernetes Device Plugin；
- Prometheus Operator、Prometheus 与 KEDA 2.20；
- 已推送的不可变 light-vllm 镜像，以及可被两个 Pod 只读挂载的模型卷。

应用前检查 `scaled-object.yaml` 中三项站点配置：

1. `serverAddress` 与实际 Prometheus Service 一致；
2. PromQL 的 `namespace` 与部署命名空间一致；
3. `threshold` 已通过单 GPU 压测按 TTFT SLO 标定。

如果 Prometheus 只选择带 release 标签的 `PodMonitor`，还要给 `pod-monitor.yaml` 增加集群要求的标签。

## 为什么用 token backlog

`light_vllm_waiting_max_remaining_tokens` 是等待请求剩余工作的保守上界。它能区分“一个 8k-token 请求”
和“一个 32-token 请求”，比请求数更接近 GPU 工作量；GPU 利用率则是滞后信号，不能直接说明队列压力。

KEDA 使用 `AverageValue`，因此目标副本数近似为：

```text
ceil(所有 Pod 的 waiting token backlog / 4096)
```

结果再限制在 `[1, 2]`。`4096` 只是可运行的初始值，不是通用最优参数；HPA 的采样周期和容差也会让触发点
不是严格的单点阈值。

## 部署与观察

不要同时应用 `examples/monitoring/hpa.yaml`。KEDA 会自己创建并管理 HPA，同一个 Deployment 只能有一个
扩缩容控制器。

```bash
kubectl apply -k deploy/kubernetes/overlays/keda-dual-gpu
kubectl get pods -l app.kubernetes.io/name=light-vllm -w
kubectl get scaledobject light-vllm
kubectl get hpa light-vllm-keda -w
```

在 Prometheus 先执行与 `scaled-object.yaml` 完全相同的查询，确认它只返回一个数值。然后准备一份能持续至少
两分钟、且能让 backlog 稳定超过阈值的 workload：

```bash
python benchmarks/remote_5090/serve_benchmark.py \
  --backend light-vllm \
  --base-url http://127.0.0.1:8000 \
  --workload /data/workloads/keda-sustained.json \
  --output /data/results/keda-scale-up.json \
  --case keda-scale-up \
  --arrival-mode poisson \
  --request-rate 8 \
  --max-connections 128 \
  --disable-keepalive
```

`--disable-keepalive` 让请求使用新 TCP 连接，使 Kubernetes Service 有机会把扩容后的流量分给新 Pod。实验要
保存客户端 TTFT、TPOT、吞吐和 `/metrics`；实际 request rate 应按单 GPU 基线调整，不能机械照抄示例值。

验收至少覆盖：

1. 空闲时只有一个 Ready Pod，GPU 0/1 中只有一张被占用；
2. backlog 持续超过阈值后，HPA 从 1 调到 2；
3. 第二个 Pod 使用另一张 GPU，完成模型加载并通过 readiness 后才接收新请求；
4. 扩容期间记录客户端 TTFT P95，区分排队时间与第二副本冷启动时间；
5. 压力停止并稳定 10 分钟后回到一个副本；
6. Prometheus 暂时不可用时，KEDA fallback 保持当前副本数而不是盲目缩容。

## 当前边界

- 已经进入 Pod A 本地队列的请求不会迁移到 Pod B；新副本只帮助后续到达的连接。生产环境通常还需要
  感知负载的网关或中心队列。
- `emptyDir` 隔离两个 Pod 的 Triton/Torch 编译缓存，避免并发写冲突，但新副本需要重新 JIT，冷启动更长。
- `minReplicaCount: 1` 有意关闭 scale-to-zero。副本为零时 Pod 自身指标也会消失，而且模型冷启动会显著伤害
  第一个请求的 TTFT。
- 单机仍是一个故障域，只能证明横向扩缩容机制，不能宣称多节点高可用。
- `preStop` 只给 Service 摘流留出 10 秒；完整的应用级 draining 和请求迁移仍未实现。
