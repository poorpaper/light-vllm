# 性能观测与扩缩容

Engine runtime 启动后直接暴露 `GET /metrics`，不需要在 token 热路径安装 Prometheus SDK。指标采集、
展示和控制仍是三个边界：`PerformanceObserver` 记录事实，Prometheus adapter 读取快照，Grafana/HPA
只消费指标。

## 本地 Prometheus / Grafana

1. 以 `--runtime engine` 启动 light-vllm。
2. 用本目录的 `prometheus.yml` 启动 Prometheus；容器不能识别
   `host.docker.internal` 时，替换成服务实际地址。
3. 在 Grafana 配置 Prometheus datasource，导入 `grafana-dashboard.json`。

看板已经包含 TTFT、TPOT、输出吞吐、请求队列、token-aware 队列和 KV cache 使用率。主要指标为：

| 指标 | 含义 |
| --- | --- |
| `light_vllm_time_to_first_token_seconds` | 请求准入到首个可见 token |
| `light_vllm_time_per_output_token_seconds` | 首 token 之后，相邻可见输出 token 的平均间隔 |
| `light_vllm_requests_waiting` | 等待 Scheduler 槽位的请求数 |
| `light_vllm_queue_tokens` | 等待请求尚未计算的保守 token 上界，适合作为 HPA backlog |
| `light_vllm_kv_cache_usage_ratio` | 不能立即回收的分页 KV slot / 总 slot |

`queue_tokens` 使用请求的最大输出预算，因此偏保守，但它不需要 tokenizer 或猜测未来停止点。连续 KV
基线没有固定容量，所以不会输出 KV 容量与使用率指标。

边界与业界常见做法保持一致：vLLM 也区分 waiting/running、TTFT/TPOT 与 KV 使用率，SGLang 把运行请求、
队列请求和 token/KV 指标收敛在独立 observability collector；light-vllm 保留这些稳定概念，但用一个很小的
快照接口隔离具体监控 SDK。可对照 [vLLM metrics logger](https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py)
和 [SGLang metrics collector](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/observability/metrics_collector.py)。

## Kubernetes HPA

`pod-monitor.yaml` 让 Prometheus Operator 按 Pod 抓取 `/metrics`，并产生 Adapter 规则需要的
`namespace` / `pod` 标签。目标 Deployment 的 Pod template 需要标签
`app.kubernetes.io/name: light-vllm`，容器端口需要命名为 `http`。

`prometheus-adapter-values.yaml` 给出 Prometheus Adapter 规则，`hpa.yaml` 使用 `autoscaling/v2` 的 Pods
`AverageValue`。三个文件共同组成 Kubernetes 路径；本地静态 `prometheus.yml` 只供单机 Grafana 体验，
不提供每 Pod 标签。阈值 `4096` 只是示例，应按模型、GPU、并发压测和 TTFT SLO 调整，并保留 scale-down
稳定窗口以减少抖动。

相关上游文档：

- [Kubernetes HPA](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter)
- [Grafana provisioning](https://grafana.com/tutorials/provision-dashboards-and-data-sources/)
