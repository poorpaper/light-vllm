# 单节点双 GPU Tensor Parallel

这个 overlay 让一个 Pod 申请两张 GPU，并由 `torchrun` 启动两个 Rank。Rank 0
运行 Engine 并 spawn 独立 HTTP/tokenizer 前端，Rank 1 只执行模型命令；它不是两个独立推理副本。

## 前置条件

- 单个 Kubernetes 节点至少有两张可调度的同型号 NVIDIA GPU；
- NVIDIA Container Runtime 和 Device Plugin 已工作，节点显示
  `nvidia.com/gpu: 2` 或更多；
- `light-vllm:dev` 已导入节点运行时，模型 PVC 已准备完成；
- 集群存在名为 `nvidia` 的 RuntimeClass。如果 NVIDIA 已是默认 runtime，
  可以在站点 overlay 中删除 `runtimeClassName`。

```bash
kubectl apply -k deploy/kubernetes/overlays/single-node-tp2
kubectl rollout status deployment/light-vllm --timeout=15m
kubectl get pod -l app.kubernetes.io/name=light-vllm -o wide
kubectl describe node | grep -A4 nvidia.com/gpu
```

`/dev/shm` 使用独立的 8 GiB 内存卷，供单机 NCCL 共享内存传输使用。Pod
仍采用 `Recreate`，升级时不会同时申请四张 GPU。

## 验收边界

性能验收必须与裸机使用同一镜像、模型、workload、KV 容量和三次正式运行，
同时保存 Pod spec、节点 GPU 拓扑、NCCL 日志和客户端原始结果。只有静态渲染
或 Pod Ready 不能证明容器化没有性能回退。

当前 overlay 只覆盖单节点 TP=2。多节点 rendezvous、跨节点网络和故障恢复
属于后续阶段，不能通过增加副本数替代。
