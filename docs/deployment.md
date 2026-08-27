# 生产部署骨架

这一版提供单节点生产运行骨架和访问本地 GPU 的 OpenAI-compatible 文本协议，不引入网关、多节点通信或 Helm。
单卡原生、Docker 和 Kubernetes 都启动同一个 `light-vllm-serve` 入口：

```text
HTTP 父进程 → ProcessEngineClient → Engine 子进程 → Worker → GPU

TP: HTTP 子进程 → ConnectionEngineClient → torchrun Rank 0 Engine → Rank Worker → GPU
```

部署配置只存在于控制面，不修改 `EngineCore`、Scheduler、Worker 或模型热路径。因此同一运行参数下，
本次代码不会增加逐 token 逻辑；容器带来的差异需要在目标 GPU 上用相同 workload 做 A/B 验证。

## 1. 原生启动

原生方式最适合先建立性能基线，也可以直接作为 systemd 服务运行：

```bash
python -m venv /opt/light-vllm/.venv
/opt/light-vllm/.venv/bin/python -m pip install "/opt/light-vllm[serve,triton]"

/opt/light-vllm/.venv/bin/light-vllm-serve \
  --host 0.0.0.0 \
  --architecture qwen2.5 \
  --loader safetensors \
  --weights /data/models/Qwen2.5-Coder-7B-Instruct \
  --tokenizer /data/models/Qwen2.5-Coder-7B-Instruct \
  --served-model-name Qwen2.5-Coder-7B-Instruct \
  --request-timeout-seconds 300 \
  --device cuda:0 \
  --dtype bfloat16 \
  --runtime engine \
  --engine-process \
  --kv-reservation blocks \
  --paged-attention-backend triton
```

systemd 示例位于 `deploy/native/`。安装前创建专用用户，并确保它有权限读取模型和访问 NVIDIA 设备；
`CacheDirectory=light-vllm` 会创建并授权 `/var/cache/light-vllm`：

```bash
sudo useradd --system --home /opt/light-vllm --shell /usr/sbin/nologin light-vllm
sudo install -d -o light-vllm -g light-vllm /etc/light-vllm
sudo install -m 0644 deploy/native/light-vllm.service /etc/systemd/system/
sudo install -m 0644 deploy/native/light-vllm.env.example /etc/light-vllm/light-vllm.env
sudo systemctl daemon-reload
sudo systemctl enable --now light-vllm
```

unit 要求 `/etc/light-vllm/light-vllm.env` 存在；站点参数只在该文件维护，缺失时服务会明确启动失败。

## 2. Docker Compose

前置条件是 Docker Compose v2.24.4+ 和 NVIDIA Container Toolkit；TP=2 覆盖层使用这一版本开始提供的
`!override` 合并标签。复制示例环境文件并填写宿主机模型目录：

```bash
cp deploy/docker/.env.example deploy/docker/.env
docker compose \
  --env-file deploy/docker/.env \
  -f deploy/docker/compose.yaml \
  up --build -d
```

镜像默认基于已验证版本组合 `PyTorch 2.11.0 + CUDA 13.0`，也可在构建时通过 `BASE_IMAGE` 替换。Compose 继承
镜像内相同的 healthcheck，不重复声明第二份。
模型只读挂载，Triton/Torch 编译缓存写入独立 volume，容器根文件系统保持只读。当前进程间通信使用 Pipe，
不依赖 `--ipc=host` 或额外共享内存权限。

单容器两卡 TP 使用额外的 Compose 配置。它以 `torchrun` 取代单卡 entrypoint，并提供 8 GiB 容器内共享内存；
模型与缓存卷仍沿用基础配置：

```bash
docker compose \
  --env-file deploy/docker/.env \
  -f deploy/docker/compose.yaml \
  -f deploy/docker/compose.tp2.yaml \
  up --build -d
```

`compose.tp2.yaml` 使用 Compose 的 `!override` 标签把单卡 GPU 申请替换成两卡，避免列表合并后同时保留 1 卡和
2 卡设备申请。TP 的 Rank 0 Engine 会自动 spawn 独立 HTTP/tokenizer 前端，因此覆盖后的命令不再使用
`--engine-process`。

## 3. Kubernetes

`deploy/kubernetes/base/` 是单 GPU、单副本的 Kustomize 基础层：

```bash
kubectl apply -k deploy/kubernetes/base
```

应用前必须完成三项站点配置：

1. 把镜像改为仓库中的不可变 tag 或 digest；
2. 根据集群 StorageClass 调整 PVC，并预先把模型放进 `light-vllm-models`；
3. 根据 GPU 和 workload 调整并发、单轮 token budget、KV 比例以及 CPU/内存 requests。

`startupProbe` 最多给模型加载 10 分钟；`readinessProbe` 检查模型与 Engine 连接是否就绪；
启动完成后的 `livenessProbe` 也检查 `/readyz`，Engine 或 TP 前端死亡时会重建 Pod。终止时 K8s 先摘除 Pod，
再给进程 180 秒完成已有请求和释放 CUDA。
已有的 Prometheus、Grafana 和 HPA 示例继续位于 `examples/monitoring/`。

一个 Pod 内的两卡 Tensor Parallel 使用独立 overlay：

```bash
kubectl apply -k deploy/kubernetes/overlays/single-node-tp2
```

它申请两张 GPU、挂载 8 GiB `/dev/shm` 并用 torchrun 启动两个 Rank；与下面的 KEDA 横向扩容是两种不同拓扑。
详细前置条件和验收边界见
[`deploy/kubernetes/overlays/single-node-tp2/README.md`](../deploy/kubernetes/overlays/single-node-tp2/README.md)。

单机两张 GPU 时，可以用可选的 KEDA overlay 验证 1→2→1 横向扩缩容：

```bash
kubectl apply -k deploy/kubernetes/overlays/keda-dual-gpu
```

它按所有 Pod 的 waiting token backlog 扩容，每个 Pod 申请一张 GPU，并把编译缓存改为 Pod 独占。这是两个各自
加载完整模型的独立副本，不是 Tensor Parallel。
详细前置条件、阈值语义、验证步骤和能力边界见
[`deploy/kubernetes/overlays/keda-dual-gpu/README.md`](../deploy/kubernetes/overlays/keda-dual-gpu/README.md)。
不要把 `examples/monitoring/hpa.yaml` 与这个 overlay 同时应用到同一个 Deployment；KEDA 会自己创建 HPA。

## 4. 烟测与性能门槛

三种方式都使用相同检查：

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/capabilities
curl -fsS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-Coder-7B-Instruct","messages":[{"role":"user","content":"Hello"}],"temperature":0,"max_tokens":16}'
```

`/v1/completions`、`/v1/chat/completions` 同时支持流式与非流式响应；`/generate`、`/generate/stream` 继续作为
token-ID 调试接口。`--tokenizer` 必须指向本地 tokenizer 目录，启动过程不联网且不执行远程代码。

正式切换前要在相同模型、请求序列、到达时间和输出 token 下比较原生与容器：

- 输出 token 必须完全一致；
- 吞吐和 TPOT 回退目标不超过 3%；
- 客户端 TTFT P95 增量目标不超过 5 ms；
- 同时保存 `/metrics` 和客户端端到端结果，不能用 Engine 内部 TTFT 代替客户端 TTFT。

当前第一版不做自动调优。只有 A/B 数据证明默认参数不合适时，才修改部署层参数；性能逻辑仍不进入推理核心。
