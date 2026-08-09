# AGENTS.md

本文件适用于整个 light-vllm 仓库。开始修改代码前，先阅读本文件和
[`docs/design.md`](docs/design.md)。当核心边界发生变化时，同步更新这两份文档。

## 项目目标

light-vllm 是一个以可维护性为第一约束的轻量 LLM 推理运行时。长期目标是：

- 轻框架：核心保持小而稳定。
- 热插拔：模型、权重加载器和后续执行后端通过明确扩展点接入。
- 高扩展：新增能力优先增加实现并注册，不修改核心分发逻辑。
- 高可读性：配置、路由、加载、模型计算和运行生命周期职责分离。
- 少边界 case：避免由功能矩阵产生的跨模块 `if-elif-else` 条件树。

仓库远端为 `https://github.com/poorpaper/light-vllm`；当前工作站的主本地副本位于
`D:\light-vllm`。

## 当前阶段

当前只完成 P0 model path：

- `ModelSpec` 描述待加载模型。
- `Catalog` / `Registry` 选择 model factory 与 loader。
- `ModelLoader` 构造模型、加载权重并准备推理。
- `ModelRunner` 管理当前模型并提供统一 `forward`。
- `TinyCausalLM` 提供 `Embedding → Linear → logits` 的最小参考实现。

Scheduler、KV cache、Paged Attention、prefill/decode 拆分、分布式执行和 serving
尚未实现。不要让这些未来能力提前污染当前契约。

## 代码地图

| 文件 | 职责 |
| --- | --- |
| `src/light_vllm/contracts.py` | 稳定的数据契约和组件 Protocol |
| `src/light_vllm/registry.py` | 通用的名字到组件映射 |
| `src/light_vllm/catalog.py` | 持有 model/loader 两类扩展点 |
| `src/light_vllm/bootstrap.py` | 内置组件的唯一装配位置 |
| `src/light_vllm/loaders/torch.py` | 基础 PyTorch 模型加载器 |
| `src/light_vllm/models/tiny.py` | 最小参考模型 |
| `src/light_vllm/runner.py` | 模型生命周期与统一 forward |
| `tests/test_minimal_forward.py` | 当前核心契约的行为测试 |

## 必须保持的架构不变量

1. `ModelRunner` 不得按 architecture、loader 名称或具体模型类型写功能分支。
2. 模型和 loader 必须经 `Catalog` 中的注册表解析。
3. 候选模型应在生命周期锁外完整构造；成功后才替换当前模型。
4. 替换 `_model` 和递增 `_generation` 必须属于同一个临界区。
5. 候选模型加载失败时，当前模型与 generation 必须保持不变。
6. `forward` 只在锁内复制当前模型引用；实际 `model(batch)` 不得长期持有生命周期锁。
7. 所有模型接受 `ForwardBatch`，返回 `ModelOutput`。
8. loader 负责把模型移动到目标 device/dtype 并切换到 `eval()`。
9. 注册同名组件默认报错；只有调用方显式传入 `replace=True` 才能覆盖。

## Runner 中锁的准确含义

当前锁是生命周期锁，只保护 `_model` 与 `_generation` 的一致性，并为并发 reload/forward
提供清晰的 happens-before 边界。`forward` 取得本地模型引用后立即释放锁，因此旧模型即使随后被
替换，本次 forward 仍持有旧对象的有效引用。

这个锁不负责：

- 串行化所有模型推理；
- 保证某个 `torch.nn.Module` 可被任意线程并发调用；
- 管理 CUDA stream、显存回收或 KV cache 迁移；
- 提供跨进程同步。

当前使用 `RLock` 只是实现选择，架构并不依赖可重入语义。若修改这一处，可以换成普通
`Lock`，但必须继续满足上面的临界区和失败回滚测试。

## Registry 与 Contracts 的边界

`Registry[T]` 是显式路由表：将稳定名字映射到某一类组件。当前有两张表：

- `catalog.models`：architecture 名称 → `ModelFactory`；
- `catalog.loaders`：loader 名称 → `ModelLoader`。

Registry 不是完整依赖注入容器，也不负责自动发现 Python package entry point。自动插件发现如需
加入，应作为 Catalog 之上的独立装配能力实现。

`contracts.py` 定义模块之间允许交换的稳定形状：

- 数据：`ModelSpec`、`ForwardBatch`、`ModelOutput`；
- 行为：`ModelFactory`、`ModelLoader` Protocol。

Protocol 主要服务于静态类型和可读性，并不会自动做完整运行时校验。必要的边界校验应放在数据
契约或真正消费该契约的位置，避免散落在具体实现中。

## 新增扩展的方式

新增模型：

1. 实现统一接收 `ModelSpec` 的 factory。
2. 返回接受 `ForwardBatch`、输出 `ModelOutput` 的 `nn.Module`。
3. 在 `bootstrap.py` 注册内置模型，或由外部 plugin 注册。
4. 增加测试，证明无需修改 `ModelRunner` 即可运行。

新增权重格式：

1. 实现 `ModelLoader.load(spec, factory)`。
2. 在 loader 内完成构造、权重加载、device/dtype 移动和 `eval()`。
3. 注册新 loader 名称；不要在 runner 中增加格式判断。
4. 覆盖成功加载、无效配置和加载失败不影响当前模型的测试。

## 开发与验证

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
```

提交前至少执行 pytest、Ruff lint 和 Ruff format check。示例和核心测试应默认可在 CPU 上运行。
除非新依赖能明显简化稳定边界或提供必要能力，否则不要增加运行时依赖。

## 下一步建议

优先设计 P1 request path：不可变 Request、Scheduler 接口和执行批次契约。先让请求如何进入批次、
批次如何交给 runner 变得清楚，再引入 KV cache 与 prefill/decode 优化。

