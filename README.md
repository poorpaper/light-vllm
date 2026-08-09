# light-vllm

一个以可维护性为第一约束的轻量 LLM 推理框架草稿。

我们的目标不是在第一天复刻 vLLM 的全部能力，而是先建立一组稳定、清晰、可组合的边界：

- **轻框架**：核心只保留注册、加载、执行三个必要动作。
- **热插拔**：模型和加载器通过 `Catalog` 注册；扩展能力不修改核心分发逻辑。
- **高扩展**：依赖接口和组合，不让功能矩阵演化成跨模块的 `if-else`。
- **高可读性**：配置、模型、权重加载、运行生命周期各自只有一个职责。
- **可验证**：每个扩展点都有小而直接的契约测试。

> 这是一个独立的实验项目，目前不是 vLLM 的兼容替代品。

## 当前最小闭环

```text
ModelSpec
   │
   ├── loader name ──────> Loader registry ──> ModelLoader
   └── architecture ─────> Model registry ──> ModelFactory
                                               │
                                               v
                                        torch.nn.Module
                                               │
ForwardBatch ─────────────────────────> ModelRunner.forward
                                               │
                                               v
                                          ModelOutput
```

首版包含：

- `TinyCausalLM`：`Embedding -> Linear` 的最简单 causal-LM forward。
- `InitModelLoader`：只初始化模型，适合测试和结构验证。
- `StateDictModelLoader`：从本地 PyTorch state dict 加载权重。
- `ModelRunner`：统一执行入口；新模型加载成功后原子替换旧模型。
- `Catalog` / `Registry`：显式扩展点，避免在核心路径增加类型判断。

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -e ".[dev]"
python examples/minimal_forward.py
python -m pytest
```

最小调用：

```python
import torch

from light_vllm import ForwardBatch, ModelSpec, create_runner

runner = create_runner()
runner.load(
    ModelSpec(
        architecture="tiny-causal-lm",
        loader="init",
        model_args={"vocab_size": 128, "hidden_size": 32},
    )
)

output = runner.forward(ForwardBatch(input_ids=torch.tensor([[1, 2, 3]])))
print(output.logits.shape)  # torch.Size([1, 3, 128])
```

## 扩展方式

第三方能力只需实现契约并注册：

```python
catalog.models.register("my-model", my_model_factory)
catalog.loaders.register("my-format", my_loader)
```

已有名称默认不可覆盖；需要有意识地替换时才传 `replace=True`。架构图与运行时序见
[`docs/design.md`](docs/design.md)，更细的边界规则见
[`docs/architecture.md`](docs/architecture.md)。

## 当前非目标

调度器、Paged KV Cache、定制 attention kernel、分布式执行和 API server 暂不进入首版。
先让模型装载和一次 forward 的边界足够稳定，再逐层增加能力。
