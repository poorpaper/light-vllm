# light-vllm 架构设计（v0.1）

这份文档描述 light-vllm 当前已经落地的最小架构，以及后续功能应该沿着哪些边界继续生长。

核心思想只有一句话：

> **稳定核心只负责编排契约；变化能力通过注册组件接入。**

![light-vllm 架构总览](assets/architecture-overview.svg)

静态文件：[SVG 矢量图](assets/architecture-overview.svg) ·
[PNG 预览图](assets/architecture-overview.png)

## 1. 设计目标

| 目标 | 在当前架构中的落实方式 |
| --- | --- |
| 轻量 | 核心闭环只有 `Catalog → Loader/Factory → ModelRunner → Model` |
| 热插拔 | 组件可以运行时注册；新模型完整加载后再原子替换旧引用 |
| 高扩展 | 模型架构和权重格式各自拥有独立注册表 |
| 高可读性 | 配置、选择、加载、执行四个职责分开 |
| 少边界 case | 功能分发依赖映射，不依赖不断扩大的 `if-elif-else` 树 |

完全消灭 `if` 不是目标。输入校验、错误处理和生命周期检查仍然应该显式存在；需要消除的是跨功能、跨后端的条件分发树。

## 2. 总体组件图

```mermaid
flowchart TB
    subgraph PublicAPI["① 公开契约层"]
        Spec["ModelSpec<br/>模型架构 / loader / 权重 / 设备"]
        Batch["ForwardBatch<br/>input_ids"]
        Output["ModelOutput<br/>logits"]
    end

    subgraph Runtime["② 稳定运行时核心"]
        Runner["ModelRunner<br/>模型生命周期 + 统一 forward"]
    end

    subgraph Extension["③ 扩展与路由层"]
        Catalog["Catalog"]
        ModelRegistry["Registry&lt;ModelFactory&gt;"]
        LoaderRegistry["Registry&lt;ModelLoader&gt;"]
        Catalog --> ModelRegistry
        Catalog --> LoaderRegistry
    end

    subgraph Implementations["④ 可替换实现层"]
        Factory["ModelFactory"]
        Loader["ModelLoader"]
        Tiny["TinyCausalLM"]
        Init["InitModelLoader"]
        StateDict["StateDictModelLoader"]
        ThirdParty["第三方 Model / Loader"]
        Factory --> Tiny
        Loader --> Init
        Loader --> StateDict
        ModelRegistry -.注册.-> ThirdParty
        LoaderRegistry -.注册.-> ThirdParty
    end

    subgraph Execution["⑤ PyTorch 执行层"]
        Model["Active torch.nn.Module"]
    end

    Spec -->|load| Runner
    Runner -->|按名字查找| Catalog
    ModelRegistry --> Factory
    LoaderRegistry --> Loader
    Factory -->|构造| Model
    Loader -->|装载权重并准备推理| Model
    Batch -->|forward| Runner
    Runner -->|调用当前模型| Model
    Model --> Output

    classDef contract fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e;
    classDef core fill:#ede9fe,stroke:#7c3aed,color:#4c1d95;
    classDef registry fill:#dcfce7,stroke:#16a34a,color:#14532d;
    classDef impl fill:#fef3c7,stroke:#d97706,color:#78350f;
    classDef execution fill:#fee2e2,stroke:#dc2626,color:#7f1d1d;
    class Spec,Batch,Output contract;
    class Runner core;
    class Catalog,ModelRegistry,LoaderRegistry registry;
    class Factory,Loader,Tiny,Init,StateDict,ThirdParty impl;
    class Model execution;
```

最关键的依赖方向是：`ModelRunner` 依赖 `ModelLoader` 和 `ModelFactory` 契约，不依赖具体的 Tiny、Llama、Safetensors 或量化实现。

## 3. 模型加载时序

```mermaid
sequenceDiagram
    autonumber
    actor App as 调用方
    participant Runner as ModelRunner
    participant Models as Model Registry
    participant Loaders as Loader Registry
    participant Loader as ModelLoader
    participant Factory as ModelFactory
    participant Candidate as Candidate Model

    App->>Runner: load(ModelSpec)
    Runner->>Models: get(spec.architecture)
    Models-->>Runner: factory
    Runner->>Loaders: get(spec.loader)
    Loaders-->>Runner: loader
    Runner->>Loader: load(spec, factory)
    Loader->>Factory: factory(spec)
    Factory-->>Loader: torch.nn.Module
    Loader->>Candidate: load weights / to(device, dtype) / eval()

    alt 候选模型加载失败
        Loader--xRunner: exception
        Note over Runner: 当前模型和 generation 保持不变
        Runner--xApp: exception
    else 候选模型加载成功
        Loader-->>Runner: candidate
        Note over Runner: 只在候选模型完整可用后进入锁
        Runner->>Runner: atomic swap + generation += 1
        Runner-->>App: success
    end
```

这个顺序刻意避免“先卸载旧模型，再尝试加载新模型”。首版保证进程内模型引用的原子替换；它还没有承诺跨模型迁移 KV cache 或请求状态。

## 4. 最简单的 forward

```mermaid
sequenceDiagram
    autonumber
    actor App as 调用方
    participant Runner as ModelRunner
    participant Model as Active Model

    App->>Runner: forward(ForwardBatch)
    Runner->>Runner: 在锁内读取当前模型引用
    Note over Runner: 随即释放锁，forward 不长期占锁
    Runner->>Model: model(batch)
    Model->>Model: Embedding(input_ids)
    Model->>Model: Linear(hidden_states)
    Model-->>Runner: ModelOutput(logits)
    Runner-->>App: ModelOutput
```

当前 `TinyCausalLM` 只实现 `Embedding → Linear`。它不是为了模拟完整 Transformer，而是为了先固定输入输出、加载和运行生命周期这三条契约。

## 5. 热替换状态机

```mermaid
stateDiagram-v2
    state "未加载" as Empty
    state "加载首个候选模型" as InitialLoading
    state "服务中" as Ready
    state "后台构造新候选模型" as Reloading
    state "原子替换引用" as Swapping

    [*] --> Empty
    Empty --> InitialLoading: load(spec)
    InitialLoading --> Empty: 加载失败
    InitialLoading --> Ready: 加载成功 / generation = 1
    Ready --> Reloading: load(new_spec)
    Reloading --> Ready: 加载失败 / 旧模型不变
    Reloading --> Swapping: 候选模型可用
    Swapping --> Ready: generation += 1
```

## 6. 为什么新增能力不改核心

```mermaid
flowchart LR
    Plugin["新插件"] --> NewModel["实现 ModelFactory"]
    Plugin --> NewLoader["实现 ModelLoader"]
    NewModel -->|register| ModelRegistry["Model Registry"]
    NewLoader -->|register| LoaderRegistry["Loader Registry"]
    ModelRegistry --> Runner["既有 ModelRunner"]
    LoaderRegistry --> Runner
    Runner --> Result["自动获得新组合"]

    NoChange["无需修改<br/>ModelRunner / 既有实现"]
    NoChange -.约束.-> Runner

    classDef plugin fill:#fef3c7,stroke:#d97706,color:#78350f;
    classDef registry fill:#dcfce7,stroke:#16a34a,color:#14532d;
    classDef core fill:#ede9fe,stroke:#7c3aed,color:#4c1d95;
    class Plugin,NewModel,NewLoader plugin;
    class ModelRegistry,LoaderRegistry registry;
    class Runner,Result,NoChange core;
```

例如新增 `SafetensorsLoader` 时，应该新增 loader 实现并注册：

```python
catalog.loaders.register("safetensors", SafetensorsLoader())
```

不应该在 `ModelRunner.load` 中加入：

```python
# 不推荐：功能矩阵最终会变成难以维护的条件树
if spec.loader == "safetensors":
    ...
elif spec.loader == "state-dict":
    ...
```

## 7. 代码映射

| 架构角色 | 当前文件 |
| --- | --- |
| 公开数据契约 | `src/light_vllm/contracts.py` |
| 通用注册表 | `src/light_vllm/registry.py` |
| 模型与 loader 目录 | `src/light_vllm/catalog.py` |
| 内置组件装配 | `src/light_vllm/bootstrap.py` |
| 模型生命周期和 forward | `src/light_vllm/runner.py` |
| 基础 loader | `src/light_vllm/loaders/torch.py` |
| 最小参考模型 | `src/light_vllm/models/tiny.py` |

## 8. 当前边界与下一步

```mermaid
flowchart LR
    P0["P0 已完成<br/>Loader + Minimal Forward"] --> P1["P1 Request Path<br/>Request / Scheduler 契约"]
    P1 --> P2["P2 Memory Path<br/>KV Cache / Block Allocator"]
    P2 --> P3["P3 Execution Path<br/>Prefill / Decode / Batch"]
    P3 --> P4["P4 Serving Path<br/>Async / Streaming / API"]

    classDef done fill:#dcfce7,stroke:#16a34a,color:#14532d;
    classDef next fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e;
    class P0 done;
    class P1,P2,P3,P4 next;
```

下一步最合理的工作不是立刻加入 CUDA kernel，而是先定义 `Request`、`Scheduler` 和执行批次之间的稳定边界，让后续 KV cache 与 prefill/decode 优化有清晰的挂载点。
