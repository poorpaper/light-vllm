from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "light_vllm"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)

    return modules


def _class_definition(path: Path, class_name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name!r} was not found in {path}")


def test_interface_modules_do_not_import_their_implementations() -> None:
    boundaries = {
        SOURCE_ROOT / "runtime" / "engine" / "interfaces.py": (
            "light_vllm.runtime.engine.core",
            "light_vllm.runtime.engine.in_process",
        ),
        SOURCE_ROOT / "runtime" / "execution" / "interfaces.py": (
            "light_vllm.runtime.execution.local",
            "light_vllm.runtime.execution.worker",
        ),
        SOURCE_ROOT / "runtime" / "generation" / "interfaces.py": (
            "light_vllm.runtime.generation.reference",
        ),
        SOURCE_ROOT / "modeling" / "loaders" / "interfaces.py": (
            "light_vllm.modeling.loaders.torch",
        ),
        SOURCE_ROOT / "modeling" / "models" / "interfaces.py": ("light_vllm.modeling.models.tiny",),
        SOURCE_ROOT / "runtime" / "scheduler" / "interfaces.py": (
            "light_vllm.runtime.scheduler.token_budget",
        ),
    }

    for path, implementations in boundaries.items():
        assert not set(implementations) & _imported_modules(path), path


def test_generation_reference_does_not_import_model_backend_details() -> None:
    modules = _imported_modules(SOURCE_ROOT / "runtime" / "generation" / "reference.py")
    forbidden_prefixes = (
        "torch",
        "light_vllm.modeling.models",
        "light_vllm.modeling.runner",
    )

    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in modules
        for prefix in forbidden_prefixes
    )


def test_engine_core_does_not_import_model_or_transport_details() -> None:
    modules = _imported_modules(SOURCE_ROOT / "runtime" / "engine" / "core.py")
    forbidden_prefixes = (
        "torch",
        "light_vllm.modeling.models",
        "light_vllm.modeling.runner",
        "light_vllm.serving",
    )

    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in modules
        for prefix in forbidden_prefixes
    )


def test_cacheable_models_delegate_softmax_to_attention_context() -> None:
    """模型可以生成 Q/K/V，但不能重新拥有 attention kernel。"""

    targets = (
        (SOURCE_ROOT / "modeling" / "models" / "tiny_attention.py", "TinyAttentionCausalLM"),
        (SOURCE_ROOT / "modeling" / "models" / "qwen2.py", "Qwen2Attention"),
    )
    for path, class_name in targets:
        class_node = _class_definition(path, class_name)
        softmax_calls = [
            node
            for node in ast.walk(class_node)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "softmax"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "softmax"
            )
        ]
        assert not softmax_calls, f"{class_name} must call AttentionContext instead"
