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


def test_api_modules_do_not_import_their_implementations() -> None:
    boundaries = {
        SOURCE_ROOT / "engine" / "api.py": (
            "light_vllm.engine.batched",
            "light_vllm.engine.in_process",
        ),
        SOURCE_ROOT / "execution" / "api.py": ("light_vllm.execution.local",),
        SOURCE_ROOT / "generation" / "api.py": ("light_vllm.generation.reference",),
        SOURCE_ROOT / "loaders" / "api.py": ("light_vllm.loaders.torch",),
        SOURCE_ROOT / "models" / "api.py": ("light_vllm.models.tiny",),
        SOURCE_ROOT / "scheduler" / "api.py": ("light_vllm.scheduler.iteration",),
    }

    for path, implementations in boundaries.items():
        assert not set(implementations) & _imported_modules(path), path


def test_generation_reference_does_not_import_model_backend_details() -> None:
    modules = _imported_modules(SOURCE_ROOT / "generation" / "reference.py")
    forbidden_prefixes = ("torch", "light_vllm.models", "light_vllm.runner")

    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in modules
        for prefix in forbidden_prefixes
    )


def test_iteration_engine_does_not_import_model_or_transport_details() -> None:
    modules = _imported_modules(SOURCE_ROOT / "engine" / "batched.py")
    forbidden_prefixes = (
        "torch",
        "light_vllm.models",
        "light_vllm.runner",
        "light_vllm.serving",
    )

    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in modules
        for prefix in forbidden_prefixes
    )
