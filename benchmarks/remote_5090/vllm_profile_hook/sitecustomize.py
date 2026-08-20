"""Benchmark-only startup hook for the vLLM CPU stage profiler."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(Path(__file__).parents[1] / "profile_vllm_cpu_stages.py")
