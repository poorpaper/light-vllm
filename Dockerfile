# syntax=docker/dockerfile:1

# 默认版本与当前 RTX 5090 验证环境一致；其他 GPU 可在构建时替换基础镜像。
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

FROM ${BASE_IMAGE}

ARG LIGHT_VLLM_UID=10001
ARG LIGHT_VLLM_GID=10001
ARG VERSION=dev
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="light-vllm" \
      org.opencontainers.image.description="Lightweight LLM inference runtime" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/poorpaper/light-vllm"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/var/cache/light-vllm \
    TRITON_CACHE_DIR=/var/cache/light-vllm/triton \
    TORCHINDUCTOR_CACHE_DIR=/var/cache/light-vllm/torchinductor \
    TORCH_EXTENSIONS_DIR=/var/cache/light-vllm/torch-extensions

# AWQ CUDA 算子在首次加载时按当前 GPU 架构编译并进入共享 extension cache。
# TP Rank 会复用同一产物，因此镜像必须提供 nvcc、C++ 编译器和 Ninja。
RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${LIGHT_VLLM_GID}" light-vllm \
    && useradd --uid "${LIGHT_VLLM_UID}" --gid "${LIGHT_VLLM_GID}" \
        --create-home --shell /usr/sbin/nologin light-vllm \
    && mkdir -p /opt/light-vllm /var/cache/light-vllm \
    && chown -R light-vllm:light-vllm /opt/light-vllm /var/cache/light-vllm

WORKDIR /opt/light-vllm

# 只复制安装所需文件，避免把 benchmark、测试产物和本地模型打进镜像。
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install ".[serve,triton]" \
    && rm -rf pyproject.toml README.md src

USER light-vllm:light-vllm
EXPOSE 8000
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=10s --timeout=3s --start-period=10m --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2).read()"]

ENTRYPOINT ["light-vllm-serve"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--architecture", "qwen2.5", "--loader", "safetensors", "--weights", "/models/model", "--tokenizer", "/models/model", "--device", "cuda:0", "--dtype", "bfloat16", "--runtime", "engine", "--engine-process", "--kv-reservation", "blocks", "--paged-attention-backend", "triton"]
