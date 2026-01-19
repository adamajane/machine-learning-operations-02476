FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu121

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen
RUN uv pip install --index-url ${TORCH_CUDA_INDEX} --extra-index-url https://pypi.org/simple torch==2.6.0+cu121 torchvision==0.21.0+cu121

ENTRYPOINT ["uv", "run", "src/rice_cnn_classifier/train.py"]
