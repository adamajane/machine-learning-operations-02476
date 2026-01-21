FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu124

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# First sync to create venv and install dependencies (including CPU PyTorch)
RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

# Replace CPU PyTorch with CUDA version (--reinstall forces replacement)
RUN .venv/bin/pip install --index-url ${TORCH_CUDA_INDEX} \
    --extra-index-url https://pypi.org/simple \
    torch==2.6.0 torchvision==0.21.0

ENTRYPOINT ["uv", "run", "src/rice_cnn_classifier/train.py"]
