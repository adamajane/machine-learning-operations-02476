FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu124

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# First sync to create venv and install dependencies (including CPU PyTorch)
RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

# Remove CPU PyTorch and install CUDA version
RUN uv pip uninstall torch torchvision && \
    uv pip install \
    --index-url ${TORCH_CUDA_INDEX} \
    --extra-index-url https://pypi.org/simple \
    torch==2.6.0 torchvision==0.21.0

# Verify CUDA is available (will print during build)
# Use .venv/bin/python directly to avoid uv sync overwriting CUDA PyTorch with CPU version
RUN .venv/bin/python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

ENTRYPOINT [".venv/bin/python", "src/rice_cnn_classifier/train.py"]
