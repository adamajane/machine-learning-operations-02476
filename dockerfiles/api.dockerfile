FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "rice_cnn_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
