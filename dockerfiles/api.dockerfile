FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

RUN uv pip install --system \
    fastapi==0.115.6 \
    google-auth>=2.36.0 \
    pydantic>=2.0.0 \
    pyyaml>=6.0.0 \
    requests>=2.32.0 \
    uvicorn==0.34.0

COPY src/rice_cnn_classifier/api.py /app/api.py
COPY config_gpu.yaml config_gpu.yaml

ENV PYTHONPATH=/app/src

ENTRYPOINT ["sh", "-c", "python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]

