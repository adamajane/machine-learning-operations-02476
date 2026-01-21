FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

RUN uv pip install \
    fastapi==0.115.6 \
    uvicorn==0.34.0 \
    google-cloud-aiplatform>=1.73.0 \
    pydantic>=2.0.0

COPY src src/

ENV PYTHONPATH=/app/src

ENTRYPOINT ["sh", "-c", "python -m uvicorn rice_cnn_classifier.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
