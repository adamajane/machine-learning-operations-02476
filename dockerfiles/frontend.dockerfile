FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

# Install PyTorch CPU version (smaller image size)
RUN uv pip install --system \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system \
    torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN uv pip install --system \
    gradio>=5.0.0 \
    google-cloud-storage>=2.14.0 \
    pillow>=10.0.0 \
    requests>=2.32.0

COPY src src/

ENV PYTHONPATH=/app/src
ENV PORT=7860
ENV MODEL_GCS_PATH=gs://rice_image_dataset/models/model.pth
ENV LOCAL_MODEL_PATH=/tmp/model.pth

EXPOSE 7860

ENTRYPOINT ["sh", "-c", "python -m rice_cnn_classifier.frontend"]
