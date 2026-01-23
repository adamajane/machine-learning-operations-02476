"""Gradio frontend for Rice CNN Classifier.

Provides two tabs:
1. Inference: Upload rice image, get classification with probabilities
2. Training: Trigger training via Cloud Run API
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import requests
import torch
import torch.nn.functional as F
from google.cloud import storage
from PIL import Image
from torchvision import transforms

from rice_cnn_classifier.data import RiceDataset
from rice_cnn_classifier.model import RiceCNN

MODEL_GCS_PATH = os.getenv("MODEL_GCS_PATH", "gs://rice_image_dataset/models/model.pth")
TRAINING_API_URL = os.getenv("TRAINING_API_URL", "")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/tmp/model.pth")

CLASS_NAMES = RiceDataset.CLASSES

_model: RiceCNN | None = None
_model_load_error: str | None = None


def download_model_from_gcs(gcs_path: str, local_path: str) -> bool:
    """Download model from GCS to local path.

    Args:
        gcs_path: GCS URI (gs://bucket/path/to/model.pth)
        local_path: Local filesystem path

    Returns:
        True if successful, False otherwise
    """
    try:
        if not gcs_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {gcs_path}")

        parts = gcs_path[5:].split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ""

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)
        return True
    except Exception as e:
        print(f"Failed to download model from GCS: {e}")
        return False


def load_model() -> tuple[RiceCNN | None, str | None]:
    """Load model from GCS or local cache.

    Returns:
        Tuple of (model, error_message). Model is None if loading failed.
    """
    local_path = Path(LOCAL_MODEL_PATH)

    if not local_path.exists():
        if not download_model_from_gcs(MODEL_GCS_PATH, str(local_path)):
            return None, f"Model not found at {MODEL_GCS_PATH}. Please train a model first."

    try:
        device = torch.device("cpu")
        model = RiceCNN(num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(str(local_path), map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


def get_inference_transform() -> transforms.Compose:
    """Get the transform for inference (same as evaluation)."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict(image: Image.Image | None) -> dict[str, float] | str:
    """Run inference on uploaded image.

    Args:
        image: PIL Image from Gradio upload

    Returns:
        Dictionary of {class_name: probability} or error string
    """
    global _model, _model_load_error

    if image is None:
        return "Please upload an image first."

    if _model is None and _model_load_error is None:
        _model, _model_load_error = load_model()

    if _model is None:
        return _model_load_error or "Model not available"

    try:
        transform = get_inference_transform()
        image_rgb = image.convert("RGB")
        tensor = transform(image_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = _model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            probs = probabilities[0].tolist()

        return {CLASS_NAMES[i]: probs[i] for i in range(len(CLASS_NAMES))}

    except Exception as e:
        return f"Prediction failed: {e}"


def trigger_training(display_name: str | None = None) -> str:
    """Trigger training job via the training API.

    Args:
        display_name: Optional job name

    Returns:
        Status message with job name or error
    """
    if not TRAINING_API_URL:
        return "Error: TRAINING_API_URL not configured. Set the environment variable to your Cloud Run API URL."

    try:
        url = f"{TRAINING_API_URL.rstrip('/')}/train"
        payload = {}
        if display_name:
            payload["display_name"] = display_name

        response = requests.post(url, json=payload, timeout=30)

        if response.ok:
            data = response.json()
            job_name = data.get("job_name", "Unknown")
            return f"Training job started!\n\nJob name: {job_name}\n\nCheck WandB for training progress."
        else:
            return f"Failed to start training: {response.status_code} - {response.text}"

    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to training API: {e}"


def create_app() -> gr.Blocks:
    """Create the Gradio application with two tabs."""
    with gr.Blocks(title="Rice CNN Classifier", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Rice Grain Classifier")
        gr.Markdown("Classify rice varieties: **Arborio**, **Basmati**, **Ipsala**, **Jasmine**, **Karacadag**")

        with gr.Tabs():
            with gr.TabItem("Classify Rice"):
                gr.Markdown("Upload an image of rice grains to identify the variety.")

                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Rice Image",
                            sources=["upload", "clipboard"],
                        )
                        classify_btn = gr.Button("Classify", variant="primary")

                    with gr.Column():
                        output_label = gr.Label(
                            label="Classification Results",
                            num_top_classes=5,
                        )

                classify_btn.click(
                    fn=predict,
                    inputs=[image_input],
                    outputs=[output_label],
                )

                gr.Markdown("### How it works")
                gr.Markdown(
                    "1. Upload an image of rice grains\n"
                    "2. Click **Classify** to get predictions\n"
                    "3. Results show probability for each rice variety"
                )

            with gr.TabItem("Train Model"):
                gr.Markdown("Trigger a new training job on Vertex AI with GPU acceleration.")

                job_name_input = gr.Textbox(
                    label="Job Name (optional)",
                    placeholder="Leave empty for auto-generated name",
                )
                train_btn = gr.Button("Start Training", variant="primary")
                training_output = gr.Textbox(
                    label="Training Status",
                    lines=5,
                    interactive=False,
                )

                train_btn.click(
                    fn=trigger_training,
                    inputs=[job_name_input],
                    outputs=[training_output],
                )

                gr.Markdown("### Training Details")
                gr.Markdown(
                    "- Runs on Vertex AI with NVIDIA T4 GPU\n"
                    "- Training progress logged to WandB\n"
                    "- Model saved to GCS bucket after training"
                )

        gr.Markdown("---")
        gr.Markdown("*Built for DTU course 02476 Machine Learning Operations*")

    return app


if __name__ == "__main__":
    print("Loading model...")
    _model, _model_load_error = load_model()
    if _model:
        print("Model loaded successfully")
    else:
        print(f"Model not loaded: {_model_load_error}")

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
    )
