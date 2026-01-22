import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import torch
import requests
from rice_cnn_classifier.frontend import get_inference_transform, predict, trigger_training, download_model_from_gcs


# 1. Test Image Transformations
def test_inference_transform():
    """Verify image transformation produces correct tensor shapes."""
    transform = get_inference_transform()
    dummy_img = Image.new("RGB", (300, 300), color="white")
    tensor = transform(dummy_img)
    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


# 2. Test GCS Logic (Success and Failure branches)
@patch("google.cloud.storage.Client")
def test_download_model_from_gcs_logic(mock_storage_client):
    """Test GCS download success and invalid path handling."""
    # Mock GCS structure
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    # Test Valid Path
    assert download_model_from_gcs("gs://bucket/model.pth", "/tmp/model.pth") is True

    # Test Invalid Path (covers the 'if not gcs_path.startswith' branch)
    assert download_model_from_gcs("http://wrong.com", "/tmp/model.pth") is False


# 3. Test Prediction (Success and Missing Input)
@patch("rice_cnn_classifier.frontend.load_model")
def test_predict_functionality(mock_load):
    """Test prediction with images and handling of None input."""
    # Setup mock model
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(1, 5)
    mock_load.return_value = (mock_model, None)

    # Success case
    dummy_img = Image.new("RGB", (224, 224), color="red")
    result = predict(dummy_img)
    assert isinstance(result, dict)
    assert len(result) == 5

    # Failure case: No image (covers 'if image is None' branch)
    assert predict(None) == "Please upload an image first."


# 4. Test API Trigger (Success, Failure, and Missing URL)
@patch("requests.post")
def test_trigger_training_scenarios(mock_post):
    """Test API triggers for success, 500 errors, and missing configuration."""
    from rice_cnn_classifier import frontend

    # Case A: Success
    mock_post.return_value.ok = True
    mock_post.return_value.json.return_value = {"job_name": "job-123"}
    frontend.TRAINING_API_URL = "http://fake-api.com"
    assert "job-123" in trigger_training("test-job")

    # Case B: API Error (covers 'else' branch for response.ok)
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = 500
    mock_post.return_value.text = "Internal Server Error"
    assert "500" in trigger_training("test-job")

    # Case C: Missing URL (covers 'if not TRAINING_API_URL' branch)
    frontend.TRAINING_API_URL = ""
    assert "not configured" in trigger_training()
