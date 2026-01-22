import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import torch
from rice_cnn_classifier.frontend import get_inference_transform, predict, trigger_training

def test_inference_transform():
    """Verify that the image transformation produces the correct tensor shape."""
    transform = get_inference_transform()
    # Create a dummy RGB image
    dummy_img = Image.new("RGB", (300, 300), color="white")
    tensor = transform(dummy_img)
    
    # Check shape: (Channels, Height, Width)
    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)

@patch("rice_cnn_classifier.frontend.load_model")
def test_predict_flow(mock_load):
    """Test the prediction logic with a mocked model."""
    # 1. Setup a fake model that returns a dummy tensor
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(1, 5) 
    mock_load.return_value = (mock_model, None)
    
    dummy_img = Image.new("RGB", (224, 224), color="red")
    
    # 2. Run prediction
    result = predict(dummy_img)
    
    # 3. Verify output is a dictionary of class probabilities
    assert isinstance(result, dict)
    assert len(result) == 5
    assert "Basmati" in result

@patch("requests.post")
def test_trigger_training_success(mock_post):
    """Test that the training trigger correctly handles a successful API response."""
    # Mock a successful FastAPI response
    mock_post.return_value.ok = True
    mock_post.return_value.json.return_value = {"job_name": "test-job-123"}
    
    with patch("os.getenv", return_value="http://fake-api.com"):
        # We need to manually set the URL for the test scope
        from rice_cnn_classifier import frontend
        frontend.TRAINING_API_URL = "http://fake-api.com"
        
        response = trigger_training("test-display-name")
        assert "Training job started" in response
        assert "test-job-123" in response