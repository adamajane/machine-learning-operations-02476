from fastapi.testclient import TestClient
from rice_cnn_classifier.api import app

client = TestClient(app)


# Ensures the API is running
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_train_missing_project_id():
    # Sending an empty request should trigger a 400 error if VERTEX_PROJECT_ID isn't set
    response = client.post("/train", json={})
    assert response.status_code == 400
    assert "Missing setting" in response.text


def test_load_config():
    # Test internal helper logic
    from rice_cnn_classifier.api import _load_config

    config = _load_config("configs/gpu.yaml", {"WANDB_API_KEY": "fake_key"})
    assert isinstance(config, dict)
