import pytest
from PIL import Image
from rice_cnn_classifier.train import train


def test_train_smoke_run(tmp_path):
    """A 'smoke test' to ensure the training loop runs without crashing."""

    # Setup paths using pytest's tmp_path fixture
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"

    # Create 10 images per class to satisfy 80/10/10 split logic
    categories = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
    for rice_class in categories:
        class_path = data_dir / rice_class
        class_path.mkdir(parents=True)
        for i in range(10):
            img = Image.new("RGB", (10, 10), color="white")
            # Use lowercase .jpg
            img.save(class_path / f"img_{i}.jpg")

    # Run a short training session
    try:
        train(
            epochs=1,
            batch_size=2,
            data_path=str(data_dir),
            model_dir=str(model_dir),
            disable_wandb=True,  # Essential: keeps tests offline
        )
    except Exception as e:
        pytest.fail(f"Training loop failed with error: {e}")

    # Verify output
    assert (model_dir / "model.pth").exists()
