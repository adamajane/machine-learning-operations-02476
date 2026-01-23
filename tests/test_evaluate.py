import warnings

import pytest
import torch
from PIL import Image

from rice_cnn_classifier.evaluate import evaluate
from rice_cnn_classifier.model import RiceCNN


@pytest.mark.filterwarnings("ignore")
def test_evaluation_pipeline(tmp_path):
    """
    Test the evaluation pipeline to ensure it loads a model,
    processes the test split, and computes metrics correctly.
    """
    warnings.filterwarnings("ignore")

    # Setup temporary directories
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Create 10 images per class (enough for split logic)
    categories = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
    for rice_class in categories:
        class_path = data_dir / rice_class
        class_path.mkdir(parents=True)
        for i in range(10):
            img = Image.new("RGB", (224, 224), color="white")
            img.save(class_path / f"img_{i}.jpg")

    # Create and save a model
    model = RiceCNN(num_classes=5)
    model.eval()  # important to avoid torch warnings

    model_path = model_dir / "test_model.pth"
    torch.save(model.state_dict(), model_path)

    # Run evaluation
    evaluate(
        model_path=str(model_path),
        data_path=str(data_dir),
        batch_size=2,
        disable_wandb=True,
    )
