import sys
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "src").resolve()
sys.path.insert(0, str(SRC))

# TEMP DEBUG (delete later)
print("Using SRC path:", SRC)
print("sys.path[0]:", sys.path[0])
print("src exists:", SRC.exists())
print("package exists:", (SRC / "rice_cnn_classifier").exists())

import torch
import pytest
from rice_cnn_classifier.model import RiceCNN




def test_model_is_torch_module():
    model = RiceCNN(num_classes=5)
    assert isinstance(model, torch.nn.Module)


def test_model_forward_output_shape_224():
    """With 224x224 input, should return (batch_size, num_classes)."""
    model = RiceCNN(num_classes=5)
    model.eval()

    x = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (4, 5)


def test_model_forward_outputs_are_finite():
    """Model outputs should not contain NaNs or Infs (for valid input size)."""
    model = RiceCNN(num_classes=5)
    model.eval()

    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    assert torch.isfinite(y).all()


def test_model_backward_pass_computes_gradients():
    """Backward pass should compute gradients without crashing (for valid input size)."""
    model = RiceCNN(num_classes=5)
    model.train()

    x = torch.rand(2, 3, 224, 224)
    y_true = torch.tensor([0, 1])

    criterion = torch.nn.CrossEntropyLoss()
    y_pred = model(x)
    loss = criterion(y_pred, y_true)

    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_model_rejects_wrong_input_size_64():
    """Your current model is hard-coded for 224x224; 64x64 should raise."""
    model = RiceCNN(num_classes=5)
    model.eval()

    x = torch.rand(1, 3, 64, 64)
    with pytest.raises(RuntimeError):
        _ = model(x)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))