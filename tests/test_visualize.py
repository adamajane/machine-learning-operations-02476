import numpy as np
import torch
import matplotlib.figure
from rice_cnn_classifier.visualize import (
    create_confusion_matrix_figure,
    create_sample_predictions_grid
)

def test_create_confusion_matrix_figure():
    """Verify confusion matrix figure generation."""
    cm = np.array([[10, 2], [1, 12]])
    classes = ["Arborio", "Basmati"]
    fig = create_confusion_matrix_figure(cm, classes)
    
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) > 0

def test_create_sample_predictions_grid():
    """Verify the denormalization and grid logic."""
    # Create 4 dummy image tensors (3, 224, 224)
    images = [torch.randn(3, 224, 224) for _ in range(4)]
    labels = [0, 1, 0, 1]
    preds = [0, 0, 0, 1]
    confs = [0.9, 0.4, 0.8, 0.9]
    classes = ["Arborio", "Basmati"]
    
    fig = create_sample_predictions_grid(images, labels, preds, confs, classes, num_samples=4)
    
    assert isinstance(fig, matplotlib.figure.Figure)
    # Check that we have the expected number of subplots
    assert len(fig.axes) >= 4