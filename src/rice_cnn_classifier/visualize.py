"""Visualization utilities for rice classification results.

All functions return matplotlib Figure objects for WandB logging.
No local file saving - figures are logged directly to WandB.
"""

from typing import Sequence

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def create_confusion_matrix_figure(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "Blues",
) -> matplotlib.figure.Figure:
    """Create a confusion matrix heatmap figure.

    Args:
        confusion_matrix: 2D numpy array of shape (n_classes, n_classes)
        class_names: List of class names for axis labels
        title: Plot title
        figsize: Figure size tuple
        cmap: Colormap name

    Returns:
        matplotlib Figure object ready for WandB logging
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    plt.tight_layout()
    return fig


def create_per_class_metrics_figure(
    metrics_dict: dict[str, dict[str, float]],
    class_names: Sequence[str],
    figsize: tuple[int, int] = (12, 6),
) -> matplotlib.figure.Figure:
    """Create bar charts for per-class precision, recall, F1.

    Args:
        metrics_dict: Dict with keys 'precision', 'recall', 'f1'
                      each containing {class_name: value}
        class_names: List of class names
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    metric_names = ["precision", "recall", "f1"]
    colors = ["#2ecc71", "#3498db", "#9b59b6"]

    for ax, metric_name, color in zip(axes, metric_names, colors):
        values = [metrics_dict[metric_name].get(cls, 0) for cls in class_names]

        bars = ax.bar(class_names, values, color=color, alpha=0.8)
        ax.set_ylabel(metric_name.capitalize())
        ax.set_xlabel("Class")
        ax.set_title(f"Per-Class {metric_name.capitalize()}")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    return fig


def create_sample_predictions_grid(
    images: list[torch.Tensor],
    true_labels: list[int],
    predicted_labels: list[int],
    confidences: list[float],
    class_names: Sequence[str],
    num_samples: int = 16,
    figsize: tuple[int, int] = (16, 16),
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> matplotlib.figure.Figure:
    """Create a grid showing sample predictions with images.

    Args:
        images: List of image tensors (C, H, W) normalized
        true_labels: List of true class indices
        predicted_labels: List of predicted class indices
        confidences: List of prediction confidence values (0-1)
        class_names: List of class names
        num_samples: Number of samples to display
        figsize: Figure size
        mean: ImageNet normalization mean (for denormalization)
        std: ImageNet normalization std (for denormalization)

    Returns:
        matplotlib Figure object
    """
    # Limit to available samples
    n = min(num_samples, len(images))
    grid_size = int(np.ceil(np.sqrt(n)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    # Denormalization tensors
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)

    for idx, ax in enumerate(axes):
        if idx < n:
            # Denormalize image
            img_tensor = images[idx].cpu()
            img_tensor = img_tensor * std_tensor + mean_tensor
            img_tensor = torch.clamp(img_tensor, 0, 1)

            # Convert to numpy for display (H, W, C)
            img_np = img_tensor.permute(1, 2, 0).numpy()

            ax.imshow(img_np)

            true_name = class_names[true_labels[idx]]
            pred_name = class_names[predicted_labels[idx]]
            conf = confidences[idx]

            # Color title based on correctness
            is_correct = true_labels[idx] == predicted_labels[idx]
            color = "green" if is_correct else "red"

            ax.set_title(
                f"True: {true_name}\nPred: {pred_name} ({conf:.1%})",
                fontsize=10,
                color=color,
            )

        ax.axis("off")

    plt.suptitle("Sample Predictions (Green=Correct, Red=Incorrect)", fontsize=14)
    plt.tight_layout()
    return fig


def create_accuracy_comparison_figure(
    class_names: Sequence[str],
    accuracies: dict[str, float],
    overall_accuracy: float,
    figsize: tuple[int, int] = (10, 6),
) -> matplotlib.figure.Figure:
    """Create a horizontal bar chart comparing per-class accuracies.

    Args:
        class_names: List of class names
        accuracies: Dict mapping class name to accuracy (0-1)
        overall_accuracy: Overall model accuracy (0-1)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(class_names))
    values = [accuracies.get(cls, 0) * 100 for cls in class_names]

    colors = ["#3498db" if v >= overall_accuracy * 100 else "#e74c3c" for v in values]

    bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
    ax.axvline(
        x=overall_accuracy * 100,
        color="black",
        linestyle="--",
        label=f"Overall: {overall_accuracy:.1%}",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy (Blue >= Overall, Red < Overall)")
    ax.set_xlim(0, 100)
    ax.legend()

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    return fig
