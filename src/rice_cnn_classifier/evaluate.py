"""Evaluation script for rice classification model.

Loads a trained model, evaluates on test set, and logs metrics
and visualizations to WandB.

Usage:
    uv run python -m rice_cnn_classifier.evaluate
    uv run python -m rice_cnn_classifier.evaluate --model-path models/model.pth
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import typer
import wandb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from rice_cnn_classifier import visualize
from rice_cnn_classifier.data import RiceDataset, get_transforms
from rice_cnn_classifier.model import RiceCNN

app = typer.Typer()


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_path: Path, device: torch.device) -> RiceCNN:
    """Load trained model from checkpoint.

    Args:
        model_path: Path to model state_dict
        device: Device to load model to

    Returns:
        Loaded RiceCNN model in eval mode
    """
    model = RiceCNN(num_classes=len(RiceDataset.CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: RiceCNN,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Run inference on dataset and collect predictions.

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Compute device

    Returns:
        Dict containing:
            - all_preds: List of predicted class indices
            - all_labels: List of true class indices
            - all_probs: List of prediction probabilities
            - all_images: List of image tensors (for visualization)
    """
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[list[float]] = []
    all_images: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

            # Store images for visualization (limit to save memory)
            if len(all_images) < 100:
                all_images.extend([img.cpu() for img in images])

    return {
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_probs": all_probs,
        "all_images": all_images[:100],
    }


def compute_metrics(
    true_labels: list[int],
    predictions: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Compute classification metrics.

    Args:
        true_labels: Ground truth labels
        predictions: Model predictions
        class_names: List of class names

    Returns:
        Dict containing overall and per-class metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions)

    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, labels=range(len(class_names))
    )

    # Macro-averaged metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="macro"
    )

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=range(len(class_names)))

    # Per-class accuracy
    per_class_accuracy = {}
    for i, cls in enumerate(class_names):
        class_mask = np.array(true_labels) == i
        if class_mask.sum() > 0:
            per_class_accuracy[cls] = float((np.array(predictions)[class_mask] == i).mean())
        else:
            per_class_accuracy[cls] = 0.0

    # Build per-class metrics dict
    per_class_metrics = {
        "precision": {cls: float(precision[i]) for i, cls in enumerate(class_names)},
        "recall": {cls: float(recall[i]) for i, cls in enumerate(class_names)},
        "f1": {cls: float(f1[i]) for i, cls in enumerate(class_names)},
    }

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics,
        "per_class_accuracy": per_class_accuracy,
        "support": {cls: int(support[i]) for i, cls in enumerate(class_names)},
    }


def log_to_wandb(
    metrics: dict[str, Any],
    inference_results: dict[str, Any],
    class_names: list[str],
) -> None:
    """Log all metrics and visualizations to WandB.

    Args:
        metrics: Computed metrics dict
        inference_results: Dict with predictions, labels, images
        class_names: List of class names
    """
    # 1. Log scalar metrics
    wandb.log(
        {
            "test/accuracy": metrics["accuracy"],
            "test/macro_precision": metrics["macro_precision"],
            "test/macro_recall": metrics["macro_recall"],
            "test/macro_f1": metrics["macro_f1"],
        }
    )

    # Log per-class metrics
    for metric_name in ["precision", "recall", "f1"]:
        for cls, value in metrics["per_class_metrics"][metric_name].items():
            wandb.log({f"test/per_class/{cls}/{metric_name}": value})

    # 2. Log confusion matrix visualization
    cm_fig = visualize.create_confusion_matrix_figure(
        confusion_matrix=metrics["confusion_matrix"],
        class_names=class_names,
        title="Test Set Confusion Matrix",
    )
    wandb.log({"test/confusion_matrix": wandb.Image(cm_fig)})

    # 3. Log per-class metrics bar charts
    metrics_fig = visualize.create_per_class_metrics_figure(
        metrics_dict=metrics["per_class_metrics"],
        class_names=class_names,
    )
    wandb.log({"test/per_class_metrics": wandb.Image(metrics_fig)})

    # 4. Log per-class accuracy comparison
    acc_fig = visualize.create_accuracy_comparison_figure(
        class_names=class_names,
        accuracies=metrics["per_class_accuracy"],
        overall_accuracy=metrics["accuracy"],
    )
    wandb.log({"test/accuracy_comparison": wandb.Image(acc_fig)})

    # 5. Log sample predictions grid
    num_images = len(inference_results["all_images"])
    confidences = [inference_results["all_probs"][i][inference_results["all_preds"][i]] for i in range(num_images)]

    samples_fig = visualize.create_sample_predictions_grid(
        images=inference_results["all_images"],
        true_labels=inference_results["all_labels"][:num_images],
        predicted_labels=inference_results["all_preds"][:num_images],
        confidences=confidences,
        class_names=class_names,
        num_samples=16,
    )
    wandb.log({"test/sample_predictions": wandb.Image(samples_fig)})

    # 6. Log WandB native confusion matrix (interactive)
    wandb.log(
        {
            "test/confusion_matrix_interactive": wandb.plot.confusion_matrix(
                y_true=inference_results["all_labels"],
                preds=inference_results["all_preds"],
                class_names=class_names,
            )
        }
    )

    # 7. Log class distribution table
    table = wandb.Table(
        columns=["Class", "Support", "Precision", "Recall", "F1", "Accuracy"],
        data=[
            [
                cls,
                metrics["support"][cls],
                f"{metrics['per_class_metrics']['precision'][cls]:.3f}",
                f"{metrics['per_class_metrics']['recall'][cls]:.3f}",
                f"{metrics['per_class_metrics']['f1'][cls]:.3f}",
                f"{metrics['per_class_accuracy'][cls]:.3f}",
            ]
            for cls in class_names
        ],
    )
    wandb.log({"test/metrics_table": table})

    # Close all matplotlib figures
    plt.close("all")


@app.command()
def evaluate(
    model_path: str = typer.Option("models/model.pth", help="Path to trained model"),
    data_path: str = typer.Option("data/processed", help="Path to processed data"),
    batch_size: int = typer.Option(32, help="Batch size for evaluation"),
    wandb_project: str = typer.Option("rice_cnn_classifier", help="WandB project name"),
    wandb_run_name: str | None = typer.Option(None, help="WandB run name (optional)"),
    disable_wandb: bool = typer.Option(False, help="Disable WandB logging for local testing"),
) -> None:
    """Evaluate trained rice classifier on test set.

    Loads model, runs inference on test data, computes metrics,
    and logs everything to WandB.
    """
    print("Starting evaluation...")

    # Setup
    device = get_device()
    print(f"Using device: {device}")

    model_path_obj = Path(model_path)
    data_path_obj = Path(data_path)
    class_names = list(RiceDataset.CLASSES)

    # Validate paths
    if not model_path_obj.exists():
        print(f"Error: Model not found at {model_path_obj}")
        raise typer.Exit(code=1)

    if not data_path_obj.exists():
        print(f"Error: Data path not found at {data_path_obj}")
        raise typer.Exit(code=1)

    # Initialize WandB
    if not disable_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or "evaluation",
            job_type="evaluation",
            config={
                "model_path": str(model_path_obj),
                "data_path": str(data_path_obj),
                "batch_size": batch_size,
                "classes": class_names,
            },
        )

    # Load model
    print(f"Loading model from {model_path_obj}...")
    model = load_model(model_path_obj, device)

    # Load test dataset
    print(f"Loading test data from {data_path_obj}...")
    test_dataset = RiceDataset(
        data_path=data_path_obj,
        split="test",
        transform=get_transforms("test"),
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Test set size: {len(test_dataset)} images")

    # Run inference
    print("Running inference...")
    inference_results = run_inference(model, test_loader, device)

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(
        true_labels=inference_results["all_labels"],
        predictions=inference_results["all_preds"],
        class_names=class_names,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
    print(f"Macro Precision:  {metrics['macro_precision']:.2%}")
    print(f"Macro Recall:     {metrics['macro_recall']:.2%}")
    print(f"Macro F1:         {metrics['macro_f1']:.2%}")
    print("\nPer-Class F1 Scores:")
    for cls in class_names:
        f1 = metrics["per_class_metrics"]["f1"][cls]
        print(f"  {cls}: {f1:.2%}")
    print("=" * 50)

    # Log to WandB
    if not disable_wandb:
        print("\nLogging to WandB...")
        log_to_wandb(metrics, inference_results, class_names)
        wandb.finish()
        print("\nEvaluation complete. Results logged to WandB.")
    else:
        print("\nEvaluation complete. WandB logging disabled.")


if __name__ == "__main__":
    app()
