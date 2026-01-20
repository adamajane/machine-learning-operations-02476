from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import typer
import wandb
from rice_cnn_classifier.model import RiceCNN
from rice_cnn_classifier.data import RiceDataset, get_transforms


def train(
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    data_path: str = "data/processed",
    model_dir: str = "models",
    wandb_project: str = "rice_cnn_classifier",
    wandb_run_name: str | None = None,
    disable_wandb: bool = False,
):
    # Hyperparameters
    print("SCRIPT STARTED: Rice Classifier Training")
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Initialize WandB
    if not disable_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "architecture": "RiceCNN",
                "num_classes": len(RiceDataset.CLASSES),
                "classes": list(RiceDataset.CLASSES),
                "device": str(device),
            },
        )

    # Paths
    print(f"Received data_path: {data_path}")
    print(f"Received model_dir: {model_dir}")

    data_path = Path(data_path)
    # Handle GCS path specifically if it comes as a string
    if str(data_path).startswith("/gcs"):
        print("Detected GCS path!")

    models_dir = Path(model_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    try:
        print("Analyzing dataset files... (This may take a few minutes on GCS)")
        train_dataset = RiceDataset(
            data_path=data_path, split="train", transform=get_transforms("train")
        )
        val_dataset = RiceDataset(
            data_path=data_path, split="val", transform=get_transforms("val")
        )
    except FileNotFoundError:
        print(
            f"Error: Data directory '{data_path}' not found. Please run 'dvc repro' first."
        )
        return

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = RiceCNN(num_classes=len(RiceDataset.CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
        )

        # Log metrics to WandB
        if not disable_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/accuracy": train_acc,
                    "val/accuracy": val_acc,
                }
            )

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = models_dir / "model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with Val Acc: {val_acc:.2f}%")

            # Log model as WandB artifact
            if not disable_wandb:
                artifact = wandb.Artifact(
                    name="rice_cnn_model",
                    type="model",
                    description=f"Best model with val_acc={val_acc:.2f}%",
                    metadata={"val_accuracy": val_acc, "epoch": epoch + 1},
                )
                artifact.add_file(str(model_path))
                wandb.log_artifact(artifact)

    # Finish WandB run
    if not disable_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    typer.run(train)
