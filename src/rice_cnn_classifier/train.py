from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rice_cnn_classifier.model import RiceCNN
from rice_cnn_classifier.data import RiceDataset, get_transforms

def train():
    # Hyperparameters
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    data_path = Path("data/processed")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Datasets
    try:
        train_dataset = RiceDataset(data_path=data_path, split="train", transform=get_transforms("train"))
        val_dataset = RiceDataset(data_path=data_path, split="val", transform=get_transforms("val"))
    except FileNotFoundError:
        print(f"Error: Data directory '{data_path}' not found. Please run 'dvc repro' first.")
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
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), models_dir / "model.pth")
            print(f"Saved best model with Val Acc: {val_acc:.2f}%")

    print("Training complete.")

if __name__ == "__main__":
    train()
