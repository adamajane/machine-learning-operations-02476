from torch import nn
import torch

class RiceCNN(nn.Module):
    """Simple CNN for rice classification."""
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

if __name__ == "__main__":
    model = RiceCNN(num_classes=5)
    x = torch.rand(1, 3, 64, 64)
    print(f"Input shape: {x.shape}")
    print(f"Output shape of model: {model(x).shape}")
