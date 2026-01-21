from pathlib import Path
from typing import Callable, Literal
import torch
import typer
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RiceDataset(Dataset):
    """Rice image classification dataset.

    Expects data organized as:
        data_path/
            class_1/
                image1.jpg
                image2.jpg
            class_2/
                ...

    Args:
        data_path: Path to the dataset directory containing class folders
        transform: Optional transform to apply to images
        split: Which split to load ('train', 'val', 'test')
    """

    CLASSES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

    def __init__(
        self,
        data_path: Path,
        transform: Callable | None = None,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split

        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Scan directory and build list of image paths and labels."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

        for class_idx, class_name in enumerate(self.CLASSES):
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                continue

            for img_path in class_dir.glob("*.jpg"):
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_path}")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return a given sample from the dataset.

        Args:
            index: Index of the sample to return

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder.

        This can include:
        - Resizing images to a standard size
        - Converting formats
        - Splitting into train/val/test sets
        - Data quality checks

        Args:
            output_folder: Path where preprocessed data should be saved
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"Preprocessing {len(self.image_paths)} images...")
        print(f"Found {len(set(self.labels))} classes")

        for class_name in self.CLASSES:
            (output_folder / class_name).mkdir(exist_ok=True)

        resize_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
            ]
        )

        for img_path, label in zip(self.image_paths, self.labels):
            try:
                image = Image.open(img_path).convert("RGB")
                image = resize_transform(image)

                class_name = self.CLASSES[label]
                output_path = output_folder / class_name / img_path.name
                image.save(output_path, quality=95)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        print(f"Preprocessing complete. Saved to {output_folder}")


def get_transforms(split: Literal["train", "val", "test"]) -> transforms.Compose:
    """Get appropriate transforms for each dataset split.

    Args:
        split: Which split ('train', 'val', 'test')

    Returns:
        Composed transforms
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def preprocess(data_path: Path, output_folder: Path) -> None:
    """Preprocess raw rice images.

    Args:
        data_path: Path to raw data directory
        output_folder: Path where processed data should be saved
    """
    print("Preprocessing data...")
    dataset = RiceDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
