import torch
from PIL import Image
from rice_cnn_classifier.data import RiceDataset, get_transforms


def test_dataset_loading(tmp_path):
    """Test that RiceDataset correctly loads and transforms dummy data."""
    # Creating a dummy directory structure
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()

    for rice_class in ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]:
        class_path = data_dir / rice_class
        class_path.mkdir()

        # Create 5 dummy images per class to satisfy the 80/10/10 split logic
        for i in range(5):
            img = Image.new("RGB", (100, 100), color="white")
            # Ensure the extension is .jpg (lowercase) as required by our data.py
            img.save(class_path / f"img_{i}.jpg")

    # 2. Initialize the dataset
    transform = get_transforms(split="train")
    dataset = RiceDataset(data_path=data_dir, transform=transform, split="train")

    # 3. Assertions
    assert len(dataset) > 0
    img_tensor, label = dataset[0]

    # Check that it turned into a torch tensor with the correct normalized shape
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
