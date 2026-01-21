import sys
import typer
from pathlib import Path

# Ensure src/ is importable when running tests directly
SRC = (Path(__file__).resolve().parents[1] / "src").resolve()
sys.path.insert(0, str(SRC))

import pytest
import torch
from PIL import Image

from rice_cnn_classifier.data import RiceDataset, get_transforms


def _make_jpg(path: Path, size=(32, 32), color=(255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, format="JPEG")


def _make_dummy_dataset(root: Path) -> Path:
    """
    Creates:
      root/Arborio/img1.jpg
      root/Basmati/img2.jpg
    """
    _make_jpg(root / "Arborio" / "a1.jpg")
    _make_jpg(root / "Basmati" / "b1.jpg")
    return root


def test_dataset_raises_if_path_missing(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        RiceDataset(missing)


def test_dataset_raises_if_no_images_found(tmp_path: Path):
    # Create folders but no jpg files
    (tmp_path / "Arborio").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        RiceDataset(tmp_path)


def test_dataset_len_and_labels(tmp_path: Path):
    data_root = _make_dummy_dataset(tmp_path)
    ds = RiceDataset(data_root)

    assert len(ds) == 2
    # Arborio is class index 0, Basmati is class index 1 per CLASSES list
    assert set(ds.labels) == {0, 1}


def test_dataset_getitem_returns_image_and_label(tmp_path: Path):
    data_root = _make_dummy_dataset(tmp_path)

    # Use a simple transform to ensure tensor output
    transform = get_transforms("val")
    ds = RiceDataset(data_root, transform=transform)

    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, int)
    assert x.shape == (3, 224, 224)
    assert torch.isfinite(x).all()
    assert y in range(len(ds.CLASSES))


def test_get_transforms_train_vs_val_output_shape():
    img = Image.new("RGB", (50, 50), color=(0, 255, 0))

    train_t = get_transforms("train")
    val_t = get_transforms("val")

    x_train = train_t(img)
    x_val = val_t(img)

    assert isinstance(x_train, torch.Tensor)
    assert isinstance(x_val, torch.Tensor)

    assert x_train.shape == (3, 224, 224)
    assert x_val.shape == (3, 224, 224)

    assert torch.isfinite(x_train).all()
    assert torch.isfinite(x_val).all()


def test_preprocess_creates_resized_images(tmp_path: Path):
    data_root = _make_dummy_dataset(tmp_path / "raw")
    out_root = tmp_path / "processed"

    ds = RiceDataset(data_root)
    ds.preprocess(out_root)

    # Check that output images exist in expected class folders
    arborio_out = list((out_root / "Arborio").glob("*.jpg"))
    basmati_out = list((out_root / "Basmati").glob("*.jpg"))

    assert len(arborio_out) == 1
    assert len(basmati_out) == 1

    # Verify size is 224x224
    img = Image.open(arborio_out[0])
    assert img.size == (224, 224)

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
