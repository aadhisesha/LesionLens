"""Automated tests for dataset pairing, preprocessing, splitting, and DataLoader.

These tests use synthetic images and masks in temporary directories, allowing
complete test execution and validation without requiring the full ~850MB ISIC dataset.
Integration tests on the full ISIC dataset are marked with @pytest.mark.integration.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import ISICDataset, get_dataloader, smoke_test_dataloader
from data.preprocessing import (
    normalize_identifier,
    pair_images_and_masks,
    preprocess_image,
    preprocess_mask,
    process_and_save_pair,
)
from scripts.prepare_dataset import create_splits


def test_identifier_normalization():
    """Verify identifier normalization across diverse file naming conventions."""
    assert normalize_identifier("ISIC_0000000.jpg") == "ISIC_0000000"
    assert normalize_identifier("ISIC_0000000.jpeg") == "ISIC_0000000"
    assert normalize_identifier("ISIC_0000000_Segmentation.png") == "ISIC_0000000"
    assert normalize_identifier("ISIC_0000042_segmentation.png") == "ISIC_0000042"
    assert normalize_identifier("path/to/nested/ISIC_0000999_Segmentation.png") == "ISIC_0000999"

    with pytest.raises(ValueError, match="could not be normalized"):
        normalize_identifier("invalid_image_name.jpg")

    with pytest.raises(ValueError, match="could not be normalized"):
        normalize_identifier("ISIC_123.jpg")  # Invalid digits count


def test_pairing_images_and_masks(tmp_path):
    """Verify pairing correctly correlates image and mask files by canonical ID."""
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir()
    msk_dir.mkdir()

    # Create synthetic files
    img1 = img_dir / "ISIC_0000001.jpg"
    img2 = img_dir / "ISIC_0000002.jpg"
    msk1 = msk_dir / "ISIC_0000001_Segmentation.png"
    msk2 = msk_dir / "ISIC_0000002_Segmentation.png"

    for f in (img1, img2, msk1, msk2):
        f.touch()

    # Reverse order to verify sorting independence
    pairs = pair_images_and_masks([img2, img1], [msk1, msk2])

    assert len(pairs) == 2
    assert pairs[0][0] == "ISIC_0000001"
    assert pairs[0][1] == img1
    assert pairs[0][2] == msk1
    assert pairs[1][0] == "ISIC_0000002"
    assert pairs[1][1] == img2
    assert pairs[1][2] == msk2


def test_pairing_unmatched_error(tmp_path):
    """Verify ValueError is raised when images or masks are missing."""
    img1 = tmp_path / "ISIC_0000001.jpg"
    msk2 = tmp_path / "ISIC_0000002_Segmentation.png"
    img1.touch()
    msk2.touch()

    with pytest.raises(ValueError, match="Image-mask pairing failure"):
        pair_images_and_masks([img1], [msk2])


def test_pairing_duplicate_error(tmp_path):
    """Verify duplicate identifiers are caught and raise an error."""
    img1 = tmp_path / "a" / "ISIC_0000001.jpg"
    img2 = tmp_path / "b" / "ISIC_0000001.png"
    img1.parent.mkdir()
    img2.parent.mkdir()
    img1.touch()
    img2.touch()

    with pytest.raises(ValueError, match="Duplicate image identifier"):
        pair_images_and_masks([img1, img2], [])


def test_image_preprocessing(tmp_path):
    """Verify RGB conversion and bilinear resizing to 256x256."""
    img_path = tmp_path / "raw.jpg"
    # Create non-square test image (120 x 180)
    arr = np.random.randint(0, 256, (120, 180, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    resized = preprocess_image(img_path, target_size=(256, 256), interpolation="bilinear")

    assert resized.size == (256, 256)
    assert resized.mode == "RGB"


def test_mask_preprocessing_nearest_neighbor(tmp_path):
    """Verify nearest-neighbor resizing and strict binarization to {0, 1}."""
    msk_path = tmp_path / "raw_mask.png"
    # Create mask with 0 and 255
    arr = np.zeros((100, 150), dtype=np.uint8)
    arr[30:70, 40:110] = 255
    Image.fromarray(arr).save(msk_path)

    proc_mask = preprocess_mask(msk_path, target_size=(256, 256), threshold=128)

    assert proc_mask.shape == (256, 256)
    assert proc_mask.dtype == np.uint8
    # Must only contain 0 and 1
    unique_vals = set(np.unique(proc_mask).tolist())
    assert unique_vals.issubset({0, 1})
    assert 1 in unique_vals and 0 in unique_vals


def test_reproducible_split(tmp_path):
    """Verify 80/20 splitting reproducibility, disjointness, and seed determinism."""
    # Create 10 synthetic sample pairs
    pairs = [
        (f"ISIC_{i:07d}", tmp_path / f"img_{i}.png", tmp_path / f"msk_{i}.png")
        for i in range(10)
    ]
    test_pairs = [
        (f"ISIC_{i:07d}", tmp_path / f"img_{i}.png", tmp_path / f"msk_{i}.png")
        for i in range(10, 14)
    ]

    config = {
        "dataset": {"name": "Test", "year": 2016, "task": "Task 1"},
        "preprocessing": {"target_height": 256, "target_width": 256},
    }

    # Split with seed 42 (val_ratio = 0.2 -> 2 val, 8 train)
    splits_dir1 = tmp_path / "splits1"
    res1 = create_splits(pairs, test_pairs, splits_dir1, config, seed=42, val_ratio=0.2)

    assert res1["train_count"] == 8
    assert res1["val_count"] == 2
    assert res1["test_count"] == 4

    with open(res1["train_path"]) as f:
        train_data1 = json.load(f)["samples"]
    with open(res1["val_path"]) as f:
        val_data1 = json.load(f)["samples"]

    train_ids1 = {x["id"] for x in train_data1}
    val_ids1 = {x["id"] for x in val_data1}

    # Verify disjointness
    assert train_ids1.isdisjoint(val_ids1)
    assert len(train_ids1 | val_ids1) == 10

    # Repeat with seed 42 -> MUST be identical
    splits_dir2 = tmp_path / "splits2"
    res2 = create_splits(pairs, test_pairs, splits_dir2, config, seed=42, val_ratio=0.2)

    with open(res2["train_path"]) as f:
        train_data2 = json.load(f)["samples"]
    with open(res2["val_path"]) as f:
        val_data2 = json.load(f)["samples"]

    assert [x["id"] for x in train_data1] == [x["id"] for x in train_data2]
    assert [x["id"] for x in val_data1] == [x["id"] for x in val_data2]


def test_isic_dataset_and_dataloader(tmp_path):
    """Verify PyTorch ISICDataset loading, tensor shapes, normalization, and DataLoader batching."""
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir()
    msk_dir.mkdir()

    samples = []
    for i in range(4):
        s_id = f"ISIC_{i:07d}"
        img_p = img_dir / f"{s_id}.png"
        msk_p = msk_dir / f"{s_id}_Segmentation.png"

        # Synthetic RGB image
        img_arr = np.full((256, 256, 3), 100 + i * 20, dtype=np.uint8)
        Image.fromarray(img_arr).save(img_p)

        # Synthetic binary mask (values 0 and 255)
        msk_arr = np.zeros((256, 256), dtype=np.uint8)
        msk_arr[50:150, 50:150] = 255
        Image.fromarray(msk_arr).save(msk_p)

        samples.append({
            "id": s_id,
            "image_path": str(img_p),
            "mask_path": str(msk_p),
        })

    manifest_p = tmp_path / "test_manifest.json"
    with open(manifest_p, "w") as f:
        json.dump({"samples": samples, "metadata": {}}, f)

    # Instantiate dataset
    dataset = ISICDataset(manifest=manifest_p, root_dir=tmp_path)
    assert len(dataset) == 4

    # Test single sample
    sample0 = dataset[0]
    assert "image" in sample0 and "mask" in sample0 and "id" in sample0
    assert sample0["id"] == "ISIC_0000000"

    img_t = sample0["image"]
    msk_t = sample0["mask"]

    # Image checks: (3, 256, 256), float32 in [0, 1]
    assert isinstance(img_t, torch.Tensor)
    assert img_t.shape == (3, 256, 256)
    assert img_t.dtype == torch.float32
    assert 0.0 <= img_t.min() <= img_t.max() <= 1.0

    # Mask checks: (1, 256, 256), float32 in {0.0, 1.0}
    assert isinstance(msk_t, torch.Tensor)
    assert msk_t.shape == (1, 256, 256)
    assert msk_t.dtype == torch.float32
    unique_vals = set(torch.unique(msk_t).tolist())
    assert unique_vals.issubset({0.0, 1.0})

    # Test DataLoader batching
    loader = get_dataloader(manifest_p, root_dir=tmp_path, batch_size=2)
    batch = next(iter(loader))

    assert batch["image"].shape == (2, 3, 256, 256)
    assert batch["mask"].shape == (2, 1, 256, 256)
    assert len(batch["id"]) == 2

    # Test smoke test helper
    smoke_res = smoke_test_dataloader(manifest_p, batch_size=2, root_dir=tmp_path)
    assert smoke_res["batch_size"] == 2
    assert smoke_res["image_shape"] == [2, 3, 256, 256]
    assert smoke_res["mask_shape"] == [2, 1, 256, 256]


def test_spatial_dimension_mismatch_error(tmp_path):
    """Verify ISICDataset catches mismatched image and mask dimensions."""
    img_p = tmp_path / "img.png"
    msk_p = tmp_path / "msk.png"

    Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(img_p)
    Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(msk_p)

    manifest_p = tmp_path / "mismatch_manifest.json"
    with open(manifest_p, "w") as f:
        json.dump({"samples": [{"id": "ISIC_0000000", "image_path": str(img_p), "mask_path": str(msk_p)}]}, f)

    dataset = ISICDataset(manifest=manifest_p)
    with pytest.raises(ValueError, match="Spatial dimension mismatch"):
        _ = dataset[0]


@pytest.mark.integration
def test_full_dataset_integration():
    """Integration test verifying full dataset manifests and loader when prepared."""
    train_manifest = Path("data/splits/train.json")
    val_manifest = Path("data/splits/val.json")
    test_manifest = Path("data/splits/test.json")

    if not (train_manifest.is_file() and val_manifest.is_file() and test_manifest.is_file()):
        pytest.skip("Full dataset manifests not yet prepared. Run prepare_dataset.py first.")

    processed_dir = Path("data/processed/train_images")
    has_processed_images = processed_dir.is_dir() and any(
        p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
        for p in processed_dir.iterdir()
    )
    if not has_processed_images:
        pytest.skip("Processed dataset images not found. Run prepare_dataset.py first.")

    # Check manifest sample counts
    with open(train_manifest) as f:
        train_samples = json.load(f)["samples"]
    with open(val_manifest) as f:
        val_samples = json.load(f)["samples"]
    with open(test_manifest) as f:
        test_samples = json.load(f)["samples"]

    assert len(train_samples) == 720
    assert len(val_samples) == 180
    assert len(test_samples) == 379

    train_ids = {s["id"] for s in train_samples}
    val_ids = {s["id"] for s in val_samples}
    test_ids = {s["id"] for s in test_samples}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids | val_ids) == 900

    # Test DataLoader batch loading on actual prepared data
    train_batch = smoke_test_dataloader(train_manifest, batch_size=4)
    assert train_batch["image_shape"] == [4, 3, 256, 256]
    assert train_batch["mask_shape"] == [4, 1, 256, 256]

    val_batch = smoke_test_dataloader(val_manifest, batch_size=4)
    assert val_batch["image_shape"] == [4, 3, 256, 256]
    assert val_batch["mask_shape"] == [4, 1, 256, 256]

    test_batch = smoke_test_dataloader(test_manifest, batch_size=4)
    assert test_batch["image_shape"] == [4, 3, 256, 256]
    assert test_batch["mask_shape"] == [4, 1, 256, 256]

