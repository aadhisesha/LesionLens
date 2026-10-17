"""PyTorch Dataset implementation for ISIC 2016 Skin Lesion Segmentation.

Features:
- Loads image-mask pairs from reproducible JSON manifests.
- Safe image and mask loading with Pillow.
- Returns RGB image tensors formatted as (C, H, W), float32 normalized to [0, 1].
- Returns mask tensors formatted as (1, H, W), float32 strictly containing {0.0, 1.0}.
- Validates matching spatial dimensions for every sample.
- Supplies canonical sample identifier ('id') for tracing and evaluation.
- Informative exceptions on missing or corrupted assets.
- Clean DataLoader factory and smoke test functionality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


class ISICDataset(Dataset):
    """PyTorch Dataset for ISIC 2016 Skin Lesion Segmentation."""

    def __init__(
        self,
        manifest: Union[str, Path, List[Dict[str, Any]]],
        root_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        standardize: bool = False,
    ) -> None:
        """Initialize ISIC Dataset.

        Args:
            manifest: Path to split JSON manifest or pre-loaded list of record dicts.
            root_dir: Project root directory used to resolve relative paths in manifests.
            transform: Optional callable for Phase 2 augmentations (None in Phase 1).
            standardize: If True, apply ImageNet mean and std normalization. Default False.
        """
        super().__init__()
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.transform = transform
        self.standardize = standardize

        # Mean and standard deviation for ImageNet
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if not manifest_path.is_file():
                # Try relative to root_dir
                alt_path = self.root_dir / manifest_path
                if alt_path.is_file():
                    manifest_path = alt_path
                else:
                    raise FileNotFoundError(f"Manifest not found: {manifest}")

            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "samples" in data:
                    self.samples = data["samples"]
                    self.metadata = data.get("metadata", {})
                elif isinstance(data, list):
                    self.samples = data
                    self.metadata = {}
                else:
                    raise ValueError(
                        f"Unsupported manifest format in {manifest_path}. "
                        f"Expected list or dict with 'samples' key."
                    )
        elif isinstance(manifest, list):
            self.samples = manifest
            self.metadata = {}
        else:
            raise TypeError(f"Expected manifest path or list of records, got {type(manifest)}")

        if len(self.samples) == 0:
            raise ValueError("Dataset manifest contains 0 samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path relative to root_dir if not absolute."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (self.root_dir / p).resolve()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Fetch and return a single sample.

        Returns:
            dict containing:
                - 'image': torch.FloatTensor of shape (3, H, W) normalized to [0, 1]
                - 'mask': torch.FloatTensor of shape (1, H, W) in {0.0, 1.0}
                - 'id': str sample identifier (e.g. 'ISIC_0000000')
        """
        record = self.samples[index]
        sample_id = record.get("id") or record.get("sample_id", f"sample_{index}")
        img_rel = record.get("image_path") or record.get("image")
        msk_rel = record.get("mask_path") or record.get("mask")

        if not img_rel or not msk_rel:
            raise KeyError(
                f"Sample at index {index} ('{sample_id}') missing 'image_path' or 'mask_path'."
            )

        img_path = self._resolve_path(img_rel)
        msk_path = self._resolve_path(msk_rel)

        if not img_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {img_path}")
        if not msk_path.is_file():
            raise FileNotFoundError(f"Mask file does not exist: {msk_path}")

        try:
            with Image.open(img_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                img_np = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
        except Exception as e:
            raise RuntimeError(f"Corrupted or unreadable image file '{img_path}': {e}") from e

        try:
            with Image.open(msk_path) as pil_msk:
                pil_msk = pil_msk.convert("L")
                msk_np = np.array(pil_msk, dtype=np.float32)
                # Binarize: ground truth pixels >= 128 become 1.0, else 0.0
                msk_np = (msk_np >= 128.0).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Corrupted or unreadable mask file '{msk_path}': {e}") from e

        # Validate matching spatial dimensions
        if img_np.shape[:2] != msk_np.shape[:2]:
            raise ValueError(
                f"Spatial dimension mismatch for sample '{sample_id}': "
                f"Image shape is {img_np.shape[:2]} but Mask shape is {msk_np.shape[:2]}"
            )

        # Reorder image to (C, H, W)
        img_np = np.transpose(img_np, (2, 0, 1))  # (3, H, W)
        msk_np = np.expand_dims(msk_np, axis=0)   # (1, H, W)

        if self.standardize:
            img_np = (img_np - self.mean) / self.std

        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(img_np).float()
        msk_tensor = torch.from_numpy(msk_np).float()

        # Final assertion on mask binary integrity
        unique_vals = torch.unique(msk_tensor)
        for v in unique_vals:
            if v.item() not in (0.0, 1.0):
                raise ValueError(
                    f"Invalid mask value {v.item()} found in sample '{sample_id}'. "
                    f"Expected only 0.0 and 1.0."
                )

        if self.transform is not None:
            img_tensor, msk_tensor = self.transform(img_tensor, msk_tensor)

        return {
            "image": img_tensor,
            "mask": msk_tensor,
            "id": sample_id,
        }


def get_dataloader(
    manifest_path: Union[str, Path],
    root_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    standardize: bool = False,
) -> DataLoader:
    """Create a PyTorch DataLoader for a given dataset manifest.

    Args:
        manifest_path: Path to manifest JSON.
        root_dir: Workspace root directory.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of background worker processes.
        pin_memory: Whether to pin memory for CUDA transfer.
        standardize: Whether to apply ImageNet normalization.

    Returns:
        Configured DataLoader instance.
    """
    dataset = ISICDataset(
        manifest=manifest_path,
        root_dir=root_dir,
        standardize=standardize,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def smoke_test_dataloader(
    manifest_path: Union[str, Path],
    batch_size: int = 4,
    root_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute a smoke test loading a single batch from a manifest.

    Returns:
        dict with batch statistics for verification.
    """
    loader = get_dataloader(
        manifest_path=manifest_path,
        root_dir=root_dir,
        batch_size=batch_size,
        shuffle=False,
    )
    batch = next(iter(loader))

    images = batch["image"]
    masks = batch["mask"]
    ids = batch["id"]

    assert images.ndim == 4, f"Expected 4D image tensor, got {images.ndim}"
    assert masks.ndim == 4, f"Expected 4D mask tensor, got {masks.ndim}"
    assert images.shape[1] == 3, f"Expected 3 image channels, got {images.shape[1]}"
    assert masks.shape[1] == 1, f"Expected 1 mask channel, got {masks.shape[1]}"
    assert images.shape[2:] == masks.shape[2:], "Height and width must match between image and mask"
    assert images.dtype == torch.float32, f"Expected float32 images, got {images.dtype}"
    assert masks.dtype == torch.float32, f"Expected float32 masks, got {masks.dtype}"

    return {
        "batch_size": images.shape[0],
        "image_shape": list(images.shape),
        "mask_shape": list(masks.shape),
        "ids": ids,
        "image_min": float(images.min()),
        "image_max": float(images.max()),
        "mask_unique": [float(x) for x in torch.unique(masks)],
    }
