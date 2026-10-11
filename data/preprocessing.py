"""Preprocessing pipeline for ISIC 2016 Task 1 Skin Lesion Segmentation.

Handles:
- Identifier normalization and image-mask pairing.
- Safe RGB image loading, bilinear resizing to target dimensions (256x256), and normalization.
- Safe mask loading in grayscale, nearest-neighbor resizing, and strict binarization.
- Batch processing with caching / skip logic and integrity checks.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import yaml

# Supported interpolation mappings
INTERPOLATION_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "box": Image.Resampling.BOX,
    "lanczos": Image.Resampling.LANCZOS,
}

# Standard ISIC pattern (e.g., ISIC_0000000)
ISIC_PATTERN = re.compile(r"^ISIC_\d{7}$")


def normalize_identifier(filename_or_path: Union[str, Path]) -> str:
    """Extract and validate normalized ISIC sample identifier from a filename or path.

    Rules:
    - Removes parent directories.
    - Strips extension (e.g., .jpg, .png).
    - Strips ground-truth suffixes (e.g., _Segmentation, _segmentation).
    - Validates that the resulting identifier adheres to ISIC conventions.

    Examples:
        'ISIC_0000000.jpg' -> 'ISIC_0000000'
        'ISIC_0000000_Segmentation.png' -> 'ISIC_0000000'
        'data/raw/train_masks/ISIC_0000004_segmentation.png' -> 'ISIC_0000004'

    Args:
        filename_or_path: File name or Path object.

    Returns:
        Canonical identifier string.

    Raises:
        ValueError: If no valid ISIC identifier can be resolved.
    """
    stem = Path(filename_or_path).stem

    # Remove known segmentation suffixes
    for suffix in ["_Segmentation", "_segmentation"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    if not ISIC_PATTERN.match(stem):
        raise ValueError(
            f"Filename '{filename_or_path}' could not be normalized to a valid "
            f"ISIC identifier (expected pattern 'ISIC_XXXXXXX', got '{stem}')."
        )

    return stem


def pair_images_and_masks(
    image_paths: List[Union[str, Path]],
    mask_paths: List[Union[str, Path]],
) -> List[Tuple[str, Path, Path]]:
    """Pair images with their corresponding ground-truth masks by identifier.

    Guarantees:
    - No sorting-based assumption; matches strictly by canonical identifier.
    - Detects and rejects duplicate identifiers within image or mask collections.
    - Reports all unmatched images or masks explicitly.

    Args:
        image_paths: Iterable of image file paths.
        mask_paths: Iterable of mask file paths.

    Returns:
        List of tuples: (sample_id, image_path, mask_path) sorted by sample_id.

    Raises:
        ValueError: If duplicates exist or if any image or mask is unmatched.
    """
    images_by_id: Dict[str, Path] = {}
    for img in image_paths:
        p = Path(img)
        sample_id = normalize_identifier(p)
        if sample_id in images_by_id:
            raise ValueError(
                f"Duplicate image identifier '{sample_id}' detected: "
                f"'{images_by_id[sample_id]}' and '{p}'"
            )
        images_by_id[sample_id] = p

    masks_by_id: Dict[str, Path] = {}
    for msk in mask_paths:
        p = Path(msk)
        sample_id = normalize_identifier(p)
        if sample_id in masks_by_id:
            raise ValueError(
                f"Duplicate mask identifier '{sample_id}' detected: "
                f"'{masks_by_id[sample_id]}' and '{p}'"
            )
        masks_by_id[sample_id] = p

    img_keys = set(images_by_id.keys())
    msk_keys = set(masks_by_id.keys())

    missing_masks = img_keys - msk_keys
    missing_images = msk_keys - img_keys

    if missing_masks or missing_images:
        err_msg = ["Image-mask pairing failure:"]
        if missing_masks:
            err_msg.append(
                f"  - {len(missing_masks)} images lack masks: {sorted(list(missing_masks))[:5]}..."
            )
        if missing_images:
            err_msg.append(
                f"  - {len(missing_images)} masks lack images: {sorted(list(missing_images))[:5]}..."
            )
        raise ValueError("\n".join(err_msg))

    paired = [
        (sample_id, images_by_id[sample_id], masks_by_id[sample_id])
        for sample_id in sorted(img_keys)
    ]
    return paired


def preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (256, 256),
    interpolation: str = "bilinear",
) -> Image.Image:
    """Load, convert to RGB, and resize an image safely.

    Args:
        image_path: Path to the raw image file.
        target_size: Tuple of (height, width).
        interpolation: Resampling filter ('bilinear', 'bicubic', etc.).

    Returns:
        Resized PIL RGB Image.

    Raises:
        FileNotFoundError: If image does not exist.
        ValueError: If interpolation method is invalid or image cannot be read.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    resample_method = INTERPOLATION_MAP.get(interpolation.lower())
    if resample_method is None:
        raise ValueError(
            f"Unsupported image interpolation '{interpolation}'. "
            f"Allowed: {list(INTERPOLATION_MAP.keys())}"
        )

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            # PIL resize takes (width, height)
            target_w, target_h = target_size[1], target_size[0]
            resized = img.resize((target_w, target_h), resample=resample_method)
            return resized
    except Exception as e:
        raise ValueError(f"Failed to load/process image '{path}': {e}") from e


def preprocess_mask(
    mask_path: Union[str, Path],
    target_size: Tuple[int, int] = (256, 256),
    threshold: int = 128,
) -> np.ndarray:
    """Load, resize using nearest-neighbor only, and strictly binarize a mask.

    Strict Requirements:
    - Masks are loaded in grayscale ('L').
    - Interpolation MUST be nearest-neighbor (NEAREST) to prevent intermediate labels.
    - Output is strictly binary (values in {0, 1}).

    Args:
        mask_path: Path to raw ground truth mask.
        target_size: Tuple of (height, width).
        threshold: Intensity threshold (default 128 for 0-255 masks).

    Returns:
        Binary numpy array of shape (height, width), dtype uint8, containing only 0 and 1.

    Raises:
        FileNotFoundError: If mask does not exist.
        ValueError: If mask processing fails or result is non-binary.
    """
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask not found: {path}")

    try:
        with Image.open(path) as msk:
            msk = msk.convert("L")
            target_w, target_h = target_size[1], target_size[0]
            # Nearest-neighbor resize ONLY
            resized_msk = msk.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
            arr = np.array(resized_msk, dtype=np.uint8)

            # Binarize: values >= threshold become 1, else 0
            binary_arr = (arr >= threshold).astype(np.uint8)

            # Validate binary integrity
            unique_vals = set(np.unique(binary_arr).tolist())
            if not unique_vals.issubset({0, 1}):
                raise ValueError(
                    f"Processed mask '{path}' contains unexpected values: {unique_vals}. "
                    f"Only binary values {{0, 1}} are permitted."
                )

            return binary_arr
    except Exception as e:
        raise ValueError(f"Failed to load/process mask '{path}': {e}") from e


def process_and_save_pair(
    sample_id: str,
    raw_img_path: Union[str, Path],
    raw_msk_path: Union[str, Path],
    out_img_dir: Union[str, Path],
    out_msk_dir: Union[str, Path],
    target_size: Tuple[int, int] = (256, 256),
    image_interpolation: str = "bilinear",
    mask_threshold: int = 128,
    force: bool = False,
) -> Tuple[Path, Path, bool]:
    """Preprocess and save a single image-mask pair to the processed directory.

    Masks are saved as standard 8-bit PNG images with pixel values (0 and 255)
    for easy visualization and lossless storage, while dataset loader converts
    them to model-ready binary tensors {0.0, 1.0}.

    Args:
        sample_id: Canonical sample identifier (e.g. 'ISIC_0000000').
        raw_img_path: Path to raw image.
        raw_msk_path: Path to raw mask.
        out_img_dir: Output directory for processed images.
        out_msk_dir: Output directory for processed masks.
        target_size: Target (H, W).
        image_interpolation: Resampling method for images.
        mask_threshold: Binarization cutoff.
        force: If True, overwrite existing processed files.

    Returns:
        Tuple of (processed_img_path, processed_msk_path, was_processed).
    """
    out_img_dir = Path(out_img_dir)
    out_msk_dir = Path(out_msk_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    dest_img = out_img_dir / f"{sample_id}.png"
    dest_msk = out_msk_dir / f"{sample_id}_Segmentation.png"

    if not force and dest_img.is_file() and dest_msk.is_file():
        # Validate existing files
        try:
            with Image.open(dest_img) as img:
                if img.size == (target_size[1], target_size[0]):
                    with Image.open(dest_msk) as msk:
                        if msk.size == (target_size[1], target_size[0]):
                            return dest_img, dest_msk, False
        except Exception:
            pass  # Fall through to reprocess if corrupt

    # Preprocess image
    proc_img = preprocess_image(
        raw_img_path, target_size=target_size, interpolation=image_interpolation
    )
    proc_img.save(dest_img, format="PNG")

    # Preprocess mask
    proc_msk_arr = preprocess_mask(
        raw_msk_path, target_size=target_size, threshold=mask_threshold
    )
    # Save as 0 / 255 PNG for clean display
    msk_img = Image.fromarray((proc_msk_arr * 255).astype(np.uint8), mode="L")
    msk_img.save(dest_msk, format="PNG")

    return dest_img, dest_msk, True
