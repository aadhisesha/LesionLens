"""Inference and mask prediction pipeline for TransAttUNet baseline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.transattunet import TransAttUNet
from training.trainer_utils import select_device


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> TransAttUNet:
    """Load TransAttUNet model weights from a trained checkpoint file.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        device: Target compute device.

    Returns:
        TransAttUNet model ready for inference in eval mode.
    """
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")

    device = device or select_device("auto")
    checkpoint = torch.load(ckpt_p, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    in_channels = model_cfg.get("in_channels", 3)
    num_classes = model_cfg.get("num_classes", 1)
    bilinear = model_cfg.get("bilinear", True)

    model = TransAttUNet(in_channels=in_channels, num_classes=num_classes, bilinear=bilinear)

    state_dict = checkpoint["model_state_dict"]
    # Handle module. prefix if trained with DataParallel
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[key] = v

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_input_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[torch.Tensor, Tuple[int, int], Image.Image]:
    """Load and normalize an image for model input.

    Args:
        image_input: Image path, PIL Image, or NumPy RGB array.
        target_size: (H, W) model resolution.

    Returns:
        Tuple of (input_tensor_3xHxW, original_size_WxH, original_pil_image).
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            raise FileNotFoundError(f"Input image not found: {p}")
        pil_img = Image.open(p).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    orig_size = pil_img.size  # (W, H)

    # Resize using bilinear interpolation
    resized_img = pil_img.resize((target_size[1], target_size[0]), resample=Image.Resampling.BILINEAR)

    # Convert to float32 tensor [0, 1] formatted as (1, 3, H, W)
    np_img = np.array(resized_img, dtype=np.float32) / 255.0
    np_img = np.transpose(np_img, (2, 0, 1))  # (3, H, W)
    tensor = torch.from_numpy(np_img).unsqueeze(0).float()

    return tensor, orig_size, pil_img


@torch.no_grad()
def predict_mask(
    model: nn.Module,
    image_input: Union[str, Path, Image.Image, np.ndarray],
    threshold: float = 0.5,
    target_size: Tuple[int, int] = (256, 256),
    resize_to_original: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run model inference to generate probability map and binary segmentation mask.

    Args:
        model: Loaded TransAttUNet model.
        image_input: Path or image object.
        threshold: Probability cutoff for binary mask (default 0.5).
        target_size: Model spatial resolution (256, 256).
        resize_to_original: Whether to restore mask to original input image size.
        device: Compute device.

    Returns:
        Dict with keys:
            - 'prob_map_256': NumPy float array (256, 256) probabilities [0, 1].
            - 'mask_256': NumPy uint8 binary array (256, 256) in {0, 1}.
            - 'mask_original': NumPy uint8 binary array matching original input size.
            - 'original_size': (width, height) of input image.
    """
    device = device or next(model.parameters()).device
    model.eval()

    tensor, orig_size, orig_pil = preprocess_input_image(image_input, target_size=target_size)
    tensor = tensor.to(device)

    logits = model(tensor)
    logits_np = logits.squeeze().cpu().numpy()  # (256, 256)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (256, 256)
    binary_mask_256 = (probs >= threshold).astype(np.uint8)

    # Compute detailed prediction diagnostics
    unique_vals, val_counts = np.unique(binary_mask_256, return_counts=True)
    mask_distribution = {int(k): int(v) for k, v in zip(unique_vals, val_counts)}

    diagnostics: Dict[str, Any] = {
        "logit_min": float(np.min(logits_np)),
        "logit_max": float(np.max(logits_np)),
        "logit_mean": float(np.mean(logits_np)),
        "logit_std": float(np.std(logits_np)),
        "prob_min": float(np.min(probs)),
        "prob_max": float(np.max(probs)),
        "prob_mean": float(np.mean(probs)),
        "prob_std": float(np.std(probs)),
        "pixels_ge_threshold": int(np.sum(probs >= threshold)),
        "total_pixels": int(probs.size),
        "pct_ge_threshold": float(np.mean(probs >= threshold) * 100.0),
        "coverage_pct": float((np.sum(binary_mask_256) / binary_mask_256.size) * 100.0),
        "mask_unique_values": [int(x) for x in unique_vals],
        "mask_distribution": mask_distribution,
    }

    result: Dict[str, Any] = {
        "logits_256": logits_np,
        "prob_map_256": probs,
        "mask_256": binary_mask_256,
        "original_size": orig_size,
        "diagnostics": diagnostics,
    }

    if resize_to_original:
        # Resize binary mask back to original dimensions using nearest-neighbor
        mask_pil = Image.fromarray((binary_mask_256 * 255).astype(np.uint8), mode="L")
        mask_orig_pil = mask_pil.resize(orig_size, resample=Image.Resampling.NEAREST)
        mask_orig = (np.array(mask_orig_pil) >= 128).astype(np.uint8)
        result["mask_original"] = mask_orig

    return result



def save_prediction_visualization(
    original_image: Union[str, Path, Image.Image],
    prediction_dict: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Save side-by-side visualization: Original Image | Probability Map | Binary Mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(original_image, (str, Path)):
        orig_img = Image.open(original_image).convert("RGB")
    elif isinstance(original_image, Image.Image):
        orig_img = original_image.convert("RGB")
    else:
        orig_img = Image.fromarray(original_image).convert("RGB")

    prob_map = prediction_dict["prob_map_256"]
    mask_256 = prediction_dict["mask_256"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(orig_img)
    axes[0].set_title(f"Input Image ({orig_img.size[0]}x{orig_img.size[1]})")
    axes[0].axis("off")

    im1 = axes[1].imshow(prob_map, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Predicted Probability Map")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(mask_256, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Binary Lesion Mask (Threshold 0.5)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_p, dpi=150)
    plt.close(fig)

    return out_p


def main() -> int:
    """CLI entry point for prediction."""
    parser = argparse.ArgumentParser(description="Run TransAttUNet inference on skin lesion image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint .pth")
    parser.add_argument("--image", type=str, required=True, help="Path to input lesion image")
    parser.add_argument("--output", type=str, default="results/predictions/prediction_sample.png", help="Path to save prediction visualization")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization probability threshold")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto/cpu/cuda)")

    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Loading checkpoint from: {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device=device)

    print(f"Running inference on: {args.image}")
    result = predict_mask(model, args.image, threshold=args.threshold, device=device)

    diag = result["diagnostics"]
    print("\n--- PREDICTION DIAGNOSTICS ---")
    print("Logits:")
    print(f"  Min:  {diag['logit_min']:.4f}")
    print(f"  Max:  {diag['logit_max']:.4f}")
    print(f"  Mean: {diag['logit_mean']:.4f}")
    print(f"  Std:  {diag['logit_std']:.4f}")
    print("Probabilities:")
    print(f"  Min:  {diag['prob_min']:.4f}")
    print(f"  Max:  {diag['prob_max']:.4f}")
    print(f"  Mean: {diag['prob_mean']:.4f}")
    print(f"  Std:  {diag['prob_std']:.4f}")
    print(f"Thresholding (cutoff = {args.threshold}):")
    print(f"  Pixels >= {args.threshold}:       {diag['pixels_ge_threshold']} / {diag['total_pixels']} ({diag['pct_ge_threshold']:.2f}%)")
    print(f"  Foreground coverage: {diag['coverage_pct']:.2f}% ({diag['pixels_ge_threshold']}/{diag['total_pixels']} pixels)")
    print(f"  Mask unique values:  {diag['mask_unique_values']}")
    print(f"  Mask distribution:   {diag['mask_distribution']}\n")

    saved_p = save_prediction_visualization(args.image, result, args.output)
    print(f"Visualization saved to: {saved_p.resolve()}")
    return 0



if __name__ == "__main__":
    sys.exit(main())
