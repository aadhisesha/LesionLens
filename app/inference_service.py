"""Inference service for LesionLens segmentation and morphological analysis."""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.model_factory import build_model
from app.tta_utils import predict_with_tta, TTA_TRANSFORMS

IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL: Optional[torch.nn.Module] = None
_CHECKPOINT_METADATA: Dict[str, Any] = {}


def find_checkpoint() -> Path:
    """Locate the best available trained checkpoint."""
    candidates = [
        REPO_ROOT / "checkpoints/phase3_efficient_gsa/best_epoch_26.pth",
        REPO_ROOT / "checkpoints/phase3_efficient_gsa/transattunet_efficient_gsa_best.pth",
        REPO_ROOT / "checkpoints/transattunet_baseline_best.pth",
        REPO_ROOT / "checkpoints/smoke_test_checkpoint.pth",
    ]
    env_ckpt = os.environ.get("LESIONLENS_CHECKPOINT", "").strip()
    if env_ckpt:
        candidates.insert(0, Path(env_ckpt).expanduser())

    for c in candidates:
        if c.is_file():
            return c.resolve()

    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not locate a trained model checkpoint. Checked:\n{checked}")


def get_model() -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Lazy load and return the segmentation model in eval mode."""
    global _MODEL, _CHECKPOINT_METADATA
    if _MODEL is not None:
        return _MODEL, _CHECKPOINT_METADATA

    ckpt_path = find_checkpoint()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config", {})
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    _MODEL = model
    _CHECKPOINT_METADATA = {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_name": ckpt_path.name,
        "epoch": checkpoint.get("best_epoch", checkpoint.get("epoch", "unknown")),
        "best_val_dice": float(checkpoint.get("best_val_dice", -1.0)),
        "device": str(DEVICE),
        "total_params_m": round(total_params / 1e6, 2),
        "model_name": config.get("model", {}).get("name", "transattunet"),
        "use_efficient_gsa": config.get("model", {}).get("use_efficient_gsa", False),
    }
    print(f"[InferenceService] Model loaded from {ckpt_path.name} on {DEVICE} ({_CHECKPOINT_METADATA['total_params_m']}M params)")
    return _MODEL, _CHECKPOINT_METADATA


def pil_to_base64_png(image: Image.Image) -> str:
    """Convert PIL image to base64-encoded PNG data URI."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def color_heatmap(values: np.ndarray, colormap: str = "magma") -> Image.Image:
    """Render a continuous [0, 1] 2D numpy array as an RGB color heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clipped = np.clip(values, 0.0, 1.0)
    cmap = plt.get_cmap(colormap)
    rgba = cmap(clipped)
    rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def compute_abcde_morphology(
    original_rgb: np.ndarray,
    binary_mask: np.ndarray,
    uncertainty_map: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute dermatological ABCDE criteria and morphological descriptors.

    Args:
        original_rgb: (H, W, 3) uint8 image.
        binary_mask: (H, W) boolean or uint8 (0, 1) mask.
        uncertainty_map: Optional (H, W) float32 [0, 1] map.

    Returns:
        Structured dictionary of quantitative morphological features.
    """
    mask_bool = binary_mask > 0
    total_pixels = binary_mask.size
    lesion_pixels = int(np.sum(mask_bool))
    coverage_pct = float((lesion_pixels / total_pixels) * 100.0)

    if lesion_pixels == 0:
        return {
            "lesion_detected": False,
            "coverage_pct": 0.0,
            "area_pixels": 0,
            "perimeter_pixels": 0,
            "compactness": 0.0,
            "asymmetry": {"score": 0.0, "horizontal": 0.0, "vertical": 0.0},
            "border": {"score": 0.0, "tortuosity": 0.0, "regularity": "N/A"},
            "color": {"score": 0.0, "zones_detected": [], "std_rgb": [0.0, 0.0, 0.0]},
            "diameter": {"major_axis_px": 0, "minor_axis_px": 0, "equiv_diam_px": 0, "est_diam_mm": 0.0},
            "uncertainty": {"mean_disagreement": 0.0, "high_disagreement_pct": 0.0},
            "risk_assessment": {"level": "Zero Lesion", "badge": "secondary", "description": "No foreground lesion pixels detected at current threshold."},
        }

    # 1. Asymmetry (A)
    y_idx, x_idx = np.where(mask_bool)
    cy, cx = int(np.mean(y_idx)), int(np.mean(x_idx))

    # Horizontal reflection across vertical axis through centroid
    mask_flipped_h = np.fliplr(mask_bool)
    xor_h = np.logical_xor(mask_bool, mask_flipped_h).sum()
    asym_h = min(1.0, float(xor_h / (2.0 * lesion_pixels)))

    # Vertical reflection across horizontal axis through centroid
    mask_flipped_v = np.flipud(mask_bool)
    xor_v = np.logical_xor(mask_bool, mask_flipped_v).sum()
    asym_v = min(1.0, float(xor_v / (2.0 * lesion_pixels)))

    asymmetry_score = round(((asym_h + asym_v) / 2.0) * 100.0, 1)

    # 2. Border Irregularity (B)
    mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
    edges = mask_img.filter(ImageFilter.FIND_EDGES)
    edge_bool = np.asarray(edges) > 0
    perimeter = int(np.sum(edge_bool))

    # Compactness (Isoperimetric Quotient: 4*pi*Area / P^2). Perfect circle = 1.0; irregular < 0.6
    compactness = (4.0 * np.pi * lesion_pixels) / max(1.0, float(perimeter ** 2))
    compactness = min(1.0, max(0.0, float(compactness)))

    # Irregularity score (0 = smooth circular; 100 = highly irregular/tortuous)
    border_irregularity_score = round((1.0 - compactness) * 100.0, 1)
    regularity_label = "Smooth / Regular" if border_irregularity_score < 35 else ("Moderately Irregular" if border_irregularity_score < 65 else "Markedly Irregular")

    # 3. Color Variegation (C)
    lesion_colors = original_rgb[mask_bool].astype(np.float32)
    std_r = float(np.std(lesion_colors[:, 0]))
    std_g = float(np.std(lesion_colors[:, 1]))
    std_b = float(np.std(lesion_colors[:, 2]))
    mean_std = float(np.mean([std_r, std_g, std_b]))

    # Detect dermoscopic color tones
    r_mean = np.mean(lesion_colors[:, 0])
    g_mean = np.mean(lesion_colors[:, 1])
    b_mean = np.mean(lesion_colors[:, 2])

    detected_zones = []
    if np.any(lesion_colors[:, 0] > 180) and np.any(lesion_colors[:, 1] < 120):
        detected_zones.append("Erythema / Red")
    if np.any(np.all(lesion_colors < 60, axis=1)):
        detected_zones.append("Deep Pigment / Black")
    if np.any((lesion_colors[:, 0] > 100) & (lesion_colors[:, 1] > 60) & (lesion_colors[:, 2] < 60)):
        detected_zones.append("Light / Medium Brown")
    if np.any((lesion_colors[:, 2] > lesion_colors[:, 0] + 10) & (lesion_colors[:, 2] > 70)):
        detected_zones.append("Blue-Gray Veil")
    if len(detected_zones) == 0:
        detected_zones.append("Uniform Pigmentation")

    color_score = min(100.0, round((mean_std / 50.0) * 100.0, 1))

    # 4. Diameter & Dimensions (D)
    min_y, max_y = int(np.min(y_idx)), int(np.max(y_idx))
    min_x, max_x = int(np.min(x_idx)), int(np.max(x_idx))
    height_px = max_y - min_y + 1
    width_px = max_x - min_x + 1
    major_axis_px = max(height_px, width_px)
    minor_axis_px = min(height_px, width_px)
    equiv_diam_px = float(np.sqrt(4.0 * lesion_pixels / np.pi))

    # Standard clinical dermatoscope calibration estimate (~10 pixels per mm for typical 256x256 framing)
    est_diam_mm = round(equiv_diam_px * 0.05, 2)

    # 5. Evolution / Uncertainty & Attention Index (E)
    mean_disagreement = 0.0
    high_disagreement_pct = 0.0
    if uncertainty_map is not None:
        mean_disagreement = round(float(uncertainty_map.mean() * 100.0), 1)
        high_disagreement_pct = round(float((uncertainty_map >= 0.5).mean() * 100.0), 1)

    # Composite Heuristic Atypical Score
    atypical_index = (asymmetry_score * 0.3) + (border_irregularity_score * 0.35) + (color_score * 0.25) + (min(100.0, est_diam_mm * 15.0) * 0.1)
    if atypical_index < 35.0:
        risk_level = "Low Atypical Features"
        risk_badge = "success"
        risk_desc = "Lesion exhibits uniform pigmentation, symmetric boundary geometry, and well-demarcated contours."
    elif atypical_index < 60.0:
        risk_level = "Moderate Morphological Variation"
        risk_badge = "warning"
        risk_desc = "Mild boundary irregularity or color variegation detected. Routine dermatological observation recommended."
    else:
        risk_level = "Elevated Atypical Profile"
        risk_badge = "danger"
        risk_desc = "Notable asymmetry, border tortuosity, and multi-tone color variance detected. Clinical dermoscopic correlation advised."

    return {
        "lesion_detected": True,
        "coverage_pct": round(coverage_pct, 2),
        "area_pixels": lesion_pixels,
        "perimeter_pixels": perimeter,
        "compactness": round(compactness, 3),
        "asymmetry": {
            "score": asymmetry_score,
            "horizontal": round(asym_h * 100.0, 1),
            "vertical": round(asym_v * 100.0, 1),
            "centroid": [cx, cy],
        },
        "border": {
            "score": border_irregularity_score,
            "compactness": round(compactness, 3),
            "regularity": regularity_label,
        },
        "color": {
            "score": color_score,
            "zones_detected": detected_zones,
            "std_rgb": [round(std_r, 1), round(std_g, 1), round(std_b, 1)],
            "mean_std": round(mean_std, 1),
        },
        "diameter": {
            "major_axis_px": major_axis_px,
            "minor_axis_px": minor_axis_px,
            "equiv_diam_px": round(equiv_diam_px, 1),
            "est_diam_mm": est_diam_mm,
            "bbox": [min_x, min_y, max_x, max_y],
        },
        "uncertainty": {
            "mean_disagreement": mean_disagreement,
            "high_disagreement_pct": high_disagreement_pct,
        },
        "risk_assessment": {
            "level": risk_level,
            "badge": risk_badge,
            "atypical_score": round(atypical_index, 1),
            "description": risk_desc,
        },
    }


def analyze_lesion(
    image: Image.Image,
    threshold: float = 0.5,
    use_tta: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end segmentation, uncertainty estimation, and morphological profiling.

    Args:
        image: Uploaded PIL Image in RGB format.
        threshold: Decision threshold for binarization (default 0.50).
        use_tta: If True, uses 8-view Test-Time Augmentation ensemble.

    Returns:
        Comprehensive analysis dictionary with Base64 visualizations and metrics.
    """
    t_start = time.time()
    model, metadata = get_model()

    original_rgb = image.convert("RGB")
    original_size = original_rgb.size  # (width, height)

    # Preprocess to model resolution (256x256)
    resized_rgb = original_rgb.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
    img_np = np.asarray(resized_rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    # Inference with or without TTA
    if use_tta:
        prob_256, uncertainty_256 = predict_with_tta(model, tensor)
    else:
        with torch.inference_mode():
            logits = model(tensor)
            prob_256 = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
            uncertainty_256 = np.zeros_like(prob_256)

    # Interpolate probability and uncertainty to original image dimensions
    prob_map_img = Image.fromarray(prob_256, mode="F").resize(original_size, resample=Image.Resampling.BILINEAR)
    prob_full = np.asarray(prob_map_img, dtype=np.float32).clip(0.0, 1.0)

    uncertainty_map_img = Image.fromarray(uncertainty_256, mode="F").resize(original_size, resample=Image.Resampling.BILINEAR)
    uncertainty_full = np.asarray(uncertainty_map_img, dtype=np.float32).clip(0.0, 1.0)

    # Threshold binary mask
    binary_mask = (prob_full >= threshold).astype(np.uint8)
    mask_img = Image.fromarray(binary_mask * 255, mode="L")

    # Render High-Contrast Red Overlay
    orig_np = np.asarray(original_rgb, dtype=np.float32)
    overlay_np = orig_np.copy()
    mask_bool = binary_mask > 0
    red_color = np.array([255.0, 50.0, 75.0], dtype=np.float32)
    overlay_np[mask_bool] = 0.50 * overlay_np[mask_bool] + 0.50 * red_color
    overlay_img = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8), mode="RGB")

    # Render Glowing Cyan Contour Boundary Overlay
    edges_img = mask_img.filter(ImageFilter.FIND_EDGES)
    edge_bool = np.asarray(edges_img) > 0
    contour_np = orig_np.copy()
    cyan_color = np.array([0.0, 242.0, 254.0], dtype=np.float32)
    contour_np[edge_bool] = cyan_color
    contour_img = Image.fromarray(np.clip(contour_np, 0, 255).astype(np.uint8), mode="RGB")

    # Probability and Uncertainty Heatmaps
    prob_heatmap_img = color_heatmap(prob_full, "magma")
    uncert_heatmap_img = color_heatmap(uncertainty_full, "inferno")

    # Lesion Crop (Isolated Lesion Interior)
    lesion_crop_np = orig_np.copy()
    lesion_crop_np[~mask_bool] = [12, 18, 30]  # dark glass background
    lesion_crop_img = Image.fromarray(lesion_crop_np.astype(np.uint8), mode="RGB")

    # Compute ABCDE morphological metrics
    morphology = compute_abcde_morphology(orig_np, binary_mask, uncertainty_full)

    inference_duration = time.time() - t_start

    return {
        "metadata": {
            **metadata,
            "original_width": original_size[0],
            "original_height": original_size[1],
            "threshold": threshold,
            "use_tta": use_tta,
            "tta_views": len(TTA_TRANSFORMS) if use_tta else 1,
            "inference_duration_s": round(inference_duration, 3),
        },
        "morphology": morphology,
        "visualizations": {
            "original": pil_to_base64_png(original_rgb),
            "mask": pil_to_base64_png(mask_img),
            "overlay": pil_to_base64_png(overlay_img),
            "contour": pil_to_base64_png(contour_img),
            "probability_map": pil_to_base64_png(prob_heatmap_img),
            "uncertainty_map": pil_to_base64_png(uncert_heatmap_img),
            "lesion_crop": pil_to_base64_png(lesion_crop_img),
        },
    }
