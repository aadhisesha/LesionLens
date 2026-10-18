"""LesionLens: Efficient-GSA skin-lesion segmentation with TTA disagreement.

Run from the LesionLens repository root:
    python app/gradio_demo.py

Optional environment variables:
    LESIONLENS_CHECKPOINT=/path/to/best_epoch_26.pth
    GRADIO_SHARE=true          # enables a temporary public link; local images are safer
    GRADIO_SERVER_NAME=0.0.0.0 # useful for Colab/container use
    PORT=7860

This is a research demo. TTA disagreement is a heuristic uncertainty proxy, not a
calibrated probability of error or a medical diagnosis.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.model_factory import build_model  # noqa: E402
from app.tta_utils import (  # noqa: E402
    INVERSE_TRANSFORM, TTA_TRANSFORMS, apply_transform, predict_with_tta,
)

IMAGE_SIZE = 256
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def find_checkpoint() -> Path:
    """Locate the trained Phase 3 checkpoint, preferring an explicit env path."""
    candidates: list[Path] = []
    configured = os.environ.get("LESIONLENS_CHECKPOINT", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())

    candidates.extend(
        [
            REPO_ROOT / "checkpoints/phase3_efficient_gsa/transattunet_efficient_gsa_best.pth",
            REPO_ROOT / "checkpoints/phase3_efficient_gsa/best_epoch_26.pth",
            Path(
                "/content/drive_lesionlens/MyDrive/LesionLens/phase3/runs/"
                "efficient_gsa_20261017_145306/best/best_epoch_26.pth"
            ),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find the trained Phase 3 checkpoint. Set LESIONLENS_CHECKPOINT "
        "to the .pth file, or copy it into the repository checkpoint folder. Checked:\n"
        + checked
    )


def load_model() -> tuple[torch.nn.Module, Path, int | str]:
    """Build the architecture from checkpoint config and load trained weights."""
    checkpoint_path = find_checkpoint()
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint.get("config", {})
    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    if not model_config.get("use_efficient_gsa", False):
        raise ValueError(
            "This demo requires the Phase 3 Efficient GSA checkpoint. The selected "
            "checkpoint config does not enable use_efficient_gsa."
        )

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", "unknown"))
    print(f"Loaded Phase 3 checkpoint: {checkpoint_path}")
    print(f"Device: {DEVICE}; best epoch: {best_epoch}")
    print(f"TTA views per image: {len(TTA_TRANSFORMS)}")
    return model, checkpoint_path, best_epoch


MODEL, CHECKPOINT_PATH, BEST_EPOCH = load_model()


def color_heatmap(values: np.ndarray, cmap_name: str = "magma") -> Image.Image:
    """Render a numeric map using fixed [0, 1] color scaling."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgba = plt.get_cmap(cmap_name)(np.clip(values, 0.0, 1.0))
    rgb = (rgba[..., :3] * 255).round().astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def resize_float_map(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a floating point map to (width, height), preserving continuous values."""
    map_image = Image.fromarray(values.astype(np.float32), mode="F")
    return np.asarray(
        map_image.resize(size, resample=Image.Resampling.BILINEAR), dtype=np.float32
    ).clip(0.0, 1.0)


def segment_image(image: Image.Image | None):
    """Segment an uploaded image and return mask, overlay, maps, download and status."""
    if image is None:
        raise gr.Error("Upload a dermoscopic skin-lesion image first.")

    original = image.convert("RGB")
    resized = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR
    )
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0).float()
    tensor = tensor.to(DEVICE)

    try:
        mean_probability_256, uncertainty_256 = predict_with_tta(MODEL, tensor)
    except RuntimeError as exc:
        if DEVICE.type == "cuda" and "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise gr.Error(
                "GPU memory ran out during the eight-view prediction. Try again or "
                "run this app on CPU / use a smaller model input."
            ) from exc
        raise gr.Error(f"Inference failed: {exc}") from exc

    binary_256 = (mean_probability_256 >= THRESHOLD).astype(np.uint8)
    original_size = original.size  # width, height
    mean_probability = resize_float_map(mean_probability_256, original_size)
    uncertainty = resize_float_map(uncertainty_256, original_size)

    binary_image = Image.fromarray(binary_256 * 255, mode="L")
    mask_output = binary_image.resize(original_size, resample=Image.Resampling.NEAREST)
    mask_binary_original = np.asarray(mask_output) >= 128

    original_array = np.asarray(original, dtype=np.float32)
    overlay_array = original_array.copy()
    red = np.array([255.0, 55.0, 70.0], dtype=np.float32)
    overlay_array[mask_binary_original] = (
        0.55 * overlay_array[mask_binary_original] + 0.45 * red
    )
    overlay = Image.fromarray(np.clip(overlay_array, 0, 255).astype(np.uint8), mode="RGB")

    probability_heatmap = color_heatmap(mean_probability, "magma")
    uncertainty_heatmap = color_heatmap(uncertainty, "inferno")

    fd, output_path = tempfile.mkstemp(prefix="lesionlens_tta_mask_", suffix=".png")
    os.close(fd)
    mask_output.save(output_path, format="PNG")

    foreground_percent = float(mask_binary_original.mean() * 100.0)
    mean_disagreement_percent = float(uncertainty.mean() * 100.0)
    high_disagreement_percent = float((uncertainty >= 0.5).mean() * 100.0)
    status = (
        f"**Segmentation complete.** Model input: {IMAGE_SIZE} × {IMAGE_SIZE}; "
        f"TTA views: {len(TTA_TRANSFORMS)}; mask threshold: {THRESHOLD:.2f}.\n\n"
        f"Predicted foreground coverage: **{foreground_percent:.2f}%**. "
        f"Mean scaled TTA disagreement: **{mean_disagreement_percent:.2f}%**; "
        f"pixels with scaled disagreement ≥ 0.5: **{high_disagreement_percent:.2f}%**.\n\n"
        "**Interpreting uncertainty:** bright regions mark pixels whose predictions "
        "differ more across transformed views. This is a TTA-based disagreement proxy, "
        "not a calibrated probability of error or a diagnostic confidence score. "
        "This research demo does not classify lesions as benign or malignant and is "
        "not a medical diagnosis."
    )
    return (
        mask_output,
        overlay,
        probability_heatmap,
        uncertainty_heatmap,
        output_path,
        status,
    )


with gr.Blocks(
    title="LesionLens — Segmentation and TTA Uncertainty",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # LesionLens
        ### Skin lesion segmentation · TransAttUNet with Efficient GSA

        Upload a dermoscopic image to view the predicted lesion mask, red overlay,
        mean probability map, and **TTA disagreement heatmap**. The app runs the
        previously trained Phase 3 checkpoint; no retraining is performed.

        **Uncertainty note:** the disagreement map compares predictions from eight
        flipped/rotated views. It highlights transformation-sensitive pixels; it is
        not a calibrated confidence score and cannot rule out model errors.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            uploaded_image = gr.Image(type="pil", label="Upload a skin-lesion image")
            run_button = gr.Button("Segment + estimate uncertainty", variant="primary")
        with gr.Column(scale=2):
            with gr.Row():
                mask_view = gr.Image(type="pil", label="Predicted binary mask")
                overlay_view = gr.Image(type="pil", label="Mask overlay")
            with gr.Row():
                probability_view = gr.Image(type="pil", label="Mean probability map")
                uncertainty_view = gr.Image(
                    type="pil", label="TTA disagreement / uncertainty proxy"
                )
            mask_download = gr.File(label="Download predicted mask (PNG)")
            status_view = gr.Markdown()

    run_button.click(
        fn=segment_image,
        inputs=[uploaded_image],
        outputs=[
            mask_view,
            overlay_view,
            probability_view,
            uncertainty_view,
            mask_download,
            status_view,
        ],
    )


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "false").lower() in {"1", "true", "yes"}
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("PORT", "7860"))
    demo.queue().launch(
        server_name=server_name,
        server_port=port,
        share=share,
        show_error=True,
    )
