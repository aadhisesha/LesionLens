"""Test-time augmentation helpers for LesionLens segmentation inference."""
from __future__ import annotations

import numpy as np
import torch

TTA_TRANSFORMS = (
    "identity",
    "horizontal_flip",
    "vertical_flip",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "transpose",
    "anti_transpose",
)

INVERSE_TRANSFORM = {
    "identity": "identity",
    "horizontal_flip": "horizontal_flip",
    "vertical_flip": "vertical_flip",
    "rotate_90": "rotate_270",
    "rotate_180": "rotate_180",
    "rotate_270": "rotate_90",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}


def apply_transform(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Apply one of the eight square-image symmetries to a BCHW tensor."""
    if name == "identity":
        return tensor
    if name == "horizontal_flip":
        return torch.flip(tensor, dims=(-1,))
    if name == "vertical_flip":
        return torch.flip(tensor, dims=(-2,))
    if name == "rotate_90":
        return torch.rot90(tensor, k=1, dims=(-2, -1))
    if name == "rotate_180":
        return torch.rot90(tensor, k=2, dims=(-2, -1))
    if name == "rotate_270":
        return torch.rot90(tensor, k=3, dims=(-2, -1))
    if name == "transpose":
        return tensor.transpose(-2, -1)
    if name == "anti_transpose":
        return torch.flip(tensor.transpose(-2, -1), dims=(-2, -1))
    raise ValueError(f"Unknown TTA transformation: {name}")


def predict_with_tta(
    model: torch.nn.Module, tensor: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean probability and normalized pixel-wise TTA disagreement.

    Predictions are inverse-transformed to the original coordinate system. The
    mean probabilities drive the final segmentation; the standard deviation across
    views is a heuristic uncertainty proxy. Probability standard deviation is at
    most 0.5, so division by 0.5 scales the display map to [0, 1].
    """
    aligned_probabilities: list[torch.Tensor] = []
    with torch.inference_mode():
        for transform_name in TTA_TRANSFORMS:
            transformed = apply_transform(tensor, transform_name)
            logits = model(transformed)
            probabilities = torch.sigmoid(logits)
            aligned = apply_transform(
                probabilities, INVERSE_TRANSFORM[transform_name]
            )
            aligned_probabilities.append(aligned)

    stack = torch.stack(aligned_probabilities, dim=0)
    mean_probability = stack.mean(dim=0)[0, 0]
    probability_std = stack.std(dim=0, unbiased=False)[0, 0]
    normalized_disagreement = torch.clamp(probability_std / 0.5, 0.0, 1.0)
    return (
        mean_probability.detach().cpu().numpy().astype(np.float32),
        normalized_disagreement.detach().cpu().numpy().astype(np.float32),
    )
