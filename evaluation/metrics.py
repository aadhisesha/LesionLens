"""Data-independent segmentation metrics for skin lesion segmentation.

Provides PyTorch-native implementations of:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU) / Jaccard Index

Design Specifications:
- Accepts torch.Tensor or numpy.ndarray.
- Supports shapes: (H, W), (1, H, W), (B, 1, H, W), (B, C, H, W), or flattened (N,).
- Applies explicit thresholding for continuous predictions / probability maps.
- Configurable reduction modes ('mean', 'sum', 'none').
- Robust policy for empty masks:
  When both prediction and ground truth are empty (sum == 0), the match is perfect,
  yielding 1.0. If one is empty and the other is not, the metric yields 0.0.
"""

from __future__ import annotations

from typing import Union, Dict
import numpy as np
import torch


def _to_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Safely convert numpy array or tensor to torch.Tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")


def _binarize_inputs(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float | None = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and binarize pred and target tensors into float {0.0, 1.0}."""
    pred = _to_tensor(pred).float()
    target = _to_tensor(target).float()

    if pred.shape != target.shape:
        # Check if unsqueezing or squeezing single-channel dimension resolves shape
        if pred.ndim == target.ndim + 1 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        elif target.ndim == pred.ndim + 1 and target.shape[1] == 1:
            target = target.squeeze(1)
        elif pred.ndim == target.ndim - 1:
            pred = pred.unsqueeze(0)
        elif target.ndim == pred.ndim - 1:
            target = target.unsqueeze(0)

        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch between prediction {tuple(pred.shape)} "
                f"and target {tuple(target.shape)}"
            )

    if threshold is not None:
        pred_bin = (pred >= threshold).float()
    else:
        pred_bin = (pred > 0.5).float()

    target_bin = (target > 0.5).float()
    return pred_bin, target_bin


def dice_coefficient(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float | None = 0.5,
    eps: float = 1e-7,
    reduction: str = "mean",
    empty_score: float = 1.0,
) -> Union[torch.Tensor, float]:
    """Compute the Dice Similarity Coefficient (F1 Score).

    Formula:
        Dice = (2 * |X ∩ Y|) / (|X| + |Y|)

    Empty Mask Policy:
        - Both empty (sum(pred)==0 and sum(target)==0): returns `empty_score` (1.0).
        - One empty, other non-empty: returns 0.0.

    Args:
        pred: Predicted tensor or array (binary or probabilities).
        target: Ground truth tensor or array (binary).
        threshold: Binarization cutoff for probabilities. If None, expects binary.
        eps: Small smoothing constant to avoid zero division.
        reduction: 'mean' (default), 'sum', or 'none'.
        empty_score: Score to assign when both prediction and ground truth are empty.

    Returns:
        Scalar tensor/float if reduction is 'mean' or 'sum', or tensor of shape (B,) if 'none'.
    """
    pred_bin, target_bin = _binarize_inputs(pred, target, threshold)

    # Standardize to (B, N) for batched evaluation
    if pred_bin.ndim == 2:  # (H, W) -> (1, H*W)
        pred_flat = pred_bin.view(1, -1)
        target_flat = target_bin.view(1, -1)
    elif pred_bin.ndim == 3:  # (B, H, W) or (1, H, W)
        pred_flat = pred_bin.reshape(pred_bin.shape[0], -1)
        target_flat = target_bin.reshape(target_bin.shape[0], -1)
    elif pred_bin.ndim == 4:  # (B, C, H, W)
        pred_flat = pred_bin.reshape(pred_bin.shape[0], -1)
        target_flat = target_bin.reshape(target_bin.shape[0], -1)
    elif pred_bin.ndim == 1:  # (N,) -> (1, N)
        pred_flat = pred_bin.view(1, -1)
        target_flat = target_bin.view(1, -1)
    else:
        raise ValueError(f"Unsupported tensor dimension: {pred_bin.ndim}")

    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)
    total_cardinality = pred_sum + target_sum

    # Calculate dice with epsilon smoothing
    dice = (2.0 * intersection + eps) / (total_cardinality + eps)

    # Handle boundary conditions explicitly:
    both_empty = (pred_sum == 0) & (target_sum == 0)
    one_empty = (pred_sum == 0) ^ (target_sum == 0)

    dice[both_empty] = empty_score
    dice[one_empty] = 0.0

    if reduction == "mean":
        return dice.mean().item() if isinstance(pred, np.ndarray) else dice.mean()
    elif reduction == "sum":
        return dice.sum().item() if isinstance(pred, np.ndarray) else dice.sum()
    elif reduction == "none":
        return dice.numpy() if isinstance(pred, np.ndarray) else dice
    else:
        raise ValueError(f"Invalid reduction '{reduction}'. Must be 'mean', 'sum', or 'none'.")


def iou_score(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float | None = 0.5,
    eps: float = 1e-7,
    reduction: str = "mean",
    empty_score: float = 1.0,
) -> Union[torch.Tensor, float]:
    """Compute Intersection over Union (IoU) / Jaccard Index.

    Formula:
        IoU = |X ∩ Y| / |X ∪ Y| = |X ∩ Y| / (|X| + |Y| - |X ∩ Y|)

    Empty Mask Policy:
        - Both empty (sum(pred)==0 and sum(target)==0): returns `empty_score` (1.0).
        - One empty, other non-empty: returns 0.0.

    Args:
        pred: Predicted tensor or array (binary or probabilities).
        target: Ground truth tensor or array (binary).
        threshold: Binarization cutoff for probabilities. If None, expects binary.
        eps: Small smoothing constant to avoid zero division.
        reduction: 'mean' (default), 'sum', or 'none'.
        empty_score: Score to assign when both prediction and ground truth are empty.

    Returns:
        Scalar tensor/float if reduction is 'mean' or 'sum', or tensor of shape (B,) if 'none'.
    """
    pred_bin, target_bin = _binarize_inputs(pred, target, threshold)

    if pred_bin.ndim == 2:
        pred_flat = pred_bin.view(1, -1)
        target_flat = target_bin.view(1, -1)
    elif pred_bin.ndim == 3:
        pred_flat = pred_bin.reshape(pred_bin.shape[0], -1)
        target_flat = target_bin.reshape(target_bin.shape[0], -1)
    elif pred_bin.ndim == 4:
        pred_flat = pred_bin.reshape(pred_bin.shape[0], -1)
        target_flat = target_bin.reshape(target_bin.shape[0], -1)
    elif pred_bin.ndim == 1:
        pred_flat = pred_bin.view(1, -1)
        target_flat = target_bin.view(1, -1)
    else:
        raise ValueError(f"Unsupported tensor dimension: {pred_bin.ndim}")

    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)
    union = pred_sum + target_sum - intersection

    iou = (intersection + eps) / (union + eps)

    both_empty = (pred_sum == 0) & (target_sum == 0)
    one_empty = (pred_sum == 0) ^ (target_sum == 0)

    iou[both_empty] = empty_score
    iou[one_empty] = 0.0

    if reduction == "mean":
        return iou.mean().item() if isinstance(pred, np.ndarray) else iou.mean()
    elif reduction == "sum":
        return iou.sum().item() if isinstance(pred, np.ndarray) else iou.sum()
    elif reduction == "none":
        return iou.numpy() if isinstance(pred, np.ndarray) else iou
    else:
        raise ValueError(f"Invalid reduction '{reduction}'. Must be 'mean', 'sum', or 'none'.")


# Alias
jaccard_index = iou_score


def evaluate_segmentation(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float | None = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Compute dictionary of standard evaluation metrics.

    Args:
        pred: Predicted tensor or probability map.
        target: Ground truth mask.
        threshold: Threshold cutoff.
        eps: Epsilon smoothing.

    Returns:
        dict with float values for 'dice' and 'iou'.
    """
    d = dice_coefficient(pred, target, threshold=threshold, eps=eps, reduction="mean")
    j = iou_score(pred, target, threshold=threshold, eps=eps, reduction="mean")
    d_val = float(d.item()) if isinstance(d, torch.Tensor) else float(d)
    j_val = float(j.item()) if isinstance(j, torch.Tensor) else float(j)
    return {
        "dice": d_val,
        "iou": j_val,
    }
