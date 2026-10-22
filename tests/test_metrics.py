"""Unit tests for segmentation metrics (Dice and IoU).

Tests:
- Perfect predictions
- Complete mismatches
- Partial overlap
- Empty masks (both empty vs one empty)
- Continuous probability maps with explicit thresholds
- Tensor shape handling: 2D, 3D, 4D
- Batched reduction modes: 'mean', 'sum', 'none'
- NumPy array compatibility
"""

import pytest
import numpy as np
import torch

from evaluation.metrics import dice_coefficient, iou_score, evaluate_segmentation


def test_perfect_prediction():
    """Verify metrics return 1.0 for identical non-empty masks."""
    target = torch.zeros((10, 10))
    target[2:7, 2:7] = 1.0
    pred = target.clone()

    dice = dice_coefficient(pred, target)
    iou = iou_score(pred, target)

    assert pytest.approx(dice.item(), rel=1e-5) == 1.0
    assert pytest.approx(iou.item(), rel=1e-5) == 1.0


def test_complete_mismatch():
    """Verify metrics return 0.0 when prediction and target have no overlap."""
    target = torch.zeros((10, 10))
    target[0:4, 0:4] = 1.0

    pred = torch.zeros((10, 10))
    pred[5:9, 5:9] = 1.0

    dice = dice_coefficient(pred, target)
    iou = iou_score(pred, target)

    assert pytest.approx(dice.item(), abs=1e-5) == 0.0
    assert pytest.approx(iou.item(), abs=1e-5) == 0.0


def test_partial_overlap():
    """Verify metrics match analytic calculation for partial overlap."""
    # Target has 20 active pixels: [0:4, 0:5] = 20
    target = torch.zeros((10, 10))
    target[0:4, 0:5] = 1.0

    # Pred has 20 active pixels: [2:6, 0:5] = 20
    pred = torch.zeros((10, 10))
    pred[2:6, 0:5] = 1.0

    # Overlap is [2:4, 0:5] = 2 * 5 = 10 pixels
    # Total cardinality = 20 + 20 = 40
    # Expected Dice = 2 * 10 / 40 = 0.5
    # Expected Union = 20 + 20 - 10 = 30
    # Expected IoU = 10 / 30 = 1/3 ~ 0.333333
    dice = dice_coefficient(pred, target)
    iou = iou_score(pred, target)

    assert pytest.approx(dice.item(), rel=1e-4) == 0.5
    assert pytest.approx(iou.item(), rel=1e-4) == 1.0 / 3.0


def test_empty_mask_policy():
    """Verify empty mask policies: both empty vs one empty."""
    empty_target = torch.zeros((10, 10))
    empty_pred = torch.zeros((10, 10))
    non_empty = torch.zeros((10, 10))
    non_empty[2:5, 2:5] = 1.0

    # Both empty -> perfect agreement (1.0)
    assert pytest.approx(dice_coefficient(empty_pred, empty_target).item(), rel=1e-5) == 1.0
    assert pytest.approx(iou_score(empty_pred, empty_target).item(), rel=1e-5) == 1.0

    # False positive (pred has lesion, target is empty) -> 0.0
    assert pytest.approx(dice_coefficient(non_empty, empty_target).item(), abs=1e-5) == 0.0
    assert pytest.approx(iou_score(non_empty, empty_target).item(), abs=1e-5) == 0.0

    # False negative (pred is empty, target has lesion) -> 0.0
    assert pytest.approx(dice_coefficient(empty_pred, non_empty).item(), abs=1e-5) == 0.0
    assert pytest.approx(iou_score(empty_pred, non_empty).item(), abs=1e-5) == 0.0


def test_probability_maps_and_threshold():
    """Verify continuous probability maps are thresholded properly."""
    target = torch.zeros((6, 6))
    target[1:5, 1:5] = 1.0

    # Probabilities: 0.8 above threshold 0.5, 0.3 below threshold 0.5
    pred_probs = torch.full((6, 6), 0.2)
    pred_probs[1:5, 1:5] = 0.85

    dice = dice_coefficient(pred_probs, target, threshold=0.5)
    iou = iou_score(pred_probs, target, threshold=0.5)

    assert pytest.approx(dice.item(), rel=1e-5) == 1.0
    assert pytest.approx(iou.item(), rel=1e-5) == 1.0

    # If threshold is 0.9, all predictions become 0 -> mismatch
    dice_high_thresh = dice_coefficient(pred_probs, target, threshold=0.9)
    assert pytest.approx(dice_high_thresh.item(), abs=1e-5) == 0.0


def test_tensor_shapes_and_batching():
    """Verify 2D, 3D, and 4D tensor inputs with various reduction modes."""
    # Batch of 2 samples of shape (1, 8, 8)
    b_target = torch.zeros((2, 1, 8, 8))
    b_pred = torch.zeros((2, 1, 8, 8))

    # Sample 0: perfect match
    b_target[0, 0, 1:4, 1:4] = 1.0
    b_pred[0, 0, 1:4, 1:4] = 1.0

    # Sample 1: complete mismatch
    b_target[1, 0, 0:3, 0:3] = 1.0
    b_pred[1, 0, 5:8, 5:8] = 1.0

    # Reduction = 'none' -> should return tensor of shape [2] with [1.0, 0.0]
    dice_none = dice_coefficient(b_pred, b_target, reduction="none")
    assert dice_none.shape == (2,)
    assert pytest.approx(dice_none[0].item(), rel=1e-5) == 1.0
    assert pytest.approx(dice_none[1].item(), abs=1e-5) == 0.0

    # Reduction = 'mean' -> (1.0 + 0.0) / 2 = 0.5
    dice_mean = dice_coefficient(b_pred, b_target, reduction="mean")
    assert pytest.approx(dice_mean.item(), rel=1e-5) == 0.5

    # Reduction = 'sum' -> 1.0 + 0.0 = 1.0
    dice_sum = dice_coefficient(b_pred, b_target, reduction="sum")
    assert pytest.approx(dice_sum.item(), rel=1e-5) == 1.0


def test_numpy_compatibility():
    """Verify numpy arrays work seamlessly."""
    target_np = np.zeros((12, 12), dtype=np.uint8)
    target_np[2:8, 2:8] = 1

    pred_np = target_np.copy()

    dice = dice_coefficient(pred_np, target_np)
    iou = iou_score(pred_np, target_np)

    assert isinstance(dice, float)
    assert isinstance(iou, float)
    assert pytest.approx(dice, rel=1e-5) == 1.0
    assert pytest.approx(iou, rel=1e-5) == 1.0


def test_evaluate_segmentation_dict():
    """Verify dictionary helper function."""
    mask = torch.ones((5, 5))
    metrics = evaluate_segmentation(mask, mask)

    assert "dice" in metrics
    assert "iou" in metrics
    assert pytest.approx(metrics["dice"], rel=1e-5) == 1.0
    assert pytest.approx(metrics["iou"], rel=1e-5) == 1.0
