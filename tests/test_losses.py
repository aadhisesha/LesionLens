"""Unit tests for segmentation loss functions."""

import pytest
import torch
import torch.nn as nn

from training.losses import SoftDiceLoss, CombinedBCEDiceLoss, get_loss_function


def test_loss_finite_and_forward():
    """Verify loss outputs finite scalars on standard inputs."""
    criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    logits = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss = criterion(logits, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_loss_perfect_prediction():
    """Verify that highly confident, matching predictions yield a very low loss."""
    criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1.0

    # High positive logits where target == 1, high negative logits where target == 0
    logits = torch.where(target == 1.0, 15.0, -15.0)

    loss = criterion(logits, target)
    components = criterion.get_components()

    # Soft dice loss should be ~0.0, BCE should be ~0.0
    assert loss.item() < 0.01
    assert components["dice"] < 0.01
    assert components["bce"] < 0.01


def test_loss_incorrect_prediction():
    """Verify that completely inverted predictions yield high loss."""
    criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1.0

    # Opposite logits
    logits = torch.where(target == 1.0, -15.0, 15.0)

    loss = criterion(logits, target)
    components = criterion.get_components()

    # Dice loss should approach 1.0
    assert components["dice"] > 0.95
    assert loss.item() > 1.0


def test_loss_empty_target_mask():
    """Verify loss handles completely empty target masks gracefully without NaN."""
    criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    empty_target = torch.zeros(2, 1, 32, 32)

    # Model predicting background (large negative logits)
    logits_bg = torch.full((2, 1, 32, 32), -10.0)
    loss_correct_empty = criterion(logits_bg, empty_target)
    assert torch.isfinite(loss_correct_empty)
    assert loss_correct_empty.item() < 0.05

    # Model predicting foreground (large positive logits on empty target)
    logits_fg = torch.full((2, 1, 32, 32), 10.0)
    loss_incorrect_empty = criterion(logits_fg, empty_target)
    assert torch.isfinite(loss_incorrect_empty)
    assert loss_incorrect_empty.item() > 1.0


def test_loss_shape_mismatch():
    """Verify informative ValueError on dimension mismatch."""
    criterion = CombinedBCEDiceLoss()
    logits = torch.randn(2, 1, 32, 32)
    mismatched_target = torch.randn(2, 1, 64, 64)

    with pytest.raises(ValueError, match="Shape mismatch"):
        _ = criterion(logits, mismatched_target)


def test_loss_backward_propagation():
    """Verify that loss propagates gradients back to model logits."""
    criterion = CombinedBCEDiceLoss()
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 32, 32)).float()

    loss = criterion(logits, target)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad != 0).any()


def test_get_loss_factory():
    """Verify loss factory function."""
    loss_fn = get_loss_function("dice_bce", bce_weight=0.5, dice_weight=1.5)
    assert isinstance(loss_fn, CombinedBCEDiceLoss)
    assert loss_fn.bce_weight == 0.5
    assert loss_fn.dice_weight == 1.5

    with pytest.raises(ValueError, match="Unknown loss name"):
        _ = get_loss_function("unknown_loss")
