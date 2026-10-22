"""Loss functions for skin lesion segmentation training.

Provides:
- SoftDiceLoss: Differentiable soft Dice loss computed over probability maps.
- CombinedBCEDiceLoss: Weighted combination of BCEWithLogitsLoss and SoftDiceLoss.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Differentiable Soft Dice Loss with Laplace smoothing.

    Formula:
        Dice = (2 * sum(P * T) + smooth) / (sum(P) + sum(T) + smooth)
        Loss = 1 - mean(Dice)

    Args:
        smooth: Smoothing constant (Laplace smoothing, default 1.0) preventing
                zero-division and handling empty masks gracefully.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Soft Dice Loss.

        Args:
            logits: Predicted logits of shape (B, 1, H, W) or (B, H, W).
            target: Ground truth binary mask {0, 1} of matching spatial shape.

        Returns:
            Scalar loss tensor with attached computation graph.
        """
        # Align dimensions
        if logits.ndim == target.ndim + 1 and logits.shape[1] == 1:
            target = target.unsqueeze(1)
        elif target.ndim == logits.ndim + 1 and target.shape[1] == 1:
            logits = logits.unsqueeze(1)
        elif logits.ndim != target.ndim:
            raise ValueError(
                f"Dimension mismatch between logits {tuple(logits.shape)} and target {tuple(target.shape)}"
            )

        if logits.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: logits {tuple(logits.shape)} vs target {tuple(target.shape)}"
            )

        probs = torch.sigmoid(logits)
        target = target.float()

        # Flatten spatial dimensions per batch item: (B, -1)
        probs_flat = probs.reshape(probs.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        intersection = (probs_flat * target_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice.mean()
        return loss



class CombinedBCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy with Logits and Soft Dice Loss.

    Formula:
        total_loss = bce_weight * bce_loss + dice_weight * dice_loss
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        smooth_val = kwargs.get("eps", smooth)
        self.dice_loss_fn = SoftDiceLoss(smooth=smooth_val)

        self.last_components: Dict[str, float] = {}


    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: Predicted logits of shape (B, 1, H, W) or (B, H, W).
            target: Ground truth binary mask of matching shape.

        Returns:
            Scalar total loss tensor.
        """
        # Align dimensions
        if logits.ndim == target.ndim + 1 and logits.shape[1] == 1:
            target = target.unsqueeze(1)
        elif target.ndim == logits.ndim + 1 and target.shape[1] == 1:
            logits = logits.unsqueeze(1)

        if logits.shape != target.shape:
            raise ValueError(
                f"Shape mismatch between logits {tuple(logits.shape)} and target {tuple(target.shape)}"
            )

        target = target.float()

        bce = self.bce_loss_fn(logits, target)
        dice = self.dice_loss_fn(logits, target)

        total = self.bce_weight * bce + self.dice_weight * dice

        self.last_components = {
            "bce": float(bce.detach().item()),
            "dice": float(dice.detach().item()),
            "total": float(total.detach().item()),
        }

        return total

    def get_components(self) -> Dict[str, float]:
        """Return the component losses from the most recent forward pass."""
        return self.last_components


# Factory helper
def get_loss_function(
    loss_name: str = "dice_bce",
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    **kwargs,
) -> nn.Module:
    """Build loss function based on name."""
    name = loss_name.lower().replace("-", "_")
    if name in ("dice_bce", "bce_dice", "combined"):
        return CombinedBCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight, **kwargs)
    elif name == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    elif name == "dice":
        return SoftDiceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss name: '{loss_name}'. Available: ['dice_bce', 'bce', 'dice']")
