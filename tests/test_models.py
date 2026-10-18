"""Automated unit and integration tests for TransAttUNet architecture."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from models.transattunet import (
    TransAttUNet,
    EfficientGlobalContextAttention,
)
from models.model_factory import build_model
from training.losses import CombinedBCEDiceLoss
from data.dataset import get_dataloader


def test_transattunet_construction():
    """Verify default constructor and layer initialization."""
    model = TransAttUNet(in_channels=3, num_classes=1)
    assert isinstance(model, nn.Module)
    assert model.in_channels == 3
    assert model.num_classes == 1
    assert hasattr(model, "pam")
    assert hasattr(model, "sdpa")
    assert hasattr(model, "pos")


def test_transattunet_forward_shape():
    """Verify model produces (B, 1, 256, 256) raw logits for (B, 3, 256, 256) input."""
    model = TransAttUNet(in_channels=3, num_classes=1)
    model.eval()

    with torch.no_grad():
        x = torch.randn(2, 3, 256, 256)
        logits = model(x)

    assert logits.shape == (2, 1, 256, 256)
    assert torch.isfinite(logits).all()


def test_transattunet_batch_size_one():
    """Verify model works on a single sample without batch dimension issues."""
    model = TransAttUNet(in_channels=3, num_classes=1)
    model.eval()

    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256)
        logits = model(x)

    assert logits.shape == (1, 1, 256, 256)
    assert torch.isfinite(logits).all()


def test_transattunet_backward_cpu():
    """Verify backward propagation computes finite gradients for all trainable layers."""
    model = TransAttUNet(in_channels=3, num_classes=1)
    model.train()

    x = torch.randn(2, 3, 128, 128)  # 128x128 for fast CPU test
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    # Check encoder conv gradient
    assert model.inc.double_conv[0].weight.grad is not None
    assert torch.isfinite(model.inc.double_conv[0].weight.grad).all()

    # Check GSA (PAM) gradient
    assert model.pam.gamma.grad is not None

    # Check decoder outc gradient
    assert model.outc.conv.weight.grad is not None


def test_model_factory_build():
    """Verify build_model instantiates TransAttUNet from configuration."""
    model = build_model("transattunet", in_channels=3, num_classes=1)
    assert isinstance(model, TransAttUNet)

    cfg = {"model": {"name": "transattunet", "in_channels": 3, "num_classes": 1}}
    model_cfg = build_model(cfg)
    assert isinstance(model_cfg, TransAttUNet)




def test_efficient_gsa_variant_forward_backward():
    config = {
        "model": {
            "name": "transattunet",
            "in_channels": 3,
            "num_classes": 1,
            "bilinear": True,
            "use_efficient_gsa": True,
        }
    }
    model = build_model(config)

    assert isinstance(model.pam, EfficientGlobalContextAttention)
    assert model.use_efficient_gsa is True

    model.train()
    logits = model(torch.randn(1, 3, 128, 128))
    assert logits.shape == (1, 1, 128, 128)
    assert torch.isfinite(logits).all()

    logits.square().mean().backward()
    assert model.pam.gamma.grad is not None
    assert torch.isfinite(model.pam.gamma.grad).all()
    assert model.outc.conv.weight.grad is not None

@pytest.mark.integration
def test_dataset_model_loss_integration():
    """Integration test: One batch from dataset -> forward pass -> loss -> backward pass."""
    manifest_p = "data/splits/train.json"
    processed_dir = Path("data/processed/train_images")
    has_processed_images = processed_dir.is_dir() and any(
        p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
        for p in processed_dir.iterdir()
    )
    if not Path(manifest_p).is_file() or not has_processed_images:
        pytest.skip("Full dataset not yet prepared. Run prepare_dataset.py first.")
    loader = get_dataloader(manifest_p, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    images = batch["image"]
    masks = batch["mask"]

    assert images.shape == (2, 3, 256, 256)
    assert masks.shape == (2, 1, 256, 256)

    model = TransAttUNet(in_channels=3, num_classes=1)
    criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

    logits = model(images)
    loss = criterion(logits, masks)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert model.inc.double_conv[0].weight.grad is not None
