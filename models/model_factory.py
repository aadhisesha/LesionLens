"""Model factory for building segmentation model architectures."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import torch.nn as nn

from models.transattunet import TransAttUNet


def build_model(
    config_or_name: Union[str, Dict[str, Any]] = "transattunet",
    in_channels: int = 3,
    num_classes: int = 1,
    **kwargs,
) -> nn.Module:
    """Instantiate a segmentation model from configuration dictionary or name.

    Args:
        config_or_name: Model name string or dictionary with 'model' config block.
        in_channels: Number of input channels (default 3 for RGB).
        num_classes: Number of output channels (default 1 for binary mask logits).
        kwargs: Additional model arguments.

    Returns:
        Instantiated PyTorch nn.Module.
    """
    efficient_flag_override = kwargs.pop("use_efficient_gsa", None)

    if isinstance(config_or_name, dict):
        model_cfg = config_or_name.get("model", config_or_name)
        name = model_cfg.get("name", "transattunet").lower()
        in_channels = model_cfg.get("in_channels", in_channels)
        num_classes = model_cfg.get("num_classes", num_classes)
        bilinear = model_cfg.get("bilinear", True)
        use_efficient_gsa = bool(model_cfg.get("use_efficient_gsa", False))
        if efficient_flag_override is not None:
            use_efficient_gsa = bool(efficient_flag_override)
    elif isinstance(config_or_name, str):
        name = config_or_name.lower()
        bilinear = kwargs.get("bilinear", True)
        use_efficient_gsa = bool(efficient_flag_override or False)
    else:
        raise TypeError(f"Expected str or dict, got {type(config_or_name)}")

    if name in ("transattunet", "transatt_unet", "baseline", "unet_attention_transformer_multiscale"):
        return TransAttUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            bilinear=bilinear,
            use_efficient_gsa=use_efficient_gsa,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model name: '{name}'. Available: ['transattunet']")
