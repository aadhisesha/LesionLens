"""TransAttUnet: Multi-level Attention-guided U-Net with Transformer for Medical Image Segmentation.

Reference:
- Paper: "TransAttUnet: Multi-level Attention-guided U-Net with Transformer for Medical Image Segmentation"
  Authors: Bingzhi Chen, Yishu Liu, Zheng Zhang, Guangming Lu, David Zhang
  arXiv:2107.05274 (2021)
- Source code: https://github.com/YishuLiu/TransAttUnet

Key Architecture Components:
1. Encoder: Standard U-Net hierarchical convolution blocks.
2. Self-Aware Attention (SAA) Bottleneck:
   - PositionEmbeddingLearned: 2D learned positional encoding.
   - Global Spatial Attention (GSA / PAM_Module): Non-local spatial position attention.
   - Transformer Self-Attention (TSA / ScaledDotProductAttention): Cross-channel self-attention.
3. Multi-scale Skip Connections: Inter-scale feature aggregation across decoder stages.
4. Output: 1x1 convolution producing unscaled logits (no sigmoid applied during training).
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Encoder & Decoder Convolution Blocks
# -------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks: (Conv2d => BatchNorm => ReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool2d(2) followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv, matching the original TransAttUnet decoder."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad to handle any odd dimension mismatch
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_x != 0 or diff_y != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping to desired number of output channels/classes."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# -------------------------------------------------------------------------
# Self-Aware Attention (SAA) Bridge Modules
# -------------------------------------------------------------------------

class PAM_Module(nn.Module):
    """Global Spatial Attention (GSA) module based on Position Attention Module.

    Calculates all-to-all spatial attention across spatial positions (H x W).
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m_batchsize, c, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, c, height, width)

        out = self.gamma * out + x
        return out



class EfficientGlobalContextAttention(nn.Module):
    """Global spatial attention with pooled keys and values."""

    def __init__(self, in_dim: int, context_size: int = 4):
        super().__init__()
        if in_dim < 8 or context_size < 1:
            raise ValueError("in_dim must be >= 8 and context_size >= 1")

        self.context_size = context_size
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape

        context = F.adaptive_avg_pool2d(
            x,
            (min(self.context_size, height), min(self.context_size, width)),
        )

        query = self.query_conv(x).flatten(2).transpose(1, 2)
        key = self.key_conv(context).flatten(2)
        value = self.value_conv(context).flatten(2)

        attention = self.softmax(
            torch.bmm(query, key) * (query.shape[-1] ** -0.5)
        )
        output = torch.bmm(value, attention.transpose(1, 2))
        output = output.reshape(batch, channels, height, width)

        return x + self.gamma * output


class PositionEmbeddingLearned(nn.Module):
    """Learned 2D positional embedding for the bottleneck feature map."""

    def __init__(self, num_pos_feats: int = 256, len_embedding: int = 64):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.len_embedding = len_embedding
        self.row_embed = nn.Embedding(len_embedding, num_pos_feats)
        self.col_embed = nn.Embedding(len_embedding, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h > self.len_embedding or w > self.len_embedding:
            # Dynamically handle larger bottleneck if necessary
            i = torch.linspace(0, self.len_embedding - 1, w, device=x.device).long()
            j = torch.linspace(0, self.len_embedding - 1, h, device=x.device).long()
        else:
            i = torch.arange(w, device=x.device)
            j = torch.arange(h, device=x.device)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return pos


class ScaledDotProductAttention(nn.Module):
    """Transformer Self-Attention (TSA) module operating across channel dimensions."""

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature ** 0.5
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        m_batchsize, d, height, width = x.size()
        q = x.view(m_batchsize, d, -1)
        k = x.view(m_batchsize, d, -1).permute(0, 2, 1)
        v = x.view(m_batchsize, d, -1)

        attn = torch.matmul(q / self.temperature, k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        output = output.view(m_batchsize, d, height, width)
        return output


# -------------------------------------------------------------------------
# TransAttUNet Main Model
# -------------------------------------------------------------------------

class TransAttUNet(nn.Module):
    """Original TransAttUnet Baseline Architecture.

    Incorporates:
    - 4-stage encoder
    - Self-Aware Attention (SAA) bottleneck: Learned Position Embedding + GSA (PAM) + TSA (SDPA)
    - 4-stage decoder with multi-scale skip connections
    - Raw unscaled logits output
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        bilinear: bool = True,
        use_efficient_gsa: bool = False,
        **kwargs,
    ):
        super().__init__()
        # Support alternative argument names if passed
        n_channels = kwargs.get("n_channels", in_channels)
        n_classes = kwargs.get("n_classes", num_classes)

        self.in_channels = n_channels
        self.num_classes = n_classes
        self.bilinear = bilinear
        self.use_efficient_gsa = bool(use_efficient_gsa)

        factor = 2 if bilinear else 1

        # Encoder stages
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)  # 512 channels

        # Decoder stages
        self.up1 = Up(1024, 512 // factor, bilinear)  # In: 1024, Out: 256
        self.up2 = Up(1024, 256 // factor, bilinear)  # In: 1024, Out: 128
        self.up3 = Up(512, 128 // factor, bilinear)   # In: 512,  Out: 64
        self.up4 = Up(256, 64, bilinear)              # In: 256,  Out: 64

        # Output head
        self.outc = OutConv(128, n_classes)

        # Self-Aware Attention (SAA) Bridge
        bottleneck_channels = 1024 // factor
        self.pos = PositionEmbeddingLearned(num_pos_feats=bottleneck_channels // 2, len_embedding=64)
        self.pam = (
            EfficientGlobalContextAttention(bottleneck_channels)
            if self.use_efficient_gsa
            else PAM_Module(bottleneck_channels)
        )
        self.sdpa = ScaledDotProductAttention(bottleneck_channels)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        # Encoder forward
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Self-Aware Attention (SAA) Bridge
        x5_pam = self.pam(x5)
        x5_pos = self.pos(x5)
        x5 = x5 + x5_pos

        x5_sdpa = self.sdpa(x5)
        x5 = x5_sdpa + x5_pam

        # Multi-scale decoder with inter-scale skip connections
        x6 = self.up1(x5, x4)
        x5_scale = F.interpolate(x5, size=x6.shape[2:], mode="bilinear", align_corners=True)
        x6_cat = torch.cat((x5_scale, x6), dim=1)

        x7 = self.up2(x6_cat, x3)
        x6_scale = F.interpolate(x6, size=x7.shape[2:], mode="bilinear", align_corners=True)
        x7_cat = torch.cat((x6_scale, x7), dim=1)

        x8 = self.up3(x7_cat, x2)
        x7_scale = F.interpolate(x7, size=x8.shape[2:], mode="bilinear", align_corners=True)
        x8_cat = torch.cat((x7_scale, x8), dim=1)

        x9 = self.up4(x8_cat, x1)
        x8_scale = F.interpolate(x8, size=x9.shape[2:], mode="bilinear", align_corners=True)
        x9_cat = torch.cat((x8_scale, x9), dim=1)

        logits = self.outc(x9_cat)
        return logits


# Alias matching author's original class name for backward compatibility
UNet_Attention_Transformer_Multiscale = TransAttUNet
