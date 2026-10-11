# Architecture Specification: TransAttUNet Baseline (Phase 2)

## 1. Overview & Source Attribution

The baseline model is an adaptation of **TransAttUnet: Multi-level Attention-guided U-Net with Transformer for Medical Image Segmentation** (Chen et al., 2021).

- **Reference Paper**: [arXiv:2107.05274](https://arxiv.org/abs/2107.05274)
- **Reference Code**: [https://github.com/YishuLiu/TransAttUnet](https://github.com/YishuLiu/TransAttUnet)
- **Authors**: Bingzhi Chen, Yishu Liu, Zheng Zhang, Guangming Lu, David Zhang

---

## 2. Core Architecture Components

```
                   Input (B, 3, 256, 256)
                            │
       Encoder 1 (64) ──────┼─────────────────────> Skip to Decoder 4 (up4)
            │ (MaxPool)     │                            ▲
       Encoder 2 (128) ─────┼───────────────> Skip to Decoder 3 (up3)
            │ (MaxPool)     │                      ▲
       Encoder 3 (256) ─────┼─────────> Skip to Decoder 2 (up2)
            │ (MaxPool)     │                ▲
       Encoder 4 (512) ─────┼───> Skip to Decoder 1 (up1)
            │ (MaxPool)     │          ▲
       Bottleneck (512) ────┴──────────┘
            │
    ┌───────┴─────────────────────────────────────────┐
    │ Self-Aware Attention (SAA) Module               │
    │  1. Learned Positional Encoding (x5 + pos)      │
    │  2. Global Spatial Attention GSA (PAM_Module)   │
    │  3. Transformer Self-Attention TSA (SDPA)       │
    │  4. Feature Fusion: SAA = TSA(x5) + GSA(x5)     │
    └───────────────────┬─────────────────────────────┘
                        ▼
                Multi-Scale Decoder
         (Inter-scale upsampling & fusion)
                        │
                        ▼
              OutConv 1x1 (128 -> 1)
                        │
                        ▼
               Raw Logits (B, 1, 256, 256)
```

### A. Encoder Backbone
- 4 downsampling stages with `DoubleConv` (Conv3x3 -> BatchNorm2d -> ReLU -> Conv3x3 -> BatchNorm2d -> ReLU).
- Feature channel progression: $3 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512 \rightarrow 512$ (bottleneck with bilinear `factor=2`).

### B. Self-Aware Attention (SAA) Bottleneck Bridge
The SAA bridge connects the deepest encoder stage to the decoder and models both spatial and channel-wise long-range dependencies:
1. **Learned 2D Positional Embeddings (`PositionEmbeddingLearned`)**:
   - Row and column learned embedding tables ($64 \times 256$ each).
   - Injected additively: $x_5 = x_5 + \text{pos}$.
2. **Global Spatial Attention (`PAM_Module`)**:
   - Calculates all-to-all spatial affinity matrix of shape $(B, N, N)$ where $N = H \times W = 16 \times 16 = 256$.
   - Energy: $E = \text{Softmax}(Q^T K)$, Output: $V \cdot E^T \cdot \gamma + x_5$.
   - **Note for Phase 3**: This $\mathcal{O}(N^2)$ spatial attention matrix is the target for the Efficient GSA replacement.
3. **Transformer Self-Attention (`ScaledDotProductAttention`)**:
   - Cross-channel self-attention operating over channels $d=512$:
     $$\text{Attn} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$
4. **Bridge Summation**:
   - Combines spatial and channel representations: $x_5 = \text{TSA}(x_5) + \text{GSA}(x_5)$.

### C. Multi-Scale Decoder with Inter-Scale Skips
Instead of isolated single-level skip connections, the decoder progressively aggregates multi-scale semantic representations:
- Stage 1: $x_6 = \text{Up}_1(x_5, x_4)$, then concatenates with upsampled $x_5$ ($x_{6\text{cat}} = [x_5^{\uparrow}, x_6]$, 768 channels).
- Stage 2: $x_7 = \text{Up}_2(x_{6\text{cat}}, x_3)$, then concatenates with upsampled $x_6$ ($x_{7\text{cat}} = [x_6^{\uparrow}, x_7]$, 384 channels).
- Stage 3: $x_8 = \text{Up}_3(x_{7\text{cat}}, x_2)$, then concatenates with upsampled $x_7$ ($x_{8\text{cat}} = [x_7^{\uparrow}, x_8]$, 192 channels).
- Stage 4: $x_9 = \text{Up}_4(x_{8\text{cat}}, x_1)$, then concatenates with upsampled $x_8$ ($x_{9\text{cat}} = [x_8^{\uparrow}, x_9]$, 128 channels).
- Output: $1 \times 1$ conv maps $128 \rightarrow 1$ raw logits.

---

## 3. Adaptations from Original Implementation

| Component | Author's Original Code | LesionLens Phase 2 Implementation | Rationale |
|---|---|---|---|
| **Input Channels** | Configurable via `n_channels` | Default `in_channels=3` (RGB) | ISIC 2016 dermoscopy images are 3-channel RGB. |
| **Output Classes** | Configurable via `n_classes` | Default `num_classes=1` | Binary lesion segmentation uses single output channel. |
| **Output Activation** | None (logits) | Raw unscaled logits | Numerically stable with PyTorch `BCEWithLogitsLoss`. |
| **Position Embedding** | Fixed `len_embedding=32` | Configurable `len_embedding=64` with dynamic coordinate fallback | Supports diverse input resolutions up to $1024 \times 1024$ without embedding index out-of-bounds errors. |
| **Constructor API** | `UNet_Attention_Transformer_Multiscale(n_channels, n_classes)` | `TransAttUNet(in_channels=3, num_classes=1, bilinear=True)` (with backward-compatible alias) | Conforms strictly to standard modular project specification. |
| **Model Factory** | Not present | `models/model_factory.py: build_model(config)` | Facilitates plug-and-play model swapping for Phase 3 and Phase 4. |
