# LesionLens: Efficient TransAttUNet for Skin Lesion Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python)](https://python.org)
[![License: CC-0](https://img.shields.io/badge/Dataset-ISIC%202016%20CC--0-green.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Tests](https://img.shields.io/badge/Tests-31%20Passed-brightgreen.svg)](#running-automated-tests)

An efficient, transformer-augmented attention U-Net architecture engineered for automated dermoscopic skin lesion boundary segmentation on the ISIC 2016 Task 1 benchmark.

---

## Project Overview

Skin lesion segmentation is a vital prerequisite for computer-aided diagnosis of malignant melanoma. Standard convolutional neural networks (e.g. U-Net) struggle to balance local edge precision with long-range semantic context, while standard vision transformers incur quadratic $\mathcal{O}(N^2)$ computational complexity on high-resolution clinical imagery.

**Project Roadmap**:
1. **Phase 1 (Completed)**: Dataset acquisition, 256×256 preprocessing, and reproducible 80/20 train/validation splitting.
2. **Phase 2 (Completed — Current Baseline)**: Original TransAttUNet model, combined BCE + Soft Dice loss, training/validation loops, checkpointing, and inference.
3. **Phase 3 (Next)**: Efficient Guided Self-Attention (GSA) replacement to achieve $\mathcal{O}(N)$ complexity.
4. **Phase 4**: Auxiliary boundary refinement head.
5. **Phase 5**: Comparative benchmarking on the isolated 379-image official test set.
6. **Phase 6**: Clinical interactive web frontend.

---

## Phase 2: TransAttUNet Baseline Architecture

The baseline model implements the architecture from:
> **TransAttUnet: Multi-level Attention-guided U-Net with Transformer for Medical Image Segmentation**  
> *Bingzhi Chen, Yishu Liu, Zheng Zhang, Guangming Lu, David Zhang*  
> Reference Paper: [arXiv:2107.05274](https://arxiv.org/abs/2107.05274) | Source Code: [https://github.com/YishuLiu/TransAttUnet](https://github.com/YishuLiu/TransAttUnet)

### Architecture Highlights
- **Encoder**: 4 downsampling stages with `DoubleConv` (Conv-BN-ReLU).
- **Self-Aware Attention (SAA) Bridge**:
  - **Learned 2D Position Embedding**: Additive coordinate embeddings for bottleneck features.
  - **Global Spatial Attention (GSA / `PAM_Module`)**: $\mathcal{O}(N^2)$ all-to-all spatial attention across bottleneck feature positions.
  - **Transformer Self-Attention (TSA / `ScaledDotProductAttention`)**: Cross-channel self-attention with scaled dot-product.
  - **Fusion**: $x_5 = \text{TSA}(x_5) + \text{GSA}(x_5)$.
- **Multi-Scale Decoder**: Inter-scale progressive concatenation aggregating upsampled deep features with shallower features before upsampling.
- **Output**: Single-channel raw logits (no sigmoid applied during training for numerical stability).

---

## Environment Setup

### 1. Requirements
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```
For CPU-only PyTorch on Windows:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Dataset Pipeline (Phase 1)

The dataset uses the official **ISIC 2016 Task 1 (Lesion Boundary Segmentation)** benchmark:
- 900 training image-mask pairs (720 train / 180 validation split, Seed 42).
- 379 official test image-mask pairs (kept isolated for Phase 5 benchmark).
- Target dimensions: $256 \times 256$ pixels.

```bash
# Check archive availability
python scripts/prepare_dataset.py --check-archives

# Preprocess and generate splits
python scripts/prepare_dataset.py --prepare

# Validate dataset integrity
python scripts/prepare_dataset.py --validate
```

---

## Phase 2 Execution Commands

### 1. Automated Testing
Run the complete automated test suite (31 unit and integration tests covering model architecture, loss functions, dataset loaders, and metrics):
```bash
# Run model unit tests
python -m pytest tests/test_models.py -v

# Run loss unit tests
python -m pytest tests/test_losses.py -v

# Run all test suites
python -m pytest tests/ -v
```

### 2. Training the TransAttUNet Baseline

#### A. One-Epoch Smoke Test (Fast CPU verification)
Runs 2 training batches and 2 validation batches, saving a dedicated smoke checkpoint without overwriting production checkpoints:
```bash
python training/train.py --config configs/baseline.yaml --epochs 1 --max-train-batches 2 --max-val-batches 2
```

#### B. Full Baseline Training
Runs complete 30-epoch training on available hardware (automatically selects CUDA if available, otherwise CPU):
```bash
python training/train.py --config configs/baseline.yaml
```

**Training Options**:
- `--config`: Path to YAML configuration (default: `configs/baseline.yaml`).
- `--epochs`: Override number of epochs.
- `--batch-size`: Override batch size (default: 4).
- `--lr`: Override learning rate (default: 0.0001).
- `--device`: Compute device (`auto`, `cuda`, or `cpu`).
- `--resume`: Path to checkpoint to resume training.

**Checkpoints Saved**:
- Best validation model: `checkpoints/transattunet_baseline_best.pth`
- Latest epoch model: `checkpoints/transattunet_baseline_latest.pth`
- Smoke test model: `checkpoints/smoke_test_checkpoint.pth`

### 3. Inference on Single Images
Run inference on an arbitrary lesion image to generate a predicted lesion probability map and binary mask:
```bash
python inference/predict.py --checkpoint checkpoints/smoke_test_checkpoint.pth --image data/raw/train_images/ISIC_0000000.jpg --output results/predictions/prediction_sample.png
```
Visual results (Input Image, Probability Map, Binary Mask) are saved to `results/predictions/`.

---

## Verification & Acceptance Criteria

| Requirement | Implementation Status | Verified Output |
|---|---|---|
| Adapted TransAttUNet Model | `models/transattunet.py` | Produces `(B, 1, 256, 256)` raw logits from `(B, 3, 256, 256)` RGB input |
| SAA Bridge (GSA + TSA + Pos) | `models/transattunet.py` | Verified in forward and backward passes |
| Combined BCE + Soft Dice Loss | `training/losses.py` | Stable gradient propagation, handles empty masks |
| Training Engine | `training/train.py` | Epoch loop, validation loop, AdamW optimizer, Cosine Annealing |
| Checkpointing | `training/trainer_utils.py` | State dict, optimizer state, metrics, resume support |
| Metrics Integration | `evaluation/metrics.py` | Dice coefficient & IoU with explicit thresholding |
| Single-image Inference | `inference/predict.py` | Generates probability maps, binary masks, and visualizations |
| Automated Test Coverage | `tests/` | **31 passed, 0 skipped, 0 failed** in 21s |
| Git Hygiene | `.gitignore` | Checkpoints, raw images, and prediction masks properly ignored |

---

## Hardware Notes & Phase 3 Outlook

- The baseline model comprises **22.68M trainable parameters**.
- For local HP laptops (8 GB RAM, Intel i3/i5 CPUs), run unit tests and short smoke tests locally.
- For full 30-epoch baseline convergence, execute on a CUDA-enabled GPU (e.g. Google Colab / cloud instance).
- **Phase 3 Goal**: Replace the $\mathcal{O}(N^2)$ Global Spatial Attention (`PAM_Module`) with an Efficient GSA module to significantly reduce latency and FLOPs while maintaining segmentation accuracy.
