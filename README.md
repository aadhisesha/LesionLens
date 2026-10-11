# Efficient TransAttUNet for Skin Lesion Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python)](https://python.org)
[![License: CC-0](https://img.shields.io/badge/Dataset-ISIC%202016%20CC--0-green.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](#running-automated-tests)

An efficient, transformer-augmented attention U-Net architecture engineered for automated dermoscopic skin lesion boundary segmentation on the ISIC 2016 Task 1 benchmark.

---

## Project Overview

Skin lesion segmentation is a vital prerequisite for computer-aided diagnosis of malignant melanoma. Standard convolutional neural networks (e.g. U-Net) struggle to balance local edge precision with long-range semantic context, while standard vision transformers incur quadratic $\mathcal{O}(N^2)$ computational complexity on high-resolution clinical imagery.

**Efficient TransAttUNet** addresses these bottlenecks through:
1. **Hierarchical Encoder-Decoder**: Residual convolutional blocks capture fine lesion boundaries.
2. **Efficient Guided Self-Attention (GSA)**: Reduces transformer attention complexity to linear/low-rank $\mathcal{O}(N)$ while maintaining global context.
3. **Auxiliary Boundary Head**: Dual-task architecture that explicitly reinforces sharp lesion contours.

---

## Phase 1 Scope & Status

> **Important**: This repository implements **Phase 1: Dataset Acquisition, Preprocessing, Splitting, and Validation**.
> Neural network training and deployment are isolated to subsequent phases.

### Phase 1 Deliverables
- [x] Official ISIC 2016 Task 1 archive acquisition & verification pipeline.
- [x] Strict canonical identifier normalization and image-mask pairing (`ISIC_XXXXXXX`).
- [x] Preprocessing to $256 \times 256$ pixels (Bilinear for RGB images, Nearest-Neighbor for binary masks).
- [x] Reproducible 80/20 train/validation split (720 train, 180 val, Seed 42).
- [x] Independent test split isolation (379 official test pairs).
- [x] PyTorch `ISICDataset` and `DataLoader` batch streaming with data integrity assertions.
- [x] Data-independent segmentation metrics: Dice Similarity Coefficient & IoU (Jaccard Index).
- [x] Automated test suite with fast synthetic in-memory fixtures.
- [x] Automated end-to-end dataset validation script generating machine-readable JSON reports.

---

## Repository Structure

```
efficient-transattunet/
├── README.md                      # Project overview and instructions
├── requirements.txt               # Dependencies
├── .gitignore                     # Git ignore rules
├── configs/
│   ├── data.yaml                  # Dataset, preprocessing, and split configuration
│   ├── baseline.yaml              # Baseline TransAttUNet configuration (Phase 2)
│   ├── efficient.yaml             # Efficient TransAttUNet configuration (Phase 2)
│   └── boundary.yaml              # Boundary-Aware TransAttUNet configuration (Phase 2)
├── data/
│   ├── README.md                  # Detailed ISIC 2016 Task 1 documentation
│   ├── dataset.py                 # PyTorch Dataset & DataLoader
│   ├── preprocessing.py           # Resizing, binarization, and pairing logic
│   ├── archives/                  # Downloaded raw zip archives (gitignored)
│   ├── raw/                       # Extracted images & masks (gitignored)
│   │   ├── train_images/
│   │   ├── train_masks/
│   │   ├── test_images/
│   │   └── test_masks/
│   ├── processed/                 # 256x256 normalized PNG files (gitignored)
│   │   ├── train_images/
│   │   ├── train_masks/
│   │   ├── test_images/
│   │   └── test_masks/
│   └── splits/                    # Reproducible manifest JSONs
│       ├── train.json             # 720 training pairs
│       ├── val.json               # 180 validation pairs
│       └── test.json              # 379 test pairs
├── models/                        # Model architectures (Phase 2 placeholders)
├── training/                      # Training engine and losses (Phase 2 placeholders)
├── evaluation/
│   └── metrics.py                 # Dice and IoU metric functions
├── inference/                     # Inference pipelines (Phase 5 placeholders)
├── app/                           # Web application (Phase 6 placeholders)
├── scripts/
│   ├── prepare_dataset.py         # Main dataset preparation & validation CLI
│   ├── run_baseline.sh            # Baseline training script
│   ├── run_efficient.sh           # Efficient model training script
│   └── run_boundary.sh            # Boundary-aware model training script
├── tests/
│   ├── test_dataset.py            # Unit tests for pairing, preprocessing & dataset loader
│   ├── test_metrics.py            # Unit tests for Dice & IoU metrics
│   ├── test_models.py             # Phase 2 placeholder tests
│   └── test_losses.py             # Phase 2 placeholder tests
├── results/
│   └── metrics/
│       └── validation_report.json # Automated dataset validation report
└── docs/
    ├── architecture.md            # System architecture details
    ├── experiment_plan.md         # Phased experiment roadmap
    └── team_interfaces.md        # API contracts & data schemas
```

---

## Environment Setup

### 1. Clone repository
```bash
git clone <repo-url>
cd LesionLens
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
*Note: For CPU-only PyTorch on Windows:*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Dataset Preparation Commands

The unified entry point is `scripts/prepare_dataset.py`.

### Step 1: Check archives
Inspect whether the four ISIC 2016 Task 1 archives exist in `data/archives/` and verify their zip integrity:
```bash
python scripts/prepare_dataset.py --check-archives
```

### Step 2: Download archives
Automatically download any missing archives directly from the official ISIC AWS S3 challenge repository:
```bash
python scripts/prepare_dataset.py --download
```

### Step 3: Extract archives
Extract raw archives directly into `data/raw/` (stripping nested archive folders and skipping if already extracted):
```bash
python scripts/prepare_dataset.py --extract
```

### Step 4: Pair, Preprocess, and Split
Preprocess images to $256 \times 256$ using bilinear interpolation, masks to $256 \times 256$ using nearest-neighbor interpolation, and generate reproducible train (720), validation (180), and test (379) manifests:
```bash
python scripts/prepare_dataset.py --prepare
```

### Step 5: Validate dataset
Run the complete automated validation checklist and generate `results/metrics/validation_report.json`:
```bash
python scripts/prepare_dataset.py --validate
```

### End-to-End Workflow
Execute all steps sequentially in a single command:
```bash
python scripts/prepare_dataset.py --all
```

---

## Running Automated Tests

Run pytest across all unit tests:
```bash
python -m pytest tests/ -v
```

All dataset and metric unit tests use lightweight synthetic data and execute in seconds without requiring the full ISIC dataset.

---

## Verification & Acceptance Criteria

| Check | Expected | Phase 1 Status |
|---|---|---|
| Training images & masks | 900 pairs | Supported & Verified |
| Test images & masks | 379 pairs | Supported & Verified |
| Train / Validation Split | 720 / 180 (Seed 42) | Reproducible & Disjoint |
| Processed Image Dimensions | $256 \times 256 \times 3$ (Bilinear) | Implemented |
| Processed Mask Dimensions | $256 \times 256 \times 1$ (Nearest-Neighbor, Binary $\{0, 1\}$) | Implemented |
| PyTorch DataLoader Batch Test | Successful tensor shapes | Verified |
| Dice & IoU Unit Tests | All edge cases passing | 100% Passed |
