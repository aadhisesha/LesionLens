# Architecture Specification: Efficient TransAttUNet

## 1. Overview
Efficient TransAttUNet is an advanced medical image segmentation architecture designed for dermoscopic skin lesion boundary delineation (ISIC 2016 Task 1).

The model combines:
1. **Hierarchical Encoder-Decoder Backbone**: Multi-scale feature extraction based on residual convolution blocks.
2. **Transformer Attention Modules**: Captures long-range spatial dependencies across lesion regions.
3. **Efficient Guided Self-Attention (GSA)**: Reduces attention computational complexity from $\mathcal{O}(N^2)$ to linear/low-rank $\mathcal{O}(N)$.
4. **Auxiliary Boundary Head**: Explicitly refines fuzzy, irregular lesion boundaries.

## 2. Directory Layout & Module Responsibilities
- `configs/`: YAML configuration files defining data, model, and training hyperparameters.
- `data/`: Dataset loaders, preprocessing pipelines, and split manifests.
  - `dataset.py`: PyTorch `ISICDataset` yielding `(3, 256, 256)` float32 images and `(1, 256, 256)` binary masks.
  - `preprocessing.py`: Bilinear image resizing, nearest-neighbor mask resizing, binarization.
- `models/`: Neural network architecture implementations (Phase 2+).
- `training/`: Training engine, loss functions, and optimization loops (Phase 2+).
- `evaluation/`: Independent metrics calculation (`Dice`, `IoU`) and benchmarking suites.
- `inference/`: Prediction pipelines and mask overlay visualizers (Phase 5+).
- `app/`: Interactive deployment services (Phase 6).
