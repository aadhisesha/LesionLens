# ISIC 2016 Task 1 Dataset: Lesion Boundary Segmentation

This directory manages the acquisition, raw storage, preprocessing, and reproducible manifest generation for the **ISIC 2016 Challenge Task 1 (Lesion Boundary Segmentation)**.

---

## 1. Official Dataset Source

- **Challenge Portal**: [https://challenge.isic-archive.com/data/#2016](https://challenge.isic-archive.com/data/#2016)
- **Year & Task**: 2016 Challenge, **Task 1: Lesion Boundary Segmentation** (First row in the challenge table)
- **License**: Creative Commons CC-0 (Public Domain Dedication)

### The Four Required Archives
| Split / Content | Archive Filename | Approx. Size | Expected Samples | Format | Official Direct Download URL |
|---|---|---|---|---|---|
| **Training Data** | `ISBI2016_ISIC_Part1_Training_Data.zip` | 602 MB | 900 | JPEG (`.jpg`) | [Download Training Images](https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Training_Data.zip) |
| **Training Ground Truth** | `ISBI2016_ISIC_Part1_Training_GroundTruth.zip` | 6 MB | 900 | PNG (`.png`) | [Download Training Masks](https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip) |
| **Test Data** | `ISBI2016_ISIC_Part1_Test_Data.zip` | 232 MB | 379 | JPEG (`.jpg`) | [Download Test Images](https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Test_Data.zip) |
| **Test Ground Truth** | `ISBI2016_ISIC_Part1_Test_GroundTruth.zip` | 2 MB | 379 | PNG (`.png`) | [Download Test Masks](https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip) |

> **Important**: Do not confuse Task 1 with Task 2 (superpixels) or Task 3 (classification), or with datasets from other challenge years (2017–2024).

---

## 2. Directory Layout

The `data/` folder is organized into clean functional tiers:

```
data/
├── README.md               # This documentation
├── dataset.py              # PyTorch Dataset and DataLoader implementation
├── preprocessing.py        # Image/mask preprocessing and pairing logic
├── archives/               # Storage for downloaded zip archives (gitignored)
│   ├── ISBI2016_ISIC_Part1_Training_Data.zip
│   ├── ISBI2016_ISIC_Part1_Training_GroundTruth.zip
│   ├── ISBI2016_ISIC_Part1_Test_Data.zip
│   └── ISBI2016_ISIC_Part1_Test_GroundTruth.zip
├── raw/                    # Extracted raw files preserving original dimensions (gitignored)
│   ├── train_images/       # 900 raw JPEG files (ISIC_XXXXXXX.jpg)
│   ├── train_masks/        # 900 raw PNG ground truth masks (ISIC_XXXXXXX_Segmentation.png)
│   ├── test_images/        # 379 raw JPEG files (ISIC_XXXXXXX.jpg)
│   └── test_masks/         # 379 raw PNG ground truth masks (ISIC_XXXXXXX_Segmentation.png)
├── processed/              # Normalized, 256x256 resolution assets (gitignored)
│   ├── train_images/       # 900 preprocessed PNG files (256x256 RGB)
│   ├── train_masks/        # 900 preprocessed PNG binary masks (256x256, values 0 & 255)
│   ├── test_images/        # 379 preprocessed PNG files (256x256 RGB)
│   └── test_masks/         # 379 preprocessed PNG binary masks (256x256, values 0 & 255)
└── splits/                 # Version-controlled split manifests
    ├── train.json          # 720 training pairs
    ├── val.json            # 180 validation pairs
    └── test.json           # 379 test pairs
```

---

## 3. Image-Mask Pairing Rules

Images and masks are paired **strictly by sample identifier**, never by arbitrary alphabetical alignment:
1. **Identifier Extraction**:
   - `ISIC_0000000.jpg` $\rightarrow$ `ISIC_0000000`
   - `ISIC_0000000_Segmentation.png` $\rightarrow$ `ISIC_0000000`
2. **Strict Validation**:
   - Filenames must match pattern `ISIC_\d{7}`.
   - Suffixes `_Segmentation` and `_segmentation` are stripped for ground truth masks.
   - Duplicate identifiers within any set raise an immediate `ValueError`.
   - Any missing image or missing mask raises an explicit pairing exception halts the workflow.

---

## 4. Preprocessing Specifications

- **Target Dimensions**: $256 \times 256$ pixels (configurable in [configs/data.yaml](file:///d:/Study%20materials/sem7/imageprocessing/project/LesionLens/configs/data.yaml)).
- **RGB Images**:
  - Loaded safely via Pillow and converted to `RGB`.
  - Resized using **bilinear interpolation** (`Image.Resampling.BILINEAR`).
  - Saved losslessly as PNG in `data/processed/` preserving sample IDs.
  - Normalized to $[0.0, 1.0]$ `float32` tensors when loaded via `ISICDataset`.
- **Segmentation Masks**:
  - Loaded in single-channel grayscale (`L`).
  - Resized using **nearest-neighbor interpolation ONLY** (`Image.Resampling.NEAREST`) to prevent synthetic boundary values.
  - Binarized with threshold 128: pixels $\ge 128 \rightarrow 1$, else $0$.
  - Stored as 8-bit PNG (values 0 and 255) for visualization, converted to $\{0.0, 1.0\}$ `float32` tensors upon loading.
  - Strict validation checks ensure no values outside $\{0, 1\}$ exist.

---

## 5. Dataset Splitting Strategy

- **Train/Val Split**:
  - Derived strictly from the **900 training pairs** (80% train / 20% validation).
  - **720 pairs** for training (`train.json`).
  - **180 pairs** for validation (`val.json`).
  - Deterministic random seed: **42** (with identifiers sorted before shuffling).
  - Train and validation sets are strictly disjoint ($Train \cap Val = \emptyset$).
  - Union contains all 900 samples ($Train \cup Val = All\ 900$).
- **Test Set Isolation**:
  - The **379 official test pairs** are kept in `test.json`.
  - Zero test samples or masks are used in model training, selection, or threshold tuning.

---

## 6. Execution Commands

From the repository root:

```bash
# 1. Check archive status
python scripts/prepare_dataset.py --check-archives

# 2. Download missing archives (automatic via official URLs)
python scripts/prepare_dataset.py --download

# 3. Extract archives to data/raw/
python scripts/prepare_dataset.py --extract

# 4. Pair, preprocess to 256x256, and generate split manifests
python scripts/prepare_dataset.py --prepare

# 5. Run complete validation suite and generate results/metrics/validation_report.json
python scripts/prepare_dataset.py --validate

# Or run the entire pipeline end-to-end:
python scripts/prepare_dataset.py --all
```
