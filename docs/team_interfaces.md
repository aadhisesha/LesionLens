# Team Interfaces & Data Contracts

This document formalizes the input/output contracts between components for team collaboration across project phases.

---

## 1. Data Contracts (Phase 1)

### Split Manifest Contract (`data/splits/{train,val,test}.json`)
```json
{
  "metadata": {
    "split": "train",
    "dataset_name": "ISIC Challenge",
    "year": 2016,
    "task": "Task 1: Lesion Boundary Segmentation",
    "num_samples": 720,
    "random_seed": 42,
    "created_at": "ISO-8601 string",
    "preprocessing": {
      "target_height": 256,
      "target_width": 256,
      "image_interpolation": "bilinear",
      "mask_interpolation": "nearest"
    }
  },
  "samples": [
    {
      "id": "ISIC_0000000",
      "image_path": "data/processed/train_images/ISIC_0000000.png",
      "mask_path": "data/processed/train_masks/ISIC_0000000_Segmentation.png"
    }
  ]
}
```

### PyTorch Dataset Output Contract (`ISICDataset.__getitem__`)
```python
{
    "image": torch.Tensor,  # shape: (3, 256, 256), dtype: torch.float32, range: [0.0, 1.0]
    "mask": torch.Tensor,   # shape: (1, 256, 256), dtype: torch.float32, values: {0.0, 1.0}
    "id": str               # canonical identifier e.g. "ISIC_0000000"
}
```

---

## 2. Evaluation Metric Contracts (`evaluation/metrics.py`)

- `dice_coefficient(pred, target, threshold=0.5, eps=1e-7, reduction='mean') -> float | torch.Tensor`
- `iou_score(pred, target, threshold=0.5, eps=1e-7, reduction='mean') -> float | torch.Tensor`
- `evaluate_segmentation(pred, target, threshold=0.5) -> dict[str, float]`
  - Returns `{"dice": float, "iou": float}`
