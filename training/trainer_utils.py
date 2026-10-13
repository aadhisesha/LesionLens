"""Trainer utilities: reproducibility seeding, device selection, checkpointing, and metrics plotting."""

from __future__ import annotations

import csv
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(preference: str = "auto") -> torch.device:
    """Determine the optimal compute device.

    Args:
        preference: 'auto', 'cuda', 'cpu', or specific cuda device e.g. 'cuda:0'.

    Returns:
        torch.device instance.
    """
    pref = preference.lower().strip()
    if pref == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    elif pref.startswith("cuda"):
        if torch.cuda.is_available():
            device_str = pref
        else:
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            device_str = "cpu"
    else:
        device_str = "cpu"

    device = torch.device(device_str)
    return device


def get_scheduler_t_max(scheduler: Any) -> Optional[int]:
    """Extract T_max from a CosineAnnealingLR scheduler, if available.

    Args:
        scheduler: A learning-rate scheduler instance.

    Returns:
        T_max integer or None if not applicable / not present.
    """
    return getattr(scheduler, "T_max", None)


def check_scheduler_compatibility(
    scheduler: Any,
    checkpoint_scheduler_state: Optional[Dict[str, Any]],
    total_epochs: int,
) -> Tuple[bool, str]:
    """Verify that a saved scheduler state is compatible with the current scheduler.

    A CosineAnnealingLR state is considered *incompatible* when the T_max stored
    in the checkpoint differs from the T_max of the freshly constructed scheduler.
    Loading such a state would silently corrupt the learning-rate schedule for the
    entire formal run.

    For example, a T_max=1 calibration checkpoint loaded into a T_max=30 formal run
    would restart the cosine cycle after every step, keeping the LR near eta_min.

    Args:
        scheduler: Freshly constructed scheduler for the current run.
        checkpoint_scheduler_state: scheduler_state_dict loaded from a checkpoint,
            or None if the checkpoint does not contain one.
        total_epochs: Intended total number of training epochs for this run.

    Returns:
        Tuple of (is_compatible, reason).  is_compatible is True when it is safe
        to call scheduler.load_state_dict(checkpoint_scheduler_state).
    """
    if checkpoint_scheduler_state is None:
        return True, "No scheduler state in checkpoint; skipping restore."

    current_t_max = get_scheduler_t_max(scheduler)
    if current_t_max is None:
        return True, "Scheduler does not have T_max; compatibility check skipped."

    saved_t_max = checkpoint_scheduler_state.get("T_max")
    if saved_t_max is None:
        msg = (
            f"Checkpoint scheduler state has no T_max field; "
            f"allowing restore (current T_max={current_t_max})."
        )
        return True, msg

    if int(saved_t_max) != int(current_t_max):
        msg = (
            f"Scheduler T_max mismatch: checkpoint has T_max={saved_t_max}, "
            f"but current run uses T_max={current_t_max} (total_epochs={total_epochs}). "
            f"Loading this state would corrupt the cosine LR schedule. "
            f"Starting with a fresh scheduler state for this run."
        )
        return False, msg

    return True, f"Scheduler T_max={current_t_max} matches checkpoint; state restored."


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_dice: float,
    val_iou: float,
    config: Dict[str, Any],
    scheduler: Optional[Any] = None,
    seed: int = 42,
    best_epoch: Optional[int] = None,
    best_val_iou: Optional[float] = None,
    metrics_history: Optional[List[Dict[str, Any]]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save full training checkpoint for reproduction and resumption.

    Args:
        checkpoint_path: Target path for the checkpoint file.
        model: Trained model instance.
        optimizer: Optimizer instance.
        epoch: Current epoch index (last completed epoch).
        best_val_dice: Best validation Dice score achieved so far.
        val_iou: Current epoch validation IoU (kept for backwards compatibility).
        config: Full configuration dictionary.
        scheduler: Optional learning rate scheduler.
        seed: Random seed used for training.
        best_epoch: Optional best epoch index.
        best_val_iou: IoU corresponding to the best-Dice epoch. Stored
            separately from val_iou so that resumption restores the correct
            best-model IoU rather than the most-recent epoch IoU.
        metrics_history: Optional history list of per-epoch metrics.
        extra_metadata: Optional additional metadata dict.

    Returns:
        Resolved Path of the saved checkpoint.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DataParallel if present
    model_to_save = model.module if hasattr(model, "module") else model

    state = {
        "epoch": epoch,
        "best_epoch": int(best_epoch) if best_epoch is not None else epoch,
        "best_val_dice": float(best_val_dice),
        # val_iou: current epoch IoU (kept for backwards compatibility on load)
        "val_iou": float(val_iou),
        # best_val_iou: IoU at the best-Dice epoch (distinct when not the best epoch)
        "best_val_iou": float(best_val_iou) if best_val_iou is not None else float(val_iou),
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics_history": list(metrics_history) if metrics_history is not None else [],
        "config": config,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra_metadata:
        state["extra_metadata"] = extra_metadata

    torch.save(state, path)
    return path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    total_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Load model weights and training state from checkpoint.

    Scheduler state is restored only when the saved T_max is compatible with the
    freshly constructed scheduler.  An incompatible state (e.g. T_max=1 from a
    1-epoch calibration run loaded into a 30-epoch formal run) is rejected with
    a printed warning -- the scheduler keeps its fresh state so the cosine LR
    schedule is correct for the intended total_epochs.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        model: Model instance to load weights into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.
        device: Device to map weights onto.
        total_epochs: Intended total epochs for the current run, used for
            scheduler compatibility checking.  When None, the scheduler's
            own T_max attribute is used instead.

    Returns:
        Checkpoint dictionary with metadata.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    map_location = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Validate scheduler T_max before restoring state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        ckpt_sched_state = checkpoint["scheduler_state_dict"]
        t_max_for_check = total_epochs if total_epochs is not None else get_scheduler_t_max(scheduler)
        is_compat, reason = check_scheduler_compatibility(
            scheduler, ckpt_sched_state, t_max_for_check or 0
        )
        if is_compat and ckpt_sched_state is not None:
            scheduler.load_state_dict(ckpt_sched_state)
            print(f"  [Checkpoint] Scheduler state restored. {reason}")
        else:
            warnings.warn(
                f"[Checkpoint] Scheduler state NOT restored. {reason}",
                UserWarning,
                stacklevel=2,
            )
            print(f"  [Checkpoint] WARNING - Scheduler state skipped. {reason}")

    return checkpoint


def log_metrics(
    metrics_history: List[Dict[str, Any]],
    json_path: Union[str, Path],
    csv_path: Union[str, Path],
) -> None:
    """Save epoch-level metrics in JSON and CSV formats.

    The full metrics_history list is written atomically on each call.  Because
    the list is always built by appending new epoch records to a prior history
    restored from checkpoint (see train.py resumption logic), writing the full
    list is equivalent to appending -- no previously completed epoch records
    are discarded on resume.

    Args:
        metrics_history: List of per-epoch metric dicts in chronological order.
        json_path: Output path for JSON metrics file.
        csv_path: Output path for CSV metrics file.
    """
    json_p = Path(json_path)
    csv_p = Path(csv_path)
    json_p.parent.mkdir(parents=True, exist_ok=True)
    csv_p.parent.mkdir(parents=True, exist_ok=True)

    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)

    if metrics_history:
        fieldnames = list(metrics_history[0].keys())
        with open(csv_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_history)


def plot_training_curves(
    metrics_history: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> Optional[Path]:
    """Generate and save loss and validation Dice plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping curve plotting.")
        return None

    if not metrics_history:
        return None

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    epochs = [m["epoch"] for m in metrics_history]
    train_losses = [m["train_loss"] for m in metrics_history]
    val_losses = [m.get("val_loss", 0.0) for m in metrics_history]
    val_dices = [m.get("val_dice", 0.0) for m in metrics_history]
    val_ious = [m.get("val_iou", 0.0) for m in metrics_history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    axes[0].plot(epochs, train_losses, "b-o", label="Train Loss", linewidth=1.5, markersize=4)
    axes[0].plot(epochs, val_losses, "r-s", label="Val Loss", linewidth=1.5, markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Metric plot
    axes[1].plot(epochs, val_dices, "g-^", label="Val Dice", linewidth=1.5, markersize=4)
    axes[1].plot(epochs, val_ious, "m-d", label="Val IoU", linewidth=1.5, markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Segmentation Metrics")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_p, dpi=150)
    plt.close(fig)
    return out_p
