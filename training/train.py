"""Training pipeline for TransAttUNet baseline on ISIC 2016 skin lesion segmentation."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import get_dataloader
from models.model_factory import build_model
from training.losses import CombinedBCEDiceLoss, get_loss_function
from training.trainer_utils import (
    load_checkpoint,
    log_metrics,
    plot_training_curves,
    save_checkpoint,
    select_device,
    set_seed,
)
from evaluation.metrics import dice_coefficient, iou_score


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load training configuration YAML."""
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_batches: Optional[int] = None,
) -> float:
    """Train for a single epoch.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    total_len = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1

        if (i + 1) % max(1, total_len // 5) == 0 or (i + 1) == total_len:
            print(f"  [Epoch {epoch:2d}] Batch {i + 1:3d}/{total_len:3d} | Batch Loss: {loss_val:.4f}")

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate model on validation split.

    Returns:
        dict containing 'loss', 'dice', and 'iou'.
    """
    model.eval()
    total_loss = 0.0
    all_dices = []
    all_ious = []
    num_batches = 0

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(logits)
        # Compute sample-level metrics with reduction='none'
        batch_dice = dice_coefficient(probs, masks, threshold=threshold, reduction="none")
        batch_iou = iou_score(probs, masks, threshold=threshold, reduction="none")

        if isinstance(batch_dice, torch.Tensor):
            all_dices.extend(batch_dice.cpu().tolist())
        else:
            all_dices.extend(list(batch_dice))

        if isinstance(batch_iou, torch.Tensor):
            all_ious.extend(batch_iou.cpu().tolist())
        else:
            all_ious.extend(list(batch_iou))

    avg_loss = total_loss / max(1, num_batches)
    avg_dice = float(sum(all_dices) / max(1, len(all_dices)))
    avg_iou = float(sum(all_ious) / max(1, len(all_ious)))

    return {
        "loss": avg_loss,
        "dice": avg_dice,
        "iou": avg_iou,
    }


def train(
    config: Dict[str, Any],
    epochs: Optional[int] = None,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    device_override: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    seed_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Main training execution function."""
    start_time = time.time()

    # 1. Configuration & Seeding
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    loss_cfg = config.get("loss", {})
    ckpt_cfg = config.get("checkpoint", {})
    log_cfg = config.get("logging", {})

    seed = seed_override if seed_override is not None else train_cfg.get("seed", 42)
    set_seed(seed)

    device_str = device_override or train_cfg.get("device", "auto")
    device = select_device(device_str)
    print(f"\n=======================================================")
    print(f"Starting TransAttUNet Training")
    print(f"=======================================================")
    print(f"Device: {device}")
    print(f"Random seed: {seed}")

    # 2. Dataset loaders
    train_split_json = data_cfg.get("train_split", "data/splits/train.json")
    val_split_json = data_cfg.get("val_split", "data/splits/val.json")
    batch_size = train_cfg.get("batch_size") or data_cfg.get("batch_size", 4)
    num_workers = data_cfg.get("num_workers", 0)

    train_loader = get_dataloader(train_split_json, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = get_dataloader(val_split_json, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Loaded train loader: {len(train_loader.dataset)} samples ({len(train_loader)} batches, bs={batch_size})")
    print(f"Loaded val loader:   {len(val_loader.dataset)} samples ({len(val_loader)} batches, bs={batch_size})")

    # 3. Model construction
    model = build_model(config)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params / 1e6:.2f}M trainable parameters")

    # 4. Loss & Optimizer
    bce_weight = loss_cfg.get("bce_weight", 1.0)
    dice_weight = loss_cfg.get("dice_weight", 1.0)
    criterion = CombinedBCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight).to(device)

    lr = float(train_cfg.get("learning_rate", 0.0001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0001))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_epochs = epochs if epochs is not None else train_cfg.get("epochs", 30)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # 5. Checkpoint paths
    ckpt_dir = Path(ckpt_cfg.get("directory", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    is_smoke = max_train_batches is not None and max_train_batches < len(train_loader)
    best_name = "smoke_test_checkpoint.pth" if is_smoke else ckpt_cfg.get("best_filename", "transattunet_baseline_best.pth")
    latest_name = "smoke_test_latest.pth" if is_smoke else ckpt_cfg.get("latest_filename", "transattunet_baseline_latest.pth")

    best_checkpoint_path = ckpt_dir / best_name
    latest_checkpoint_path = ckpt_dir / latest_name

    json_path = log_cfg.get("metrics_json", "results/logs/train_metrics.json")
    csv_path = log_cfg.get("metrics_csv", "results/logs/train_metrics.csv")
    plot_path = log_cfg.get("plot_path", "results/figures/training_curves.png")

    start_epoch = 1
    best_val_dice = -1.0
    best_val_iou = -1.0
    best_epoch = 0
    metrics_history: List[Dict[str, Any]] = []

    # Resumption
    if resume_checkpoint and Path(resume_checkpoint).is_file():
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        chk = load_checkpoint(
            resume_checkpoint,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            total_epochs=total_epochs,
        )
        completed_epoch = chk.get("epoch", 0)
        start_epoch = completed_epoch + 1
        best_val_dice = float(chk.get("best_val_dice", -1.0))
        # Prefer the explicit best_val_iou field; fall back to val_iou for older checkpoints
        best_val_iou = float(chk.get("best_val_iou", chk.get("val_iou", -1.0)))
        best_epoch = int(chk.get("best_epoch", completed_epoch))

        # Restore metrics history from checkpoint (authoritative source)
        if "metrics_history" in chk and isinstance(chk["metrics_history"], list) and chk["metrics_history"]:
            metrics_history = [dict(m) for m in chk["metrics_history"] if m.get("epoch", 0) <= completed_epoch]
            print(
                f"  [Resume] Restored {len(metrics_history)} epoch records from checkpoint."
            )
        elif Path(json_path).is_file():
            # Fallback: read from JSON log if checkpoint has no history
            print(
                "  [Resume] Checkpoint has no metrics_history; falling back to JSON log."
            )
            try:
                import json as _json
                with open(json_path, "r", encoding="utf-8") as f:
                    past_metrics = _json.load(f)
                    if isinstance(past_metrics, list):
                        metrics_history = [dict(m) for m in past_metrics if m.get("epoch", 0) <= completed_epoch]
            except Exception as hist_err:
                print(f"  [Resume] WARNING: Could not load JSON history: {hist_err}. Starting with empty history.")
                metrics_history = []
        else:
            print(
                "  [Resume] No metrics_history in checkpoint and no JSON log found. "
                "Starting with empty metrics history."
            )
            metrics_history = []

        print(
            f"Resumed from epoch {completed_epoch} -> starting epoch {start_epoch}/{total_epochs} "
            f"(Best Val Dice: {best_val_dice:.4f} at epoch {best_epoch}, Restored {len(metrics_history)} history records)"
        )
        if start_epoch > total_epochs:
            print(
                f"NOTICE: Checkpoint already completed {completed_epoch} epochs, which is >= requested total_epochs ({total_epochs}). "
                f"No additional epochs to run. To train further, specify a higher --epochs."
            )

    # 6. Training loop
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        curr_lr = optimizer.param_groups[0]["lr"]

        print(f"\n--- Epoch {epoch}/{total_epochs} (lr={curr_lr:.6f}) ---")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_batches=max_train_batches,
        )

        val_metrics = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_val_batches,
        )

        scheduler.step()
        epoch_duration = time.time() - epoch_start

        val_loss = val_metrics["loss"]
        val_dice = val_metrics["dice"]
        val_iou = val_metrics["iou"]

        print(
            f"Epoch {epoch:2d} Summary | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f} | Time: {epoch_duration:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "lr": curr_lr,
            "duration_seconds": round(epoch_duration, 2),
        }
        metrics_history.append(record)

        # Check for best validation Dice BEFORE saving so best_val_dice and best_epoch are updated
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            best_val_iou = val_iou
            best_epoch = epoch

        # Save latest checkpoint on every epoch (reflects updated best-metric state)
        save_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_dice=best_val_dice,
            val_iou=val_iou,
            best_val_iou=best_val_iou,
            config=config,
            scheduler=scheduler,
            seed=seed,
            best_epoch=best_epoch,
            metrics_history=metrics_history,
        )

        # Save best checkpoint whenever Dice improved
        if is_best:
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_dice=best_val_dice,
                val_iou=val_iou,
                best_val_iou=best_val_iou,
                config=config,
                scheduler=scheduler,
                seed=seed,
                best_epoch=best_epoch,
                metrics_history=metrics_history,
            )
            print(f"  * New best validation Dice: {best_val_dice:.4f}! Saved to {best_checkpoint_path}")

        # Update log files and plots incrementally on every epoch
        log_metrics(metrics_history, json_path=json_path, csv_path=csv_path)
        plot_training_curves(metrics_history, output_path=plot_path)

    # 7. Final sync of metrics & plot curves
    log_metrics(metrics_history, json_path=json_path, csv_path=csv_path)
    plot_training_curves(metrics_history, output_path=plot_path)

    total_duration = time.time() - start_time

    # 8. Concise Final Summary
    print("\n=======================================================")
    print("TransAttUNet Training Finished")
    print("=======================================================")
    print(f"Selected Checkpoint:       {best_checkpoint_path.resolve()}")
    print(f"Best Epoch:                {best_epoch}")
    print(f"Best Validation Dice:      {best_val_dice:.4f}")
    print(f"Validation IoU (Best Ep):  {best_val_iou:.4f}")
    print(f"Total Training Duration:   {total_duration:.1f}s ({total_duration / 60:.2f} min)")
    print(f"Compute Device Used:       {device}")
    print(f"Epochs Run:                {len(metrics_history)}")
    print(f"=======================================================\n")

    return {
        "best_checkpoint": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "total_duration": total_duration,
        "device": str(device),
        "history": metrics_history,
    }


def main() -> int:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train TransAttUNet Baseline on ISIC 2016")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches per epoch (smoke test)")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit val batches per epoch (smoke test)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Override compute device (auto/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Configuration Error: {e}")
        return 1

    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr

    train(
        config=config,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        device_override=args.device,
        resume_checkpoint=args.resume,
        seed_override=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
