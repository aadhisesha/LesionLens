"""Cloud GPU training helper for LesionLens TransAttUNet baseline.

Provides dedicated entry points for:
1. Verifying CUDA hardware and environment
2. Running a calibrated GPU smoke test and measuring throughput
3. Running a 1-epoch calibration pass on the full dataset
4. Resuming and completing the 30-epoch baseline run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train import train, load_config
from training.trainer_utils import select_device


def verify_gpu() -> Dict[str, Any]:
    """Verify CUDA GPU availability, device name, and memory."""
    print("\n" + "=" * 60)
    print("LESIONLENS — GPU ENVIRONMENT VERIFICATION")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"PyTorch Version:         {torch.__version__}")
    print(f"CUDA Available:          {cuda_available}")

    if not cuda_available:
        print("\nWARNING: No CUDA-capable GPU detected by PyTorch!")
        print("Running on CPU. Ensure CUDA drivers and GPU runtime are enabled.")
        return {"cuda_available": False, "device": "cpu"}

    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    device_count = torch.cuda.device_count()
    capability = torch.cuda.get_device_capability(current_device)
    total_mem_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)

    print(f"GPU Device Count:        {device_count}")
    print(f"Primary GPU Name:        {device_name}")
    print(f"Compute Capability:      {capability[0]}.{capability[1]}")
    print(f"Total GPU Memory:        {total_mem_gb:.2f} GB")
    print(f"cuDNN Enabled:           {torch.backends.cudnn.enabled}")
    print(f"cuDNN Version:           {torch.backends.cudnn.version()}")
    print("=" * 60 + "\n")

    return {
        "cuda_available": True,
        "device_name": device_name,
        "device_count": device_count,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_memory_gb": round(total_mem_gb, 2),
    }


def run_gpu_smoke_test(config_path: str = "configs/baseline.yaml", batch_size: int = 4) -> Dict[str, Any]:
    """Run a 2-batch smoke test on GPU and report throughput."""
    print("\n" + "=" * 60)
    print("RUNNING GPU SMOKE TEST & THROUGHPUT BENCHMARK")
    print("=" * 60)

    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    results = train(
        config=config,
        epochs=1,
        max_train_batches=2,
        max_val_batches=2,
        device_override=device,
    )
    total_time = time.time() - t0

    # 2 batches * batch_size images
    total_images = 2 * batch_size
    throughput = total_images / max(1e-6, total_time)

    peak_mem_mb = 0.0
    if device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("\n--- GPU SMOKE TEST RESULTS ---")
    print(f"Device Used:             {device}")
    print(f"Total Wall Time:         {total_time:.2f}s")
    print(f"Batches Processed:       2 train + 2 val (bs={batch_size})")
    print(f"Training Throughput:     {throughput:.2f} images/sec")
    print(f"Average Batch Time:      {(total_time / 4) * 1000:.1f} ms/batch")
    if device == "cuda":
        print(f"Peak GPU VRAM:           {peak_mem_mb:.1f} MB")
    print(f"Smoke Checkpoint:        checkpoints/smoke_test_checkpoint.pth")
    print("=" * 60 + "\n")

    return {
        "total_time": total_time,
        "throughput_img_per_sec": throughput,
        "peak_vram_mb": peak_mem_mb,
    }


def run_epoch_calibration(config_path: str = "configs/baseline.yaml", batch_size: int = 4) -> Dict[str, Any]:
    """Run 1 full training epoch on the full dataset (720 train + 180 val)."""
    print("\n" + "=" * 60)
    print("RUNNING 1-EPOCH BASELINE CALIBRATION (720 Train / 180 Val)")
    print("=" * 60)

    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    results = train(
        config=config,
        epochs=1,
        max_train_batches=None,
        max_val_batches=None,
        device_override=device,
    )
    epoch_duration = time.time() - t0
    estimated_30_epochs = (epoch_duration * 30) / 60

    print("\n--- EPOCH 1 CALIBRATION SUMMARY ---")
    print(f"Epoch 1 Duration:        {epoch_duration:.1f}s ({epoch_duration / 60:.2f} min)")
    print(f"Best Validation Dice:    {results['best_val_dice']:.4f}")
    print(f"Validation IoU:          {results['best_val_iou']:.4f}")
    print(f"Best Checkpoint:         {results['best_checkpoint']}")
    print(f"Estimated 30 Epochs:     {estimated_30_epochs:.1f} minutes")
    print("=" * 60 + "\n")

    return {
        "epoch1_duration": epoch_duration,
        "best_val_dice": results["best_val_dice"],
        "best_val_iou": results["best_val_iou"],
        "estimated_30_epochs_min": estimated_30_epochs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cloud GPU Training Helper for LesionLens")
    parser.add_argument("--verify-gpu", action="store_true", help="Verify CUDA GPU hardware")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-batch GPU smoke test")
    parser.add_argument("--calibrate-epoch", action="store_true", help="Run 1 full epoch on all 720/180 samples")
    parser.add_argument("--full-run", action="store_true", help="Run full 30 epochs (resuming if checkpoint exists)")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")

    args = parser.parse_args()

    if args.verify_gpu:
        verify_gpu()
        return 0

    if args.smoke_test:
        verify_gpu()
        run_gpu_smoke_test(config_path=args.config, batch_size=args.batch_size)
        return 0

    if args.calibrate-epoch if hasattr(args, "calibrate-epoch") else False or args.calibrate_epoch:
        verify_gpu()
        run_epoch_calibration(config_path=args.config, batch_size=args.batch_size)
        return 0

    if args.full_run:
        verify_gpu()
        config = load_config(args.config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        resume_p = args.resume or "checkpoints/transattunet_baseline_latest.pth"
        if not Path(resume_p).is_file():
            resume_p = None
        train(config=config, epochs=30, device_override=device, resume_checkpoint=resume_p)
        return 0

    # Default action if no flags provided: verify GPU
    verify_gpu()
    return 0


if __name__ == "__main__":
    sys.exit(main())
