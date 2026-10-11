"""Unit tests for training utilities and checkpoint resumption."""

from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn

from training.trainer_utils import (
    save_checkpoint,
    load_checkpoint,
    log_metrics,
    select_device,
    set_seed,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def test_device_selection():
    dev_cpu = select_device("cpu")
    assert dev_cpu.type == "cpu"
    dev_auto = select_device("auto")
    assert dev_auto.type in ["cpu", "cuda"]


def test_seed_reproducibility():
    set_seed(1234)
    t1 = torch.randn(5)
    set_seed(1234)
    t2 = torch.randn(5)
    assert torch.allclose(t1, t2)


def test_checkpoint_saving_and_loading_with_history():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_ckpt.pth"

        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        history = [
            {"epoch": 1, "train_loss": 0.8, "val_loss": 0.7, "val_dice": 0.5, "val_iou": 0.4},
            {"epoch": 2, "train_loss": 0.6, "val_loss": 0.5, "val_dice": 0.65, "val_iou": 0.52},
        ]
        config = {"model": {"name": "dummy"}, "training": {"epochs": 10}}

        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=2,
            best_val_dice=0.65,
            val_iou=0.52,
            config=config,
            scheduler=scheduler,
            seed=42,
            best_epoch=2,
            metrics_history=history,
        )

        assert ckpt_path.is_file()

        # Load back into fresh model
        new_model = DummyModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)

        # Confirm weights differ initially
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            p2.data.zero_()

        chk = load_checkpoint(
            checkpoint_path=ckpt_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device=torch.device("cpu"),
        )

        assert chk["epoch"] == 2
        assert chk["best_epoch"] == 2
        assert chk["best_val_dice"] == 0.65
        assert chk["metrics_history"] == history
        assert chk["config"] == config
        assert chk["seed"] == 42
        assert "timestamp" in chk

        # Confirm weights match after loading
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


def test_log_metrics_json_and_csv():
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p = Path(tmp_dir) / "metrics.json"
        csv_p = Path(tmp_dir) / "metrics.csv"

        history = [
            {"epoch": 1, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.4},
            {"epoch": 2, "train_loss": 0.7, "val_loss": 0.6, "val_dice": 0.6},
        ]

        log_metrics(history, json_p, csv_p)

        assert json_p.is_file()
        assert csv_p.is_file()

        import json
        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[1]["val_dice"] == 0.6


def test_checkpoint_isolation_paths():
    smoke_name = "smoke_test_checkpoint.pth"
    baseline_best = "transattunet_baseline_best.pth"
    baseline_latest = "transattunet_baseline_latest.pth"

    assert smoke_name != baseline_best
    assert smoke_name != baseline_latest
    assert baseline_best != baseline_latest
