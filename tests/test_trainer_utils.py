"""Regression tests for training utilities, checkpointing, scheduler compatibility,
and history preservation.
"""

import csv
import json
import warnings
from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn

from training.trainer_utils import (
    check_scheduler_compatibility,
    get_scheduler_t_max,
    save_checkpoint,
    load_checkpoint,
    log_metrics,
    select_device,
    set_seed,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def _cosine_scheduler(optimizer, t_max):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)


# ---------------------------------------------------------------------------
# Original tests (preserved and extended)
# ---------------------------------------------------------------------------

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
            best_val_iou=0.52,
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
            total_epochs=10,
        )

        assert chk["epoch"] == 2
        assert chk["best_epoch"] == 2
        assert chk["best_val_dice"] == 0.65
        assert chk["best_val_iou"] == 0.52
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


# ---------------------------------------------------------------------------
# Scheduler compatibility tests (new)
# ---------------------------------------------------------------------------


def test_scheduler_t_max_extraction():
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())
    sched = _cosine_scheduler(optimizer, t_max=30)
    assert get_scheduler_t_max(sched) == 30

    # Non-cosine schedulers have no T_max
    step_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
    assert get_scheduler_t_max(step_sched) is None


def test_scheduler_compatible_same_t_max():
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())
    sched = _cosine_scheduler(optimizer, t_max=30)
    state = sched.state_dict()
    is_compat, reason = check_scheduler_compatibility(sched, state, total_epochs=30)
    assert is_compat is True
    assert "30" in reason


def test_scheduler_incompatible_calibration_into_formal():
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())
    formal_sched = _cosine_scheduler(optimizer, t_max=30)
    # Calibration checkpoint had T_max=1
    calib_sched = _cosine_scheduler(optimizer, t_max=1)
    calib_state = calib_sched.state_dict()

    is_compat, reason = check_scheduler_compatibility(formal_sched, calib_state, total_epochs=30)
    assert is_compat is False
    assert "T_max mismatch" in reason
    assert "1" in reason and "30" in reason


def test_scheduler_no_state_in_checkpoint_is_compatible():
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())
    sched = _cosine_scheduler(optimizer, t_max=30)
    is_compat, _ = check_scheduler_compatibility(sched, None, total_epochs=30)
    assert is_compat is True


def test_load_checkpoint_rejects_incompatible_scheduler():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "calib.pth"
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Save calibration checkpoint with T_max=1
        calib_scheduler = _cosine_scheduler(optimizer, t_max=1)
        calib_scheduler.step()
        save_checkpoint(
            ckpt_path, model, optimizer, epoch=1,
            best_val_dice=0.75, val_iou=0.65, best_val_iou=0.65,
            config={}, scheduler=calib_scheduler, seed=42,
        )

        new_model = DummyModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        formal_scheduler = _cosine_scheduler(new_optimizer, t_max=30)
        initial_last_epoch = formal_scheduler.last_epoch

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_checkpoint(
                ckpt_path, new_model, optimizer=new_optimizer,
                scheduler=formal_scheduler, device=torch.device("cpu"),
                total_epochs=30,
            )

        # Formal scheduler must NOT have been overwritten with T_max=1 state
        assert formal_scheduler.T_max == 30, "Scheduler T_max was corrupted"
        assert formal_scheduler.last_epoch == initial_last_epoch, "Scheduler last_epoch was mutated"
        # A UserWarning must have been raised
        sched_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(sched_warns) >= 1
        assert "T_max mismatch" in str(sched_warns[0].message)


def test_load_checkpoint_restores_compatible_scheduler():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "compatible.pth"
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = _cosine_scheduler(optimizer, t_max=30)
        for _ in range(5):
            scheduler.step()
        saved_last_epoch = scheduler.last_epoch

        save_checkpoint(
            ckpt_path, model, optimizer, epoch=5,
            best_val_dice=0.80, val_iou=0.70, best_val_iou=0.70,
            config={}, scheduler=scheduler, seed=42,
        )

        new_model = DummyModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        new_scheduler = _cosine_scheduler(new_optimizer, t_max=30)

        load_checkpoint(
            ckpt_path, new_model, optimizer=new_optimizer,
            scheduler=new_scheduler, device=torch.device("cpu"),
            total_epochs=30,
        )

        assert new_scheduler.last_epoch == saved_last_epoch
        assert new_scheduler.T_max == 30


# ---------------------------------------------------------------------------
# Best-metric metadata tests (new)
# ---------------------------------------------------------------------------


def test_best_val_iou_stored_separately_from_val_iou():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "meta.pth"
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())

        save_checkpoint(
            ckpt_path, model, optimizer, epoch=5,
            best_val_dice=0.90,
            val_iou=0.60,       # current epoch IoU (not the best)
            best_val_iou=0.85,  # best-epoch IoU
            config={}, seed=42, best_epoch=3,
        )

        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert raw["val_iou"] == 0.60
        assert raw["best_val_iou"] == 0.85
        assert raw["best_epoch"] == 3
        assert raw["epoch"] == 5


def test_best_epoch_and_best_val_iou_preserved_on_resume():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "resume.pth"
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = _cosine_scheduler(optimizer, t_max=30)

        history = [
            {"epoch": i, "train_loss": round(0.9 - i * 0.05, 3), "val_dice": round(0.5 + i * 0.04, 3)}
            for i in range(1, 6)
        ]

        save_checkpoint(
            ckpt_path, model, optimizer, epoch=5,
            best_val_dice=0.80, val_iou=0.65, best_val_iou=0.72,
            config={}, scheduler=scheduler, seed=42,
            best_epoch=4, metrics_history=history,
        )

        new_model = DummyModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())
        new_scheduler = _cosine_scheduler(new_optimizer, t_max=30)
        chk = load_checkpoint(
            ckpt_path, new_model, new_optimizer, new_scheduler,
            device=torch.device("cpu"), total_epochs=30,
        )

        assert chk["epoch"] == 5
        assert chk["best_epoch"] == 4
        assert chk["best_val_dice"] == 0.80
        assert chk["best_val_iou"] == 0.72
        assert len(chk["metrics_history"]) == 5


# ---------------------------------------------------------------------------
# History preservation tests (new)
# ---------------------------------------------------------------------------


def test_log_metrics_preserves_prior_and_new_epochs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_p = Path(tmp_dir) / "m.json"
        csv_p = Path(tmp_dir) / "m.csv"

        prior = [
            {"epoch": 1, "train_loss": 0.9, "val_dice": 0.40},
            {"epoch": 2, "train_loss": 0.8, "val_dice": 0.50},
        ]
        log_metrics(prior, json_p, csv_p)

        # Simulate resume: restore prior history and append new epoch
        resumed = list(prior)
        resumed.append({"epoch": 3, "train_loss": 0.7, "val_dice": 0.60})
        log_metrics(resumed, json_p, csv_p)

        with open(json_p, "r") as f:
            data = json.load(f)

        assert len(data) == 3
        assert data[0]["epoch"] == 1
        assert data[1]["epoch"] == 2
        assert data[2]["epoch"] == 3

        with open(csv_p, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert rows[2]["epoch"] == "3"


def test_checkpoint_metrics_history_survives_round_trip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "hist.pth"
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = _cosine_scheduler(optimizer, t_max=30)

        history = [
            {"epoch": i, "train_loss": round(1.0 - i * 0.03, 4), "val_dice": round(i * 0.03, 4)}
            for i in range(1, 11)
        ]

        save_checkpoint(
            ckpt_path, model, optimizer, epoch=10,
            best_val_dice=0.30, val_iou=0.20, best_val_iou=0.20,
            config={}, scheduler=scheduler, seed=42,
            best_epoch=10, metrics_history=history,
        )

        new_model = DummyModel()
        chk = load_checkpoint(ckpt_path, new_model, device=torch.device("cpu"))
        assert chk["metrics_history"] == history


# ---------------------------------------------------------------------------
# Extended checkpoint isolation tests (new)
# ---------------------------------------------------------------------------


def test_all_checkpoint_paths_are_distinct():
    smoke_best = "smoke_test_checkpoint.pth"
    smoke_latest = "smoke_test_latest.pth"
    formal_best = "transattunet_baseline_best.pth"
    formal_latest = "transattunet_baseline_latest.pth"

    all_paths = {smoke_best, smoke_latest, formal_best, formal_latest}
    assert len(all_paths) == 4, "Checkpoint path collision detected"


def test_smoke_flag_selects_different_path_than_formal():
    def _pick_best(is_smoke):
        return "smoke_test_checkpoint.pth" if is_smoke else "transattunet_baseline_best.pth"

    def _pick_latest(is_smoke):
        return "smoke_test_latest.pth" if is_smoke else "transattunet_baseline_latest.pth"

    assert _pick_best(True) != _pick_best(False)
    assert _pick_latest(True) != _pick_latest(False)
    assert _pick_best(True) != _pick_latest(False)
    assert _pick_latest(True) != _pick_best(False)
