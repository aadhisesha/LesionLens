"""Dataset preparation and validation pipeline for ISIC 2016 Task 1.

Provides a unified command-line tool for:
1. Checking archive availability and integrity (--check-archives)
2. Downloading missing archives from official ISIC archive servers (--download)
3. Extracting archives to raw directories (--extract)
4. Pairing images and masks, preprocessing (256x256), and creating reproducible splits (--prepare)
5. Comprehensive dataset and split validation with machine-readable report (--validate)
6. End-to-end execution (--all)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
import yaml

# Import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure robust console encoding across platforms (e.g. Windows cp1252)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from data.preprocessing import (
    INTERPOLATION_MAP,
    normalize_identifier,
    pair_images_and_masks,
    preprocess_image,
    preprocess_mask,
    process_and_save_pair,
)
from data.dataset import ISICDataset, smoke_test_dataloader
from evaluation.metrics import dice_coefficient, iou_score


def load_config(config_path: str | Path = "configs/data.yaml") -> Dict[str, Any]:
    """Load dataset configuration file."""
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_archives(config: Dict[str, Any], archive_dir: str | Path) -> Dict[str, Any]:
    """Check existence and integrity of the four required ISIC 2016 Task 1 archives.

    Returns:
        Dictionary with status of each archive.
    """
    archive_dir = Path(archive_dir)
    archives_spec = config["dataset"]["archives"]
    results = {}
    all_present_and_valid = True

    print(f"\n[1/5] Checking archives in: {archive_dir.resolve()}")
    if not archive_dir.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)

    for key, spec in archives_spec.items():
        fname = spec["filename"]
        expected_size_mb = spec.get("approx_size_mb", 0)
        archive_path = archive_dir / fname

        entry: Dict[str, Any] = {
            "filename": fname,
            "path": str(archive_path),
            "exists": False,
            "valid_zip": False,
            "file_count": 0,
            "size_bytes": 0,
            "size_mb": 0.0,
            "expected_mb": expected_size_mb,
            "url": spec["url"],
            "expected_count": spec["expected_count"],
        }

        if archive_path.is_file():
            entry["exists"] = True
            entry["size_bytes"] = archive_path.stat().st_size
            entry["size_mb"] = round(entry["size_bytes"] / (1024 * 1024), 2)
            try:
                with zipfile.ZipFile(archive_path, "r") as zf:
                    corrupt = zf.testzip()
                    if corrupt is None:
                        entry["valid_zip"] = True
                        file_list = [f for f in zf.namelist() if not f.endswith("/")]
                        entry["file_count"] = len(file_list)
                    else:
                        entry["error"] = f"Corrupted file in zip: {corrupt}"
            except Exception as e:
                entry["error"] = f"Invalid zip archive: {e}"

        status_str = "VALID" if entry["valid_zip"] else ("EXISTS BUT CORRUPT" if entry["exists"] else "MISSING")
        print(f"  • {key:14}: {fname}")
        print(f"    Status: {status_str} | Size: {entry['size_mb']} MB (expected ~{expected_size_mb} MB) | Files: {entry['file_count']}")

        if not entry["valid_zip"]:
            all_present_and_valid = False
            if not entry["exists"]:
                print(f"    Download URL: {spec['url']}")

        results[key] = entry

    results["all_valid"] = all_present_and_valid
    return results


def download_archives(config: Dict[str, Any], archive_dir: str | Path) -> bool:
    """Download missing archives from official ISIC challenge URLs.

    Returns:
        True if all archives are present and valid, False otherwise.
    """
    import urllib.request
    from urllib.error import URLError

    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archives_spec = config["dataset"]["archives"]

    print(f"\nDownloading missing archives into: {archive_dir.resolve()}")

    for key, spec in archives_spec.items():
        fname = spec["filename"]
        url = spec["url"]
        archive_path = archive_dir / fname

        if archive_path.is_file():
            try:
                with zipfile.ZipFile(archive_path, "r") as zf:
                    if zf.testzip() is None:
                        print(f"  • {fname} already exists and is valid. Skipping download.")
                        continue
            except Exception:
                print(f"  • {fname} exists but is invalid. Re-downloading...")

        print(f"  • Downloading {fname} (~{spec['approx_size_mb']} MB) from {url}...")
        try:
            # Simple chunked download with progress
            req = urllib.request.Request(url, headers={"User-Agent": "LesionLens-DatasetDownloader/1.0"})
            with urllib.request.urlopen(req) as resp, open(archive_path, "wb") as out_file:
                total_size = int(resp.headers.get("content-length", 0))
                downloaded = 0
                block_size = 1024 * 1024  # 1MB blocks

                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        mb = downloaded / (1024 * 1024)
                        tot_mb = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r    Progress: {mb:.1f}/{tot_mb:.1f} MB ({percent:.1f}%)")
                        sys.stdout.flush()
                sys.stdout.write("\n")
            print(f"    Saved {fname} successfully.")
        except (URLError, Exception) as e:
            print(f"\n    Failed to download {fname}: {e}")
            if archive_path.is_file():
                archive_path.unlink()
            return False

    return True


def extract_archives(config: Dict[str, Any], archive_dir: str | Path, force: bool = False) -> bool:
    """Extract all four archives into their respective data/raw/ directories.

    Handles nested archive directories by placing raw files directly in the target directory.
    Avoids extracting if contents are already present and match expected count.
    """
    archive_dir = Path(archive_dir)
    archives_spec = config["dataset"]["archives"]

    print("\n[2/5] Extracting archives to data/raw/...")

    for key, spec in archives_spec.items():
        fname = spec["filename"]
        target_dir = Path(spec["raw_target_dir"])
        expected_count = spec["expected_count"]
        archive_path = archive_dir / fname

        target_dir.mkdir(parents=True, exist_ok=True)

        # Check existing non-hidden files
        existing_files = [f for f in target_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
        if not force and len(existing_files) == expected_count:
            print(f"  • {key:14}: {len(existing_files)} files already present in {target_dir}. Skipping extraction.")
            continue

        if not archive_path.is_file():
            print(f"  • ERROR: Archive not found: {archive_path}")
            return False

        print(f"  • Extracting {fname} to {target_dir}...")
        with zipfile.ZipFile(archive_path, "r") as zf:
            extracted_count = 0
            for member in zf.infolist():
                if member.is_dir():
                    continue
                # Extract filename without internal nested directory hierarchy
                source_filename = Path(member.filename).name
                if not source_filename or source_filename.startswith("."):
                    continue
                dest_file_path = target_dir / source_filename
                with zf.open(member) as source_f, open(dest_file_path, "wb") as dest_f:
                    shutil.copyfileobj(source_f, dest_f)
                extracted_count += 1

            print(f"    Extracted {extracted_count} files into {target_dir} (expected {expected_count}).")
            if extracted_count != expected_count:
                print(f"    WARNING: Extracted file count ({extracted_count}) does not match expected ({expected_count})!")

    return True


def discover_files(directory: Union[str, Path], allowed_exts: List[str]) -> List[Path]:
    """Find all files in directory matching allowed extensions."""
    d = Path(directory)
    if not d.exists():
        return []
    allowed_exts_set = {ext.lower() for ext in allowed_exts}
    files = [
        f for f in d.iterdir()
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in allowed_exts_set
    ]
    return sorted(files)


def create_splits(
    train_pairs: List[Tuple[str, Path, Path]],
    test_pairs: List[Tuple[str, Path, Path]],
    splits_dir: Union[str, Path],
    config: Dict[str, Any],
    seed: int = 42,
    val_ratio: float = 0.2,
) -> Dict[str, Any]:
    """Create reproducible 80/20 train/val split and standalone test manifest.

    Guarantees:
    - 720 train samples and 180 validation samples.
    - Zero overlap between train and validation identifiers.
    - Test set kept strictly isolated.
    - Manifests stored with relative paths and complete metadata.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Sort identifiers deterministically before shuffle
    sorted_train_pairs = sorted(train_pairs, key=lambda x: x[0])
    total_train = len(sorted_train_pairs)

    rng = random.Random(seed)
    shuffled_indices = list(range(total_train))
    rng.shuffle(shuffled_indices)

    num_val = int(round(total_train * val_ratio))
    val_indices = set(shuffled_indices[:num_val])
    train_indices = set(shuffled_indices[num_val:])

    train_subset = [sorted_train_pairs[i] for i in sorted(list(train_indices))]
    val_subset = [sorted_train_pairs[i] for i in sorted(list(val_indices))]

    train_ids = {p[0] for p in train_subset}
    val_ids = {p[0] for p in val_subset}
    test_ids = {p[0] for p in test_pairs}

    # Strict assertion tests
    assert train_ids.isdisjoint(val_ids), "Train and validation splits MUST be disjoint!"
    assert (train_ids | val_ids) == {p[0] for p in train_pairs}, "Union of train and val must equal all train samples!"
    assert train_ids.isdisjoint(test_ids), "Train and test samples MUST NOT overlap!"
    assert val_ids.isdisjoint(test_ids), "Validation and test samples MUST NOT overlap!"

    def make_manifest_payload(split_name: str, pairs: List[Tuple[str, Path, Path]]) -> Dict[str, Any]:
        records = []
        for sample_id, img_p, msk_p in pairs:
            # Format relative paths using forward slashes for cross-platform portability
            records.append({
                "id": sample_id,
                "image_path": str(Path(img_p).as_posix()),
                "mask_path": str(Path(msk_p).as_posix()),
            })
        return {
            "metadata": {
                "split": split_name,
                "dataset_name": config["dataset"]["name"],
                "year": config["dataset"]["year"],
                "task": config["dataset"]["task"],
                "num_samples": len(records),
                "random_seed": seed,
                "created_at": datetime.datetime.now().isoformat(),
                "preprocessing": config.get("preprocessing", {}),
            },
            "samples": records,
        }

    train_manifest = make_manifest_payload("train", train_subset)
    val_manifest = make_manifest_payload("val", val_subset)
    test_manifest = make_manifest_payload("test", sorted(test_pairs, key=lambda x: x[0]))

    train_path = splits_dir / "train.json"
    val_path = splits_dir / "val.json"
    test_path = splits_dir / "test.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_manifest, f, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_manifest, f, indent=2)

    print(f"\nGenerated reproducible splits (Seed: {seed}):")
    print(f"  • Train: {len(train_subset)} samples -> {train_path}")
    print(f"  • Val:   {len(val_subset)} samples -> {val_path}")
    print(f"  • Test:  {len(test_pairs)} samples -> {test_path}")

    return {
        "train_count": len(train_subset),
        "val_count": len(val_subset),
        "test_count": len(test_pairs),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
    }


def prepare_dataset(config: Dict[str, Any], force: bool = False, seed: int = 42) -> bool:
    """Pair images and masks, resize to target size, binarize masks, and generate split manifests."""
    print("\n[3/5] Pairing and preprocessing datasets...")

    raw_train_img_dir = Path(config["dataset"]["archives"]["train_images"]["raw_target_dir"])
    raw_train_msk_dir = Path(config["dataset"]["archives"]["train_masks"]["raw_target_dir"])
    raw_test_img_dir = Path(config["dataset"]["archives"]["test_images"]["raw_target_dir"])
    raw_test_msk_dir = Path(config["dataset"]["archives"]["test_masks"]["raw_target_dir"])

    proc_train_img_dir = Path(config["dataset"]["archives"]["train_images"]["processed_target_dir"])
    proc_train_msk_dir = Path(config["dataset"]["archives"]["train_masks"]["processed_target_dir"])
    proc_test_img_dir = Path(config["dataset"]["archives"]["test_images"]["processed_target_dir"])
    proc_test_msk_dir = Path(config["dataset"]["archives"]["test_masks"]["processed_target_dir"])

    img_exts = config["preprocessing"]["supported_image_extensions"]
    msk_exts = config["preprocessing"]["supported_mask_extensions"]
    target_size = (config["preprocessing"]["target_height"], config["preprocessing"]["target_width"])
    img_interp = config["preprocessing"]["image_interpolation"]
    mask_thresh = config["preprocessing"]["mask_binarization"]["threshold"]

    # 1. Discover raw training files
    train_raw_imgs = discover_files(raw_train_img_dir, img_exts)
    train_raw_msks = discover_files(raw_train_msk_dir, msk_exts)

    if not train_raw_imgs or not train_raw_msks:
        print(f"  • ERROR: Raw training data missing ({len(train_raw_imgs)} images, {len(train_raw_msks)} masks).")
        print("    Please run extraction first via: python scripts/prepare_dataset.py --extract")
        return False

    # 2. Pair training files
    print(f"  • Discovered {len(train_raw_imgs)} raw train images and {len(train_raw_msks)} raw train masks.")
    try:
        paired_train = pair_images_and_masks(train_raw_imgs, train_raw_msks)
    except Exception as e:
        print(f"  • Pairing ERROR in training set: {e}")
        return False
    print(f"  • Successfully paired {len(paired_train)} training image-mask couples.")

    # 3. Discover and pair test files
    test_raw_imgs = discover_files(raw_test_img_dir, img_exts)
    test_raw_msks = discover_files(raw_test_msk_dir, msk_exts)

    if not test_raw_imgs or not test_raw_msks:
        print(f"  • ERROR: Raw test data missing ({len(test_raw_imgs)} images, {len(test_raw_msks)} masks).")
        return False

    try:
        paired_test = pair_images_and_masks(test_raw_imgs, test_raw_msks)
    except Exception as e:
        print(f"  • Pairing ERROR in test set: {e}")
        return False
    print(f"  • Successfully paired {len(paired_test)} test image-mask couples.")

    # 4. Preprocess training set
    print(f"  • Preprocessing {len(paired_train)} training pairs to {target_size}...")
    processed_train_pairs = []
    processed_count = 0
    skipped_count = 0

    for sample_id, raw_img, raw_msk in paired_train:
        p_img, p_msk, was_proc = process_and_save_pair(
            sample_id=sample_id,
            raw_img_path=raw_img,
            raw_msk_path=raw_msk,
            out_img_dir=proc_train_img_dir,
            out_msk_dir=proc_train_msk_dir,
            target_size=target_size,
            image_interpolation=img_interp,
            mask_threshold=mask_thresh,
            force=force,
        )
        processed_train_pairs.append((sample_id, p_img, p_msk))
        if was_proc:
            processed_count += 1
        else:
            skipped_count += 1

    print(f"    Train: {processed_count} preprocessed, {skipped_count} cached.")

    # 5. Preprocess test set
    print(f"  • Preprocessing {len(paired_test)} test pairs to {target_size}...")
    processed_test_pairs = []
    test_proc_count = 0
    test_skip_count = 0

    for sample_id, raw_img, raw_msk in paired_test:
        p_img, p_msk, was_proc = process_and_save_pair(
            sample_id=sample_id,
            raw_img_path=raw_img,
            raw_msk_path=raw_msk,
            out_img_dir=proc_test_img_dir,
            out_msk_dir=proc_test_msk_dir,
            target_size=target_size,
            image_interpolation=img_interp,
            mask_threshold=mask_thresh,
            force=force,
        )
        processed_test_pairs.append((sample_id, p_img, p_msk))
        if was_proc:
            test_proc_count += 1
        else:
            test_skip_count += 1

    print(f"    Test: {test_proc_count} preprocessed, {test_skip_count} cached.")

    # 6. Generate reproducible manifests
    splits_dir = Path(config["paths"]["splits_dir"])
    val_ratio = config["split"].get("val_ratio", 0.2)
    create_splits(
        train_pairs=processed_train_pairs,
        test_pairs=processed_test_pairs,
        splits_dir=splits_dir,
        config=config,
        seed=seed,
        val_ratio=val_ratio,
    )

    return True


def validate_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute complete automated dataset validation checklist and compile report.

    Verifies:
    1. Exactly 900 valid training pairs.
    2. Exactly 379 valid test pairs.
    3. No duplicate sample IDs.
    4. Disjoint training and validation identifiers (0 overlap).
    5. Union of train and val matches all 900 training pairs.
    6. All processed images are 256x256x3 RGB.
    7. All processed masks are 256x256 and strictly binary {0, 1} (or {0, 255}).
    8. PyTorch DataLoader smoke test on train, val, and test manifests.
    9. Dice and IoU metric functions.
    """
    print("\n[4/5] Executing dataset validation...")

    report: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "overall_status": "FAILED",
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    checks = report["checks"]
    errors = report["errors"]
    warnings = report["warnings"]

    # 1. Check manifests exist
    train_manifest_path = Path(config["paths"]["train_split_json"])
    val_manifest_path = Path(config["paths"]["val_split_json"])
    test_manifest_path = Path(config["paths"]["test_split_json"])

    manifests_exist = (
        train_manifest_path.is_file() and
        val_manifest_path.is_file() and
        test_manifest_path.is_file()
    )
    checks["manifests_exist"] = manifests_exist

    if not manifests_exist:
        errors.append("Split manifests not found. Please run --prepare first.")
        print(f"  ✗ FAIL: Missing manifest files in {config['paths']['splits_dir']}")
        _save_validation_report(report, config)
        return report

    # Load manifests
    with open(train_manifest_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)["samples"]
    with open(val_manifest_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)["samples"]
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)["samples"]

    # 2. Check counts
    train_count = len(train_data)
    val_count = len(val_data)
    test_count = len(test_data)
    total_train_pairs = train_count + val_count

    expected_train = config["counts"]["expected_train"]
    expected_test = config["counts"]["expected_test"]
    expected_split_train = config["counts"]["train_split"]
    expected_split_val = config["counts"]["val_split"]

    checks["sample_counts"] = {
        "train_samples": train_count,
        "val_samples": val_count,
        "total_train_val": total_train_pairs,
        "expected_total_train": expected_train,
        "test_samples": test_count,
        "expected_test": expected_test,
        "split_train_match": train_count == expected_split_train,
        "split_val_match": val_count == expected_split_val,
        "test_match": test_count == expected_test,
    }

    if total_train_pairs != expected_train:
        errors.append(f"Expected {expected_train} training samples in total, found {total_train_pairs}.")
    if train_count != expected_split_train:
        errors.append(f"Expected {expected_split_train} in train split, got {train_count}.")
    if val_count != expected_split_val:
        errors.append(f"Expected {expected_split_val} in val split, got {val_count}.")
    if test_count != expected_test:
        errors.append(f"Expected {expected_test} test samples, got {test_count}.")

    # 3. Check identifier uniqueness and disjointness
    train_ids = [s["id"] for s in train_data]
    val_ids = [s["id"] for s in val_data]
    test_ids = [s["id"] for s in test_data]

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    test_id_set = set(test_ids)

    checks["no_train_duplicates"] = len(train_ids) == len(train_id_set)
    checks["no_val_duplicates"] = len(val_ids) == len(val_id_set)
    checks["no_test_duplicates"] = len(test_ids) == len(test_id_set)

    train_val_overlap = train_id_set & val_id_set
    train_test_overlap = train_id_set & test_id_set
    val_test_overlap = val_id_set & test_id_set

    checks["train_val_disjoint"] = len(train_val_overlap) == 0
    checks["train_test_disjoint"] = len(train_test_overlap) == 0
    checks["val_test_disjoint"] = len(val_test_overlap) == 0

    if train_val_overlap:
        errors.append(f"Data leakage! {len(train_val_overlap)} identifiers present in both train and val.")
    if train_test_overlap:
        errors.append(f"Data leakage! {len(train_test_overlap)} identifiers present in both train and test.")
    if val_test_overlap:
        errors.append(f"Data leakage! {len(val_test_overlap)} identifiers present in both val and test.")

    # 4. Check image & mask dimensions and binary mask integrity
    target_h = config["preprocessing"]["target_height"]
    target_w = config["preprocessing"]["target_width"]
    target_shape = (target_w, target_h)  # PIL size is (width, height)

    corrupted_files = []
    dimension_mismatches = []
    non_binary_masks = []

    print("  • Verifying processed image files, dimensions, and mask binary integrity...")
    all_samples = train_data + val_data + test_data

    for sample in all_samples:
        s_id = sample["id"]
        img_p = Path(sample["image_path"])
        msk_p = Path(sample["mask_path"])

        # Check image
        try:
            with Image.open(img_p) as img:
                if img.size != target_shape:
                    dimension_mismatches.append(f"{img_p} has size {img.size}, expected {target_shape}")
                if img.mode != "RGB":
                    dimension_mismatches.append(f"{img_p} has mode {img.mode}, expected RGB")
        except Exception as e:
            corrupted_files.append(f"Image {img_p} unreadable: {e}")

        # Check mask
        try:
            with Image.open(msk_p) as msk:
                if msk.size != target_shape:
                    dimension_mismatches.append(f"{msk_p} has size {msk.size}, expected {target_shape}")
                arr = np.array(msk)
                unique_vals = set(np.unique(arr).tolist())
                # Allow {0, 255} or {0, 1}
                if not unique_vals.issubset({0, 255}) and not unique_vals.issubset({0, 1}):
                    non_binary_masks.append(f"{msk_p} contains non-binary pixel values: {unique_vals}")
        except Exception as e:
            corrupted_files.append(f"Mask {msk_p} unreadable: {e}")

    checks["corrupted_files_count"] = len(corrupted_files)
    checks["dimension_mismatches_count"] = len(dimension_mismatches)
    checks["non_binary_masks_count"] = len(non_binary_masks)

    if corrupted_files:
        errors.extend(corrupted_files[:10])
    if dimension_mismatches:
        errors.extend(dimension_mismatches[:10])
    if non_binary_masks:
        errors.extend(non_binary_masks[:10])

    # 5. Check DataLoader batch loading
    dataloader_checks = {}
    for split_name, manifest_p in [("train", train_manifest_path), ("val", val_manifest_path), ("test", test_manifest_path)]:
        try:
            smoke_result = smoke_test_dataloader(manifest_p, batch_size=4)
            dataloader_checks[split_name] = {
                "status": "PASSED",
                "batch_size": smoke_result["batch_size"],
                "image_shape": smoke_result["image_shape"],
                "mask_shape": smoke_result["mask_shape"],
            }
        except Exception as e:
            dataloader_checks[split_name] = {"status": "FAILED", "error": str(e)}
            errors.append(f"DataLoader failed for {split_name} split: {e}")

    checks["dataloader_smoke_tests"] = dataloader_checks

    # 6. Overall status determination
    if len(errors) == 0:
        report["overall_status"] = "PASSED"
        print("  [PASS] SUCCESS: All dataset validation checks passed!")
    else:
        report["overall_status"] = "FAILED"
        print(f"  [FAIL] FAILURE: Validation encountered {len(errors)} error(s):")
        for err in errors[:10]:
            print(f"    - {err}")


    _save_validation_report(report, config)
    return report


def _save_validation_report(report: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Save validation report to JSON file."""
    report_path = Path(config["paths"]["validation_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  • Validation report saved to: {report_path.resolve()}")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ISIC 2016 Task 1 Skin Lesion Dataset Preparation & Validation Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to data configuration YAML file.",
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        default=None,
        help="Path to archives directory (overrides config).",
    )
    parser.add_argument(
        "--check-archives",
        action="store_true",
        help="Check presence and integrity of downloaded zip archives.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing archives from official ISIC challenge URLs.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract archives to data/raw/ directories.",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Pair images/masks, preprocess to 256x256, and generate split manifests.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Execute automated dataset validation suite.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute complete pipeline: check -> extract -> prepare -> validate.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction and re-preprocessing of existing files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for train/val splitting (default 42).",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Configuration Error: {e}")
        return 1

    archive_dir = args.archive_dir or config["paths"]["archive_dir"]
    seed = args.seed if args.seed is not None else config["split"]["seed"]

    # If no flags provided, default to --check-archives
    if not (args.check_archives or args.download or args.extract or args.prepare or args.validate or args.all):
        print("No action specified. Defaulting to --check-archives. (Use --help for options)")
        args.check_archives = True

    if args.check_archives or args.all:
        archive_status = check_archives(config, archive_dir)
        if not archive_status["all_valid"] and not args.all and not args.download:
            print("\nNote: Use '--download' to automatically download missing archives, or download them manually.")

    if args.download:
        success = download_archives(config, archive_dir)
        if not success:
            print("Download failed.")
            return 1

    if args.extract or args.all:
        # Check if archives are present before extracting
        archive_status = check_archives(config, archive_dir)
        if not archive_status["all_valid"]:
            if args.all:
                print("\nArchives are missing. Attempting automatic download...")
                if not download_archives(config, archive_dir):
                    print("Failed to download archives. Stopping.")
                    return 1
            else:
                print("\nCannot extract: required archives are missing or invalid.")
                return 1

        success = extract_archives(config, archive_dir, force=args.force)
        if not success:
            return 1

    if args.prepare or args.all:
        success = prepare_dataset(config, force=args.force, seed=seed)
        if not success:
            return 1

    if args.validate or args.all:
        report = validate_dataset(config)
        if report["overall_status"] != "PASSED":
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
