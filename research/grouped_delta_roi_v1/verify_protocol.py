from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


CASE_PATTERN = re.compile(r"^(BraTS-GLI-\d+)-(\d+)$")


def subject_id(case_id: str) -> str:
    match = CASE_PATTERN.fullmatch(case_id)
    return match.group(1) if match else case_id


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def dataset_folder(payload: dict[str, Any]) -> str:
    return f"Dataset{int(payload['id']):03d}_{payload['name']}"


def case_ids(directory: Path, suffix: str) -> set[str]:
    return {path.name.removesuffix(suffix) for path in directory.glob(f"*{suffix}")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the grouped DeltaT1 ablation protocol.")
    parser.add_argument("--config", default="research/grouped_delta_roi_v1/config.json")
    parser.add_argument("--post-plan", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    development = pd.read_csv(resolve(root, config["protocol"]["development_cases"]))
    locked = pd.read_csv(resolve(root, config["protocol"]["locked_test_cases"]))
    expected = set(development["case_id"].astype(str))
    locked_ids = set(locked["case_id"].astype(str))
    splits = json.loads(resolve(root, config["protocol"]["splits"]).read_text(encoding="utf-8-sig"))
    raw_root = resolve(root, config["paths"]["raw_root"])
    preprocessed_root = resolve(root, config["paths"]["preprocessed_root"])
    errors: list[str] = []
    checks: dict[str, Any] = {}

    if expected & locked_ids:
        errors.append("Development and locked-test case IDs overlap")
    if set(development["subject_id"].astype(str)) & set(locked["subject_id"].astype(str)):
        errors.append("Development and locked-test subjects overlap")

    fold_checks: list[dict[str, Any]] = []
    validation_counts = {case: 0 for case in expected}
    for index, fold in enumerate(splits):
        train = set(fold["train"])
        val = set(fold["val"])
        train_subjects = {subject_id(case) for case in train}
        val_subjects = {subject_id(case) for case in val}
        overlap = train_subjects & val_subjects
        if train & val:
            errors.append(f"Fold {index} has exact case overlap")
        if overlap:
            errors.append(f"Fold {index} has subject leakage")
        if train | val != expected:
            errors.append(f"Fold {index} does not cover development exactly")
        for case in val:
            validation_counts[case] += 1
        fold_checks.append(
            {
                "fold": index,
                "train_cases": len(train),
                "validation_cases": len(val),
                "train_subjects": len(train_subjects),
                "validation_subjects": len(val_subjects),
                "subject_overlap": len(overlap),
            }
        )
    if set(validation_counts.values()) != {1}:
        errors.append("Development cases are not validated exactly once across folds")

    dataset_checks: dict[str, Any] = {}
    for key, channels in (("baseline", 4), ("delta", 5)):
        payload = config["datasets"][key]
        folder = dataset_folder(payload)
        dataset_dir = raw_root / folder
        labels = case_ids(dataset_dir / "labelsTr", ".nii.gz")
        dataset_item: dict[str, Any] = {
            "folder": folder,
            "label_cases": len(labels),
            "channels": channels,
            "locked_test_cases_present": len(labels & locked_ids),
            "channel_case_counts": {},
        }
        if labels != expected:
            errors.append(f"{folder} label IDs do not match development")
        for channel in range(channels):
            suffix = f"_{channel:04d}.nii.gz"
            present = case_ids(dataset_dir / "imagesTr", suffix)
            dataset_item["channel_case_counts"][str(channel)] = len(present)
            if present != expected:
                errors.append(f"{folder} channel {channel} IDs do not match development")
        raw_split = json.loads(
            (dataset_dir / "splits" / "splits_final.json").read_text(encoding="utf-8-sig")
        )
        if raw_split != splits:
            errors.append(f"{folder} raw split differs from grouped protocol")
        dataset_json = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8-sig"))
        if int(dataset_json["numTraining"]) != len(expected):
            errors.append(f"{folder} dataset.json numTraining is incorrect")
        dataset_checks[key] = dataset_item

    sample_cases = sorted(expected)[:3] + sorted(expected)[-3:]
    baseline_dir = raw_root / dataset_folder(config["datasets"]["baseline"])
    delta_dir = raw_root / dataset_folder(config["datasets"]["delta"])
    delta_checks: list[dict[str, Any]] = []
    for case in sample_cases:
        base_t1 = nib.load(str(baseline_dir / "imagesTr" / f"{case}_0000.nii.gz"))
        base_t1c = nib.load(str(baseline_dir / "imagesTr" / f"{case}_0001.nii.gz"))
        delta_image = nib.load(str(delta_dir / "imagesTr" / f"{case}_0004.nii.gz"))
        t1 = np.asarray(base_t1.dataobj, dtype=np.float32)
        t1c = np.asarray(base_t1c.dataobj, dtype=np.float32)
        delta = np.asarray(delta_image.dataobj, dtype=np.float32)
        support = (t1 != 0) | (t1c != 0)
        expected_delta = np.zeros(t1.shape, dtype=np.float32)
        expected_delta[support] = t1c[support] - t1[support]
        max_error = float(np.max(np.abs(delta - expected_delta)))
        if max_error > 1e-5:
            errors.append(f"DeltaT1 mismatch for {case}: {max_error}")
        delta_checks.append({"case_id": case, "max_abs_error": max_error})

    plan_checks: dict[str, Any] = {"checked": False}
    if args.post_plan:
        plans_name = str(config["nnunet"]["plans"])
        configurations: dict[str, Any] = {}
        for key in ("baseline", "delta"):
            folder = dataset_folder(config["datasets"][key])
            preprocessed_dir = preprocessed_root / folder
            installed_split = json.loads(
                (preprocessed_dir / "splits_final.json").read_text(encoding="utf-8-sig")
            )
            if installed_split != splits:
                errors.append(f"{folder} preprocessed split differs from grouped protocol")
            plans = json.loads(
                (preprocessed_dir / f"{plans_name}.json").read_text(encoding="utf-8-sig")
            )
            configurations[key] = plans["configurations"][config["nnunet"]["configuration"]]
        comparison_keys = ("patch_size", "spacing", "architecture", "batch_size")
        mismatches = [
            key
            for key in comparison_keys
            if configurations["baseline"].get(key) != configurations["delta"].get(key)
        ]
        if mismatches:
            errors.append(f"Baseline and DeltaT1 plans differ: {mismatches}")
        plan_checks = {
            "checked": True,
            "matching_keys": [key for key in comparison_keys if key not in mismatches],
            "mismatching_keys": mismatches,
            "baseline_batch_size": configurations["baseline"].get("batch_size"),
            "delta_batch_size": configurations["delta"].get("batch_size"),
            "baseline_patch_size": configurations["baseline"].get("patch_size"),
            "delta_patch_size": configurations["delta"].get("patch_size"),
        }

    checks.update(
        {
            "folds": fold_checks,
            "datasets": dataset_checks,
            "sample_delta_checks": delta_checks,
            "plans": plan_checks,
        }
    )
    payload = {
        "ok": not errors,
        "errors": errors,
        "development_cases": len(expected),
        "development_subjects": int(development["subject_id"].nunique()),
        "locked_test_cases": len(locked_ids),
        "locked_test_subjects": int(locked["subject_id"].nunique()),
        "test_set_used": False,
        "checks": checks,
    }
    output = resolve(root, config["paths"]["experiment_dir"]) / "protocol_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
