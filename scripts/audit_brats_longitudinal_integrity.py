from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


CASE_PATTERN = re.compile(r"^(BraTS-GLI-\d+)-(\d+)$")
SPLITS = ("train", "val", "test")
METRIC_COLUMNS = (
    "mean_foreground_dice",
    "whole_foreground_labels_1_2_3_4_dice",
    "non_rc_labels_1_2_3_dice",
    "rc_label_4_dice",
    "label_1_dice",
    "label_2_dice",
    "label_3_dice",
    "label_4_dice",
)


def subject_id(case_id: str) -> str:
    match = CASE_PATTERN.fullmatch(case_id)
    return match.group(1) if match else case_id


def timepoint_id(case_id: str) -> str | None:
    match = CASE_PATTERN.fullmatch(case_id)
    return match.group(2) if match else None


def hash_ids(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_number(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def inspect_npz_root(root: Path) -> dict[str, Any]:
    exact_case_splits: dict[str, str] = {}
    crc_groups: dict[tuple[int, int, int, int], list[dict[str, str]]] = defaultdict(list)
    split_counts: dict[str, int] = {}
    sample_payload: dict[str, Any] = {}
    errors: list[str] = []

    for split in SPLITS:
        files = sorted((root / split).glob("*.npz"))
        split_counts[split] = len(files)
        for path in files:
            if path.stem in exact_case_splits:
                errors.append(
                    f"Exact case duplicate: {path.stem} in {exact_case_splits[path.stem]} and {split}"
                )
            exact_case_splits[path.stem] = split
            try:
                with zipfile.ZipFile(path) as archive:
                    image_info = archive.getinfo("image.npy")
                    seg_info = archive.getinfo("seg.npy")
                    fingerprint = (
                        image_info.CRC,
                        image_info.file_size,
                        seg_info.CRC,
                        seg_info.file_size,
                    )
                    crc_groups[fingerprint].append(
                        {"split": split, "case_id": path.stem}
                    )
            except Exception as exc:
                errors.append(f"Unreadable NPZ {path}: {exc!r}")

        if files and not sample_payload:
            with np.load(files[0], allow_pickle=True) as data:
                sample_payload = {
                    "path": str(files[0]),
                    "keys": sorted(data.files),
                    "image_shape": list(data["image"].shape),
                    "image_dtype": str(data["image"].dtype),
                    "seg_shape": list(data["seg"].shape),
                    "seg_dtype": str(data["seg"].dtype),
                    "modalities": [str(value) for value in data["modalities"].tolist()],
                    "labels": [int(value) for value in np.unique(data["seg"])],
                }

    duplicate_groups = [items for items in crc_groups.values() if len(items) > 1]
    cross_split_duplicates = [
        items for items in duplicate_groups if len({item["split"] for item in items}) > 1
    ]
    return {
        "split_counts": split_counts,
        "total_cases": sum(split_counts.values()),
        "sample": sample_payload,
        "exact_duplicate_case_errors": errors,
        "exact_content_duplicate_groups": len(duplicate_groups),
        "cross_split_exact_content_duplicate_groups": len(cross_split_duplicates),
        "cross_split_exact_content_examples": cross_split_duplicates[:20],
    }


def inspect_split_csv(split_csv: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    split_df = pd.read_csv(split_csv)
    required = {"case_id", "split"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"Split CSV missing columns: {sorted(missing)}")

    split_df = split_df.copy()
    split_df["subject_id"] = split_df["case_id"].map(subject_id)
    split_df["timepoint_id"] = split_df["case_id"].map(timepoint_id)

    subject_membership = (
        split_df.groupby("subject_id")["split"]
        .agg(lambda values: "+".join(sorted(set(values))))
        .rename("split_membership")
        .reset_index()
    )
    subject_case_counts = (
        split_df.groupby("subject_id").size().rename("case_count").reset_index()
    )
    subjects = subject_membership.merge(subject_case_counts, on="subject_id", how="left")

    split_subject_sets = {
        split: set(split_df.loc[split_df["split"] == split, "subject_id"])
        for split in SPLITS
    }
    pair_overlaps: dict[str, Any] = {}
    overlap_rows: list[dict[str, Any]] = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(split_subject_sets[left] & split_subject_sets[right])
        left_cases = int(
            split_df[
                (split_df["split"] == left) & split_df["subject_id"].isin(overlap)
            ].shape[0]
        )
        right_cases = int(
            split_df[
                (split_df["split"] == right) & split_df["subject_id"].isin(overlap)
            ].shape[0]
        )
        pair_overlaps[f"{left}_{right}"] = {
            "subject_count": len(overlap),
            f"{left}_affected_case_count": left_cases,
            f"{right}_affected_case_count": right_cases,
            "subjects": overlap,
        }
        for sid in overlap:
            overlap_rows.append(
                {"subject_id": sid, "split_a": left, "split_b": right}
            )

    label_summary: dict[str, Any] = {}
    for split in SPLITS:
        subset = split_df[split_df["split"] == split]
        item: dict[str, Any] = {
            "cases": int(len(subset)),
            "subjects": int(subset["subject_id"].nunique()),
        }
        for label in range(1, 5):
            present_column = f"has_label_{label}"
            volume_column = f"label_{label}_voxels"
            if present_column in subset:
                item[f"label_{label}_positive_cases"] = int(subset[present_column].astype(bool).sum())
            if volume_column in subset:
                positive = subset.loc[subset[volume_column] > 0, volume_column]
                item[f"label_{label}_positive_median_voxels"] = (
                    float(positive.median()) if len(positive) else None
                )
        label_summary[split] = item

    report = {
        "case_counts": {
            split: int((split_df["split"] == split).sum()) for split in SPLITS
        },
        "subject_counts": {
            split: len(split_subject_sets[split]) for split in SPLITS
        },
        "total_unique_subjects": int(split_df["subject_id"].nunique()),
        "pairwise_subject_overlap": pair_overlaps,
        "subject_split_membership_counts": {
            key: int(value)
            for key, value in Counter(subjects["split_membership"]).items()
        },
        "longitudinal_case_count_distribution": {
            str(key): int(value)
            for key, value in Counter(subjects["case_count"]).items()
        },
        "label_distribution": label_summary,
    }
    overlap_df = pd.DataFrame(overlap_rows).drop_duplicates()
    return split_df, report, overlap_df


def inspect_nnunet_splits(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    folds = json.loads(path.read_text(encoding="utf-8"))
    summaries = []
    for index, fold in enumerate(folds):
        train = list(fold["train"])
        val = list(fold["val"])
        summaries.append(
            {
                "fold": index,
                "train_cases": len(train),
                "val_cases": len(val),
                "train_hash": hash_ids(train),
                "val_hash": hash_ids(val),
            }
        )
    unique_pairs = {(item["train_hash"], item["val_hash"]) for item in summaries}
    return {
        "path": str(path),
        "exists": True,
        "declared_fold_count": len(folds),
        "unique_split_pair_count": len(unique_pairs),
        "all_folds_identical": len(folds) > 1 and len(unique_pairs) == 1,
        "folds": summaries,
    }


def inspect_nifti_geometry(label_path: Path) -> dict[str, Any]:
    if not label_path.exists():
        return {"path": str(label_path), "exists": False}
    image = nib.load(str(label_path))
    affine = np.asarray(image.affine, dtype=np.float64)
    return {
        "path": str(label_path),
        "exists": True,
        "shape": list(image.shape),
        "zooms": [float(value) for value in image.header.get_zooms()[:3]],
        "affine": affine.tolist(),
        "identity_affine": bool(np.allclose(affine, np.eye(4))),
    }


def inspect_validation_metrics(metrics_csv: Path, split_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    if not metrics_csv.exists():
        return {"path": str(metrics_csv), "exists": False}, pd.DataFrame()
    metrics = pd.read_csv(metrics_csv)
    metrics["subject_id"] = metrics["case_id"].map(subject_id)
    train_subjects = set(split_df.loc[split_df["split"] == "train", "subject_id"])
    metrics["subject_seen_in_train"] = metrics["subject_id"].isin(train_subjects)

    groups = {
        "all_validation_cases": metrics,
        "subject_seen_in_train": metrics[metrics["subject_seen_in_train"]],
        "subject_unseen_in_train": metrics[~metrics["subject_seen_in_train"]],
    }
    summaries: dict[str, Any] = {}
    for name, frame in groups.items():
        item: dict[str, Any] = {
            "case_count": int(len(frame)),
            "subject_count": int(frame["subject_id"].nunique()),
        }
        for column in METRIC_COLUMNS:
            if column in frame:
                item[f"{column}_mean"] = float(frame[column].mean())
                item[f"{column}_valid_case_count"] = int(frame[column].notna().sum())
        summaries[name] = item
    return {
        "path": str(metrics_csv),
        "exists": True,
        "note": (
            "Seen/unseen subsets are descriptive only. They have different case composition "
            "and cannot replace a patient-independent resplit and retraining."
        ),
        "groups": summaries,
    }, metrics


def build_markdown(report: dict[str, Any]) -> str:
    split = report["split_audit"]
    npz = report["npz_audit"]
    folds = report["nnunet_split_audit"]
    geometry = report["nifti_geometry_audit"]
    lines = [
        "# BraTS Longitudinal Data Integrity Audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Cases: {npz['total_cases']}",
        f"- Unique longitudinal subjects: {split['total_unique_subjects']}",
        "",
        "## Critical findings",
    ]
    for finding in report["critical_findings"]:
        lines.append(f"- {finding}")
    lines.extend(["", "## Pairwise subject overlap"])
    for name, item in split["pairwise_subject_overlap"].items():
        lines.append(
            f"- {name}: {item['subject_count']} subjects; "
            + ", ".join(
                f"{key.replace('_affected_case_count', '')}={value} affected cases"
                for key, value in item.items()
                if key.endswith("_affected_case_count")
            )
        )
    lines.extend(
        [
            "",
            "## Additional checks",
            f"- Exact cross-split content duplicate groups: {npz['cross_split_exact_content_duplicate_groups']}",
            f"- Declared nnU-Net folds: {folds.get('declared_fold_count')}",
            f"- Unique nnU-Net split pairs: {folds.get('unique_split_pair_count')}",
            f"- Sample converted NIfTI uses identity affine: {geometry.get('identity_affine')}",
            "",
            "## Required remediation",
            "1. Group all timepoints from the same subject into one split.",
            "2. Recreate true grouped folds; do not call repeated seeds on one split cross-validation.",
            "3. Retrain and select checkpoints only on the corrected patient-independent protocol.",
            "4. Report official BraTS 2024 lesion-wise Dice and HD95 for ET, NETC, SNFH, RC, TC, and WT.",
            "5. Preserve or explicitly document original NIfTI spatial metadata; NPZ-derived identity affine is not the source geometry.",
            "",
            "Existing checkpoints and reports remain useful as exploratory artifacts, but they are not valid final patient-independent evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit longitudinal BraTS split and evaluation integrity.")
    parser.add_argument(
        "--split-csv",
        default="_archive_inspect/eda/eda_outputs/brats2024_split.csv",
    )
    parser.add_argument("--npz-root", default="processed/brats2024_npz_full")
    parser.add_argument(
        "--nnunet-splits",
        default="processed/nnunet_preprocessed/Dataset501_BraTS2024GliomaNPZ/splits_final.json",
    )
    parser.add_argument(
        "--sample-label",
        default="processed/nnunet_raw/Dataset501_BraTS2024GliomaNPZ/labelsTr/BraTS-GLI-00005-101.nii.gz",
    )
    parser.add_argument(
        "--metrics-csv",
        default="experiments/nnunet_ensemble_5fold_best_val_report/val_case_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/brats_longitudinal_integrity_audit_20260716",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_df, split_report, overlap_df = inspect_split_csv(Path(args.split_csv))
    npz_report = inspect_npz_root(Path(args.npz_root))
    folds_report = inspect_nnunet_splits(Path(args.nnunet_splits))
    geometry_report = inspect_nifti_geometry(Path(args.sample_label))
    metrics_report, metrics_df = inspect_validation_metrics(Path(args.metrics_csv), split_df)

    critical_findings: list[str] = []
    overlap_subjects = set()
    for item in split_report["pairwise_subject_overlap"].values():
        overlap_subjects.update(item["subjects"])
    if overlap_subjects:
        critical_findings.append(
            f"Patient-level split leakage: {len(overlap_subjects)} unique subjects occur in multiple splits."
        )
    if folds_report.get("all_folds_identical"):
        critical_findings.append(
            "The five declared nnU-Net folds contain one unique train/validation split; this is a repeated-seed ensemble, not five-fold cross-validation."
        )
    if geometry_report.get("identity_affine"):
        critical_findings.append(
            "The NPZ-to-NIfTI conversion writes an identity affine, so original MNI spatial metadata is not preserved."
        )
    critical_findings.append(
        "Current reports use legacy voxel-wise Dice/IoU, while the official BraTS 2024 protocol uses lesion-wise Dice and HD95."
    )

    report = {
        "status": "CRITICAL_PROTOCOL_FAILURE" if critical_findings else "PASS",
        "critical_findings": critical_findings,
        "split_audit": split_report,
        "npz_audit": npz_report,
        "nnunet_split_audit": folds_report,
        "nifti_geometry_audit": geometry_report,
        "validation_metric_subset_audit": metrics_report,
        "label_semantics": {
            "1": "NETC",
            "2": "SNFH",
            "3": "ET",
            "4": "RC",
            "TC": "labels 1 + 3",
            "WT": "labels 1 + 2 + 3 (RC excluded)",
        },
        "metric_protocol": {
            "current": "legacy per-case voxel-wise Dice and IoU",
            "required": "BraTS 2024 lesion-wise Dice and lesion-wise HD95",
            "official_regions": ["ET", "NETC", "SNFH", "RC", "TC", "WT"],
        },
    }

    split_df.to_csv(output_dir / "case_split_with_subject_ids.csv", index=False)
    overlap_df.to_csv(output_dir / "cross_split_subject_overlaps.csv", index=False)
    if not metrics_df.empty:
        metrics_df.to_csv(output_dir / "validation_metrics_with_leakage_flag.csv", index=False)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(report, indent=2, default=json_number), encoding="utf-8"
    )
    (output_dir / "audit_summary.md").write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, default=json_number))


if __name__ == "__main__":
    main()
