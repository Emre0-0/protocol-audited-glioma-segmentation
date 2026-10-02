from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


CASE_PATTERN = re.compile(r"^(BraTS-GLI-\d+)-(\d+)$")
REGIONS = ("WT", "TC", "NETC", "SNFH", "ET", "RC")


@dataclass
class Check:
    check_id: str
    status: str
    severity: str
    message: str
    evidence: dict[str, Any]


class Audit:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(
        self,
        check_id: str,
        status: str,
        severity: str,
        message: str,
        **evidence: Any,
    ) -> None:
        self.checks.append(
            Check(check_id, status, severity, message, evidence)
        )

    def require(
        self,
        condition: bool,
        check_id: str,
        message_pass: str,
        message_fail: str,
        **evidence: Any,
    ) -> None:
        self.add(
            check_id,
            "PASS" if condition else "FAIL",
            "critical" if not condition else "info",
            message_pass if condition else message_fail,
            **evidence,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently audit the publication-critical BraTS protocol, "
            "data conversion, metrics, checkpoints, and locked-test record."
        )
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/publication_integrity_v1"),
    )
    parser.add_argument(
        "--array-check",
        choices=("none", "labels", "full"),
        default="none",
        help="Compare converted development NIfTI arrays against source NPZ arrays.",
    )
    parser.add_argument(
        "--array-limit",
        type=int,
        default=0,
        help="Optional deterministic case limit for an integration check; 0 means all cases.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def subject_id(case_id: str) -> str:
    match = CASE_PATTERN.fullmatch(case_id)
    if match is None:
        raise ValueError(f"Unexpected BraTS case identifier: {case_id}")
    return match.group(1)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def source_npz_index(root: Path) -> tuple[dict[str, Path], dict[str, int]]:
    index: dict[str, Path] = {}
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        files = sorted((root / split).glob("*.npz"))
        counts[split] = len(files)
        for path in files:
            if path.stem in index:
                raise ValueError(f"Duplicate case identifier across NPZ splits: {path.stem}")
            index[path.stem] = path
    return index, counts


def exact_fold_signature(fold: dict[str, list[str]]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(sorted(fold["train"])), tuple(sorted(fold["val"]))


def audit_protocol(root: Path, audit: Audit) -> dict[str, Any]:
    protocol_dir = root / "processed/brats_subject_grouped_protocol_v1"
    development = pd.read_csv(protocol_dir / "development_cases.csv")
    locked_test = pd.read_csv(protocol_dir / "locked_test_cases.csv")
    assignments = pd.read_csv(protocol_dir / "case_assignments.csv")
    folds = load_json(protocol_dir / "splits_final.json")
    nnunet_folds = load_json(
        root
        / "processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json"
    )

    source_index, source_counts = source_npz_index(
        root / "processed/brats2024_npz_full"
    )
    source_ids = set(source_index)
    development_ids = set(development["case_id"].astype(str))
    test_ids = set(locked_test["case_id"].astype(str))
    development_subjects = {subject_id(item) for item in development_ids}
    test_subjects = {subject_id(item) for item in test_ids}

    audit.require(
        len(source_ids) == 1350 and source_counts == {"train": 945, "val": 202, "test": 203},
        "source_case_inventory",
        "The source NPZ archive contains 1,350 unique cases.",
        "The source NPZ case inventory is incomplete or duplicated.",
        split_counts=source_counts,
        unique_cases=len(source_ids),
    )
    audit.require(
        not (development_ids & test_ids)
        and not (development_subjects & test_subjects)
        and development_ids | test_ids == source_ids,
        "development_test_subject_independence",
        "Development and locked test partitions are case- and subject-disjoint and cover all cases.",
        "Development/test coverage or subject independence failed.",
        development_cases=len(development_ids),
        development_subjects=len(development_subjects),
        locked_test_cases=len(test_ids),
        locked_test_subjects=len(test_subjects),
        case_overlap=len(development_ids & test_ids),
        subject_overlap=len(development_subjects & test_subjects),
        coverage=len(development_ids | test_ids),
    )
    audit.require(
        len(assignments) == 1350
        and set(assignments["case_id"].astype(str)) == source_ids
        and assignments["case_id"].astype(str).is_unique,
        "assignment_manifest_coverage",
        "The case-assignment manifest contains each source case exactly once.",
        "The case-assignment manifest is incomplete or duplicated.",
        rows=len(assignments),
    )
    audit.require(
        folds == nnunet_folds,
        "protocol_nnunet_split_identity",
        "The protocol and nnU-Net split files are semantically identical.",
        "The protocol and nnU-Net split files differ.",
        protocol_sha256=sha256(protocol_dir / "splits_final.json"),
        nnunet_sha256=sha256(
            root
            / "processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json"
        ),
    )

    validation_counts = {item: 0 for item in development_ids}
    fold_details: list[dict[str, Any]] = []
    fold_ok = len(folds) == 5
    for index, fold in enumerate(folds):
        train_ids = set(fold["train"])
        val_ids = set(fold["val"])
        train_subjects = {subject_id(item) for item in train_ids}
        val_subjects = {subject_id(item) for item in val_ids}
        current_ok = (
            not (train_ids & val_ids)
            and not (train_subjects & val_subjects)
            and train_ids | val_ids == development_ids
        )
        fold_ok = fold_ok and current_ok
        for item in val_ids:
            if item in validation_counts:
                validation_counts[item] += 1
        fold_details.append(
            {
                "fold": index,
                "train_cases": len(train_ids),
                "validation_cases": len(val_ids),
                "train_subjects": len(train_subjects),
                "validation_subjects": len(val_subjects),
                "case_overlap": len(train_ids & val_ids),
                "subject_overlap": len(train_subjects & val_subjects),
                "coverage_ok": train_ids | val_ids == development_ids,
            }
        )
    signatures = {exact_fold_signature(fold) for fold in folds}
    validated_once = set(validation_counts.values()) == {1}
    fold_ok = fold_ok and len(signatures) == 5 and validated_once
    audit.require(
        fold_ok,
        "grouped_fivefold_integrity",
        "Five unique subject-disjoint folds cover every development case exactly once in validation.",
        "The grouped five-fold protocol failed a leakage, uniqueness, or coverage check.",
        folds=fold_details,
        unique_fold_pairs=len(signatures),
        every_development_case_validated_once=validated_once,
    )

    dataset_dir = root / "processed/nnunet_raw/Dataset503_BraTS2024Grouped4Ch"
    label_ids = {item.name.removesuffix(".nii.gz") for item in (dataset_dir / "labelsTr").glob("*.nii.gz")}
    image_counts = {
        channel: len(list((dataset_dir / "imagesTr").glob(f"*_{channel:04d}.nii.gz")))
        for channel in range(4)
    }
    image_case_ids = {
        path.name.rsplit("_", 1)[0]
        for path in (dataset_dir / "imagesTr").glob("*.nii.gz")
    }
    audit.require(
        label_ids == development_ids
        and image_case_ids == development_ids
        and image_counts == {0: 1155, 1: 1155, 2: 1155, 3: 1155},
        "nnunet_development_inventory",
        "Dataset503 contains exactly the 1,155 development cases with four image channels and one label map each.",
        "Dataset503 does not exactly match the development manifest.",
        labels=len(label_ids),
        image_case_ids=len(image_case_ids),
        image_counts=image_counts,
        locked_test_labels_present=len(label_ids & test_ids),
        locked_test_images_present=len(image_case_ids & test_ids),
    )

    old_split = pd.read_csv(root / "_archive_inspect/eda_outputs/eda_outputs/brats2024_split.csv")
    old_split["subject_id"] = old_split["case_id"].astype(str).map(subject_id)
    old_subjects = {
        name: set(group["subject_id"].astype(str))
        for name, group in old_split.groupby("split")
    }
    old_overlap = {
        "train_val": len(old_subjects["train"] & old_subjects["val"]),
        "train_test": len(old_subjects["train"] & old_subjects["test"]),
        "val_test": len(old_subjects["val"] & old_subjects["test"]),
    }
    audit.add(
        "legacy_split_is_not_publishable",
        "PASS" if any(old_overlap.values()) else "FAIL",
        "critical",
        (
            "The legacy 945/202/203 case-level split is independently confirmed to contain longitudinal subject leakage "
            "and must not be used for publication claims."
        ),
        pairwise_subject_overlap=old_overlap,
    )

    old_membership = old_split[["case_id", "split"]].copy()
    old_membership["case_id"] = old_membership["case_id"].astype(str)
    locked_prior_roles = (
        old_membership.loc[old_membership["case_id"].isin(test_ids), "split"]
        .value_counts()
        .reindex(["train", "val", "test"], fill_value=0)
        .astype(int)
        .to_dict()
    )
    audit.add(
        "late_defined_holdout_prior_reuse_boundary",
        "WARN",
        "critical",
        (
            "The corrected holdout is case- and subject-disjoint from the final model's training data, "
            "but it was defined after legacy exploratory experiments. Of its 195 cases, "
            f"{locked_prior_roles['train']} previously served as legacy training cases, "
            f"{locked_prior_roles['val']} as validation cases, and {locked_prior_roles['test']} as test cases. "
            "It is therefore a late-defined retrospective holdout, not a pristine external or prospectively untouched test set."
        ),
        corrected_holdout_cases=len(test_ids),
        legacy_roles=locked_prior_roles,
        final_checkpoint_direct_training_overlap=0,
        allowed_claim=(
            "Secondary retrospective subject-disjoint holdout evidence only; primary generalization evidence must come from grouped OOF estimates."
        ),
    )
    return {
        "source_index": source_index,
        "development_ids": development_ids,
        "locked_test_ids": test_ids,
        "folds": folds,
        "dataset_dir": dataset_dir,
    }


def audit_array_fidelity(
    root: Path,
    audit: Audit,
    protocol: dict[str, Any],
    mode: str,
    limit: int,
) -> None:
    if mode == "none":
        audit.add(
            "npz_to_nifti_array_fidelity",
            "PENDING",
            "major",
            "Full array fidelity was not requested in this audit run.",
            command=(
                "py -3.10 research/publication_integrity_v1/audit_publication_integrity.py "
                "--array-check full"
            ),
        )
        return

    case_ids = sorted(protocol["development_ids"])
    if limit > 0:
        case_ids = case_ids[:limit]
    source_index: dict[str, Path] = protocol["source_index"]
    dataset_dir: Path = protocol["dataset_dir"]
    mismatches: list[dict[str, Any]] = []
    modality_mismatches: list[str] = []
    identity_affines = 0
    label_sets: set[int] = set()
    for position, case_id in enumerate(case_ids, start=1):
        with np.load(source_index[case_id], allow_pickle=False) as data:
            source_seg = np.asarray(data["seg"], dtype=np.uint8)
            modalities = [str(item) for item in np.asarray(data["modalities"]).tolist()]
            if modalities != ["t1n", "t1c", "t2w", "t2f"]:
                modality_mismatches.append(case_id)
            label_path = dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
            label_image = nib.load(str(label_path))
            converted_seg = np.asanyarray(label_image.dataobj).astype(np.uint8, copy=False)
            if np.array_equal(label_image.affine, np.eye(4)):
                identity_affines += 1
            label_sets.update(int(item) for item in np.unique(converted_seg))
            if not np.array_equal(source_seg, converted_seg):
                mismatches.append({"case_id": case_id, "part": "segmentation"})
            if mode == "full":
                source_image = np.asarray(data["image"], dtype=np.float32)
                for channel in range(4):
                    image_path = dataset_dir / "imagesTr" / f"{case_id}_{channel:04d}.nii.gz"
                    converted_image = np.asanyarray(nib.load(str(image_path)).dataobj).astype(
                        np.float32, copy=False
                    )
                    if not np.array_equal(source_image[channel], converted_image):
                        max_abs = float(
                            np.max(np.abs(source_image[channel] - converted_image))
                        )
                        mismatches.append(
                            {
                                "case_id": case_id,
                                "part": f"channel_{channel}",
                                "max_abs_error": max_abs,
                            }
                        )
        if position % 100 == 0 or position == len(case_ids):
            print(
                json.dumps(
                    {
                        "array_audit_completed": position,
                        "array_audit_total": len(case_ids),
                        "mismatches": len(mismatches),
                    }
                ),
                flush=True,
            )
    audit.require(
        not mismatches and not modality_mismatches and label_sets <= {0, 1, 2, 3, 4},
        "npz_to_nifti_array_fidelity",
        f"Converted NIfTI {mode} arrays exactly match source NPZ arrays for {len(case_ids)} development cases.",
        "NPZ-to-NIfTI array fidelity failed.",
        mode=mode,
        checked_cases=len(case_ids),
        mismatches=mismatches[:50],
        mismatch_count=len(mismatches),
        modality_order_mismatch_count=len(modality_mismatches),
        observed_labels=sorted(label_sets),
        identity_affine_label_files=identity_affines,
        geometry_limitation=(
            "Identity affines confirm voxel-array preservation only; original scanner/MNI affine provenance is unavailable."
        ),
    )


def recompute_metric_summary(case_metrics: pd.DataFrame) -> dict[str, float]:
    return {
        region: float(
            pd.to_numeric(
                case_metrics[f"{region}_LesionWise_Score_Dice"], errors="coerce"
            ).mean()
        )
        for region in REGIONS
    }


def audit_metric_directory(
    audit: Audit,
    check_id: str,
    directory: Path,
    expected_ids: set[str],
    vendor_hash: str,
) -> dict[str, Any] | None:
    required = (
        directory / "case_metrics_official.csv",
        directory / "summary_official.csv",
        directory / "metadata.json",
        directory / "failures.json",
    )
    if not all(path.is_file() for path in required):
        audit.add(
            check_id,
            "PENDING",
            "major",
            "Official metric artifacts are not complete yet.",
            directory=str(directory),
            missing=[str(path) for path in required if not path.is_file()],
        )
        return None
    case_metrics = pd.read_csv(required[0])
    summary = pd.read_csv(required[1]).set_index("region")
    metadata = load_json(required[2])
    failures = load_json(required[3])
    observed_ids = set(case_metrics["case_id"].astype(str))
    recomputed = recompute_metric_summary(case_metrics)
    reported = {
        region: float(summary.loc[region, "LesionWise_Score_Dice_mean"])
        for region in REGIONS
    }
    summary_error = max(abs(recomputed[item] - reported[item]) for item in REGIONS)
    metric_script = Path(metadata.get("metric_script", ""))
    metric_script_ok = metric_script.name == "metrics_GLI.py"
    passed = (
        observed_ids == expected_ids
        and len(case_metrics) == len(expected_ids)
        and metadata.get("successful_cases") == len(expected_ids)
        and metadata.get("failed_cases") == 0
        and not failures
        and metric_script_ok
        and summary_error < 1e-12
    )
    audit.require(
        passed,
        check_id,
        "Official BraTS lesion-wise metrics cover the exact intended case set and recompute exactly.",
        "Official metric coverage, failure count, or summary recomputation failed.",
        directory=str(directory),
        expected_cases=len(expected_ids),
        observed_cases=len(observed_ids),
        missing_cases=sorted(expected_ids - observed_ids)[:20],
        unexpected_cases=sorted(observed_ids - expected_ids)[:20],
        failures=len(failures),
        metric_repository_commit=metadata.get("metric_repository_commit"),
        vendor_metrics_sha256=vendor_hash,
        maximum_summary_recompute_error=summary_error,
        six_region_mean=float(np.mean(list(recomputed.values()))),
        regions=recomputed,
    )
    return {
        "six_region_mean": float(np.mean(list(recomputed.values()))),
        "regions": recomputed,
        "cases": len(observed_ids),
    }


def audit_metrics_and_folds(root: Path, audit: Audit, protocol: dict[str, Any]) -> None:
    vendor_script = root / "research/vendor/BraTS-2024-Metrics/metrics_GLI.py"
    vendor_hash = sha256(vendor_script)
    audit.require(
        vendor_script.is_file(),
        "official_metric_vendor_present",
        "The vendored official BraTS 2024 GLI metric implementation is present and hashable.",
        "The official BraTS 2024 GLI metric implementation is missing.",
        path=str(vendor_script),
        sha256=vendor_hash,
    )
    fold_directories = {
        0: root / "experiments/grouped_delta_roi_v1/official_metrics/baseline",
        1: root / "experiments/grouped_fivefold_cv_v11/fold_1/official_metrics",
        2: root / "experiments/grouped_fivefold_cv_v11/fold_2/official_metrics",
        3: root / "experiments/grouped_fivefold_cv_v11/fold_3/official_metrics",
        4: root / "experiments/grouped_fivefold_cv_v11/fold_4/official_metrics",
    }
    completed: list[dict[str, Any]] = []
    for fold_index, directory in fold_directories.items():
        result = audit_metric_directory(
            audit,
            f"grouped_fold_{fold_index}_official_metrics",
            directory,
            set(protocol["folds"][fold_index]["val"]),
            vendor_hash,
        )
        if result is not None:
            result["fold"] = fold_index
            completed.append(result)
    audit.add(
        "grouped_fivefold_completion",
        "PASS" if len(completed) == 5 else "PENDING",
        "major",
        (
            "All five grouped folds have official out-of-fold metrics."
            if len(completed) == 5
            else f"{len(completed)}/5 grouped folds currently have complete official metrics."
        ),
        completed_folds=[item["fold"] for item in completed],
        fold_results=completed,
    )


def audit_checkpoint_recipe(root: Path, audit: Audit) -> None:
    candidates = {
        0: root
        / "experiments/nnunet_results/Dataset503_BraTS2024Grouped4Ch/nnUNetTrainer_250epochs__nnUNetResEncUNet10p5GBPlans__3d_fullres/fold_0/checkpoint_final.pth",
        1: root
        / "experiments/nnunet_results/Dataset503_BraTS2024Grouped4Ch/nnUNetTrainer_GroupedCV250Duty80__nnUNetResEncUNet10p5GBPlans__3d_fullres/fold_1/checkpoint_final.pth",
        2: root
        / "experiments/nnunet_results/Dataset503_BraTS2024Grouped4Ch/nnUNetTrainer_GroupedCV250Duty80__nnUNetResEncUNet10p5GBPlans__3d_fullres/fold_2/checkpoint_final.pth",
        3: root
        / "experiments/nnunet_results/Dataset503_BraTS2024Grouped4Ch/nnUNetTrainer_GroupedCV250Duty80__nnUNetResEncUNet10p5GBPlans__3d_fullres/fold_3/checkpoint_final.pth",
        4: root
        / "experiments/nnunet_results/Dataset503_BraTS2024Grouped4Ch/nnUNetTrainer_GroupedCV250Duty80__nnUNetResEncUNet10p5GBPlans__3d_fullres/fold_4/checkpoint_final.pth",
    }
    complete = [index for index, path in candidates.items() if path.is_file()]
    audit.add(
        "checkpoint_completion",
        "PASS" if len(complete) == 5 else "PENDING",
        "major",
        f"Final checkpoints exist for {len(complete)}/5 grouped folds.",
        completed_folds=complete,
        missing_folds=[index for index in range(5) if index not in complete],
        note=(
            "The fixed 250-epoch budget is reproducible, but late validation improvement means the manuscript must not claim demonstrated convergence."
        ),
    )
    metadata_path = (
        root
        / "experiments/publication_integrity_v1/checkpoint_metadata/checkpoint_metadata.json"
    )
    if not metadata_path.is_file():
        audit.add(
            "checkpoint_metadata_consistency",
            "PENDING",
            "major",
            "Checkpoint recipe metadata has not been inspected yet.",
            command=(
                "py -3.10 research/publication_integrity_v1/"
                "inspect_checkpoint_metadata.py"
            ),
        )
        return
    metadata = load_json(metadata_path)
    recorded_complete = int(metadata.get("completed_folds", -1))
    metadata_status = str(metadata.get("status", "FAIL"))
    stale = recorded_complete != len(complete)
    if metadata_status == "FAIL":
        status = "FAIL"
    elif stale or len(complete) < 5:
        status = "PENDING"
    else:
        status = "PASS"
    audit.add(
        "checkpoint_metadata_consistency",
        status,
        "critical" if status == "FAIL" else "major",
        (
            "All five final/best checkpoint pairs match the fixed dataset, plan, "
            "architecture, optimizer, fold, and epoch recipe."
            if status == "PASS"
            else (
                "Checkpoint metadata inspection found a recipe mismatch."
                if status == "FAIL"
                else f"Checkpoint metadata is consistent for {len(complete)}/5 completed folds."
            )
        ),
        metadata_path=str(metadata_path),
        metadata_status=metadata_status,
        recorded_completed_folds=recorded_complete,
        current_completed_folds=len(complete),
        stale=stale,
        cross_checkpoint_checks=metadata.get("cross_checkpoint_checks", {}),
        invalid_checkpoints=metadata.get("invalid_checkpoints", []),
        reproducibility_boundary=metadata.get("reproducibility_boundary"),
    )
    recipe_path = (
        root
        / "experiments/publication_integrity_v1/training_recipe/training_recipe.json"
    )
    if not recipe_path.is_file():
        audit.add(
            "training_recipe_provenance",
            "PENDING",
            "major",
            "The publication training/inference recipe has not been exported yet.",
            command=(
                "py -3.10 research/publication_integrity_v1/"
                "build_training_recipe.py"
            ),
        )
        return
    recipe = load_json(recipe_path)
    source_records = [
        recipe["software"]["nnunet_trainer_source"],
        recipe["software"]["nnunet_250epoch_source"],
        recipe["software"]["project_duty_trainer_source"],
        recipe["software"]["plans_source"],
        recipe["software"]["dataset_json_source"],
    ]
    source_hashes_match = all(
        Path(item["path"]).is_file()
        and sha256(Path(item["path"])) == item["sha256"]
        for item in source_records
    )
    expected_recipe = (
        recipe.get("configuration") == "3d_fullres"
        and recipe["dataset"].get("num_training") == 1155
        and recipe["plan"].get("batch_size") == 2
        and recipe["plan"].get("patch_size") == [128, 192, 128]
        and recipe["training"].get("epochs") == 250
        and recipe["training"].get("initial_lr") == 0.01
        and recipe["augmentation"].get("mirror_axes") == [0, 1, 2]
        and recipe["inference"].get("tile_step_size") == 0.5
        and recipe["inference"].get("save_probabilities") is False
    )
    audit.require(
        source_hashes_match and expected_recipe,
        "training_recipe_provenance",
        "The exported training/inference recipe matches the hashed nnU-Net sources and Dataset503 plan.",
        "The exported training recipe is stale or inconsistent with its hashed sources.",
        path=str(recipe_path),
        source_hashes_match=source_hashes_match,
        expected_recipe=expected_recipe,
    )


def audit_locked_test(root: Path, audit: Audit, protocol: dict[str, Any]) -> None:
    experiment = root / "experiments/final_locked_test_v1"
    frozen_path = experiment / "frozen_model_manifest.json"
    summary_path = experiment / "test_summary.json"
    frozen = load_json(frozen_path)
    summary = load_json(summary_path)
    hash_checks: list[dict[str, Any]] = []
    hashes_ok = True
    for name, model in frozen["models"].items():
        path = Path(model["checkpoint"])
        actual = sha256(path) if path.is_file() else None
        match = actual == str(model["sha256"]).upper()
        hashes_ok = hashes_ok and match
        hash_checks.append(
            {
                "model": name,
                "path": str(path),
                "expected_sha256": model["sha256"],
                "actual_sha256": actual,
                "match": match,
            }
        )
    protocol_hash_ok = (
        sha256(root / "processed/brats_subject_grouped_protocol_v1/locked_test_cases.csv")
        == frozen["protocol"]["locked_test_csv_sha256"]
        and sha256(
            root / "processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json"
        )
        == frozen["protocol"]["split_sha256"]
    )
    audit.require(
        frozen.get("status") == "FROZEN"
        and frozen.get("selection_completed_before_test") is True
        and frozen.get("post_test_model_selection_allowed") is False
        and hashes_ok
        and protocol_hash_ok,
        "locked_test_freeze_integrity",
        "The one-time test model, checkpoints, split, and test manifest still match the pre-test freeze record.",
        "The locked-test freeze record or an artifact hash no longer matches.",
        frozen_at_utc=frozen.get("frozen_at_utc"),
        model_hashes=hash_checks,
        protocol_hash_match=protocol_hash_ok,
    )

    test_ids = protocol["locked_test_ids"]
    baseline = audit_metric_directory(
        audit,
        "locked_test_baseline_metric_integrity",
        experiment / "official_metrics/baseline",
        test_ids,
        sha256(root / "research/vendor/BraTS-2024-Metrics/metrics_GLI.py"),
    )
    final = audit_metric_directory(
        audit,
        "locked_test_final_metric_integrity",
        experiment / "official_metrics/frozen_v2",
        test_ids,
        sha256(root / "research/vendor/BraTS-2024-Metrics/metrics_GLI.py"),
    )
    audit.require(
        summary.get("status") == "COMPLETE_ONE_TIME_LOCKED_TEST"
        and summary.get("cases") == 195
        and summary.get("subjects") == 92
        and summary.get("model_selection_after_test") is False
        and baseline is not None
        and final is not None,
        "locked_test_reporting_integrity",
        "The locked-test report covers the intended 195 cases/92 subjects and is marked as one-time evaluation.",
        "The locked-test report is incomplete or inconsistent.",
        report_status=summary.get("status"),
        cases=summary.get("cases"),
        subjects=summary.get("subjects"),
    )
    ci = summary["paired_six_region_mean_delta_ci95"]
    audit.add(
        "locked_test_claim_boundary",
        "WARN",
        "critical",
        (
            "The frozen V2 gain is not statistically conclusive on the locked test: the subject-cluster 95% CI crosses zero. "
            "The final model must not be claimed superior, and the strong negative tiny-ET subgroup result must be reported."
        ),
        baseline_mean=summary["baseline_six_region_mean"],
        frozen_v2_mean=summary["final_six_region_mean"],
        paired_delta=summary["paired_six_region_mean_delta"],
        paired_delta_ci95=ci,
        tiny_et_bottom_25_delta=next(
            item["paired_delta"]
            for item in summary["subgroups"]
            if item["subgroup"].startswith("ET-positive bottom 25%")
        ),
    )
    test_time = summary_path.stat().st_mtime
    later_checkpoints = [
        str(path)
        for path in (root / "experiments/nnunet_results").rglob("checkpoint_*.pth")
        if path.stat().st_mtime > test_time + 1.0
    ]
    audit.add(
        "post_test_artifact_boundary",
        "WARN" if later_checkpoints else "PASS",
        "critical" if later_checkpoints else "info",
        (
            "Post-test checkpoints exist for completion of the pre-specified grouped cross-validation. "
            "They may be used only for development-fold analyses and must never be evaluated on the already opened locked test."
            if later_checkpoints
            else "No model checkpoint was created or modified after the one-time test report."
        ),
        test_report_time=datetime.fromtimestamp(test_time, tz=timezone.utc).isoformat(),
        later_checkpoints=later_checkpoints,
        locked_test_reuse_allowed=False,
    )


def audit_pretraining_and_manuscript(root: Path, audit: Audit) -> None:
    pretrain = load_json(root / "experiments/pretraining_pipeline_audit_20260716/summary.json")
    split_audit = load_json(
        root / "experiments/pretrain_subject_split_audit_20260716/summary.json"
    )
    audit.add(
        "pretraining_claim_boundary",
        "WARN",
        "critical",
        (
            "The original ADNI/OASIS pretraining experiment is not patient-independent and has orientation/background-objective limitations. "
            "It may be reported as an audited negative/limited ablation, not as proven anatomical transfer."
        ),
        leaked_subjects=split_audit["legacy_file_level_split"].get("overlap_subjects"),
        affected_validation_files=split_audit["legacy_file_level_split"].get("affected_validation_files"),
        audit_status=pretrain.get("status"),
    )
    provenance_path = (
        root
        / "experiments/publication_integrity_v1/data_provenance/data_provenance.json"
    )
    if not provenance_path.is_file():
        audit.add(
            "data_preprocessing_provenance",
            "PENDING",
            "major",
            "Archived HD-BET and NPZ preprocessing provenance has not been audited.",
            command=(
                "py -3.10 research/publication_integrity_v1/"
                "audit_data_provenance.py"
            ),
        )
    else:
        provenance = load_json(provenance_path)
        provenance_pass = provenance.get("status") == "PASS_WITH_PROVENANCE_LIMITATIONS"
        audit.add(
            "data_preprocessing_provenance",
            "WARN" if provenance_pass else "FAIL",
            "major" if provenance_pass else "critical",
            (
                "All 565 normal-T1 NPZ inputs trace to successful HD-BET outputs; BraTS used inherited challenge preprocessing, while OASIS command details and original affines remain unavailable."
                if provenance_pass
                else "The archived HD-BET/NPZ preprocessing chain is inconsistent."
            ),
            path=str(provenance_path),
            normal_t1=provenance.get("normal_t1_pretraining"),
            brats=provenance.get("brats_segmentation"),
            claim_boundary=provenance.get("claim_boundary"),
        )

    manuscript = root / "paper_results_assets/results_section.tex"
    text = manuscript.read_text(encoding="utf-8") if manuscript.is_file() else ""
    stale_markers = [
        marker
        for marker in (
            "945 training cases and 202 validation cases",
            "Five-model nnU-Net ensemble",
            "0.7886",
            "fixed BraTS validation split",
            "subject-weighted sensitivity estimate",
            "ground truth",
        )
        if marker in text
    ]
    required_disclosures = (
        "late-defined retrospective",
        "139",
        "post-hoc exploratory",
        "565 normal T1",
    )
    normalized_text = text.lower()
    missing_disclosures = [
        marker for marker in required_disclosures if marker.lower() not in normalized_text
    ]
    audit.require(
        not stale_markers and not missing_disclosures,
        "manuscript_uses_corrected_protocol",
        "The Results draft excludes legacy claims and discloses the late-defined holdout and exploratory subgroup boundaries.",
        "The Results draft contains a legacy claim or omits a required holdout/subgroup disclosure.",
        path=str(manuscript),
        stale_markers=stale_markers,
        missing_disclosures=missing_disclosures,
    )

    statistics_path = (
        root
        / "experiments/publication_integrity_v1/statistics/publication_statistics.json"
    )
    if not statistics_path.is_file() or not manuscript.is_file():
        audit.add(
            "manuscript_statistics_synchronized",
            "PENDING",
            "major",
            "The Results draft cannot yet be cross-checked against publication statistics.",
            statistics_path=str(statistics_path),
            manuscript_path=str(manuscript),
        )
        return

    statistics = load_json(statistics_path)
    oof = statistics["grouped_oof"]
    pooled = oof["case_weighted_cluster_bootstrap"]
    completed = int(oof["completed_fold_count"])
    expected = int(oof["expected_fold_count"])
    et_detection = oof["lesion_detection"]["ET"]
    fold_progress_marker = (
        "All five pre-specified"
        if completed == expected == 5
        else f"{completed} of the {expected} pre-specified"
    )
    expected_markers = {
        "generated_header": "Do not manually edit numeric OOF values",
        "fold_progress": fold_progress_marker,
        "oof_cases": f"{int(oof['cases'])} non-duplicated OOF scans",
        "oof_subjects": f"{int(oof['subjects'])} subjects",
        "six_region_mean": f"Dice was {float(pooled['six_region_mean']['mean']):.4f}",
        "subject_weighted_mean": (
            f"{float(oof['subject_weighted_alternative_estimand']['six_region_mean']):.4f}"
        ),
        "et_detection_counts": (
            f"{int(et_detection['true_positive_lesions'])} of "
            f"{int(et_detection['reference_lesions'])} reference"
        ),
    }
    for region in REGIONS:
        metric = pooled[f"{region}_LesionWise_Score_Dice"]
        expected_markers[f"{region}_dice"] = f"{float(metric['mean']):.4f}"
    missing_statistical_markers = {
        key: marker for key, marker in expected_markers.items() if marker not in text
    }
    subsection_count = len(re.findall(r"^\\subsection\{", text, flags=re.MULTILINE))
    synchronization_ok = subsection_count == 5 and not missing_statistical_markers
    audit.require(
        synchronization_ok,
        "manuscript_statistics_synchronized",
        "The five-part Results draft is synchronized with the machine-readable OOF statistics.",
        "The Results draft is structurally or numerically out of sync with publication statistics.",
        statistics_path=str(statistics_path),
        manuscript_path=str(manuscript),
        subsection_count=subsection_count,
        expected_subsection_count=5,
        completed_folds=completed,
        expected_folds=expected,
        missing_statistical_markers=missing_statistical_markers,
    )

    source_evidence_path = (
        root / "research/publication_integrity_v1/METHODS_SOURCE_EVIDENCE.md"
    )
    bibliography_path = root / "paper_results_assets/publication_references.bib"
    source_text = (
        source_evidence_path.read_text(encoding="utf-8")
        if source_evidence_path.is_file()
        else ""
    )
    bibliography_text = (
        bibliography_path.read_text(encoding="utf-8")
        if bibliography_path.is_file()
        else ""
    )
    required_source_markers = (
        "late-defined retrospective holdout",
        "reference standard",
        "internal testing",
        "label 1: NETC",
        "label 4: RC",
        "WT: labels 1, 2, and 3",
    )
    required_bib_keys = (
        "deVerdier2024BraTSPostTreatment",
        "Tejani2024CLAIM",
        "MaierHein2024MetricsReloaded",
        "Isensee2021nnUNet",
        "Isensee2019HDBET",
    )
    missing_source_markers = [
        marker for marker in required_source_markers if marker not in source_text
    ]
    missing_bib_keys = [
        marker for marker in required_bib_keys if marker not in bibliography_text
    ]
    audit.require(
        not missing_source_markers and not missing_bib_keys,
        "source_attribution_controls",
        "Methods claim boundaries and primary bibliography entries are archived.",
        "Methods source attribution or required bibliography entries are incomplete.",
        source_evidence_path=str(source_evidence_path),
        bibliography_path=str(bibliography_path),
        missing_source_markers=missing_source_markers,
        missing_bib_keys=missing_bib_keys,
    )


def render_markdown(payload: dict[str, Any]) -> str:
    checks_by_id = {item["check_id"]: item for item in payload["checks"]}
    fivefold_complete = (
        checks_by_id.get("grouped_fivefold_completion", {}).get("status")
        == "PASS"
    )
    if fivefold_complete:
        fold_interpretation = (
            "- The corrected Dataset503 protocol is subject-disjoint, and all five "
            "pre-specified folds have complete official OOF metrics."
        )
    else:
        fold_interpretation = (
            "- The corrected Dataset503 protocol is subject-disjoint; full five-fold "
            "evidence remains in progress."
        )

    if payload["counts"]["FAIL"]:
        submission_gate = (
            "The evidence package is not ready for submission until all FAIL checks "
            "are corrected."
        )
    elif payload["counts"]["PENDING"]:
        submission_gate = (
            "The evidence package remains in progress until all PENDING checks are "
            "resolved."
        )
    else:
        submission_gate = (
            "The automated evidence gates are complete. Any manuscript submission "
            "must retain every WARN limitation and must not claim unsupported SOTA "
            "superiority or clinical readiness."
        )

    lines = [
        "# Publication integrity audit",
        "",
        f"- Overall status: **{payload['overall_status']}**",
        f"- Generated: {payload['generated_at_utc']}",
        f"- Checks: {payload['counts']}",
        "",
        "## Critical interpretation",
        "",
        "- The legacy 945/202/203 case-level experiment is not publication-valid because longitudinal subjects cross partitions.",
        fold_interpretation,
        "- The one-time holdout has already been used and must never be reused for model selection. Because it was defined after legacy experiments, it is retrospective rather than pristine external test evidence.",
        "- The frozen V2 test gain is small and inconclusive, and tiny ET-positive cases worsened. Superiority or clinical-readiness claims are not supported.",
        "- The original normal-anatomy pretraining result is an audited limited/negative ablation, not proof of anatomy-aware transfer.",
        "",
        "## Checks",
        "",
        "| Status | Severity | Check | Finding |",
        "|---|---|---|---|",
    ]
    for item in payload["checks"]:
        message = item["message"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item['status']} | {item['severity']} | `{item['check_id']}` | {message} |"
        )
    lines.extend(
        [
            "",
            "## Submission gate",
            "",
            submission_gate,
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = Audit()
    protocol = audit_protocol(root, audit)
    audit_array_fidelity(
        root,
        audit,
        protocol,
        args.array_check,
        max(0, args.array_limit),
    )
    audit_metrics_and_folds(root, audit, protocol)
    audit_checkpoint_recipe(root, audit)
    audit_locked_test(root, audit, protocol)
    audit_pretraining_and_manuscript(root, audit)

    counts = {
        status: sum(item.status == status for item in audit.checks)
        for status in ("PASS", "WARN", "FAIL", "PENDING")
    }
    if counts["FAIL"]:
        overall = "NOT_SUBMISSION_READY"
    elif counts["PENDING"]:
        overall = "EVIDENCE_IN_PROGRESS"
    elif counts["WARN"]:
        overall = "READY_WITH_MANDATORY_LIMITATIONS"
    else:
        overall = "READY"
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "overall_status": overall,
        "counts": counts,
        "array_check": args.array_check,
        "array_limit": args.array_limit,
        "checks": [asdict(item) for item in audit.checks],
    }
    (output_dir / "audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "audit.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    if counts["FAIL"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
