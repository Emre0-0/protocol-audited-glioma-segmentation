from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def dataset_folder(payload: dict[str, Any]) -> str:
    return f"Dataset{int(payload['id']):03d}_{payload['name']}"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.stat().st_size != target.stat().st_size:
            raise RuntimeError(f"Existing target differs from source: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def save_nifti(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.ascontiguousarray(array), np.eye(4, dtype=np.float32))
    image.header.set_xyzt_units("mm")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.nii.gz")
    nib.save(image, str(temporary))
    os.replace(temporary, path)


def convert_npz_case(task: tuple[str, str, str]) -> dict[str, Any]:
    case_id, npz_value, dataset_value = task
    npz_path = Path(npz_value)
    dataset_dir = Path(dataset_value)
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    expected = [images_dir / f"{case_id}_{channel:04d}.nii.gz" for channel in range(4)]
    expected.append(labels_dir / f"{case_id}.nii.gz")
    if all(path.exists() for path in expected):
        return {"case_id": case_id, "status": "existing"}

    with np.load(npz_path, allow_pickle=True) as data:
        image = np.asarray(data["image"], dtype=np.float32)
        segmentation = np.asarray(data["seg"], dtype=np.uint8)
        stored_case = str(np.asarray(data["case_id"]).item())
    if stored_case != case_id:
        raise ValueError(f"Case ID mismatch: expected {case_id}, found {stored_case}")
    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError(f"Unexpected image shape for {case_id}: {image.shape}")
    if segmentation.shape != image.shape[1:]:
        raise ValueError(
            f"Segmentation shape mismatch for {case_id}: {segmentation.shape} vs {image.shape[1:]}"
        )
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite image values for {case_id}")
    labels = set(np.unique(segmentation).tolist())
    if not labels.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"Unexpected labels for {case_id}: {sorted(labels)}")

    for channel in range(4):
        save_nifti(image[channel], images_dir / f"{case_id}_{channel:04d}.nii.gz")
    save_nifti(segmentation, labels_dir / f"{case_id}.nii.gz")
    return {"case_id": case_id, "status": "converted", "shape": list(segmentation.shape)}


def create_delta_case(task: tuple[str, str]) -> dict[str, Any]:
    case_id, dataset_value = task
    dataset_dir = Path(dataset_value)
    images_dir = dataset_dir / "imagesTr"
    output_path = images_dir / f"{case_id}_0004.nii.gz"
    if output_path.exists():
        return {"case_id": case_id, "status": "existing"}
    t1_image = nib.load(str(images_dir / f"{case_id}_0000.nii.gz"))
    t1c_image = nib.load(str(images_dir / f"{case_id}_0001.nii.gz"))
    if t1_image.shape != t1c_image.shape:
        raise ValueError(f"T1/T1c shape mismatch for {case_id}")
    t1 = np.asarray(t1_image.dataobj, dtype=np.float32)
    t1c = np.asarray(t1c_image.dataobj, dtype=np.float32)
    support = (t1 != 0) | (t1c != 0)
    delta = np.zeros(t1.shape, dtype=np.float32)
    delta[support] = t1c[support] - t1[support]
    if not np.isfinite(delta).all():
        raise ValueError(f"Non-finite DeltaT1 values for {case_id}")
    header = t1_image.header.copy()
    header.set_data_dtype(np.float32)
    output = nib.Nifti1Image(delta, t1_image.affine, header=header)
    output.set_qform(t1_image.get_qform(), int(t1_image.header["qform_code"]))
    output.set_sform(t1_image.get_sform(), int(t1_image.header["sform_code"]))
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.nii.gz")
    nib.save(output, str(temporary))
    os.replace(temporary, output_path)
    return {
        "case_id": case_id,
        "status": "created",
        "brain_voxels": int(support.sum()),
        "mean": float(delta[support].mean()),
        "std": float(delta[support].std()),
    }


def index_npz(npz_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for path in sorted((npz_root / split).glob("*.npz")):
            if path.stem in index:
                raise RuntimeError(f"Duplicate NPZ case ID: {path.stem}")
            index[path.stem] = path
    return index


def write_dataset_json(dataset_dir: Path, name: str, case_count: int, delta: bool) -> None:
    channels = {"0": "t1n", "1": "t1c", "2": "t2w", "3": "t2f"}
    if delta:
        channels["4"] = "t1c_minus_t1n"
    payload = {
        "channel_names": channels,
        "labels": {
            "background": 0,
            "label_1": 1,
            "label_2": 2,
            "label_3": 3,
            "label_4": 4,
        },
        "numTraining": int(case_count),
        "file_ending": ".nii.gz",
        "name": name,
        "description": (
            "Patient-independent BraTS 2024 development dataset. All longitudinal scans from "
            "a subject remain in one partition. The locked subject-level test set is excluded."
        ),
    }
    atomic_json(dataset_dir / "dataset.json", payload)


def ensure_exact_case_set(dataset_dir: Path, expected_cases: set[str], channels: int) -> None:
    labels = {
        path.name.removesuffix(".nii.gz") for path in (dataset_dir / "labelsTr").glob("*.nii.gz")
    }
    if labels != expected_cases:
        missing = sorted(expected_cases - labels)
        extra = sorted(labels - expected_cases)
        raise RuntimeError(
            f"Dataset case mismatch for {dataset_dir.name}: missing={missing[:10]} extra={extra[:10]}"
        )
    for channel in range(channels):
        present = {
            path.name.removesuffix(f"_{channel:04d}.nii.gz")
            for path in (dataset_dir / "imagesTr").glob(f"*_{channel:04d}.nii.gz")
        }
        if present != expected_cases:
            raise RuntimeError(
                f"Channel {channel} case mismatch for {dataset_dir.name}: "
                f"missing={sorted(expected_cases - present)[:10]} extra={sorted(present - expected_cases)[:10]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build isolated patient-grouped 4-channel and DeltaT1 nnU-Net datasets."
    )
    parser.add_argument("--config", default="research/grouped_delta_roi_v1/config.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    root = config_path.parents[2]
    config = load_config(config_path)
    protocol_splits = resolve(root, config["protocol"]["splits"])
    development_csv = resolve(root, config["protocol"]["development_cases"])
    locked_test_csv = resolve(root, config["protocol"]["locked_test_cases"])
    splits = json.loads(protocol_splits.read_text(encoding="utf-8-sig"))
    development = pd.read_csv(development_csv)
    locked_test = pd.read_csv(locked_test_csv)
    development_cases = set(development["case_id"].astype(str))
    locked_cases = set(locked_test["case_id"].astype(str))
    if development_cases & locked_cases:
        raise RuntimeError("Development and locked-test case sets overlap")
    if len(development_cases) != int(config["protocol"]["expected_development_cases"]):
        raise RuntimeError(f"Unexpected development case count: {len(development_cases)}")
    split_cases = set().union(*(set(fold["train"]) | set(fold["val"]) for fold in splits))
    if split_cases != development_cases:
        raise RuntimeError("Protocol split cases do not match development_cases.csv")

    raw_root = resolve(root, config["paths"]["raw_root"])
    npz_root = resolve(root, config["paths"]["npz_root"])
    experiment_dir = resolve(root, config["paths"]["experiment_dir"])
    baseline_dir = raw_root / dataset_folder(config["datasets"]["baseline"])
    delta_dir = raw_root / dataset_folder(config["datasets"]["delta"])
    legacy_4ch_dir = raw_root / dataset_folder(config["datasets"]["legacy_4ch_source"])
    legacy_delta_dir = raw_root / dataset_folder(config["datasets"]["legacy_delta_source"])
    npz_index = index_npz(npz_root)
    missing_npz = sorted(development_cases - set(npz_index))
    if missing_npz:
        raise FileNotFoundError(f"Missing development NPZ files: {missing_npz[:10]}")

    baseline_images = baseline_dir / "imagesTr"
    baseline_labels = baseline_dir / "labelsTr"
    baseline_images.mkdir(parents=True, exist_ok=True)
    baseline_labels.mkdir(parents=True, exist_ok=True)
    link_counts = {"hardlink": 0, "copy": 0, "existing": 0}
    conversion_tasks: list[tuple[str, str, str]] = []
    for case_id in tqdm(sorted(development_cases), desc="build-grouped-4ch"):
        source_paths = [
            legacy_4ch_dir / "imagesTr" / f"{case_id}_{channel:04d}.nii.gz"
            for channel in range(4)
        ]
        source_paths.append(legacy_4ch_dir / "labelsTr" / f"{case_id}.nii.gz")
        if all(path.exists() for path in source_paths):
            for channel in range(4):
                status = link_or_copy(
                    source_paths[channel], baseline_images / f"{case_id}_{channel:04d}.nii.gz"
                )
                link_counts[status] += 1
            status = link_or_copy(source_paths[-1], baseline_labels / f"{case_id}.nii.gz")
            link_counts[status] += 1
        else:
            conversion_tasks.append((case_id, str(npz_index[case_id]), str(baseline_dir)))

    converted: list[dict[str, Any]] = []
    if conversion_tasks:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = [executor.submit(convert_npz_case, task) for task in conversion_tasks]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="convert-new-development-cases"
            ):
                converted.append(future.result())

    write_dataset_json(
        baseline_dir,
        str(config["datasets"]["baseline"]["name"]),
        len(development_cases),
        delta=False,
    )
    atomic_json(baseline_dir / "splits" / "splits_final.json", splits)
    ensure_exact_case_set(baseline_dir, development_cases, channels=4)

    delta_images = delta_dir / "imagesTr"
    delta_labels = delta_dir / "labelsTr"
    delta_images.mkdir(parents=True, exist_ok=True)
    delta_labels.mkdir(parents=True, exist_ok=True)
    delta_tasks: list[tuple[str, str]] = []
    for case_id in tqdm(sorted(development_cases), desc="build-grouped-delta-links"):
        for channel in range(4):
            status = link_or_copy(
                baseline_images / f"{case_id}_{channel:04d}.nii.gz",
                delta_images / f"{case_id}_{channel:04d}.nii.gz",
            )
            link_counts[status] += 1
        status = link_or_copy(
            baseline_labels / f"{case_id}.nii.gz", delta_labels / f"{case_id}.nii.gz"
        )
        link_counts[status] += 1
        legacy_delta = legacy_delta_dir / "imagesTr" / f"{case_id}_0004.nii.gz"
        target_delta = delta_images / f"{case_id}_0004.nii.gz"
        if legacy_delta.exists():
            status = link_or_copy(legacy_delta, target_delta)
            link_counts[status] += 1
        elif not target_delta.exists():
            delta_tasks.append((case_id, str(delta_dir)))

    delta_records: list[dict[str, Any]] = []
    if delta_tasks:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = [executor.submit(create_delta_case, task) for task in delta_tasks]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="create-new-delta-channels"
            ):
                delta_records.append(future.result())

    write_dataset_json(
        delta_dir,
        str(config["datasets"]["delta"]["name"]),
        len(development_cases),
        delta=True,
    )
    atomic_json(delta_dir / "splits" / "splits_final.json", splits)
    ensure_exact_case_set(delta_dir, development_cases, channels=5)

    summary = {
        "protocol": "patient-independent grouped development folds",
        "development_cases": len(development_cases),
        "development_subjects": int(development["subject_id"].nunique()),
        "locked_test_cases": len(locked_cases),
        "locked_test_subjects": int(locked_test["subject_id"].nunique()),
        "locked_test_in_training_datasets": False,
        "baseline_dataset": baseline_dir.name,
        "delta_dataset": delta_dir.name,
        "linked_or_copied_files": link_counts,
        "npz_converted_cases": len(conversion_tasks),
        "new_delta_cases": len(delta_tasks),
        "delta_formula": config["delta_channel"]["formula"],
        "fold_count": len(splits),
    }
    atomic_json(experiment_dir / "dataset_build_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
