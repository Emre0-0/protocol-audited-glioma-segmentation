from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_case_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row["case_id"]) for row in csv.DictReader(handle)}


def file_hashes(directory: Path, case_ids: set[str]) -> dict[str, str]:
    return {
        case_id: sha256(directory / f"{case_id}.nii.gz")
        for case_id in sorted(case_ids)
    }


def write_label_atomic(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name.removesuffix('.nii.gz')}.tmp.nii.gz"
    )
    if temporary.exists():
        temporary.unlink()
    image = nib.Nifti1Image(
        np.ascontiguousarray(array, dtype=np.uint8),
        affine=np.eye(4, dtype=np.float32),
    )
    image.header.set_xyzt_units("mm")
    nib.save(image, str(temporary))
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Release locked-test labels only after prediction freeze."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    args = parser.parse_args()

    root = project_root()
    config: dict[str, Any] = json.loads(
        args.config.resolve().read_text(encoding="utf-8")
    )
    experiment_dir = resolve(root, config["paths"]["experiment_dir"])
    inference_lock_path = experiment_dir / "inference_lock.json"
    release_path = experiment_dir / "test_label_release_manifest.json"
    labels_dir = resolve(root, config["paths"]["released_test_labels"])
    test_ids = read_case_ids(
        resolve(root, config["protocol"]["locked_test_cases"])
    )
    expected = int(config["protocol"]["expected_test_cases"])
    if len(test_ids) != expected:
        raise ValueError("Locked-test case count mismatch")

    lock = json.loads(inference_lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "PREDICTIONS_FROZEN" or lock.get(
        "test_labels_used"
    ) is not False:
        raise RuntimeError("Predictions were not frozen before label release")
    for name, key in (
        ("baseline", "baseline_predictions"),
        ("specialist", "specialist_predictions"),
        ("frozen_v2", "final_predictions"),
    ):
        directory = resolve(root, config["paths"][key])
        if file_hashes(directory, test_ids) != lock["prediction_sha256"][name]:
            raise RuntimeError(f"{name} predictions changed before label release")

    if release_path.exists():
        existing = json.loads(release_path.read_text(encoding="utf-8"))
        if existing.get("status") != "TEST_LABELS_RELEASED":
            raise RuntimeError("Existing test label release manifest is invalid")
        print(json.dumps(existing, indent=2))
        return

    npz_root = resolve(root, config["data"]["npz_root"])
    sources = {path.stem: path for path in npz_root.glob("*/*.npz")}
    missing = sorted(test_ids - set(sources))
    if missing:
        raise FileNotFoundError(f"Missing test NPZ files: {missing[:10]}")
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_counts = {str(label): 0 for label in range(5)}
    for index, case_id in enumerate(sorted(test_ids), start=1):
        with np.load(sources[case_id], allow_pickle=False) as archive:
            segmentation = np.asarray(archive["seg"], dtype=np.uint8)
        unique, counts = np.unique(segmentation, return_counts=True)
        invalid = set(unique.astype(int)) - {0, 1, 2, 3, 4}
        if invalid:
            raise ValueError(f"Invalid labels in {case_id}: {sorted(invalid)}")
        for label, count in zip(unique, counts):
            label_counts[str(int(label))] += int(count)
        write_label_atomic(segmentation, labels_dir / f"{case_id}.nii.gz")
        if index == 1 or index % 25 == 0 or index == expected:
            print(
                json.dumps({"completed": index, "total": expected}),
                flush=True,
            )

    if len(list(labels_dir.glob("BraTS-GLI-*.nii.gz"))) != expected:
        raise AssertionError("Released test label set is incomplete")
    manifest = {
        "status": "TEST_LABELS_RELEASED",
        "released_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": expected,
        "inference_lock_sha256": sha256(inference_lock_path),
        "prediction_hashes_verified_before_release": True,
        "labels_opened_only_after_prediction_freeze": True,
        "post_test_model_selection_allowed": False,
        "label_voxel_counts": label_counts,
        "labels_dir": str(labels_dir),
    }
    release_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
