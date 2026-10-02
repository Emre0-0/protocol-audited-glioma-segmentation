from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


CASE_PATTERN = re.compile(r"^(BraTS-GLI-\d+)-(\d+)$")


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


def read_case_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [str(row["case_id"]) for row in csv.DictReader(handle)]


def subject_id(case_id: str) -> str:
    match = CASE_PATTERN.fullmatch(case_id)
    return match.group(1) if match else case_id


def write_nifti_atomic(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name.removesuffix('.nii.gz')}.tmp.nii.gz"
    )
    if temporary.exists():
        temporary.unlink()
    image = nib.Nifti1Image(
        np.ascontiguousarray(array, dtype=np.float32),
        affine=np.eye(4, dtype=np.float32),
    )
    image.header.set_xyzt_units("mm")
    nib.save(image, str(temporary))
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage locked-test MR channels without reading segmentation arrays."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = project_root()
    config_path = args.config.resolve()
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    test_csv = resolve(root, config["protocol"]["locked_test_cases"])
    npz_root = resolve(root, config["data"]["npz_root"])
    output_dir = resolve(root, config["paths"]["input_images"])
    experiment_dir = resolve(root, config["paths"]["experiment_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    case_ids = read_case_ids(test_csv)
    expected = int(config["protocol"]["expected_test_cases"])
    expected_subjects = int(config["protocol"]["expected_test_subjects"])
    if len(case_ids) != expected or len(set(case_ids)) != expected:
        raise ValueError("Locked-test case count or uniqueness check failed")
    if len({subject_id(case_id) for case_id in case_ids}) != expected_subjects:
        raise ValueError("Locked-test subject count check failed")

    sources: dict[str, Path] = {}
    for path in npz_root.glob("*/*.npz"):
        if path.stem in sources:
            raise ValueError(f"Duplicate NPZ case ID: {path.stem}")
        sources[path.stem] = path
    missing = sorted(set(case_ids) - set(sources))
    if missing:
        raise FileNotFoundError(f"Missing locked-test NPZ cases: {missing[:10]}")

    converted = 0
    skipped = 0
    source_splits: dict[str, int] = {}
    for index, case_id in enumerate(sorted(case_ids), start=1):
        destinations = [
            output_dir / f"{case_id}_{channel:04d}.nii.gz"
            for channel in range(4)
        ]
        if not args.overwrite and all(path.is_file() for path in destinations):
            skipped += 1
        else:
            with np.load(sources[case_id], allow_pickle=False) as archive:
                # The segmentation key is intentionally never accessed here.
                image = np.asarray(archive["image"], dtype=np.float32)
            if image.ndim != 4 or image.shape[0] != 4:
                raise ValueError(
                    f"Expected (4, X, Y, Z), got {image.shape} for {case_id}"
                )
            if not np.isfinite(image).all():
                raise ValueError(f"Non-finite image values in {case_id}")
            for channel, destination in enumerate(destinations):
                write_nifti_atomic(image[channel], destination)
            converted += 1
        split_name = sources[case_id].parent.name
        source_splits[split_name] = source_splits.get(split_name, 0) + 1
        if index == 1 or index % 25 == 0 or index == expected:
            print(
                json.dumps({"completed": index, "total": expected}),
                flush=True,
            )

    staged = sorted(output_dir.glob("BraTS-GLI-*_????.nii.gz"))
    if len(staged) != expected * 4:
        raise AssertionError(
            f"Expected {expected * 4} staged channels, found {len(staged)}"
        )
    staged_case_ids = {
        path.name.removesuffix(".nii.gz")[:-5] for path in staged
    }
    if staged_case_ids != set(case_ids):
        raise AssertionError("Staged image IDs differ from the locked-test list")

    manifest = {
        "stage": "locked_test_image_staging",
        "status": "complete",
        "cases": expected,
        "subjects": expected_subjects,
        "channels_per_case": 4,
        "converted_cases": converted,
        "skipped_existing_cases": skipped,
        "source_split_counts": source_splits,
        "segmentation_key_accessed": False,
        "test_labels_used": False,
        "locked_test_csv_sha256": sha256(test_csv),
        "config_sha256": sha256(config_path),
        "output_dir": str(output_dir),
    }
    (experiment_dir / "test_image_staging_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
