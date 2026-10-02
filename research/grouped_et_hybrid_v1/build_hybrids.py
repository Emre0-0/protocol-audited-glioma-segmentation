from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


ALLOWED_LABELS = {0, 1, 2, 3, 4}
REGION_LABELS = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "NETC": (1,),
    "SNFH": (2,),
    "ET": (3,),
    "RC": (4,),
}


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


def case_ids(directory: Path) -> set[str]:
    return {
        path.name.removesuffix(".nii.gz")
        for path in directory.glob("*.nii.gz")
    }


def binary_region(array: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    return np.isin(array, labels)


def assert_geometry(
    baseline: sitk.Image,
    specialist: sitk.Image,
    case_id: str,
) -> None:
    checks = {
        "size": baseline.GetSize() == specialist.GetSize(),
        "spacing": np.allclose(
            baseline.GetSpacing(), specialist.GetSpacing(), atol=1e-6, rtol=0
        ),
        "origin": np.allclose(
            baseline.GetOrigin(), specialist.GetOrigin(), atol=1e-5, rtol=0
        ),
        "direction": np.allclose(
            baseline.GetDirection(),
            specialist.GetDirection(),
            atol=1e-6,
            rtol=0,
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Geometry mismatch for {case_id}: {failed}")


def component_filter(
    baseline: np.ndarray,
    specialist: np.ndarray,
    minimum_fraction: float,
    require_positive_overlap: bool,
    fallback: str,
) -> tuple[np.ndarray, dict[str, int]]:
    output = baseline.copy()
    baseline_et = baseline == 3
    specialist_et = specialist == 3
    components, count = ndimage.label(
        baseline_et,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    volumes = np.bincount(components.ravel(), minlength=count + 1)
    overlaps = np.bincount(
        components[specialist_et].ravel(),
        minlength=count + 1,
    )
    component_ids = np.arange(1, count + 1)
    if require_positive_overlap:
        remove_ids = component_ids[overlaps[1:] == 0]
    else:
        fractions = np.divide(
            overlaps[1:],
            np.maximum(volumes[1:], 1),
            dtype=np.float64,
        )
        remove_ids = component_ids[fractions < minimum_fraction]
    remove_mask = np.isin(components, remove_ids)
    fallback_counts = {str(label): 0 for label in range(5)}
    if fallback == "background":
        output[remove_mask] = 0
        fallback_counts["0"] = int(remove_mask.sum())
    elif fallback == "specialist_nonrc":
        replacement = specialist[remove_mask].copy()
        replacement[replacement == 4] = 0
        output[remove_mask] = replacement
        unique, counts = np.unique(replacement, return_counts=True)
        for label, count_value in zip(unique, counts):
            fallback_counts[str(int(label))] += int(count_value)
    elif fallback == "neighbor_nonrc":
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        for component_id in remove_ids:
            component_mask = components == component_id
            ring = (
                ndimage.binary_dilation(component_mask, structure=structure)
                & ~component_mask
            )
            neighbors = baseline[ring]
            neighbors = neighbors[np.isin(neighbors, (0, 1, 2))]
            fill_label = (
                int(
                    np.bincount(
                        neighbors.astype(np.int64),
                        minlength=3,
                    ).argmax()
                )
                if neighbors.size
                else 0
            )
            output[component_mask] = fill_label
            fallback_counts[str(fill_label)] += int(component_mask.sum())
    else:
        raise ValueError(f"Unknown component fallback: {fallback}")
    return output, {
        "baseline_et_components": int(count),
        "removed_et_components": int(len(remove_ids)),
        "removed_et_voxels": int(remove_mask.sum()),
        "added_et_voxels": 0,
        "fallback_label_voxels": fallback_counts,
    }


def candidate_within_baseline_tc(
    baseline: np.ndarray,
    specialist: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    output = baseline.copy()
    baseline_et = baseline == 3
    new_et = (specialist == 3) & np.isin(baseline, (1, 3))
    removed = baseline_et & ~new_et
    added = new_et & ~baseline_et
    output[baseline_et] = 1
    output[new_et] = 3
    return output, {
        "baseline_et_components": 0,
        "removed_et_components": 0,
        "removed_et_voxels": int(removed.sum()),
        "added_et_voxels": int(added.sum()),
        "fallback_label_voxels": {
            str(label): 0 for label in range(5)
        },
    }


def make_variant(
    baseline: np.ndarray,
    specialist: np.ndarray,
    variant: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    mode = variant["mode"]
    if mode == "component_filter":
        return component_filter(
            baseline,
            specialist,
            float(variant["minimum_candidate_overlap_fraction"]),
            bool(variant["require_positive_overlap"]),
            str(variant["fallback"]),
        )
    if mode == "candidate_within_baseline_tc":
        return candidate_within_baseline_tc(baseline, specialist)
    raise ValueError(f"Unknown hybrid mode: {mode}")


def write_image_atomic(
    array: np.ndarray,
    reference: sitk.Image,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name.removesuffix('.nii.gz')}.tmp.nii.gz"
    )
    if temporary.exists():
        temporary.unlink()
    image = sitk.GetImageFromArray(array.astype(np.uint8, copy=False))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(temporary), useCompression=True)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    args = parser.parse_args()

    root = project_root()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    baseline_dir = resolve(root, paths["baseline_prediction_dir"])
    specialist_dir = resolve(root, paths["specialist_prediction_dir"])
    split_path = resolve(root, config["protocol"]["split_file"])
    locked_test_path = resolve(root, config["protocol"]["locked_test_cases"])
    experiment_dir = resolve(root, paths["experiment_dir"])
    prediction_root = experiment_dir / "predictions"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    prediction_root.mkdir(parents=True, exist_ok=True)

    split = json.loads(split_path.read_text(encoding="utf-8"))
    fold = int(config["protocol"]["fold"])
    validation_ids = set(split[fold]["val"])
    training_ids = set(split[fold]["train"])
    locked_test_ids = set(
        pd.read_csv(locked_test_path)["case_id"].astype(str)
    )
    expected_count = int(config["protocol"]["expected_validation_cases"])
    if len(validation_ids) != expected_count:
        raise ValueError(
            f"Expected {expected_count} validation cases, found {len(validation_ids)}"
        )
    if training_ids & validation_ids:
        raise ValueError("Training and validation case sets overlap")
    if validation_ids & locked_test_ids:
        raise ValueError("Validation and locked-test case sets overlap")
    if case_ids(baseline_dir) != validation_ids:
        raise ValueError("Baseline predictions do not exactly match fold validation")
    if case_ids(specialist_dir) != validation_ids:
        raise ValueError("Specialist predictions do not exactly match fold validation")

    manifest: dict[str, Any] = {
        "experiment": config["experiment_name"],
        "protocol": {
            "fold": fold,
            "validation_cases": len(validation_ids),
            "train_validation_overlap": 0,
            "validation_locked_test_overlap": 0,
            "locked_test_used": False,
            "split_sha256": sha256(split_path),
        },
        "sources": {
            "baseline_prediction_dir": str(baseline_dir),
            "specialist_prediction_dir": str(specialist_dir),
            "baseline_checkpoint": {
                "path": str(resolve(root, paths["baseline_checkpoint"])),
                "sha256": sha256(resolve(root, paths["baseline_checkpoint"])),
            },
            "specialist_checkpoint": {
                "path": str(resolve(root, paths["specialist_checkpoint"])),
                "sha256": sha256(resolve(root, paths["specialist_checkpoint"])),
            },
        },
        "variants": {},
    }

    for variant in config["variants"]:
        name = str(variant["name"])
        output_dir = prediction_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        unexpected = case_ids(output_dir) - validation_ids
        if unexpected:
            raise ValueError(
                f"Unexpected existing outputs in {output_dir}: "
                f"{sorted(unexpected)[:10]}"
            )
        stats: dict[str, Any] = {
            "mode": variant["mode"],
            "output_dir": str(output_dir),
            "cases": len(validation_ids),
            "changed_cases": 0,
            "unchanged_cases": 0,
            "removed_et_components": 0,
            "removed_et_voxels": 0,
            "added_et_voxels": 0,
            "fallback_label_voxels": {
                str(label): 0 for label in range(5)
            },
            "baseline_et_voxels": 0,
            "hybrid_et_voxels": 0,
            "changed_case_ids": [],
            "voxel_invariants": {
                region: True for region in REGION_LABELS
            },
            "required_voxel_invariants": list(
                variant.get("required_voxel_invariants", [])
            ),
        }
        for case_id in sorted(validation_ids):
            baseline_path = baseline_dir / f"{case_id}.nii.gz"
            specialist_path = specialist_dir / f"{case_id}.nii.gz"
            destination = output_dir / f"{case_id}.nii.gz"
            baseline_image = sitk.ReadImage(str(baseline_path))
            specialist_image = sitk.ReadImage(str(specialist_path))
            assert_geometry(baseline_image, specialist_image, case_id)
            baseline = sitk.GetArrayFromImage(baseline_image)
            specialist = sitk.GetArrayFromImage(specialist_image)
            if set(int(v) for v in np.unique(baseline)) - ALLOWED_LABELS:
                raise ValueError(f"Invalid baseline labels in {case_id}")
            if set(int(v) for v in np.unique(specialist)) - ALLOWED_LABELS:
                raise ValueError(f"Invalid specialist labels in {case_id}")

            hybrid, case_stats = make_variant(
                baseline,
                specialist,
                variant,
            )
            if set(int(v) for v in np.unique(hybrid)) - ALLOWED_LABELS:
                raise ValueError(f"Invalid hybrid labels in {name}/{case_id}")
            for region, labels in REGION_LABELS.items():
                invariant = np.array_equal(
                    binary_region(baseline, labels),
                    binary_region(hybrid, labels),
                )
                stats["voxel_invariants"][region] &= bool(invariant)
                if (
                    region in stats["required_voxel_invariants"]
                    and not invariant
                ):
                    raise AssertionError(
                        f"{region} mask changed in {name}/{case_id}"
                    )

            changed = not np.array_equal(baseline, hybrid)
            stats["baseline_et_voxels"] += int((baseline == 3).sum())
            stats["hybrid_et_voxels"] += int((hybrid == 3).sum())
            for key in (
                "removed_et_components",
                "removed_et_voxels",
                "added_et_voxels",
            ):
                stats[key] += int(case_stats[key])
            for label, count_value in case_stats[
                "fallback_label_voxels"
            ].items():
                stats["fallback_label_voxels"][label] += int(count_value)
            if changed:
                stats["changed_cases"] += 1
                stats["changed_case_ids"].append(case_id)
                write_image_atomic(hybrid, baseline_image, destination)
            else:
                stats["unchanged_cases"] += 1
                if destination.exists():
                    destination.unlink()
                try:
                    os.link(baseline_path, destination)
                except OSError:
                    write_image_atomic(baseline, baseline_image, destination)

        if case_ids(output_dir) != validation_ids:
            raise AssertionError(f"Incomplete hybrid output for {name}")
        manifest["variants"][name] = stats
        print(json.dumps({name: stats}, indent=2))

    manifest_path = experiment_dir / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "status": "complete"}))


if __name__ == "__main__":
    main()
