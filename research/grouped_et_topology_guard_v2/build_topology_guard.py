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

from topology import rc_topology_labels, topology_digest


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


def read_array(path: Path) -> tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    return image, sitk.GetArrayFromImage(image)


def assert_geometry(first: sitk.Image, second: sitk.Image, case_id: str) -> None:
    checks = {
        "size": first.GetSize() == second.GetSize(),
        "spacing": np.allclose(first.GetSpacing(), second.GetSpacing(), atol=1e-6),
        "origin": np.allclose(first.GetOrigin(), second.GetOrigin(), atol=1e-5),
        "direction": np.allclose(
            first.GetDirection(),
            second.GetDirection(),
            atol=1e-6,
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Geometry mismatch for {case_id}: {failed}")


def component_filter(
    baseline: np.ndarray,
    specialist: np.ndarray,
    fallback: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = baseline.copy()
    components, count = ndimage.label(
        baseline == 3,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    overlap_counts = np.bincount(
        components[specialist == 3].ravel(),
        minlength=count + 1,
    )
    remove_ids = np.arange(1, count + 1)[overlap_counts[1:] == 0]
    remove_mask = np.isin(components, remove_ids)
    fallback_counts = {str(label): 0 for label in range(5)}

    if fallback == "background":
        output[remove_mask] = 0
        fallback_counts["0"] = int(remove_mask.sum())
    elif fallback == "specialist_nonrc":
        replacement = specialist[remove_mask].copy()
        replacement[replacement == 4] = 0
        output[remove_mask] = replacement
        labels, counts = np.unique(replacement, return_counts=True)
        for label, count in zip(labels, counts):
            fallback_counts[str(int(label))] += int(count)
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
                int(np.bincount(neighbors.astype(np.int64), minlength=3).argmax())
                if neighbors.size
                else 0
            )
            output[component_mask] = fill_label
            fallback_counts[str(fill_label)] += int(component_mask.sum())
    else:
        raise ValueError(f"Unknown fallback: {fallback}")

    return output, {
        "removed_et_components": int(len(remove_ids)),
        "removed_et_voxels": int(remove_mask.sum()),
        "fallback_label_voxels": fallback_counts,
    }


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
    v1_root = resolve(root, paths["v1_experiment_dir"])
    experiment_dir = resolve(root, paths["experiment_dir"])
    prediction_root = experiment_dir / "predictions"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    prediction_root.mkdir(parents=True, exist_ok=True)

    split = json.loads(split_path.read_text(encoding="utf-8"))
    fold = int(config["protocol"]["fold"])
    validation_ids = set(split[fold]["val"])
    training_ids = set(split[fold]["train"])
    locked_test_ids = set(pd.read_csv(locked_test_path)["case_id"].astype(str))
    expected = int(config["protocol"]["expected_validation_cases"])
    if len(validation_ids) != expected:
        raise ValueError(f"Expected {expected} validation cases")
    if training_ids & validation_ids:
        raise ValueError("Training and validation case sets overlap")
    if validation_ids & locked_test_ids:
        raise ValueError("Validation and locked-test case sets overlap")
    if case_ids(baseline_dir) != validation_ids:
        raise ValueError("Baseline predictions do not match validation split")
    if case_ids(specialist_dir) != validation_ids:
        raise ValueError("Specialist predictions do not match validation split")

    manifest: dict[str, Any] = {
        "experiment": config["experiment_name"],
        "method": (
            "Prediction-only ET component consensus with case-level rollback "
            "when the official prediction-side RC lesion partition changes."
        ),
        "ground_truth_used_during_build": False,
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
            "baseline_checkpoint_sha256": sha256(
                resolve(root, paths["baseline_checkpoint"])
            ),
            "specialist_checkpoint_sha256": sha256(
                resolve(root, paths["specialist_checkpoint"])
            ),
        },
        "variants": {},
    }

    baseline_topology_cache: dict[str, np.ndarray] = {}
    for variant in config["variants"]:
        name = str(variant["name"])
        v1_variant = str(variant["v1_variant"])
        fallback = str(variant["fallback"])
        output_dir = prediction_root / name
        v1_prediction_dir = v1_root / "predictions" / v1_variant
        output_dir.mkdir(parents=True, exist_ok=True)
        if case_ids(v1_prediction_dir) != validation_ids:
            raise ValueError(f"Incomplete v1 prediction source: {v1_variant}")

        stats: dict[str, Any] = {
            "fallback": fallback,
            "v1_variant": v1_variant,
            "output_dir": str(output_dir),
            "cases": len(validation_ids),
            "proposed_changed_cases": 0,
            "final_changed_cases": 0,
            "topology_rollback_cases": 0,
            "topology_rollback_case_ids": [],
            "proposed_removed_et_components": 0,
            "proposed_removed_et_voxels": 0,
            "final_removed_et_components": 0,
            "final_removed_et_voxels": 0,
            "rolled_back_et_components": 0,
            "rolled_back_et_voxels": 0,
            "fallback_label_voxels": {str(label): 0 for label in range(5)},
            "changed_case_ids": [],
            "matches_v1_case_ids": [],
            "matches_baseline_case_ids": [],
            "rc_topology_invariant_all_cases": True,
            "required_voxel_invariants": list(
                variant.get("required_voxel_invariants", [])
            ),
            "voxel_invariants": {region: True for region in REGION_LABELS},
            "topology_digests": {},
        }

        for index, case_id in enumerate(sorted(validation_ids), start=1):
            baseline_path = baseline_dir / f"{case_id}.nii.gz"
            specialist_path = specialist_dir / f"{case_id}.nii.gz"
            v1_path = v1_prediction_dir / f"{case_id}.nii.gz"
            destination = output_dir / f"{case_id}.nii.gz"
            baseline_image, baseline = read_array(baseline_path)
            specialist_image, specialist = read_array(specialist_path)
            assert_geometry(baseline_image, specialist_image, case_id)
            if set(int(v) for v in np.unique(baseline)) - ALLOWED_LABELS:
                raise ValueError(f"Invalid baseline labels in {case_id}")
            if set(int(v) for v in np.unique(specialist)) - ALLOWED_LABELS:
                raise ValueError(f"Invalid specialist labels in {case_id}")

            proposal, case_stats = component_filter(
                baseline,
                specialist,
                fallback,
            )
            proposed_changed = not np.array_equal(proposal, baseline)
            stats["proposed_changed_cases"] += int(proposed_changed)
            stats["proposed_removed_et_components"] += case_stats[
                "removed_et_components"
            ]
            stats["proposed_removed_et_voxels"] += case_stats[
                "removed_et_voxels"
            ]

            if case_id not in baseline_topology_cache:
                baseline_topology_cache[case_id] = rc_topology_labels(baseline)
            baseline_topology = baseline_topology_cache[case_id]
            proposal_topology = (
                rc_topology_labels(proposal)
                if proposed_changed
                else baseline_topology
            )
            rollback = not np.array_equal(
                baseline_topology,
                proposal_topology,
            )
            output = baseline.copy() if rollback else proposal
            final_topology = (
                baseline_topology
                if rollback or not proposed_changed
                else proposal_topology
            )
            topology_invariant = np.array_equal(
                baseline_topology,
                final_topology,
            )
            if not topology_invariant:
                raise AssertionError(f"RC topology guard failed for {name}/{case_id}")

            if rollback:
                stats["topology_rollback_cases"] += 1
                stats["topology_rollback_case_ids"].append(case_id)
                stats["rolled_back_et_components"] += case_stats[
                    "removed_et_components"
                ]
                stats["rolled_back_et_voxels"] += case_stats[
                    "removed_et_voxels"
                ]
            else:
                stats["final_removed_et_components"] += case_stats[
                    "removed_et_components"
                ]
                stats["final_removed_et_voxels"] += case_stats[
                    "removed_et_voxels"
                ]
                for label, count in case_stats["fallback_label_voxels"].items():
                    stats["fallback_label_voxels"][label] += int(count)

            final_changed = not np.array_equal(output, baseline)
            if final_changed:
                stats["final_changed_cases"] += 1
                stats["changed_case_ids"].append(case_id)

            _, v1_array = read_array(v1_path)
            if np.array_equal(output, v1_array):
                stats["matches_v1_case_ids"].append(case_id)
            if np.array_equal(output, baseline):
                stats["matches_baseline_case_ids"].append(case_id)

            for region, labels in REGION_LABELS.items():
                invariant = np.array_equal(
                    np.isin(baseline, labels),
                    np.isin(output, labels),
                )
                stats["voxel_invariants"][region] &= bool(invariant)
                if (
                    region in stats["required_voxel_invariants"]
                    and not invariant
                ):
                    raise AssertionError(
                        f"{region} mask changed in {name}/{case_id}"
                    )

            if rollback:
                stats["topology_digests"][case_id] = {
                    "baseline": topology_digest(baseline_topology),
                    "proposal": topology_digest(proposal_topology),
                    "final": topology_digest(final_topology),
                }

            if final_changed:
                write_image_atomic(output, baseline_image, destination)
            else:
                if destination.exists():
                    destination.unlink()
                try:
                    os.link(baseline_path, destination)
                except OSError:
                    write_image_atomic(baseline, baseline_image, destination)

            if index == 1 or index % 25 == 0 or index == len(validation_ids):
                print(
                    json.dumps(
                        {
                            "variant": name,
                            "completed": index,
                            "total": len(validation_ids),
                            "rollbacks": stats["topology_rollback_cases"],
                        }
                    ),
                    flush=True,
                )

        if case_ids(output_dir) != validation_ids:
            raise AssertionError(f"Incomplete outputs for {name}")
        if (
            len(stats["matches_v1_case_ids"])
            + len(stats["topology_rollback_case_ids"])
            < len(validation_ids)
        ):
            unresolved = (
                validation_ids
                - set(stats["matches_v1_case_ids"])
                - set(stats["topology_rollback_case_ids"])
            )
            raise AssertionError(
                f"Unexpected v2 output provenance for {name}: {sorted(unresolved)[:5]}"
            )
        manifest["variants"][name] = stats

    (experiment_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(experiment_dir / "build_manifest.json"),
                "status": "complete",
            }
        )
    )


if __name__ == "__main__":
    main()
