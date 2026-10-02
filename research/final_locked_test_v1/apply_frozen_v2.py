from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def prediction_ids(directory: Path) -> set[str]:
    return {
        path.name.removesuffix(".nii.gz")
        for path in directory.glob("BraTS-GLI-*.nii.gz")
    }


def file_hashes(directory: Path, case_ids: set[str]) -> dict[str, str]:
    return {
        case_id: sha256(directory / f"{case_id}.nii.gz")
        for case_id in sorted(case_ids)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen V2 topology guard without ground truth."
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
    model_lock_path = experiment_dir / "frozen_model_manifest.json"
    inference_lock_path = experiment_dir / "inference_lock.json"
    label_release_path = experiment_dir / "test_label_release_manifest.json"
    if label_release_path.exists():
        raise RuntimeError("Test labels are released; predictions are immutable")
    model_lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    if model_lock.get("status") != "FROZEN":
        raise RuntimeError("Model lock is missing or invalid")

    baseline_dir = resolve(root, config["paths"]["baseline_predictions"])
    specialist_dir = resolve(root, config["paths"]["specialist_predictions"])
    output_dir = resolve(root, config["paths"]["final_predictions"])
    output_dir.mkdir(parents=True, exist_ok=True)
    test_ids = read_case_ids(
        resolve(root, config["protocol"]["locked_test_cases"])
    )
    expected = int(config["protocol"]["expected_test_cases"])
    if len(test_ids) != expected:
        raise ValueError("Locked-test list count mismatch")
    if prediction_ids(baseline_dir) != test_ids:
        raise ValueError("Baseline predictions do not match locked-test IDs")
    if prediction_ids(specialist_dir) != test_ids:
        raise ValueError("Specialist predictions do not match locked-test IDs")

    if inference_lock_path.exists():
        locked = json.loads(inference_lock_path.read_text(encoding="utf-8"))
        if locked.get("status") != "PREDICTIONS_FROZEN":
            raise RuntimeError("Existing inference lock is invalid")
        if file_hashes(output_dir, test_ids) != locked["prediction_sha256"][
            "frozen_v2"
        ]:
            raise RuntimeError("Frozen V2 predictions changed after lock")
        print(json.dumps(locked, indent=2))
        return

    v2_dir = root / "research" / "grouped_et_topology_guard_v2"
    sys.path.insert(0, str(v2_dir))
    from build_topology_guard import (
        assert_geometry,
        component_filter,
        read_array,
        write_image_atomic,
    )
    from topology import rc_topology_labels

    stats = {
        "cases": expected,
        "proposed_changed_cases": 0,
        "final_changed_cases": 0,
        "topology_rollback_cases": 0,
        "topology_rollback_case_ids": [],
        "proposed_removed_et_components": 0,
        "proposed_removed_et_voxels": 0,
        "final_removed_et_components": 0,
        "final_removed_et_voxels": 0,
        "rc_voxel_invariant_all_cases": True,
        "rc_topology_invariant_all_cases": True,
    }
    fallback = str(config["frozen_candidate"]["fallback"])
    for index, case_id in enumerate(sorted(test_ids), start=1):
        baseline_path = baseline_dir / f"{case_id}.nii.gz"
        specialist_path = specialist_dir / f"{case_id}.nii.gz"
        destination = output_dir / f"{case_id}.nii.gz"
        baseline_image, baseline = read_array(baseline_path)
        specialist_image, specialist = read_array(specialist_path)
        assert_geometry(baseline_image, specialist_image, case_id)
        if set(np.unique(baseline).astype(int)) - {0, 1, 2, 3, 4}:
            raise ValueError(f"Invalid baseline labels in {case_id}")
        if set(np.unique(specialist).astype(int)) - {0, 1, 2, 3, 4}:
            raise ValueError(f"Invalid specialist labels in {case_id}")

        proposal, case_stats = component_filter(
            baseline, specialist, fallback
        )
        proposed_changed = not np.array_equal(proposal, baseline)
        stats["proposed_changed_cases"] += int(proposed_changed)
        stats["proposed_removed_et_components"] += int(
            case_stats["removed_et_components"]
        )
        stats["proposed_removed_et_voxels"] += int(
            case_stats["removed_et_voxels"]
        )
        baseline_topology = rc_topology_labels(baseline)
        proposal_topology = (
            rc_topology_labels(proposal)
            if proposed_changed
            else baseline_topology
        )
        rollback = not np.array_equal(baseline_topology, proposal_topology)
        output = baseline if rollback else proposal
        if rollback:
            stats["topology_rollback_cases"] += 1
            stats["topology_rollback_case_ids"].append(case_id)
        else:
            stats["final_removed_et_components"] += int(
                case_stats["removed_et_components"]
            )
            stats["final_removed_et_voxels"] += int(
                case_stats["removed_et_voxels"]
            )
        stats["final_changed_cases"] += int(
            not np.array_equal(output, baseline)
        )
        stats["rc_voxel_invariant_all_cases"] &= bool(
            np.array_equal(output == 4, baseline == 4)
        )
        stats["rc_topology_invariant_all_cases"] &= bool(
            np.array_equal(rc_topology_labels(output), baseline_topology)
        )
        if not stats["rc_voxel_invariant_all_cases"]:
            raise AssertionError(f"RC mask changed for {case_id}")
        if not stats["rc_topology_invariant_all_cases"]:
            raise AssertionError(f"RC topology changed for {case_id}")

        if np.array_equal(output, baseline):
            if destination.exists():
                destination.unlink()
            try:
                os.link(baseline_path, destination)
            except OSError:
                write_image_atomic(output, baseline_image, destination)
        else:
            write_image_atomic(output, baseline_image, destination)
        if index == 1 or index % 25 == 0 or index == expected:
            print(
                json.dumps({"completed": index, "total": expected}),
                flush=True,
            )

    if prediction_ids(output_dir) != test_ids:
        raise AssertionError("Frozen V2 test outputs are incomplete")
    prediction_hashes = {
        "baseline": file_hashes(baseline_dir, test_ids),
        "specialist": file_hashes(specialist_dir, test_ids),
        "frozen_v2": file_hashes(output_dir, test_ids),
    }
    lock = {
        "status": "PREDICTIONS_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_manifest_sha256": sha256(model_lock_path),
        "cases": expected,
        "selected_variant": config["frozen_candidate"]["selected_variant"],
        "fallback": fallback,
        "test_labels_used": False,
        "post_test_prediction_changes_allowed": False,
        "statistics": stats,
        "prediction_sha256": prediction_hashes,
    }
    inference_lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    (output_dir / "prediction_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "cases": expected,
                "test_labels_used": False,
                "statistics": stats,
                "inference_lock": str(inference_lock_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
