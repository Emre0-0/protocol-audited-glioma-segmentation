from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REGION_LABELS = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "NETC": (1,),
    "SNFH": (2,),
    "ET": (3,),
    "RC": (4,),
}
VARIANT = "et_cc_topology_guard_specialist_nonrc_reproduced"


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
        for path in directory.glob("BraTS-GLI-*.nii.gz")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
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
    candidate = config["frozen_candidate"]
    baseline_dir = resolve(root, candidate["baseline_validation_dir"])
    specialist_dir = resolve(root, candidate["specialist_validation_dir"])
    experiment_dir = resolve(root, config["paths"]["experiment_dir"]) / (
        "reproduced_validation"
    )
    output_dir = experiment_dir / "predictions" / VARIANT
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = int(candidate["expected_validation_cases"])
    validation_ids = case_ids(baseline_dir)
    if len(validation_ids) != expected or case_ids(specialist_dir) != validation_ids:
        raise ValueError("Reproduced validation sources are incomplete")

    v2_dir = root / "research" / "grouped_et_topology_guard_v2"
    sys.path.insert(0, str(v2_dir))
    from build_topology_guard import (
        assert_geometry,
        component_filter,
        read_array,
        write_image_atomic,
    )
    from topology import rc_topology_labels

    stats: dict[str, Any] = {
        "fallback": "specialist_nonrc",
        "cases": expected,
        "proposed_changed_cases": 0,
        "final_changed_cases": 0,
        "topology_rollback_cases": 0,
        "topology_rollback_case_ids": [],
        "proposed_removed_et_components": 0,
        "proposed_removed_et_voxels": 0,
        "final_removed_et_components": 0,
        "final_removed_et_voxels": 0,
        "required_voxel_invariants": ["RC"],
        "voxel_invariants": {region: True for region in REGION_LABELS},
        "rc_topology_invariant_all_cases": True,
    }
    for index, case_id in enumerate(sorted(validation_ids), start=1):
        baseline_path = baseline_dir / f"{case_id}.nii.gz"
        specialist_path = specialist_dir / f"{case_id}.nii.gz"
        destination = output_dir / f"{case_id}.nii.gz"
        baseline_image, baseline = read_array(baseline_path)
        specialist_image, specialist = read_array(specialist_path)
        assert_geometry(baseline_image, specialist_image, case_id)
        proposal, case_stats = component_filter(
            baseline, specialist, "specialist_nonrc"
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
        for region, labels in REGION_LABELS.items():
            stats["voxel_invariants"][region] &= bool(
                np.array_equal(np.isin(output, labels), np.isin(baseline, labels))
            )
        stats["rc_topology_invariant_all_cases"] &= bool(
            np.array_equal(rc_topology_labels(output), baseline_topology)
        )
        if not stats["voxel_invariants"]["RC"]:
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
            print(json.dumps({"completed": index, "total": expected}), flush=True)

    if case_ids(output_dir) != validation_ids:
        raise AssertionError("Reproduced V2 validation output is incomplete")
    manifest = {
        "experiment": "final_locked_test_v1_reproduced_validation",
        "ground_truth_used_during_build": False,
        "locked_test_used": False,
        "sources": {
            "baseline_checkpoint_sha256": sha256(
                resolve(root, config["models"]["baseline"]["checkpoint"])
            ),
            "specialist_checkpoint_sha256": sha256(
                resolve(root, config["models"]["specialist"]["checkpoint"])
            ),
        },
        "variants": {VARIANT: stats},
    }
    (experiment_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
