from __future__ import annotations

import argparse
import hashlib
import json
import sys
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


def case_ids(directory: Path) -> set[str]:
    return {
        path.name.removesuffix(".nii.gz")
        for path in directory.glob("BraTS-GLI-*.nii.gz")
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that the recreated specialist reproduces frozen V2."
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
    candidate = config["frozen_candidate"]
    baseline_dir = resolve(root, candidate["baseline_validation_dir"])
    specialist_dir = resolve(root, candidate["specialist_validation_dir"])
    frozen_dir = resolve(root, candidate["source_prediction_dir"])
    output_dir = resolve(root, config["paths"]["experiment_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    expected = int(candidate["expected_validation_cases"])
    expected_ids = case_ids(frozen_dir)
    if len(expected_ids) != expected:
        raise ValueError("Frozen V2 validation prediction set is incomplete")
    for name, directory in (
        ("baseline", baseline_dir),
        ("specialist", specialist_dir),
    ):
        if case_ids(directory) != expected_ids:
            raise ValueError(f"{name} validation predictions do not match V2")

    v2_dir = root / "research" / "grouped_et_topology_guard_v2"
    sys.path.insert(0, str(v2_dir))
    from build_topology_guard import (
        assert_geometry,
        component_filter,
        read_array,
    )
    from topology import rc_topology_labels

    mismatched_cases: list[str] = []
    mismatched_voxels = 0
    changed_cases = 0
    rollback_cases = 0
    removed_et_components = 0
    removed_et_voxels = 0
    for index, case_id in enumerate(sorted(expected_ids), start=1):
        baseline_image, baseline = read_array(
            baseline_dir / f"{case_id}.nii.gz"
        )
        specialist_image, specialist = read_array(
            specialist_dir / f"{case_id}.nii.gz"
        )
        frozen_image, frozen = read_array(frozen_dir / f"{case_id}.nii.gz")
        assert_geometry(baseline_image, specialist_image, case_id)
        assert_geometry(baseline_image, frozen_image, case_id)
        proposal, stats = component_filter(
            baseline, specialist, str(candidate["fallback"])
        )
        proposed_changed = not np.array_equal(proposal, baseline)
        if proposed_changed and not np.array_equal(
            rc_topology_labels(baseline), rc_topology_labels(proposal)
        ):
            rebuilt = baseline
            rollback_cases += 1
        else:
            rebuilt = proposal
            removed_et_components += int(stats["removed_et_components"])
            removed_et_voxels += int(stats["removed_et_voxels"])
        changed_cases += int(not np.array_equal(rebuilt, baseline))
        difference = int(np.count_nonzero(rebuilt != frozen))
        if difference:
            mismatched_cases.append(case_id)
            mismatched_voxels += difference
        if index == 1 or index % 25 == 0 or index == expected:
            print(
                json.dumps({"completed": index, "total": expected}),
                flush=True,
            )

    checkpoint = resolve(root, config["models"]["specialist"]["checkpoint"])
    recreated_hash = sha256(checkpoint)
    original_hash = str(config["models"]["specialist"]["original_sha256"])
    exact = not mismatched_cases
    report = {
        "status": "PASS" if exact else "FAIL",
        "validation_cases": expected,
        "voxel_exact_reproduction": exact,
        "mismatched_cases": len(mismatched_cases),
        "mismatched_voxels": mismatched_voxels,
        "mismatched_case_ids": mismatched_cases,
        "rebuilt_changed_cases": changed_cases,
        "rebuilt_topology_rollback_cases": rollback_cases,
        "rebuilt_removed_et_components": removed_et_components,
        "rebuilt_removed_et_voxels": removed_et_voxels,
        "original_specialist_checkpoint_sha256": original_hash,
        "recreated_specialist_checkpoint_sha256": recreated_hash,
        "checkpoint_byte_hash_match": recreated_hash == original_hash,
        "test_set_used": False,
        "acceptance_rule": (
            "All 223 rebuilt V2 validation masks must be voxel-identical to "
            "the preserved frozen V2 masks."
        ),
    }
    report_path = output_dir / "specialist_reproduction_verification.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
