from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import SimpleITK as sitk


def official_rc_difference_cases(
    baseline_metrics: Path,
    candidate_metrics: Path,
) -> set[str]:
    column = "RC_LesionWise_Score_Dice"
    baseline = pd.read_csv(
        baseline_metrics / "case_metrics_official.csv",
        usecols=["case_id", column],
    ).set_index("case_id")
    candidate = pd.read_csv(
        candidate_metrics / "case_metrics_official.csv",
        usecols=["case_id", column],
    ).set_index("case_id")
    aligned = baseline.join(candidate, lsuffix="_baseline", rsuffix="_candidate")
    return set(
        aligned.index[
            aligned[f"{column}_baseline"] != aligned[f"{column}_candidate"]
        ]
    )


def arrays_equal(first: Path, second: Path) -> bool:
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(first)))
    b = sitk.GetArrayFromImage(sitk.ReadImage(str(second)))
    return bool((a == b).all())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    experiment = root / paths["experiment_dir"]
    v1 = root / paths["v1_experiment_dir"]
    baseline_predictions = root / paths["baseline_prediction_dir"]
    baseline_metrics = root / paths["baseline_official_metrics"]
    manifest = json.loads(
        (experiment / "build_manifest.json").read_text(encoding="utf-8")
    )
    audit: dict[str, object] = {
        "ground_truth_used_by_topology_guard": False,
        "variants": {},
    }

    for variant in config["variants"]:
        name = str(variant["name"])
        v1_name = str(variant["v1_variant"])
        stats = manifest["variants"][name]
        signature_cases = set(stats["topology_rollback_case_ids"])
        official_cases = official_rc_difference_cases(
            baseline_metrics,
            v1 / "official_metrics" / v1_name,
        )
        if not official_cases.issubset(signature_cases):
            raise AssertionError(
                f"Topology audit mismatch for {name}: "
                f"signature={sorted(signature_cases)}, "
                f"official={sorted(official_cases)}"
            )

        output_dir = experiment / "predictions" / name
        for case_id in stats["topology_rollback_case_ids"]:
            if not arrays_equal(
                output_dir / f"{case_id}.nii.gz",
                baseline_predictions / f"{case_id}.nii.gz",
            ):
                raise AssertionError(f"Rollback output differs from baseline: {case_id}")

        audit["variants"][name] = {
            "prediction_only_signature_mismatch_cases": sorted(signature_cases),
            "v1_official_rc_difference_cases": sorted(official_cases),
            "all_observed_official_rc_changes_captured": True,
            "conservatively_protected_extra_cases": sorted(
                signature_cases - official_cases
            ),
            "rollback_outputs_equal_baseline": True,
        }

    output = experiment / "topology_audit.json"
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
