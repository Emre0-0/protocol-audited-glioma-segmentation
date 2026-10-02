from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk


REGIONS = ("WT", "TC", "NETC", "SNFH", "ET", "RC")
SUMMARY_COLUMN = "LesionWise_Score_Dice_mean"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def nifti_ids(directory: Path) -> set[str]:
    return {
        path.name.removesuffix(".nii.gz")
        for path in directory.glob("*.nii.gz")
    }


def cache_ids(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.json")}


def arrays_equal(first: Path, second: Path) -> bool:
    first_array = sitk.GetArrayFromImage(sitk.ReadImage(str(first)))
    second_array = sitk.GetArrayFromImage(sitk.ReadImage(str(second)))
    return bool(np.array_equal(first_array, second_array))


def assert_numeric_columns_equal(
    first: pd.DataFrame,
    second: pd.DataFrame,
    columns: list[str],
) -> None:
    for column in columns:
        a = pd.to_numeric(first[column], errors="coerce").to_numpy()
        b = pd.to_numeric(second[column], errors="coerce").to_numpy()
        if not np.allclose(a, b, atol=0.0, rtol=0.0, equal_nan=True):
            raise AssertionError(f"Column differs: {column}")


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
    v1_root = root / paths["v1_experiment_dir"]
    baseline_predictions = root / paths["baseline_prediction_dir"]
    baseline_metrics = root / paths["baseline_official_metrics"]
    split_path = root / config["protocol"]["split_file"]
    locked_test_path = root / config["protocol"]["locked_test_cases"]
    manifest = json.loads(
        (experiment / "build_manifest.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (experiment / "comparison" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )
    selected = comparison["selected_variant"]
    if comparison["decision"] != "GO_TOPOLOGY_GUARDED_VALIDATION_CANDIDATE":
        raise AssertionError("Decision gate did not select a candidate")
    if not selected:
        raise AssertionError("Selected variant is missing")

    variant_config = next(
        item for item in config["variants"] if item["name"] == selected
    )
    stats = manifest["variants"][selected]
    predictions = experiment / "predictions" / selected
    metric_dir = experiment / "official_metrics" / selected
    v1_predictions = (
        v1_root / "predictions" / variant_config["v1_variant"]
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    fold = int(config["protocol"]["fold"])
    validation_ids = set(split[fold]["val"])
    training_ids = set(split[fold]["train"])
    locked_test_ids = set(
        pd.read_csv(locked_test_path)["case_id"].astype(str)
    )

    checks: dict[str, bool] = {}
    checks["prediction_ids_exact_validation"] = (
        nifti_ids(predictions) == validation_ids
    )
    checks["train_validation_overlap_zero"] = not (
        training_ids & validation_ids
    )
    checks["validation_locked_test_overlap_zero"] = not (
        validation_ids & locked_test_ids
    )
    checks["metric_cache_ids_exact_validation"] = (
        cache_ids(metric_dir / "case_cache") == validation_ids
    )
    failures = json.loads(
        (metric_dir / "failures.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (metric_dir / "metadata.json").read_text(encoding="utf-8")
    )
    checks["official_failures_zero"] = not failures
    checks["official_successes_223"] = (
        int(metadata["successful_cases"]) == len(validation_ids) == 223
    )
    checks["official_vendor_commit_locked"] = (
        metadata["metric_repository_commit"]
        == "f1131e340e155908553c226675644a6363b998b7"
    )
    checks["ground_truth_not_used_during_build"] = (
        manifest["ground_truth_used_during_build"] is False
    )
    checks["rc_topology_invariant_all_cases"] = bool(
        stats["rc_topology_invariant_all_cases"]
    )

    rollback_ids = set(stats["topology_rollback_case_ids"])
    provenance_mismatches: list[str] = []
    for case_id in sorted(validation_ids):
        output = predictions / f"{case_id}.nii.gz"
        expected = (
            baseline_predictions / f"{case_id}.nii.gz"
            if case_id in rollback_ids
            else v1_predictions / f"{case_id}.nii.gz"
        )
        if not arrays_equal(output, expected):
            provenance_mismatches.append(case_id)
    checks["all_prediction_arrays_match_declared_source"] = not (
        provenance_mismatches
    )

    case_metrics = pd.read_csv(
        metric_dir / "case_metrics_official.csv"
    ).sort_values("case_id").reset_index(drop=True)
    baseline_case_metrics = pd.read_csv(
        baseline_metrics / "case_metrics_official.csv"
    ).sort_values("case_id").reset_index(drop=True)
    checks["case_metric_ids_exact_validation"] = (
        set(case_metrics["case_id"]) == validation_ids
        and len(case_metrics) == len(validation_ids)
    )
    checks["baseline_candidate_case_order_aligned"] = bool(
        case_metrics["case_id"].equals(baseline_case_metrics["case_id"])
    )
    rc_columns = [
        column
        for column in case_metrics.columns
        if column.startswith("RC_") and column in baseline_case_metrics
    ]
    assert_numeric_columns_equal(
        baseline_case_metrics,
        case_metrics,
        rc_columns,
    )
    checks["all_case_level_rc_metrics_exact_baseline"] = True

    summary = pd.read_csv(metric_dir / "summary_official.csv")
    summary_by_region = summary.set_index("region")
    recomputed = {}
    for region in REGIONS:
        value = float(
            pd.to_numeric(
                case_metrics[f"{region}_LesionWise_Score_Dice"],
                errors="coerce",
            ).mean()
        )
        recomputed[region] = value
        reported = float(summary_by_region.loc[region, SUMMARY_COLUMN])
        if not np.isclose(value, reported, atol=1e-15, rtol=0.0):
            raise AssertionError(
                f"Summary mismatch for {region}: {value} != {reported}"
            )
    checks["summary_recomputed_exactly"] = True

    selected_comparison: dict[str, Any] = next(
        item
        for item in comparison["comparisons"]
        if item["variant"] == selected
    )
    checks["strict_gate_passed"] = bool(selected_comparison["passes_gate"])
    checks["all_six_region_point_estimates_non_degraded"] = (
        selected_comparison["non_degraded_region_count"] == 6
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Final audit failed: {failed}")

    payload = {
        "status": "PASS",
        "selected_variant": selected,
        "selected_prediction_dir": str(predictions.resolve()),
        "config_sha256": sha256(args.config),
        "validation_cases": len(validation_ids),
        "locked_test_used": False,
        "topology_rollback_case_ids": sorted(rollback_ids),
        "prediction_provenance_mismatches": provenance_mismatches,
        "checks": checks,
        "official_validation_metrics": recomputed,
        "six_region_mean": float(np.mean(list(recomputed.values()))),
        "six_region_mean_gain": selected_comparison["mean_gain"],
        "paired_bootstrap": selected_comparison["paired_bootstrap"],
    }
    output = experiment / "final_audit.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
