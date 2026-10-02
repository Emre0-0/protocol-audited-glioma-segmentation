from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REGIONS = ("WT", "TC", "NETC", "SNFH", "ET", "RC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize official BraTS 2024 lesion-wise metrics, small-volume cases, and empty-class behavior."
    )
    parser.add_argument(
        "--case-metrics",
        default="experiments/nnunet_ensemble_5seed_official_brats2024_metrics/case_metrics_official.csv",
    )
    parser.add_argument(
        "--legacy-assignment",
        default="experiments/brats_longitudinal_integrity_audit_20260716/case_split_with_subject_ids.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/nnunet_ensemble_5seed_official_brats2024_metrics/failure_mode_summary",
    )
    return parser.parse_args()


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def region_summary(frame: pd.DataFrame, region: str) -> dict[str, object]:
    volume = numeric(frame, f"{region}_GT_Complete_Volume")
    positive = frame.loc[volume > 0]
    negative = frame.loc[volume <= 0]
    tp = float(numeric(frame, f"{region}_Num_TP").sum())
    fp = float(numeric(frame, f"{region}_Num_FP").sum())
    fn = float(numeric(frame, f"{region}_Num_FN").sum())
    negative_fp = numeric(negative, f"{region}_Num_FP") > 0
    positive_fn = numeric(positive, f"{region}_Num_FN") > 0
    return {
        "region": region,
        "all_cases": int(len(frame)),
        "gt_positive_cases": int(len(positive)),
        "gt_negative_cases": int(len(negative)),
        "lesion_wise_dice_all": float(
            numeric(frame, f"{region}_LesionWise_Score_Dice").mean()
        ),
        "lesion_wise_dice_positive": float(
            numeric(positive, f"{region}_LesionWise_Score_Dice").mean()
        ),
        "legacy_dice_all": float(numeric(frame, f"{region}_Legacy_Dice").mean()),
        "legacy_dice_positive": float(
            numeric(positive, f"{region}_Legacy_Dice").mean()
        ),
        "lesion_wise_hd95_all": float(
            numeric(frame, f"{region}_LesionWise_Score_HD95").mean()
        ),
        "lesion_wise_hd95_positive": float(
            numeric(positive, f"{region}_LesionWise_Score_HD95").mean()
        ),
        "lesion_tp": tp,
        "lesion_fp": fp,
        "lesion_fn": fn,
        "lesion_precision": safe_ratio(tp, tp + fp),
        "lesion_recall": safe_ratio(tp, tp + fn),
        "gt_negative_fp_cases": int(negative_fp.sum()),
        "gt_negative_fp_case_rate": safe_ratio(float(negative_fp.sum()), float(len(negative))),
        "gt_positive_cases_with_fn": int(positive_fn.sum()),
        "gt_positive_case_fn_rate": safe_ratio(float(positive_fn.sum()), float(len(positive))),
    }


def subgroup_summaries(frame: pd.DataFrame, region: str) -> list[dict[str, object]]:
    volume_column = f"{region}_GT_Complete_Volume"
    positive = frame.loc[numeric(frame, volume_column) > 0].copy()
    if positive.empty:
        return []
    positive[volume_column] = numeric(positive, volume_column)
    rows: list[dict[str, object]] = []
    for fraction in (0.10, 0.25, 0.50):
        threshold = float(positive[volume_column].quantile(fraction))
        subset = positive.loc[positive[volume_column] <= threshold]
        tp = float(numeric(subset, f"{region}_Num_TP").sum())
        fp = float(numeric(subset, f"{region}_Num_FP").sum())
        fn = float(numeric(subset, f"{region}_Num_FN").sum())
        rows.append(
            {
                "region": region,
                "positive_volume_quantile": fraction,
                "max_gt_volume_voxels": threshold,
                "case_count": int(len(subset)),
                "lesion_wise_dice": float(
                    numeric(subset, f"{region}_LesionWise_Score_Dice").mean()
                ),
                "legacy_dice": float(numeric(subset, f"{region}_Legacy_Dice").mean()),
                "lesion_wise_hd95": float(
                    numeric(subset, f"{region}_LesionWise_Score_HD95").mean()
                ),
                "lesion_tp": tp,
                "lesion_fp": fp,
                "lesion_fn": fn,
                "lesion_precision": safe_ratio(tp, tp + fp),
                "lesion_recall": safe_ratio(tp, tp + fn),
            }
        )
    return rows


def legacy_overlap_summary(
    metrics: pd.DataFrame, assignment_path: Path
) -> pd.DataFrame:
    if not assignment_path.exists():
        return pd.DataFrame()
    assignment = pd.read_csv(assignment_path)
    required = {"case_id", "split", "subject_id"}
    if not required.issubset(assignment.columns):
        return pd.DataFrame()
    train_subjects = set(assignment.loc[assignment["split"] == "train", "subject_id"])
    val = assignment.loc[assignment["split"] == "val", ["case_id", "subject_id"]].copy()
    val["subject_seen_in_train"] = val["subject_id"].isin(train_subjects)
    merged = metrics.merge(val, on="case_id", how="left")
    rows: list[dict[str, object]] = []
    for seen, group in merged.groupby("subject_seen_in_train", dropna=False):
        row: dict[str, object] = {
            "subject_seen_in_train": seen,
            "case_count": int(len(group)),
        }
        for region in REGIONS:
            row[f"{region}_lesion_wise_dice"] = float(
                numeric(group, f"{region}_LesionWise_Score_Dice").mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.case_metrics).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(metrics_path)
    if len(frame) == 0:
        raise ValueError(f"No metric rows in {metrics_path}")

    global_rows = [region_summary(frame, region) for region in REGIONS]
    subgroup_rows = [
        row for region in REGIONS for row in subgroup_summaries(frame, region)
    ]
    global_frame = pd.DataFrame(global_rows)
    subgroup_frame = pd.DataFrame(subgroup_rows)
    overlap_frame = legacy_overlap_summary(frame, Path(args.legacy_assignment).resolve())
    global_frame.to_csv(output_dir / "region_summary.csv", index=False)
    subgroup_frame.to_csv(output_dir / "small_volume_summary.csv", index=False)
    if not overlap_frame.empty:
        overlap_frame.to_csv(output_dir / "legacy_overlap_descriptive_only.csv", index=False)

    payload = {
        "case_count": int(len(frame)),
        "regions": global_rows,
        "small_volume_subgroups": subgroup_rows,
        "legacy_overlap_is_descriptive_only": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Official BraTS 2024 failure-mode summary",
        "",
        f"- Cases: {len(frame)}",
        "- Metrics: official lesion-wise Dice and HD95 plus legacy voxel Dice.",
        "- Warning: these predictions come from the invalid legacy patient-leaking split and are exploratory only.",
        "",
        "## Region-level results",
        "",
        "| Region | LW Dice | Legacy Dice | LW HD95 | TP | FP | FN | Lesion recall | Negative FP rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows:
        negative_rate = row["gt_negative_fp_case_rate"]
        negative_text = "n/a" if np.isnan(negative_rate) else f"{negative_rate:.3f}"
        lines.append(
            f"| {row['region']} | {row['lesion_wise_dice_all']:.4f} | "
            f"{row['legacy_dice_all']:.4f} | {row['lesion_wise_hd95_all']:.2f} | "
            f"{row['lesion_tp']:.0f} | {row['lesion_fp']:.0f} | {row['lesion_fn']:.0f} | "
            f"{row['lesion_recall']:.3f} | {negative_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Lesion-wise scores expose missed and false-positive lesions that were hidden by aggregate voxel Dice. Small-volume subgroup tables are stored in `small_volume_summary.csv`.",
            "The seen/unseen-subject table is descriptive only; it cannot repair the invalid split or estimate leakage inflation because subgroup case mix differs.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
