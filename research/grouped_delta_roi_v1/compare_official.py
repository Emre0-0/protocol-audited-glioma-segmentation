from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGIONS = ("WT", "TC", "NETC", "SNFH", "ET", "RC")


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def region_metrics(frame: pd.DataFrame, region: str) -> dict[str, float]:
    volume = numeric(frame, f"{region}_GT_Complete_Volume")
    positive = frame.loc[volume > 0]
    negative = frame.loc[volume <= 0]
    tp = float(numeric(frame, f"{region}_Num_TP").sum())
    fn = float(numeric(frame, f"{region}_Num_FN").sum())
    negative_fp_cases = float((numeric(negative, f"{region}_Num_FP") > 0).sum())
    return {
        "lesion_wise_dice": float(numeric(frame, f"{region}_LesionWise_Score_Dice").mean()),
        "legacy_dice": float(numeric(frame, f"{region}_Legacy_Dice").mean()),
        "lesion_recall": safe_ratio(tp, tp + fn),
        "negative_fp_case_rate": safe_ratio(negative_fp_cases, float(len(negative))),
        "positive_cases": float(len(positive)),
        "negative_cases": float(len(negative)),
    }


def small_metrics(frame: pd.DataFrame, region: str, quantile: float) -> dict[str, float]:
    volume_column = f"{region}_GT_Complete_Volume"
    positive = frame.loc[numeric(frame, volume_column) > 0].copy()
    positive[volume_column] = numeric(positive, volume_column)
    threshold = float(positive[volume_column].quantile(quantile))
    subset = positive.loc[positive[volume_column] <= threshold]
    tp = float(numeric(subset, f"{region}_Num_TP").sum())
    fn = float(numeric(subset, f"{region}_Num_FN").sum())
    return {
        "quantile": quantile,
        "threshold_voxels": threshold,
        "case_count": float(len(subset)),
        "lesion_wise_dice": float(
            numeric(subset, f"{region}_LesionWise_Score_Dice").mean()
        ),
        "legacy_dice": float(numeric(subset, f"{region}_Legacy_Dice").mean()),
        "lesion_recall": safe_ratio(tp, tp + fn),
    }


def delta_payload(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in baseline
        if key in candidate and np.isfinite(baseline[key]) and np.isfinite(candidate[key])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare grouped 4-channel and DeltaT1 official metrics.")
    parser.add_argument("--config", default="research/grouped_delta_roi_v1/config.json")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parents[2]
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8-sig"))
    experiment_dir = root / config["paths"]["experiment_dir"]
    baseline_path = experiment_dir / "official_metrics" / "baseline" / "case_metrics_official.csv"
    delta_path = experiment_dir / "official_metrics" / "delta" / "case_metrics_official.csv"
    baseline_frame = pd.read_csv(baseline_path).sort_values("case_id").reset_index(drop=True)
    delta_frame = pd.read_csv(delta_path).sort_values("case_id").reset_index(drop=True)
    if baseline_frame["case_id"].tolist() != delta_frame["case_id"].tolist():
        raise RuntimeError("Baseline and DeltaT1 official metrics cover different validation cases")

    regions: dict[str, Any] = {}
    small: dict[str, Any] = {}
    for region in REGIONS:
        baseline_region = region_metrics(baseline_frame, region)
        delta_region = region_metrics(delta_frame, region)
        regions[region] = {
            "baseline": baseline_region,
            "delta_t1": delta_region,
            "difference": delta_payload(baseline_region, delta_region),
        }
        for quantile in (0.10, 0.25):
            key = f"{region}_bottom_{int(quantile * 100)}pct"
            baseline_small = small_metrics(baseline_frame, region, quantile)
            delta_small = small_metrics(delta_frame, region, quantile)
            small[key] = {
                "baseline": baseline_small,
                "delta_t1": delta_small,
                "difference": delta_payload(baseline_small, delta_small),
            }

    baseline_mean = float(np.mean([regions[region]["baseline"]["lesion_wise_dice"] for region in REGIONS]))
    delta_mean = float(np.mean([regions[region]["delta_t1"]["lesion_wise_dice"] for region in REGIONS]))
    thresholds = config["go_no_go"]
    et_small = small["ET_bottom_10pct"]["difference"]
    tc_small = small["TC_bottom_10pct"]["difference"]
    et_global_gain = regions["ET"]["difference"]["lesion_wise_dice"]
    rc_negative_fp_delta = regions["RC"]["difference"]["negative_fp_case_rate"]
    delta_signal_pass = any(
        (
            et_global_gain >= float(thresholds["delta_et_global_lw_dice_gain"]),
            et_small["lesion_wise_dice"] >= float(thresholds["delta_small_et_lw_dice_gain"]),
            et_small["lesion_recall"] >= float(thresholds["delta_small_et_recall_gain"]),
            tc_small["lesion_wise_dice"] >= float(thresholds["delta_small_tc_lw_dice_gain"]),
            tc_small["lesion_recall"] >= float(thresholds["delta_small_tc_recall_gain"]),
        )
    )
    safety_pass = (
        delta_mean - baseline_mean
        >= -float(thresholds["maximum_mean_region_lw_dice_drop"])
        and (
            np.isnan(rc_negative_fp_delta)
            or rc_negative_fp_delta
            <= float(thresholds["maximum_rc_negative_fp_rate_increase"])
        )
    )
    decision = {
        "delta_signal_for_et_tc_branch": "GO" if delta_signal_pass else "NO_GO",
        "delta_as_global_model": "GO" if delta_signal_pass and safety_pass else "NO_GO",
        "next_stage": (
            "independent_et_tc_localizer_with_delta_prior"
            if delta_signal_pass
            else "independent_et_tc_localizer_without_delta_prior"
        ),
        "test_set_used": False,
    }
    payload = {
        "case_count": len(baseline_frame),
        "protocol": "patient-independent grouped fold 0",
        "trainer": config["nnunet"]["trainer"],
        "mean_region_lesion_wise_dice": {
            "baseline": baseline_mean,
            "delta_t1": delta_mean,
            "difference": delta_mean - baseline_mean,
        },
        "regions": regions,
        "small_volume": small,
        "gates": {
            "delta_signal_pass": delta_signal_pass,
            "safety_pass": safety_pass,
            "thresholds": thresholds,
        },
        "decision": decision,
    }
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "official_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Grouped DeltaT1 ablation",
        "",
        "- Protocol: patient-independent grouped fold 0",
        f"- Validation cases: {len(baseline_frame)}",
        f"- Trainer: `{config['nnunet']['trainer']}`",
        "- Locked test used: no",
        "",
        "| Region | Baseline LW Dice | DeltaT1 LW Dice | Difference | Baseline recall | DeltaT1 recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for region in REGIONS:
        item = regions[region]
        lines.append(
            f"| {region} | {item['baseline']['lesion_wise_dice']:.4f} | "
            f"{item['delta_t1']['lesion_wise_dice']:.4f} | "
            f"{item['difference']['lesion_wise_dice']:+.4f} | "
            f"{item['baseline']['lesion_recall']:.4f} | "
            f"{item['delta_t1']['lesion_recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- DeltaT1 signal for ET/TC branch: **{decision['delta_signal_for_et_tc_branch']}**",
            f"- DeltaT1 as the global model: **{decision['delta_as_global_model']}**",
            f"- Next stage: `{decision['next_stage']}`",
            "",
            "The decision is based on official BraTS lesion-wise metrics. It does not use the locked test set.",
        ]
    )
    (experiment_dir / "official_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["decision"], indent=2))


if __name__ == "__main__":
    main()
