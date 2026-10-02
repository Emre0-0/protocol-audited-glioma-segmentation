from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


REGIONS = ("WT", "TC", "NETC", "SNFH", "ET", "RC")
CASE_PATTERN = re.compile(r"^(BraTS-GLI-\d+)-(\d+)$")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def subject_id(case_id: str) -> str:
    match = CASE_PATTERN.fullmatch(case_id)
    return match.group(1) if match else case_id


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def cluster_bootstrap(
    frame: pd.DataFrame,
    value_function: Callable[[pd.DataFrame], float],
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    subjects = sorted(frame["subject_id"].unique())
    groups = {
        subject: frame.index[frame["subject_id"] == subject].to_numpy()
        for subject in subjects
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        row_indices = np.concatenate([groups[subject] for subject in sampled])
        estimates[index] = value_function(frame.loc[row_indices])
    finite = estimates[np.isfinite(estimates)]
    if not finite.size:
        return float("nan"), float("nan"), float("nan")
    lower, upper = np.quantile(finite, (0.025, 0.975))
    probability_positive = float(np.mean(finite > 0))
    return float(lower), float(upper), probability_positive


def paired_delta(frame: pd.DataFrame, region: str, suffix: str) -> float:
    final = pd.to_numeric(frame[f"{region}_{suffix}_final"], errors="coerce")
    baseline = pd.to_numeric(
        frame[f"{region}_{suffix}_baseline"], errors="coerce"
    )
    return finite_mean((final - baseline).to_numpy())


def group_summary(
    frame: pd.DataFrame,
    name: str,
    mask: pd.Series,
    metric_region: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    group = frame.loc[mask].copy()
    suffix = "LesionWise_Score_Dice"
    baseline = pd.to_numeric(
        group[f"{metric_region}_{suffix}_baseline"], errors="coerce"
    ).to_numpy()
    final = pd.to_numeric(
        group[f"{metric_region}_{suffix}_final"], errors="coerce"
    ).to_numpy()
    delta = finite_mean(final - baseline)
    if group["subject_id"].nunique() >= 2:
        lower, upper, probability = cluster_bootstrap(
            group,
            lambda sample: paired_delta(sample, metric_region, suffix),
            samples,
            seed,
        )
    else:
        lower = upper = probability = float("nan")
    return {
        "subgroup": name,
        "metric": f"{metric_region} lesion-wise Dice",
        "cases": int(len(group)),
        "subjects": int(group["subject_id"].nunique()),
        "baseline_mean": finite_mean(baseline),
        "final_mean": finite_mean(final),
        "paired_delta": delta,
        "delta_ci95_lower": lower,
        "delta_ci95_upper": upper,
        "bootstrap_probability_delta_gt_zero": probability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the one-time locked-test official metrics."
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
    release = json.loads(
        (experiment_dir / "test_label_release_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if release.get("status") != "TEST_LABELS_RELEASED":
        raise RuntimeError("Locked-test labels were not released correctly")

    baseline_path = resolve(
        root, config["paths"]["baseline_official_metrics"]
    ) / "case_metrics_official.csv"
    final_path = resolve(root, config["paths"]["final_official_metrics"]) / (
        "case_metrics_official.csv"
    )
    baseline = pd.read_csv(baseline_path)
    final = pd.read_csv(final_path)
    expected = int(config["protocol"]["expected_test_cases"])
    if len(baseline) != expected or len(final) != expected:
        raise ValueError("Official metric case count does not match locked test")
    if set(baseline["case_id"]) != set(final["case_id"]):
        raise ValueError("Baseline and final official metric IDs differ")

    frame = baseline.merge(
        final,
        on="case_id",
        how="inner",
        suffixes=("_baseline", "_final"),
        validate="one_to_one",
    )
    frame["subject_id"] = frame["case_id"].map(subject_id)
    samples = int(config["statistics"]["bootstrap_samples"])
    seed = int(config["statistics"]["bootstrap_seed"])

    region_rows: list[dict[str, Any]] = []
    for region_index, region in enumerate(REGIONS):
        dice_suffix = "LesionWise_Score_Dice"
        hd_suffix = "LesionWise_Score_HD95"
        baseline_dice = pd.to_numeric(
            frame[f"{region}_{dice_suffix}_baseline"], errors="coerce"
        ).to_numpy()
        final_dice = pd.to_numeric(
            frame[f"{region}_{dice_suffix}_final"], errors="coerce"
        ).to_numpy()
        baseline_hd = pd.to_numeric(
            frame[f"{region}_{hd_suffix}_baseline"], errors="coerce"
        ).to_numpy()
        final_hd = pd.to_numeric(
            frame[f"{region}_{hd_suffix}_final"], errors="coerce"
        ).to_numpy()
        lower, upper, probability = cluster_bootstrap(
            frame,
            lambda sample, current=region: paired_delta(
                sample, current, dice_suffix
            ),
            samples,
            seed + region_index,
        )
        region_rows.append(
            {
                "region": region,
                "cases": expected,
                "subjects": int(frame["subject_id"].nunique()),
                "baseline_lesion_wise_dice": finite_mean(baseline_dice),
                "final_lesion_wise_dice": finite_mean(final_dice),
                "dice_delta": finite_mean(final_dice - baseline_dice),
                "dice_delta_ci95_lower": lower,
                "dice_delta_ci95_upper": upper,
                "bootstrap_probability_delta_gt_zero": probability,
                "baseline_lesion_wise_hd95": finite_mean(baseline_hd),
                "final_lesion_wise_hd95": finite_mean(final_hd),
                "hd95_delta": finite_mean(final_hd - baseline_hd),
                "baseline_legacy_dice": finite_mean(
                    pd.to_numeric(
                        frame[f"{region}_Legacy_Dice_baseline"],
                        errors="coerce",
                    ).to_numpy()
                ),
                "final_legacy_dice": finite_mean(
                    pd.to_numeric(
                        frame[f"{region}_Legacy_Dice_final"],
                        errors="coerce",
                    ).to_numpy()
                ),
            }
        )
    region_table = pd.DataFrame(region_rows)

    for model in ("baseline", "final"):
        columns = [
            f"{region}_LesionWise_Score_Dice_{model}" for region in REGIONS
        ]
        frame[f"six_region_mean_{model}"] = frame[columns].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
    six_delta = finite_mean(
        (
            frame["six_region_mean_final"]
            - frame["six_region_mean_baseline"]
        ).to_numpy()
    )
    six_lower, six_upper, six_probability = cluster_bootstrap(
        frame,
        lambda sample: finite_mean(
            (
                sample["six_region_mean_final"]
                - sample["six_region_mean_baseline"]
            ).to_numpy()
        ),
        samples,
        seed + 100,
    )

    et_volume = pd.to_numeric(
        frame["ET_GT_Complete_Volume_final"], errors="coerce"
    )
    rc_volume = pd.to_numeric(
        frame["RC_GT_Complete_Volume_final"], errors="coerce"
    )
    wt_volume = pd.to_numeric(
        frame["WT_GT_Complete_Volume_final"], errors="coerce"
    )
    total_volume = wt_volume.fillna(0) + rc_volume.fillna(0)
    et_positive = et_volume > 0
    rc_positive = rc_volume > 0

    subgroup_rows: list[dict[str, Any]] = []
    subgroup_specs: list[tuple[str, pd.Series, str]] = []
    for percent in (0.10, 0.25):
        total_threshold = float(total_volume.quantile(percent))
        subgroup_specs.append(
            (
                f"Total foreground bottom {int(percent * 100)}% "
                f"(<= {total_threshold:.0f} voxels)",
                total_volume <= total_threshold,
                "ET",
            )
        )
        if et_positive.any():
            et_threshold = float(et_volume[et_positive].quantile(percent))
            subgroup_specs.append(
                (
                    f"ET-positive bottom {int(percent * 100)}% "
                    f"(<= {et_threshold:.0f} voxels)",
                    et_positive & (et_volume <= et_threshold),
                    "ET",
                )
            )
        if rc_positive.any():
            rc_threshold = float(rc_volume[rc_positive].quantile(percent))
            subgroup_specs.append(
                (
                    f"RC-positive bottom {int(percent * 100)}% "
                    f"(<= {rc_threshold:.0f} voxels)",
                    rc_positive & (rc_volume <= rc_threshold),
                    "RC",
                )
            )
    subgroup_specs.extend(
        [
            ("ET-negative", ~et_positive, "ET"),
            ("RC-negative", ~rc_positive, "RC"),
        ]
    )
    for index, (name, mask, region) in enumerate(subgroup_specs):
        subgroup_rows.append(
            group_summary(
                frame,
                name,
                mask,
                region,
                samples,
                seed + 200 + index,
            )
        )
    subgroup_table = pd.DataFrame(subgroup_rows)

    region_path = experiment_dir / "test_region_metrics.csv"
    subgroup_path = experiment_dir / "test_subgroup_metrics.csv"
    case_path = experiment_dir / "test_case_metrics_paired.csv"
    region_table.to_csv(region_path, index=False)
    subgroup_table.to_csv(subgroup_path, index=False)
    frame.to_csv(case_path, index=False)

    summary = {
        "status": "COMPLETE_ONE_TIME_LOCKED_TEST",
        "cases": expected,
        "subjects": int(frame["subject_id"].nunique()),
        "model_selection_after_test": False,
        "baseline_six_region_mean": finite_mean(
            frame["six_region_mean_baseline"].to_numpy()
        ),
        "final_six_region_mean": finite_mean(
            frame["six_region_mean_final"].to_numpy()
        ),
        "paired_six_region_mean_delta": six_delta,
        "paired_six_region_mean_delta_ci95": [six_lower, six_upper],
        "bootstrap_probability_six_region_delta_gt_zero": six_probability,
        "bootstrap": {
            "samples": samples,
            "seed": seed,
            "unit": config["statistics"]["bootstrap_unit"],
        },
        "regions": region_rows,
        "subgroups": subgroup_rows,
        "artifacts": {
            "region_metrics": str(region_path),
            "subgroup_metrics": str(subgroup_path),
            "paired_case_metrics": str(case_path),
        },
    }
    (experiment_dir / "test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# One-time locked-test result",
        "",
        f"- Cases/subjects: {expected}/{summary['subjects']}",
        "- Test policy: predictions frozen before labels; no post-test selection.",
        f"- Baseline six-region mean Dice: {summary['baseline_six_region_mean']:.6f}",
        f"- Frozen V2 six-region mean Dice: {summary['final_six_region_mean']:.6f}",
        f"- Paired delta: {six_delta:+.6f} "
        f"(95% subject-cluster bootstrap CI {six_lower:+.6f} to {six_upper:+.6f})",
        "",
        "## Region metrics",
        "",
        "| Region | Baseline Dice | Final Dice | Delta | 95% CI | Baseline HD95 | Final HD95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in region_rows:
        lines.append(
            f"| {row['region']} | {row['baseline_lesion_wise_dice']:.4f} | "
            f"{row['final_lesion_wise_dice']:.4f} | {row['dice_delta']:+.4f} | "
            f"[{row['dice_delta_ci95_lower']:+.4f}, {row['dice_delta_ci95_upper']:+.4f}] | "
            f"{row['baseline_lesion_wise_hd95']:.2f} | {row['final_lesion_wise_hd95']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Small/absent-region subgroups",
            "",
            "| Subgroup | Metric | N | Baseline | Final | Delta | 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in subgroup_rows:
        lines.append(
            f"| {row['subgroup']} | {row['metric']} | {row['cases']} | "
            f"{row['baseline_mean']:.4f} | {row['final_mean']:.4f} | "
            f"{row['paired_delta']:+.4f} | "
            f"[{row['delta_ci95_lower']:+.4f}, {row['delta_ci95_upper']:+.4f}] |"
        )
    (experiment_dir / "test_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
