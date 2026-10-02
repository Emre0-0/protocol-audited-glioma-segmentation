from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CASE_PATTERN = re.compile(r"^(BraTS-GLI-\d+)-(\d+)$")
REGIONS = ("WT", "TC", "NETC", "SNFH", "ET", "RC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build publication statistics from grouped OOF and frozen locked-test metrics."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/publication_integrity_v1/statistics"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def subject_id(case_id: str) -> str:
    match = CASE_PATTERN.fullmatch(case_id)
    if match is None:
        raise ValueError(f"Unexpected case identifier: {case_id}")
    return match.group(1)


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values))


def detection_metrics_from_counts(tp: float, fp: float, fn: float) -> dict[str, float]:
    sensitivity = tp / (tp + fn) if tp + fn > 0 else float("nan")
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    f1 = (
        2.0 * sensitivity * precision / (sensitivity + precision)
        if sensitivity + precision > 0
        else float("nan")
    )
    return {
        "true_positive_lesions": float(tp),
        "false_positive_lesions": float(fp),
        "false_negative_lesions": float(fn),
        "reference_lesions": float(tp + fn),
        "predicted_lesions": float(tp + fp),
        "sensitivity": float(sensitivity),
        "precision": float(precision),
        "f1": float(f1),
    }


def cluster_bootstrap_detection(
    frame: pd.DataFrame,
    region: str,
    samples: int,
    seed: int,
    suffix: str = "",
) -> dict[str, Any]:
    columns = [
        f"{region}_Num_TP{suffix}",
        f"{region}_Num_FP{suffix}",
        f"{region}_Num_FN{suffix}",
    ]
    work = frame[["subject_id", *columns]].copy()
    work[columns] = work[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    subjects = sorted(work["subject_id"].astype(str).unique())
    blocks = {
        subject: work.loc[work["subject_id"].astype(str) == subject, columns]
        .sum(axis=0)
        .to_numpy(dtype=float)
        for subject in subjects
    }
    point_counts = work[columns].sum(axis=0).to_numpy(dtype=float)
    point = detection_metrics_from_counts(*point_counts)
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(samples, dtype=float) for name in ("sensitivity", "precision", "f1")}
    for index in range(samples):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        counts = np.sum([blocks[item] for item in sampled], axis=0)
        metrics = detection_metrics_from_counts(*counts)
        for name in draws:
            draws[name][index] = metrics[name]
    for name in draws:
        point[f"{name}_ci95_lower"] = float(np.nanpercentile(draws[name], 2.5))
        point[f"{name}_ci95_upper"] = float(np.nanpercentile(draws[name], 97.5))
    return point


def cluster_bootstrap_paired_detection(
    frame: pd.DataFrame,
    region: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_columns = [
        f"{region}_Num_TP_baseline",
        f"{region}_Num_FP_baseline",
        f"{region}_Num_FN_baseline",
    ]
    final_columns = [
        f"{region}_Num_TP_final",
        f"{region}_Num_FP_final",
        f"{region}_Num_FN_final",
    ]
    columns = baseline_columns + final_columns
    work = frame[["subject_id", *columns]].copy()
    work[columns] = work[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    subjects = sorted(work["subject_id"].astype(str).unique())
    blocks = {
        subject: work.loc[work["subject_id"].astype(str) == subject, columns]
        .sum(axis=0)
        .to_numpy(dtype=float)
        for subject in subjects
    }
    totals = work[columns].sum(axis=0).to_numpy(dtype=float)
    baseline = detection_metrics_from_counts(*totals[:3])
    final = detection_metrics_from_counts(*totals[3:])
    result: dict[str, Any] = {"baseline": baseline, "final": final}
    rng = np.random.default_rng(seed)
    delta_draws = {
        name: np.empty(samples, dtype=float)
        for name in ("sensitivity", "precision", "f1")
    }
    for index in range(samples):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        counts = np.sum([blocks[item] for item in sampled], axis=0)
        baseline_draw = detection_metrics_from_counts(*counts[:3])
        final_draw = detection_metrics_from_counts(*counts[3:])
        for name in delta_draws:
            delta_draws[name][index] = final_draw[name] - baseline_draw[name]
    result["delta"] = {}
    for name in delta_draws:
        result["delta"][name] = float(final[name] - baseline[name])
        result["delta"][f"{name}_ci95_lower"] = float(
            np.nanpercentile(delta_draws[name], 2.5)
        )
        result["delta"][f"{name}_ci95_upper"] = float(
            np.nanpercentile(delta_draws[name], 97.5)
        )
    return result


def subject_weighted_mean(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    grouped = frame.groupby("subject_id", sort=True)[columns].mean(numeric_only=True)
    return {column: float(grouped[column].mean()) for column in columns}


def cluster_bootstrap_means(
    frame: pd.DataFrame,
    columns: list[str],
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    subjects = sorted(frame["subject_id"].unique())
    blocks = {
        subject: frame.loc[frame["subject_id"] == subject, columns].to_numpy(dtype=float)
        for subject in subjects
    }
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(columns)), dtype=np.float64)
    for index in range(samples):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        values = np.concatenate([blocks[item] for item in sampled], axis=0)
        draws[index] = np.nanmean(values, axis=0)
    result: dict[str, dict[str, float]] = {}
    for index, column in enumerate(columns):
        result[column] = {
            "mean": float(np.nanmean(frame[column].to_numpy(dtype=float))),
            "ci95_lower": float(np.nanpercentile(draws[:, index], 2.5)),
            "ci95_upper": float(np.nanpercentile(draws[:, index], 97.5)),
        }
    return result


def cluster_bootstrap_paired(
    frame: pd.DataFrame,
    baseline_columns: list[str],
    final_columns: list[str],
    names: list[str],
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    subjects = sorted(frame["subject_id"].unique())
    baseline_blocks = {
        subject: frame.loc[frame["subject_id"] == subject, baseline_columns].to_numpy(dtype=float)
        for subject in subjects
    }
    final_blocks = {
        subject: frame.loc[frame["subject_id"] == subject, final_columns].to_numpy(dtype=float)
        for subject in subjects
    }
    rng = np.random.default_rng(seed)
    baseline_draws = np.empty((samples, len(names)), dtype=np.float64)
    final_draws = np.empty_like(baseline_draws)
    for index in range(samples):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        baseline_values = np.concatenate([baseline_blocks[item] for item in sampled], axis=0)
        final_values = np.concatenate([final_blocks[item] for item in sampled], axis=0)
        baseline_draws[index] = np.nanmean(baseline_values, axis=0)
        final_draws[index] = np.nanmean(final_values, axis=0)
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        baseline = finite_mean(frame[baseline_columns[index]].to_numpy(dtype=float))
        final = finite_mean(frame[final_columns[index]].to_numpy(dtype=float))
        delta_draws = final_draws[:, index] - baseline_draws[:, index]
        result[name] = {
            "baseline": baseline,
            "final": final,
            "delta": final - baseline,
            "baseline_ci95_lower": float(np.nanpercentile(baseline_draws[:, index], 2.5)),
            "baseline_ci95_upper": float(np.nanpercentile(baseline_draws[:, index], 97.5)),
            "final_ci95_lower": float(np.nanpercentile(final_draws[:, index], 2.5)),
            "final_ci95_upper": float(np.nanpercentile(final_draws[:, index], 97.5)),
            "delta_ci95_lower": float(np.nanpercentile(delta_draws, 2.5)),
            "delta_ci95_upper": float(np.nanpercentile(delta_draws, 97.5)),
        }
    return result


def sign_flip_p_value(subject_differences: np.ndarray, samples: int, seed: int) -> float:
    values = np.asarray(subject_differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    batch_size = 5000
    while completed < samples:
        count = min(batch_size, samples - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(values)))
        permuted = np.abs((signs * values[None, :]).mean(axis=1))
        exceed += int(np.count_nonzero(permuted >= observed))
        completed += count
    return float((exceed + 1) / (samples + 1))


def cluster_case_weighted_sign_flip_p_value(
    frame: pd.DataFrame,
    baseline_column: str,
    final_column: str,
    samples: int,
    seed: int,
) -> float:
    differences = (
        pd.to_numeric(frame[final_column], errors="coerce")
        - pd.to_numeric(frame[baseline_column], errors="coerce")
    )
    valid = differences.notna()
    if not valid.any():
        return float("nan")
    work = pd.DataFrame(
        {
            "subject_id": frame.loc[valid, "subject_id"].astype(str),
            "difference": differences.loc[valid].astype(float),
        }
    )
    block_sums = work.groupby("subject_id", sort=True)["difference"].sum().to_numpy()
    case_count = len(work)
    observed = abs(float(work["difference"].mean()))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    batch_size = 5000
    while completed < samples:
        count = min(batch_size, samples - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(block_sums)))
        permuted = np.abs((signs * block_sums[None, :]).sum(axis=1) / case_count)
        exceed += int(np.count_nonzero(permuted >= observed))
        completed += count
    return float((exceed + 1) / (samples + 1))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    finite = [(name, value) for name, value in p_values.items() if np.isfinite(value)]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted: dict[str, float] = {name: float("nan") for name in p_values}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def completed_fold_paths(root: Path) -> list[tuple[int, Path]]:
    candidates = [
        (0, root / "experiments/grouped_delta_roi_v1/official_metrics/baseline/case_metrics_official.csv"),
        (1, root / "experiments/grouped_fivefold_cv_v11/fold_1/official_metrics/case_metrics_official.csv"),
        (2, root / "experiments/grouped_fivefold_cv_v11/fold_2/official_metrics/case_metrics_official.csv"),
        (3, root / "experiments/grouped_fivefold_cv_v11/fold_3/official_metrics/case_metrics_official.csv"),
        (4, root / "experiments/grouped_fivefold_cv_v11/fold_4/official_metrics/case_metrics_official.csv"),
    ]
    return [(fold, path) for fold, path in candidates if path.is_file()]


def build_oof_statistics(
    root: Path,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    split_path = root / "processed/brats_subject_grouped_protocol_v1/splits_final.json"
    folds = json.loads(split_path.read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    fold_hd95_rows: list[dict[str, Any]] = []
    dice_columns = [f"{region}_LesionWise_Score_Dice" for region in REGIONS]
    hd_columns = [f"{region}_LesionWise_Score_HD95" for region in REGIONS]
    for fold, path in completed_fold_paths(root):
        frame = pd.read_csv(path)
        frame["subject_id"] = frame["case_id"].astype(str).map(subject_id)
        frame["fold"] = fold
        expected = set(folds[fold]["val"])
        observed = set(frame["case_id"].astype(str))
        if expected != observed:
            raise RuntimeError(f"Fold {fold} metric IDs do not match its validation split")
        for column in dice_columns + hd_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["six_region_mean"] = frame[dice_columns].mean(axis=1)
        frames.append(frame)
        row: dict[str, Any] = {
            "fold": fold,
            "cases": len(frame),
            "subjects": frame["subject_id"].nunique(),
        }
        for region, column in zip(REGIONS, dice_columns, strict=True):
            row[region] = float(frame[column].mean())
        row["six_region_mean"] = float(np.mean([row[item] for item in REGIONS]))
        fold_rows.append(row)
        fold_hd95 = cluster_bootstrap_means(
            frame,
            hd_columns,
            bootstrap_samples,
            seed + 1000 + fold,
        )
        for region, column in zip(REGIONS, hd_columns, strict=True):
            fold_hd95_rows.append(
                {
                    "fold": fold,
                    "cases": len(frame),
                    "subjects": int(frame["subject_id"].nunique()),
                    "region": region,
                    **fold_hd95[column],
                }
            )
    oof = pd.concat(frames, ignore_index=True)
    if oof["case_id"].duplicated().any():
        raise RuntimeError("A development case appears in more than one OOF metric file")
    metric_columns = dice_columns + hd_columns + ["six_region_mean"]
    for column in metric_columns:
        oof[column] = pd.to_numeric(oof[column], errors="coerce")
    bootstrap = cluster_bootstrap_means(
        oof,
        metric_columns,
        bootstrap_samples,
        seed,
    )
    subject_weighted = subject_weighted_mean(oof, metric_columns)
    lesion_detection = {
        region: cluster_bootstrap_detection(
            oof,
            region,
            bootstrap_samples,
            seed + 500 + index,
        )
        for index, region in enumerate(REGIONS)
    }
    return (
        {
            "status": "COMPLETE" if len(frames) == 5 else "INTERIM",
            "completed_folds": [row["fold"] for row in fold_rows],
            "completed_fold_count": len(frames),
            "expected_fold_count": 5,
            "cases": len(oof),
            "subjects": int(oof["subject_id"].nunique()),
            "case_weighted_cluster_bootstrap": bootstrap,
            "subject_weighted_alternative_estimand": subject_weighted,
            "lesion_detection": lesion_detection,
            "folds": fold_rows,
            "fold_hd95": fold_hd95_rows,
            "warning": (
                None
                if len(frames) == 5
                else "Do not report these pooled OOF values as final five-fold results until all folds are complete."
            ),
        },
        oof,
    )


def build_locked_test_statistics(
    root: Path,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    path = root / "experiments/final_locked_test_v1/test_case_metrics_paired.csv"
    frame = pd.read_csv(path)
    if "subject_id" not in frame:
        frame["subject_id"] = frame["case_id"].astype(str).map(subject_id)
    baseline_columns = [f"{region}_LesionWise_Score_Dice_baseline" for region in REGIONS]
    final_columns = [f"{region}_LesionWise_Score_Dice_final" for region in REGIONS]
    hd95_baseline_columns = [
        f"{region}_LesionWise_Score_HD95_baseline" for region in REGIONS
    ]
    hd95_final_columns = [
        f"{region}_LesionWise_Score_HD95_final" for region in REGIONS
    ]
    for column in baseline_columns + final_columns + hd95_baseline_columns + hd95_final_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["six_region_mean_baseline"] = frame[baseline_columns].mean(axis=1)
    frame["six_region_mean_final"] = frame[final_columns].mean(axis=1)
    names = list(REGIONS) + ["Six-region mean"]
    baseline_columns.append("six_region_mean_baseline")
    final_columns.append("six_region_mean_final")
    paired = cluster_bootstrap_paired(
        frame,
        baseline_columns,
        final_columns,
        names,
        bootstrap_samples,
        seed,
    )
    hd95_paired = cluster_bootstrap_paired(
        frame,
        hd95_baseline_columns,
        hd95_final_columns,
        list(REGIONS),
        bootstrap_samples,
        seed + 50,
    )
    subject_level = frame.groupby("subject_id", sort=True)[baseline_columns + final_columns].mean()
    raw_subject_weighted_p: dict[str, float] = {}
    raw_case_weighted_p: dict[str, float] = {}
    for index, name in enumerate(names):
        differences = (
            subject_level[final_columns[index]] - subject_level[baseline_columns[index]]
        ).to_numpy(dtype=float)
        raw_subject_weighted_p[name] = sign_flip_p_value(
            differences,
            permutation_samples,
            seed + 100 + index,
        )
        raw_case_weighted_p[name] = cluster_case_weighted_sign_flip_p_value(
            frame,
            baseline_columns[index],
            final_columns[index],
            permutation_samples,
            seed + 200 + index,
        )
    adjusted_subject_weighted = holm_adjust(raw_subject_weighted_p)
    adjusted_case_weighted = holm_adjust(raw_case_weighted_p)
    for name in names:
        paired[name]["cluster_sign_flip_p_case_weighted_two_sided"] = raw_case_weighted_p[name]
        paired[name]["cluster_sign_flip_holm_adjusted_p"] = adjusted_case_weighted[name]
        paired[name]["subject_weighted_sign_flip_p_two_sided"] = raw_subject_weighted_p[name]
        paired[name]["subject_weighted_sign_flip_holm_adjusted_p"] = adjusted_subject_weighted[name]
    baseline_subject = subject_weighted_mean(frame, baseline_columns)
    final_subject = subject_weighted_mean(frame, final_columns)
    subject_weighted: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        baseline = baseline_subject[baseline_columns[index]]
        final = final_subject[final_columns[index]]
        subject_weighted[name] = {
            "baseline": baseline,
            "final": final,
            "delta": final - baseline,
        }
    hd95_baseline_subject = subject_weighted_mean(frame, hd95_baseline_columns)
    hd95_final_subject = subject_weighted_mean(frame, hd95_final_columns)
    hd95_subject_weighted: dict[str, dict[str, float]] = {}
    for index, region in enumerate(REGIONS):
        baseline = hd95_baseline_subject[hd95_baseline_columns[index]]
        final = hd95_final_subject[hd95_final_columns[index]]
        hd95_subject_weighted[region] = {
            "baseline": baseline,
            "final": final,
            "delta": final - baseline,
        }
    lesion_detection = {
        region: cluster_bootstrap_paired_detection(
            frame,
            region,
            bootstrap_samples,
            seed + 700 + index,
        )
        for index, region in enumerate(REGIONS)
    }
    summary = json.loads(
        (root / "experiments/final_locked_test_v1/test_summary.json").read_text(
            encoding="utf-8"
        )
    )
    return (
        {
            "status": "ONE_TIME_RETROSPECTIVE_HOLDOUT_CLOSED",
            "evidence_class": (
                "Secondary retrospective subject-disjoint holdout; not pristine external evidence because the partition was defined after legacy exploration."
            ),
            "legacy_case_roles": {"train": 139, "val": 27, "test": 29},
            "cases": len(frame),
            "subjects": int(frame["subject_id"].nunique()),
            "case_weighted_subject_cluster_bootstrap": paired,
            "subject_weighted_alternative_estimand": subject_weighted,
            "hd95_case_weighted_subject_cluster_bootstrap": hd95_paired,
            "hd95_subject_weighted_alternative_estimand": hd95_subject_weighted,
            "lesion_detection_comparison": lesion_detection,
            "multiplicity": {
                "method": "Holm family-wise correction",
                "family": names,
                "primary_test": (
                    "two-sided subject-cluster sign-flip permutation test using the case-weighted mean statistic"
                ),
                "sensitivity_test": (
                    "two-sided sign-flip permutation test on subject-level mean differences"
                ),
                "permutations": permutation_samples,
            },
            "exploratory_post_hoc_subgroups": summary["subgroups"],
            "claim_boundary": (
                "No adjusted superiority claim is supported. The frozen V2 result must be reported with the tiny ET-positive harm, "
                "and subgroup analyses must be labeled post-hoc exploratory because they were not preregistered before opening the holdout."
            ),
        },
        frame,
    )


def write_tables(
    output_dir: Path,
    oof: dict[str, Any],
    locked: dict[str, Any],
) -> None:
    pd.DataFrame(oof["folds"]).to_csv(output_dir / "grouped_fold_metrics.csv", index=False)
    oof_rows = []
    for metric, values in oof["case_weighted_cluster_bootstrap"].items():
        oof_rows.append(
            {
                "metric": metric,
                **values,
                "subject_weighted_mean": oof["subject_weighted_alternative_estimand"][metric],
            }
        )
    pd.DataFrame(oof_rows).to_csv(output_dir / "grouped_oof_summary.csv", index=False)
    pooled_hd95_rows = []
    for region in REGIONS:
        metric = f"{region}_LesionWise_Score_HD95"
        pooled_hd95_rows.append(
            {
                "region": region,
                **oof["case_weighted_cluster_bootstrap"][metric],
                "subject_weighted_mean": oof["subject_weighted_alternative_estimand"][metric],
            }
        )
    pd.DataFrame(pooled_hd95_rows).to_csv(
        output_dir / "grouped_oof_hd95.csv",
        index=False,
    )
    pd.DataFrame(oof["fold_hd95"]).to_csv(
        output_dir / "grouped_fold_hd95.csv",
        index=False,
    )
    detection_rows = []
    for region, values in oof["lesion_detection"].items():
        detection_rows.append({"region": region, **values})
    pd.DataFrame(detection_rows).to_csv(
        output_dir / "grouped_oof_lesion_detection.csv",
        index=False,
    )
    locked_rows = []
    for metric, values in locked["case_weighted_subject_cluster_bootstrap"].items():
        locked_rows.append(
            {
                "metric": metric,
                **values,
                "subject_weighted_baseline": locked["subject_weighted_alternative_estimand"][metric]["baseline"],
                "subject_weighted_final": locked["subject_weighted_alternative_estimand"][metric]["final"],
                "subject_weighted_delta": locked["subject_weighted_alternative_estimand"][metric]["delta"],
            }
        )
    pd.DataFrame(locked_rows).to_csv(output_dir / "locked_test_comparison.csv", index=False)
    locked_hd95_rows = []
    for region in REGIONS:
        locked_hd95_rows.append(
            {
                "region": region,
                **locked["hd95_case_weighted_subject_cluster_bootstrap"][region],
                "subject_weighted_baseline": locked[
                    "hd95_subject_weighted_alternative_estimand"
                ][region]["baseline"],
                "subject_weighted_final": locked[
                    "hd95_subject_weighted_alternative_estimand"
                ][region]["final"],
                "subject_weighted_delta": locked[
                    "hd95_subject_weighted_alternative_estimand"
                ][region]["delta"],
            }
        )
    pd.DataFrame(locked_hd95_rows).to_csv(
        output_dir / "locked_test_hd95_comparison.csv",
        index=False,
    )
    holdout_detection_rows = []
    for region, values in locked["lesion_detection_comparison"].items():
        holdout_detection_rows.append(
            {
                "region": region,
                **{f"baseline_{key}": value for key, value in values["baseline"].items()},
                **{f"final_{key}": value for key, value in values["final"].items()},
                **{f"delta_{key}": value for key, value in values["delta"].items()},
            }
        )
    pd.DataFrame(holdout_detection_rows).to_csv(
        output_dir / "retrospective_holdout_lesion_detection.csv",
        index=False,
    )
    pd.DataFrame(locked["exploratory_post_hoc_subgroups"]).to_csv(
        output_dir / "locked_test_subgroups.csv", index=False
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    oof, _ = build_oof_statistics(root, args.bootstrap_samples, args.seed)
    locked, _ = build_locked_test_statistics(
        root,
        args.bootstrap_samples,
        args.permutation_samples,
        args.seed,
    )
    payload = {
        "schema_version": 2,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "unit": "subject",
            "seed": args.seed,
        },
        "grouped_oof": oof,
        "locked_test": locked,
    }
    (output_dir / "publication_statistics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_tables(output_dir, oof, locked)
    print(json.dumps({
        "output_dir": str(output_dir),
        "oof_status": oof["status"],
        "completed_folds": oof["completed_folds"],
        "oof_cases": oof["cases"],
        "locked_test_status": locked["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
