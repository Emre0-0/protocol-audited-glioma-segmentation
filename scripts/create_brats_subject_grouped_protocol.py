from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


CASE_PATTERN = re.compile(r"^(BraTS-GLI-\d+)-(\d+)$")
LABELS = (1, 2, 3, 4)


def subject_id(case_id: str) -> str:
    match = CASE_PATTERN.fullmatch(case_id)
    return match.group(1) if match else case_id


def make_feature_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = [np.ones(len(frame), dtype=np.float64)]
    names = ["case_count"]
    for label in LABELS:
        presence = frame[f"has_label_{label}"].astype(bool).to_numpy(dtype=np.float64)
        volume = frame[f"label_{label}_voxels"].to_numpy(dtype=np.float64)
        columns.extend(
            [
                presence,
                ((volume > 0) & (volume <= 500)).astype(np.float64),
                ((volume > 500) & (volume <= 2000)).astype(np.float64),
                np.log1p(volume),
            ]
        )
        names.extend(
            [
                f"label_{label}_positive_cases",
                f"label_{label}_tiny_cases_le_500",
                f"label_{label}_small_cases_501_2000",
                f"label_{label}_log_volume_sum",
            ]
        )
    return np.column_stack(columns), names


def balance_score(selected: np.ndarray, total: np.ndarray, target_fraction: float) -> float:
    target = total * target_fraction
    scale = np.maximum(target, 1.0)
    relative = (selected - target) / scale
    return float(np.mean(relative**2))


def choose_locked_test(
    frame: pd.DataFrame,
    features: np.ndarray,
    test_fraction: float,
    candidates: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    groups = frame["subject_id"].to_numpy()
    total = features.sum(axis=0)
    splitter = GroupShuffleSplit(
        n_splits=candidates,
        test_size=test_fraction,
        random_state=random_state,
    )
    best_indices: np.ndarray | None = None
    best_score = float("inf")
    for _, test_indices in splitter.split(frame, groups=groups):
        score = balance_score(features[test_indices].sum(axis=0), total, test_fraction)
        case_fraction = len(test_indices) / len(frame)
        score += 4.0 * (case_fraction - test_fraction) ** 2
        if score < best_score:
            best_score = score
            best_indices = test_indices
    if best_indices is None:
        raise RuntimeError("No grouped test split candidate was generated")
    return best_indices, {
        "candidate_count": candidates,
        "balance_score": best_score,
        "random_state": random_state,
    }


def stratification_labels(frame: pd.DataFrame) -> np.ndarray:
    # NETC presence has two sufficiently populated strata. Rarer joint label
    # patterns are balanced by the multi-feature candidate score below instead
    # of being passed to StratifiedGroupKFold as under-populated classes.
    return frame["has_label_1"].astype(bool).to_numpy(dtype=np.int64)


def choose_development_folds(
    frame: pd.DataFrame,
    features: np.ndarray,
    n_splits: int,
    seed_candidates: int,
    random_state: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    groups = frame["subject_id"].to_numpy()
    y = stratification_labels(frame)
    total = features.sum(axis=0)
    best_folds: list[tuple[np.ndarray, np.ndarray]] | None = None
    best_score = float("inf")
    best_seed: int | None = None
    for offset in range(seed_candidates):
        seed = random_state + offset
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        folds = list(splitter.split(frame, y=y, groups=groups))
        scores = [
            balance_score(features[val_indices].sum(axis=0), total, 1.0 / n_splits)
            for _, val_indices in folds
        ]
        score = float(np.mean(scores) + np.std(scores))
        if score < best_score:
            best_score = score
            best_folds = folds
            best_seed = seed
    if best_folds is None:
        raise RuntimeError("No grouped fold candidate was generated")
    return best_folds, {
        "seed_candidates": seed_candidates,
        "selected_seed": best_seed,
        "balance_score": best_score,
    }


def summarize_partition(frame: pd.DataFrame, feature_names: list[str]) -> dict[str, Any]:
    features, _ = make_feature_matrix(frame)
    sums = features.sum(axis=0)
    return {
        "cases": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "features": {
            name: float(value) for name, value in zip(feature_names, sums, strict=True)
        },
    }


def verify_protocol(
    all_cases: pd.DataFrame,
    development: pd.DataFrame,
    locked_test: pd.DataFrame,
    folds_payload: list[dict[str, list[str]]],
) -> dict[str, Any]:
    all_ids = set(all_cases["case_id"])
    development_ids = set(development["case_id"])
    test_ids = set(locked_test["case_id"])
    development_subjects = set(development["subject_id"])
    test_subjects = set(locked_test["subject_id"])
    errors: list[str] = []
    if development_ids & test_ids:
        errors.append("Exact case overlap between development and locked test")
    if development_subjects & test_subjects:
        errors.append("Subject overlap between development and locked test")
    if development_ids | test_ids != all_ids:
        errors.append("Development and locked test do not cover every case exactly once")

    val_counts: dict[str, int] = {case_id: 0 for case_id in development_ids}
    for index, fold in enumerate(folds_payload):
        train_ids = set(fold["train"])
        val_ids = set(fold["val"])
        if train_ids & val_ids:
            errors.append(f"Fold {index} has exact train/validation overlap")
        if train_ids | val_ids != development_ids:
            errors.append(f"Fold {index} does not cover the full development set")
        train_subjects = set(
            development.loc[development["case_id"].isin(train_ids), "subject_id"]
        )
        val_subjects = set(
            development.loc[development["case_id"].isin(val_ids), "subject_id"]
        )
        if train_subjects & val_subjects:
            errors.append(f"Fold {index} has longitudinal subject leakage")
        for case_id in val_ids:
            val_counts[case_id] += 1
    if set(val_counts.values()) != {1}:
        errors.append("Each development case must appear in validation exactly once")
    unique_val_sets = {tuple(sorted(fold["val"])) for fold in folds_payload}
    if len(unique_val_sets) != len(folds_payload):
        errors.append("Fold validation sets are not unique")
    return {
        "ok": not errors,
        "errors": errors,
        "development_test_subject_overlap": len(development_subjects & test_subjects),
        "development_test_case_overlap": len(development_ids & test_ids),
        "unique_fold_validation_sets": len(unique_val_sets),
        "development_cases_validated_exactly_once": set(val_counts.values()) == {1},
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Patient-independent BraTS protocol v1",
        "",
        f"- Verification: **{'PASS' if summary['verification']['ok'] else 'FAIL'}**",
        f"- Total cases: {summary['total']['cases']}",
        f"- Total subjects: {summary['total']['subjects']}",
        f"- Development: {summary['development']['cases']} cases / {summary['development']['subjects']} subjects",
        f"- Locked test: {summary['locked_test']['cases']} cases / {summary['locked_test']['subjects']} subjects",
        f"- True grouped folds: {len(summary['folds'])}",
        "",
        "## Fold sizes",
    ]
    for fold in summary["folds"]:
        lines.append(
            f"- Fold {fold['fold']}: train {fold['train']['cases']} cases / {fold['train']['subjects']} subjects; "
            f"validation {fold['validation']['cases']} cases / {fold['validation']['subjects']} subjects"
        )
    lines.extend(
        [
            "",
            "## Rules",
            "- Every timepoint from one subject remains in exactly one partition.",
            "- The locked test partition must not be used for model selection or ablation.",
            "- Every development case is validation data in exactly one of the five folds.",
            "- Final reporting must include official BraTS 2024 lesion-wise Dice and HD95.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a leak-free longitudinal BraTS protocol.")
    parser.add_argument(
        "--input-csv",
        default="_archive_inspect/eda_outputs/eda_outputs/brats2024_split.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="processed/brats_subject_grouped_protocol_v1",
    )
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--test-candidates", type=int, default=5000)
    parser.add_argument("--fold-seed-candidates", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20260716)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    frame = frame.copy()
    frame["subject_id"] = frame["case_id"].map(subject_id)
    frame = frame.sort_values("case_id").reset_index(drop=True)
    features, feature_names = make_feature_matrix(frame)

    test_indices, test_search = choose_locked_test(
        frame,
        features,
        args.test_fraction,
        args.test_candidates,
        args.random_state,
    )
    test_mask = np.zeros(len(frame), dtype=bool)
    test_mask[test_indices] = True
    locked_test = frame[test_mask].copy().reset_index(drop=True)
    development = frame[~test_mask].copy().reset_index(drop=True)
    development_features, _ = make_feature_matrix(development)

    folds, fold_search = choose_development_folds(
        development,
        development_features,
        args.folds,
        args.fold_seed_candidates,
        args.random_state,
    )
    folds_payload: list[dict[str, list[str]]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_index, (train_indices, val_indices) in enumerate(folds):
        train = development.iloc[train_indices].copy()
        validation = development.iloc[val_indices].copy()
        folds_payload.append(
            {
                "train": sorted(train["case_id"].tolist()),
                "val": sorted(validation["case_id"].tolist()),
            }
        )
        fold_summaries.append(
            {
                "fold": fold_index,
                "train": summarize_partition(train, feature_names),
                "validation": summarize_partition(validation, feature_names),
            }
        )

    verification = verify_protocol(frame, development, locked_test, folds_payload)
    summary = {
        "protocol": "patient-independent longitudinal holdout plus grouped five-fold development CV",
        "random_state": args.random_state,
        "test_fraction_target": args.test_fraction,
        "test_search": test_search,
        "fold_search": fold_search,
        "total": summarize_partition(frame, feature_names),
        "development": summarize_partition(development, feature_names),
        "locked_test": summarize_partition(locked_test, feature_names),
        "folds": fold_summaries,
        "verification": verification,
        "label_semantics": {
            "1": "NETC",
            "2": "SNFH",
            "3": "ET",
            "4": "RC",
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment = frame[[column for column in frame.columns if column != "split"]].copy()
    assignment["protocol_partition"] = np.where(
        assignment["case_id"].isin(set(locked_test["case_id"])),
        "locked_test",
        "development",
    )
    assignment.to_csv(output_dir / "case_assignments.csv", index=False)
    development.to_csv(output_dir / "development_cases.csv", index=False)
    locked_test.to_csv(output_dir / "locked_test_cases.csv", index=False)
    (output_dir / "splits_final.json").write_text(
        json.dumps(folds_payload, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not verification["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
