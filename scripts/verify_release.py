"""Verify the distributable without MRI data, PyTorch, or other dependencies."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ("WT", "TC", "NETC", "SNFH", "ET", "RC")
DATASET = "Dataset503_BraTS2024Grouped4Ch"
MANIFEST = "release_manifest.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_hashes(root, hashes):
    root = Path(root).resolve()
    for name, expected in hashes.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), f"Path outside release: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"SHA-256 mismatch: {name}")


def validate_protocol(splits, development, holdout):
    require(len(splits) == 5, "Expected five folds")
    for rows in (development, holdout):
        require(len({r['case_id'] for r in rows}) == len(rows), "Duplicate scan ID")
    dev = {r["case_id"]: r["subject_id"] for r in development}
    held = {r["case_id"]: r["subject_id"] for r in holdout}
    require(not (dev.keys() & held.keys()), "Development/holdout scan overlap")
    require(not (set(dev.values()) & set(held.values())), "Development/holdout subject overlap")
    validation_counts = Counter()
    fold_counts = []
    for i, fold in enumerate(splits):
        train, val = set(fold["train"]), set(fold["val"])
        require(len(train) == len(fold["train"]) and len(val) == len(fold["val"]), "Duplicate fold ID")
        require(not train & val, f"Train/validation scan overlap: fold {i}")
        require(train | val == set(dev), f"Fold {i} does not cover development")
        require(not ({dev[c] for c in train} & {dev[c] for c in val}), f"Subject leakage: fold {i}")
        validation_counts.update(val)
        fold_counts.append((len(val), len({dev[c] for c in val})))
    require(set(validation_counts) == set(dev), "OOF coverage differs")
    require(set(validation_counts.values()) == {1}, "Each scan must be OOF exactly once")
    return fold_counts


def verify_protocol(root=ROOT):
    protocol = root / "processed/brats_subject_grouped_protocol_v1"
    development = read_csv(protocol / "development_cases.csv")
    holdout = read_csv(protocol / "locked_test_cases.csv")
    for name in ("development_cases.csv", "locked_test_cases.csv", "case_assignments.csv"):
        rows = read_csv(protocol / name)
        require(set(rows[0]) <= {"case_id", "subject_id", "protocol_partition"}, f"Unexpected public fields: {name}")
    splits = json.loads((protocol / "splits_final.json").read_text(encoding="utf-8-sig"))
    plan_splits = json.loads((root / "processed/nnunet_preprocessed" / DATASET / "splits_final.json").read_text(encoding="utf-8-sig"))
    require(splits == plan_splits, "Training splits differ from audited protocol")
    counts = validate_protocol(splits, development, holdout)
    require((len(development), len({r['subject_id'] for r in development})) == (1155, 521), "Development count differs")
    require((len(holdout), len({r['subject_id'] for r in holdout})) == (195, 92), "Holdout count differs")
    require(counts == [(223, 103), (247, 107), (243, 105), (225, 105), (217, 101)], "Fold counts differ")
    return counts


def verify_statistics(root=ROOT):
    folder = root / "experiments/publication_integrity_v1/statistics"
    dice = read_csv(folder / "grouped_oof_summary.csv")
    require([r["metric"] for r in dice[:6]] == [r + "_LesionWise_Score_Dice" for r in REGIONS], "Dice region order differs")
    for row in dice:
        if "HD95" in row["metric"]:
            continue
        mean, low, high = (float(row[c]) for c in ("mean", "ci95_lower", "ci95_upper"))
        require(0 <= low <= mean <= high <= 1, "Invalid Dice interval")
    mean_row = dice[-1]
    require(abs(float(mean_row["mean"]) - 0.763401466019467) < 1e-12, "Pooled Dice differs")
    for name, expected in (("grouped_oof_hd95.csv", 6), ("grouped_fold_hd95.csv", 30), ("locked_test_hd95_comparison.csv", 6)):
        rows = read_csv(folder / name)
        require(len(rows) == expected, f"HD95 rows differ: {name}")
        require([r["region"] for r in rows] == list(REGIONS) * (expected // 6), f"HD95 region order differs: {name}")
        for row in rows:
            prefixes = ("baseline", "final") if "baseline" in row else ("mean",)
            for prefix in prefixes:
                ci = "" if prefix == "mean" else prefix + "_"
                value, low, high = float(row[prefix]), float(row[ci + "ci95_lower"]), float(row[ci + "ci95_upper"])
                require(all(math.isfinite(v) for v in (value, low, high)) and 0 <= low <= value <= high, f"Invalid HD95 CI: {name}")


def release_files(root):
    excluded = {".git", ".venv", "__pycache__", "outputs", ".pytest_cache"}
    allowed = {".py", ".json", ".csv", ".md", ".txt", ".yml", ".yaml", ".cff", ".bib"}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or excluded.intersection(relative.parts) or path.name == MANIFEST:
            continue
        if path.suffix in allowed or path.name in {"LICENSE", ".gitignore", ".gitattributes"}:
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true", help="Maintainer: record hashes after reviewed edits")
    args = parser.parse_args()
    folds = verify_protocol()
    verify_statistics()
    vendor = ROOT / "research/vendor/BraTS-2024-Metrics"
    upstream = json.loads((vendor / "UPSTREAM.json").read_text())
    verify_hashes(vendor, upstream["files_sha256"])
    manifest_path = ROOT / MANIFEST
    if args.write_manifest:
        hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in release_files(ROOT)}
        manifest_path.write_text(json.dumps({"algorithm": "sha256", "files": hashes}, indent=2) + "\n", encoding="utf-8")
    require(manifest_path.is_file(), "Missing release_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    verify_hashes(ROOT, manifest["files"])
    print(json.dumps({"status": "verified", "files": len(manifest["files"]), "OOF_scans": 1155, "OOF_subjects": 521, "holdout_scans": 195, "fold_validation_scans_subjects": folds, "official_metric_commit": upstream["commit"]}, indent=2))


if __name__ == "__main__":
    main()
