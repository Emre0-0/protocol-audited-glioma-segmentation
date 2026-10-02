from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_mri_project.data import (
    list_pretrain_files,
    pretrain_subject_id,
    split_pretrain_files,
)


def legacy_file_split(
    files: list[Path], val_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[Path]] = {}
    for path in files:
        by_dataset.setdefault(path.parent.name, []).append(path)
    train: list[Path] = []
    val: list[Path] = []
    for dataset_files in by_dataset.values():
        shuffled = list(dataset_files)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_fraction)))
        val.extend(shuffled[:val_count])
        train.extend(shuffled[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def summarize(train: list[Path], val: list[Path]) -> dict[str, Any]:
    report: dict[str, Any] = {"train_files": len(train), "val_files": len(val)}
    train_subjects = {pretrain_subject_id(path) for path in train}
    val_subjects = {pretrain_subject_id(path) for path in val}
    overlap = train_subjects & val_subjects
    report.update(
        {
            "train_subjects": len(train_subjects),
            "val_subjects": len(val_subjects),
            "overlap_subjects": len(overlap),
            "overlap_subject_ids": sorted(overlap),
            "affected_validation_files": sum(
                pretrain_subject_id(path) in overlap for path in val
            ),
            "by_dataset": {},
        }
    )
    for dataset in sorted({path.parent.name for path in train + val}):
        dataset_train = [path for path in train if path.parent.name == dataset]
        dataset_val = [path for path in val if path.parent.name == dataset]
        dataset_train_subjects = {pretrain_subject_id(path) for path in dataset_train}
        dataset_val_subjects = {pretrain_subject_id(path) for path in dataset_val}
        dataset_overlap = dataset_train_subjects & dataset_val_subjects
        report["by_dataset"][dataset] = {
            "train_files": len(dataset_train),
            "val_files": len(dataset_val),
            "train_subjects": len(dataset_train_subjects),
            "val_subjects": len(dataset_val_subjects),
            "overlap_subjects": len(dataset_overlap),
            "affected_validation_files": sum(
                pretrain_subject_id(path) in dataset_overlap for path in dataset_val
            ),
        }
    return report


def build_markdown(payload: dict[str, Any]) -> str:
    old = payload["legacy_file_level_split"]
    new = payload["grouped_subject_level_split"]
    return "\n".join(
        [
            "# Normal-brain pretraining split audit",
            "",
            "## Legacy checkpoint protocol",
            f"- Train/validation files: {old['train_files']}/{old['val_files']}",
            f"- Cross-split subjects: **{old['overlap_subjects']}**",
            f"- Affected validation files: **{old['affected_validation_files']}**",
            "",
            "## Corrected protocol",
            f"- Train/validation files: {new['train_files']}/{new['val_files']}",
            f"- Cross-split subjects: **{new['overlap_subjects']}**",
            f"- Affected validation files: **{new['affected_validation_files']}**",
            "",
            "The legacy pretraining checkpoint remains an exploratory initialization, but its validation loss and best-epoch claim are not patient-independent. Future pretraining must use the corrected grouped split.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ADNI/OASIS longitudinal split leakage.")
    parser.add_argument("--root", default="processed/pretrain_t1_npz")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="experiments/pretrain_subject_split_audit_20260716",
    )
    args = parser.parse_args()

    files = list_pretrain_files(args.root)
    legacy_train, legacy_val = legacy_file_split(files, args.val_fraction, args.seed)
    grouped_train, grouped_val = split_pretrain_files(
        files, val_fraction=args.val_fraction, seed=args.seed
    )
    payload = {
        "total_files": len(files),
        "legacy_file_level_split": summarize(legacy_train, legacy_val),
        "grouped_subject_level_split": summarize(grouped_train, grouped_val),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(build_markdown(payload), encoding="utf-8")
    (output_dir / "grouped_train_files.txt").write_text(
        "\n".join(str(path) for path in grouped_train) + "\n", encoding="utf-8"
    )
    (output_dir / "grouped_val_files.txt").write_text(
        "\n".join(str(path) for path in grouped_val) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if payload["grouped_subject_level_split"]["overlap_subjects"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
