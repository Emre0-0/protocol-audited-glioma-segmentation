"""Portable entry point for the fixed five-fold baseline (no holdout evaluation)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from verify_release import DATASET, ROOT, verify_protocol

PLAN = "nnUNetResEncUNet10p5GBPlans"


def trainer_name(fold):
    return "nnUNetTrainer_250epochs" if fold == 0 else "nnUNetTrainer_GroupedCV250Duty80"


def environment(workers):
    env = os.environ.copy()
    env.update({
        "nnUNet_raw": str(ROOT / "processed/nnunet_raw"),
        "nnUNet_preprocessed": str(ROOT / "processed/nnunet_preprocessed"),
        "nnUNet_results": str(ROOT / "experiments/nnunet_results"),
        "nnUNet_n_proc_DA": str(workers),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "nnUNet_compile": "false",
    })
    return env


def commands(args):
    py = sys.executable
    if args.mode == "preprocess":
        return [
            ["nnUNetv2_extract_fingerprint", "-d", "503", "-np", str(args.workers), "--verify_dataset_integrity"],
            ["nnUNetv2_preprocess", "-d", "503", "-plans_name", PLAN, "-c", "3d_fullres", "-np", str(args.workers)],
        ]
    fold = args.fold
    if args.mode == "metrics":
        prediction = ROOT / "experiments/nnunet_results" / DATASET / f"{trainer_name(fold)}__{PLAN}__3d_fullres" / f"fold_{fold}" / "validation"
        output = ROOT / ("experiments/grouped_delta_roi_v1/official_metrics/baseline" if fold == 0 else f"experiments/grouped_fivefold_cv_v11/fold_{fold}/official_metrics")
        output = args.output_dir.resolve() if args.output_dir else output
        if not args.dry_run and (output / "case_cache").exists():
            raise ValueError("Metric cache already exists. Choose a NEW --output-dir to avoid stale results.")
        if not args.dry_run:
            splits = json.loads((ROOT / "processed/nnunet_preprocessed" / DATASET / "splits_final.json").read_text())
            actual = {p.name.removesuffix('.nii.gz') for p in prediction.glob('*.nii.gz')}
            if actual != set(splits[fold]["val"]):
                raise ValueError("Predictions do not exactly cover this validation fold")
        return [[py, "scripts/evaluate_brats2024_official_metrics.py", "--prediction-dir", str(prediction), "--ground-truth-dir", str(ROOT / "processed/nnunet_raw" / DATASET / "labelsTr"), "--output-dir", str(output), "--workers", str(args.workers)]]
    if fold == 0:
        cmd = ["nnUNetv2_train", "503", "3d_fullres", "0", "-tr", trainer_name(0), "-p", PLAN, "-device", args.device]
        if args.mode == "validate":
            cmd += ["--val", "--val_best"]
        elif args.resume:
            cmd += ["--c"]
    else:
        cmd = [py, "research/grouped_fivefold_cv_v11/run_fold.py", args.mode, "--fold", str(fold), "--device", args.device]
        if args.mode == "train" and args.resume:
            cmd += ["--continue-training"]
    return [cmd]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preprocess", "train", "validate", "metrics"))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mode != "preprocess" and args.fold is None:
        parser.error("--fold is required")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.resume and args.mode != "train":
        parser.error("--resume is only valid for training")
    verify_protocol()
    for command in commands(args):
        print(json.dumps({"command": command, "dry_run": args.dry_run}), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=environment(args.workers), check=True)


if __name__ == "__main__":
    main()
