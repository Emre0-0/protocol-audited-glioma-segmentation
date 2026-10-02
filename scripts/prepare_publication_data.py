"""Convert only the fixed development cases; never read holdout scan contents."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.grouped_delta_roi_v1.build_datasets import (
    convert_npz_case, ensure_exact_case_set, write_dataset_json,
)
from verify_release import DATASET, read_csv, verify_protocol


def index_development(root, expected):
    index = {}
    for path in sorted(Path(root).rglob("*.npz")):
        if path.stem not in expected:
            continue
        if path.stem in index:
            raise ValueError(f"Duplicate development scan: {path.stem}")
        index[path.stem] = path.resolve()
    missing = expected - set(index)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} development NPZ files: {sorted(missing)[:5]}")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Reuse completed files only for the SAME input archive")
    args = parser.parse_args()
    verify_protocol()
    expected = {r["case_id"] for r in read_csv(ROOT / "processed/brats_subject_grouped_protocol_v1/development_cases.csv")}
    index = index_development(args.npz_root, expected)
    target = ROOT / "processed/nnunet_raw" / DATASET
    if target.exists() and any(target.iterdir()) and not args.resume:
        parser.error("Output is not empty. Use --resume only for the identical source archive.")
    tasks = [(case, str(index[case]), str(target)) for case in sorted(expected)]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, _ in enumerate(pool.map(convert_npz_case, tasks), 1):
            if i % 25 == 0 or i == len(tasks):
                print(f"Converted/reused {i}/{len(tasks)} development scans", flush=True)
    ensure_exact_case_set(target, expected, channels=4)
    write_dataset_json(target, "BraTS2024Grouped4Ch", len(expected), delta=False)
    print(json.dumps({"development_scans": len(expected), "holdout_converted": 0, "output": str(target)}, indent=2))


if __name__ == "__main__":
    main()
