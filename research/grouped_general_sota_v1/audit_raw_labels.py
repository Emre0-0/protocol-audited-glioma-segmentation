from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk


EXPECTED = {0, 1, 2, 3, 4}


def inspect(path: Path) -> dict[str, object]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayViewFromImage(image)
    labels, counts = np.unique(array, return_counts=True)
    observed = {int(value) for value in labels}
    return {
        "path": str(path),
        "labels": sorted(observed),
        "invalid_labels": sorted(observed - EXPECTED),
        "counts": {
            str(int(label)): int(count)
            for label, count in zip(labels, counts, strict=True)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--expected-cases", type=int, default=1155)
    args = parser.parse_args()

    paths = sorted(args.label_dir.glob("*.nii.gz"))
    if len(paths) != args.expected_cases:
        raise RuntimeError(
            f"Expected {args.expected_cases} masks, found {len(paths)}."
        )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(inspect, paths))
    failures = [row for row in rows if row["invalid_labels"]]
    payload = {
        "case_count": len(rows),
        "expected_labels": sorted(EXPECTED),
        "invalid_case_count": len(failures),
        "invalid_cases": failures,
        "ok": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
