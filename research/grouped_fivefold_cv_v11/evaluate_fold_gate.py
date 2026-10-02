from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REGIONS = ("WT", "TC", "NETC", "SNFH", "ET", "RC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--minimum-mean", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = {row["region"]: row for row in csv.DictReader(handle)}
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    region_scores = {
        region: float(rows[region]["LesionWise_Score_Dice_mean"])
        for region in REGIONS
    }
    mean_score = sum(region_scores.values()) / len(region_scores)
    checks = {
        "all_regions_present": all(region in rows for region in REGIONS),
        "all_scores_finite": all(
            math.isfinite(value) for value in region_scores.values()
        ),
        "successful_cases_match": (
            int(metadata["successful_cases"]) == args.expected_cases
        ),
        "failed_cases_zero": int(metadata["failed_cases"]) == 0,
        "catastrophic_regression_absent": mean_score >= args.minimum_mean,
    }
    passed = all(checks.values())
    result = {
        "status": "PASS_FOR_REVIEW" if passed else "STOP",
        "expected_cases": args.expected_cases,
        "six_region_lesion_wise_dice_mean": mean_score,
        "region_scores": region_scores,
        "minimum_mean_gate": args.minimum_mean,
        "checks": checks,
        "note": (
            "A pass only confirms convergence and metric integrity. "
            "It does not authorize a test-set evaluation or prove superiority."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
