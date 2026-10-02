from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REGIONS = ("WT", "TC", "NETC", "SNFH", "ET", "RC")
METRIC = "LesionWise_Score_Dice_mean"


def load_summary(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    values = dict(zip(frame["region"], frame[METRIC]))
    missing = sorted(set(REGIONS) - set(values))
    if missing:
        raise ValueError(f"Missing regions in {path}: {missing}")
    return {region: float(values[region]) for region in REGIONS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    gate = config["decision_gate"]
    baseline = load_summary(args.baseline / "summary_official.csv")
    candidate = load_summary(args.candidate / "summary_official.csv")
    deltas = {
        region: candidate[region] - baseline[region] for region in REGIONS
    }
    baseline_mean = sum(baseline.values()) / len(REGIONS)
    candidate_mean = sum(candidate.values()) / len(REGIONS)
    mean_gain = candidate_mean - baseline_mean
    non_degraded = sum(delta >= 0 for delta in deltas.values())

    checks = {
        "mean_gain": (
            mean_gain
            >= gate["minimum_mean_six_region_lesion_wise_dice_gain"]
        ),
        "region_breadth": (
            non_degraded >= gate["minimum_non_degraded_region_count"]
        ),
        "worst_region_guard": (
            min(deltas.values())
            >= -gate["maximum_any_region_lesion_wise_dice_drop"]
        ),
        "et_rc_guard": (
            min(deltas["ET"], deltas["RC"])
            >= -gate["maximum_et_or_rc_lesion_wise_dice_drop"]
        ),
    }
    decision = "GO_FULL_TRAINING" if all(checks.values()) else "NO_GO_REDESIGN"
    payload = {
        "protocol": "patient-independent grouped fold 0; locked test untouched",
        "metric": "macro mean of six official lesion-wise Dice regions",
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "mean_gain": mean_gain,
        "regions": {
            region: {
                "baseline": baseline[region],
                "candidate": candidate[region],
                "difference": deltas[region],
            }
            for region in REGIONS
        },
        "non_degraded_region_count": non_degraded,
        "checks": checks,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Grouped General-Performance Decision Gate",
        "",
        "- Protocol: patient-independent grouped fold 0",
        "- Locked test used: no",
        f"- Baseline six-region mean: {baseline_mean:.6f}",
        f"- Candidate six-region mean: {candidate_mean:.6f}",
        f"- Difference: {mean_gain:+.6f}",
        f"- Decision: **{decision}**",
        "",
        "| Region | Baseline | Candidate | Difference |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {region} | {baseline[region]:.6f} | {candidate[region]:.6f} | "
        f"{deltas[region]:+.6f} |"
        for region in REGIONS
    )
    lines.extend(
        [
            "",
            "The long full-training stage is authorized only when every "
            "predeclared gate check passes.",
        ]
    )
    (args.output_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
