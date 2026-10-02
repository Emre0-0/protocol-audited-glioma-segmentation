from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGIONS = ("WT", "TC", "NETC", "SNFH", "ET", "RC")
SUMMARY_METRIC = "LesionWise_Score_Dice_mean"


def load_summary(directory: Path) -> dict[str, float]:
    frame = pd.read_csv(directory / "summary_official.csv")
    values = dict(zip(frame["region"], frame[SUMMARY_METRIC]))
    missing = sorted(set(REGIONS) - set(values))
    if missing:
        raise ValueError(f"Missing regions in {directory}: {missing}")
    return {region: float(values[region]) for region in REGIONS}


def load_cases(directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "case_metrics_official.csv")
    if len(frame) != frame["case_id"].nunique():
        raise ValueError(f"Duplicate case IDs in {directory}")
    required = [
        "case_id",
        *[f"{region}_LesionWise_Score_Dice" for region in REGIONS],
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing case columns in {directory}: {missing}")
    return frame[required].sort_values("case_id").reset_index(drop=True)


def paired_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not baseline["case_id"].equals(candidate["case_id"]):
        raise ValueError("Baseline and candidate case IDs are not aligned")
    columns = [f"{region}_LesionWise_Score_Dice" for region in REGIONS]
    baseline_values = baseline[columns].to_numpy(dtype=np.float64)
    candidate_values = candidate[columns].to_numpy(dtype=np.float64)
    macro_delta = (candidate_values - baseline_values).mean(axis=1)
    et_delta = (
        candidate["ET_LesionWise_Score_Dice"].to_numpy(dtype=np.float64)
        - baseline["ET_LesionWise_Score_Dice"].to_numpy(dtype=np.float64)
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(baseline),
        size=(samples, len(baseline)),
    )
    macro_bootstrap = macro_delta[indices].mean(axis=1)
    et_bootstrap = et_delta[indices].mean(axis=1)

    def interval(values: np.ndarray) -> list[float]:
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]

    return {
        "paired_case_count": len(baseline),
        "samples": samples,
        "seed": seed,
        "macro_mean_delta": float(macro_delta.mean()),
        "macro_mean_delta_95ci": interval(macro_bootstrap),
        "et_mean_delta": float(et_delta.mean()),
        "et_mean_delta_95ci": interval(et_bootstrap),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    experiment_dir = root / paths["experiment_dir"]
    manifest = json.loads(
        (experiment_dir / "build_manifest.json").read_text(encoding="utf-8")
    )
    baseline_dir = root / paths["baseline_official_metrics"]
    baseline_summary = load_summary(baseline_dir)
    baseline_cases = load_cases(baseline_dir)
    baseline_mean = float(np.mean(list(baseline_summary.values())))
    gate = config["selection_gate"]
    comparisons: list[dict[str, Any]] = []

    for variant in config["variants"]:
        name = variant["name"]
        metric_dir = experiment_dir / "official_metrics" / name
        failures = json.loads(
            (metric_dir / "failures.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (metric_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if failures:
            raise RuntimeError(f"Official metric failures for {name}: {failures[:3]}")
        if int(metadata["successful_cases"]) != len(baseline_cases):
            raise ValueError(f"Incomplete official metrics for {name}")

        candidate_summary = load_summary(metric_dir)
        candidate_cases = load_cases(metric_dir)
        deltas = {
            region: candidate_summary[region] - baseline_summary[region]
            for region in REGIONS
        }
        candidate_mean = float(np.mean(list(candidate_summary.values())))
        mean_gain = candidate_mean - baseline_mean
        tolerance = 1e-12
        non_degraded = sum(
            delta >= -tolerance for delta in deltas.values()
        )
        build_stats = manifest["variants"][name]
        required_invariants = build_stats["required_voxel_invariants"]
        invariants_ok = all(
            build_stats["voxel_invariants"][region]
            for region in required_invariants
        )
        checks = {
            "voxel_invariants": invariants_ok,
            "mean_gain": (
                mean_gain >= float(gate["minimum_six_region_mean_gain"])
            ),
            "et_gain": deltas["ET"] >= float(gate["minimum_et_gain"]),
            "region_breadth": (
                non_degraded
                >= int(gate["minimum_non_degraded_region_count"])
            ),
            "worst_region_guard": (
                min(deltas.values())
                >= -float(gate["maximum_any_region_drop"])
            ),
            "rc_guard": (
                deltas["RC"] >= -float(gate["maximum_rc_drop"])
            ),
        }
        comparisons.append(
            {
                "variant": name,
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "mean_gain": mean_gain,
                "regions": {
                    region: {
                        "baseline": baseline_summary[region],
                        "candidate": candidate_summary[region],
                        "difference": deltas[region],
                    }
                    for region in REGIONS
                },
                "non_degraded_region_count": non_degraded,
                "changed_cases": build_stats["changed_cases"],
                "checks": checks,
                "passes_gate": all(checks.values()),
                "paired_bootstrap": paired_bootstrap(
                    baseline_cases,
                    candidate_cases,
                    int(config["bootstrap_samples"]),
                    int(config["bootstrap_seed"]),
                ),
            }
        )

    passing = [item for item in comparisons if item["passes_gate"]]
    best = (
        max(
            passing,
            key=lambda item: (
                item["mean_gain"],
                item["regions"]["ET"]["difference"],
            ),
        )
        if passing
        else None
    )
    payload = {
        "protocol": "patient-independent grouped fold 0",
        "locked_test_used": False,
        "selection_scope": "three predeclared validation-only ET hybrid variants",
        "baseline_mean": baseline_mean,
        "selection_gate": gate,
        "comparisons": comparisons,
        "decision": (
            "GO_HYBRID_VALIDATION_CANDIDATE"
            if best is not None
            else "NO_GO_KEEP_BASELINE"
        ),
        "selected_variant": None if best is None else best["variant"],
    }
    output_dir = experiment_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    rows = []
    for item in comparisons:
        row = {
            "variant": item["variant"],
            "mean_gain": item["mean_gain"],
            "passes_gate": item["passes_gate"],
            "changed_cases": item["changed_cases"],
        }
        for region in REGIONS:
            row[f"{region}_difference"] = item["regions"][region]["difference"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "comparison.csv", index=False)

    lines = [
        "# ET Specialist Hybrid Validation",
        "",
        "- Protocol: patient-independent grouped fold 0",
        "- Locked test used: no",
        f"- Baseline six-region mean: {baseline_mean:.6f}",
        f"- Decision: **{payload['decision']}**",
        f"- Selected variant: **{payload['selected_variant']}**",
        "",
        "| Variant | Mean delta | ET delta | NETC delta | RC delta | Gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['variant']} | {item['mean_gain']:+.6f} | "
            f"{item['regions']['ET']['difference']:+.6f} | "
            f"{item['regions']['NETC']['difference']:+.6f} | "
            f"{item['regions']['RC']['difference']:+.6f} | "
            f"{'PASS' if item['passes_gate'] else 'FAIL'} |"
        )
    (output_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
