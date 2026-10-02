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
    return {region: float(values[region]) for region in REGIONS}


def load_cases(directory: Path) -> pd.DataFrame:
    columns = [
        "case_id",
        *[f"{region}_LesionWise_Score_Dice" for region in REGIONS],
    ]
    frame = pd.read_csv(directory / "case_metrics_official.csv", usecols=columns)
    if len(frame) != frame["case_id"].nunique():
        raise ValueError(f"Duplicate case IDs in {directory}")
    return frame.sort_values("case_id").reset_index(drop=True)


def paired_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not baseline["case_id"].equals(candidate["case_id"]):
        raise ValueError("Case IDs are not aligned")
    columns = [f"{region}_LesionWise_Score_Dice" for region in REGIONS]
    deltas = (
        candidate[columns].to_numpy(dtype=np.float64)
        - baseline[columns].to_numpy(dtype=np.float64)
    )
    macro = deltas.mean(axis=1)
    et = deltas[:, REGIONS.index("ET")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(macro), size=(samples, len(macro)))

    def interval(values: np.ndarray) -> list[float]:
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]

    return {
        "samples": samples,
        "seed": seed,
        "macro_mean_delta": float(macro.mean()),
        "macro_mean_delta_95ci": interval(macro[indices].mean(axis=1)),
        "et_mean_delta": float(et.mean()),
        "et_mean_delta_95ci": interval(et[indices].mean(axis=1)),
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
    experiment = root / paths["experiment_dir"]
    baseline_dir = root / paths["baseline_official_metrics"]
    baseline_summary = load_summary(baseline_dir)
    baseline_cases = load_cases(baseline_dir)
    baseline_mean = float(np.mean(list(baseline_summary.values())))
    manifest = json.loads(
        (experiment / "build_manifest.json").read_text(encoding="utf-8")
    )
    gate = config["selection_gate"]
    comparisons: list[dict[str, Any]] = []

    for variant in config["variants"]:
        name = str(variant["name"])
        metric_dir = experiment / "official_metrics" / name
        failures = json.loads(
            (metric_dir / "failures.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (metric_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if failures or int(metadata["successful_cases"]) != len(baseline_cases):
            raise RuntimeError(f"Incomplete official evaluation for {name}")
        candidate_summary = load_summary(metric_dir)
        candidate_cases = load_cases(metric_dir)
        deltas = {
            region: candidate_summary[region] - baseline_summary[region]
            for region in REGIONS
        }
        candidate_mean = float(np.mean(list(candidate_summary.values())))
        mean_gain = candidate_mean - baseline_mean
        tolerance = 1e-12
        non_degraded = sum(delta >= -tolerance for delta in deltas.values())
        stats = manifest["variants"][name]
        checks = {
            "rc_topology_invariant": bool(
                stats["rc_topology_invariant_all_cases"]
            ),
            "voxel_invariants": all(
                stats["voxel_invariants"][region]
                for region in stats["required_voxel_invariants"]
            ),
            "mean_gain": (
                mean_gain >= float(gate["minimum_six_region_mean_gain"])
            ),
            "et_gain": deltas["ET"] >= float(gate["minimum_et_gain"]),
            "all_regions_non_degraded": (
                non_degraded
                >= int(gate["minimum_non_degraded_region_count"])
            ),
            "worst_region_guard": (
                min(deltas.values())
                >= -float(gate["maximum_any_region_drop"])
            ),
            "rc_guard": (
                deltas["RC"] >= -float(gate["maximum_rc_drop"]) - tolerance
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
                "topology_rollback_cases": stats["topology_rollback_cases"],
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
    selected = (
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
        "ground_truth_used_during_candidate_build": False,
        "baseline_mean": baseline_mean,
        "selection_gate": gate,
        "comparisons": comparisons,
        "decision": (
            "GO_TOPOLOGY_GUARDED_VALIDATION_CANDIDATE"
            if selected is not None
            else "NO_GO_KEEP_BASELINE"
        ),
        "selected_variant": None if selected is None else selected["variant"],
    }
    output_dir = experiment / "comparison"
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
            "topology_rollback_cases": item["topology_rollback_cases"],
        }
        for region in REGIONS:
            row[f"{region}_difference"] = item["regions"][region]["difference"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "comparison.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
