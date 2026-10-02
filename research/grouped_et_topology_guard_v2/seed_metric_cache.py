from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def cache_files(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in directory.glob("*.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = root / config["paths"]["experiment_dir"]
    v1_root = root / config["paths"]["v1_experiment_dir"]
    baseline_cache = (
        root
        / config["paths"]["baseline_official_metrics"]
        / "case_cache"
    )
    manifest = json.loads(
        (experiment / "build_manifest.json").read_text(encoding="utf-8")
    )
    stats = manifest["variants"][args.variant]
    v1_cache = (
        v1_root
        / "official_metrics"
        / stats["v1_variant"]
        / "case_cache"
    )
    baseline_files = cache_files(baseline_cache)
    v1_files = cache_files(v1_cache)
    expected = int(manifest["protocol"]["validation_cases"])
    if len(baseline_files) != expected or len(v1_files) != expected:
        raise ValueError("Source metric caches are incomplete")

    output_cache = args.output_dir / "case_cache"
    output_cache.mkdir(parents=True, exist_ok=True)
    for stale in output_cache.glob("*.json"):
        stale.unlink()

    rollback = set(stats["topology_rollback_case_ids"])
    matches_v1 = set(stats["matches_v1_case_ids"])
    seeded_v1: list[str] = []
    left_for_clean_evaluation: list[str] = []
    for case_id in sorted(baseline_files):
        if case_id in rollback:
            left_for_clean_evaluation.append(case_id)
        elif case_id in matches_v1:
            shutil.copy2(v1_files[case_id], output_cache / f"{case_id}.json")
            seeded_v1.append(case_id)
        else:
            raise AssertionError(f"No exact cache provenance for {case_id}")

    payload = {
        "variant": args.variant,
        "seeded_from_exact_v1_predictions": len(seeded_v1),
        "clean_cases_to_evaluate": len(left_for_clean_evaluation),
        "clean_case_ids": left_for_clean_evaluation,
        "rollback_cases_intentionally_not_seeded": True,
        "baseline_cache_available_but_not_used_for_rollback": str(
            baseline_cache.resolve()
        ),
        "metric_algorithm_unchanged": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cache_seed_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
