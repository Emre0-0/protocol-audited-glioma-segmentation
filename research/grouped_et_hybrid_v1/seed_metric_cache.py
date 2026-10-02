from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    variant = manifest["variants"][args.variant]
    changed = set(variant["changed_case_ids"])
    source_files = {
        path.stem: path for path in args.baseline_cache.glob("*.json")
    }
    expected_count = int(manifest["protocol"]["validation_cases"])
    if len(source_files) != expected_count:
        raise ValueError(
            f"Expected {expected_count} baseline cache files, "
            f"found {len(source_files)}"
        )

    output_cache = args.output_dir / "case_cache"
    output_cache.mkdir(parents=True, exist_ok=True)
    seeded: list[str] = []
    cleared: list[str] = []
    for case_id, source in sorted(source_files.items()):
        destination = output_cache / source.name
        if case_id in changed:
            if destination.exists():
                destination.unlink()
                cleared.append(case_id)
            continue
        shutil.copy2(source, destination)
        seeded.append(case_id)

    payload = {
        "variant": args.variant,
        "baseline_cache": str(args.baseline_cache.resolve()),
        "seeded_unchanged_cases": len(seeded),
        "cleared_changed_case_caches": len(cleared),
        "changed_cases_to_evaluate": len(changed),
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
