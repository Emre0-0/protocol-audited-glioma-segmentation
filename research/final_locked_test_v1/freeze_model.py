from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_revalidation(
    root: Path,
    config: dict[str, Any],
    model_hashes: dict[str, str],
) -> dict[str, Any]:
    settings = config["revalidation"]
    comparison_path = require_file(resolve(root, settings["comparison"]))
    build_path = require_file(resolve(root, settings["build_manifest"]))
    metadata_path = require_file(
        resolve(root, settings["official_metrics_metadata"])
    )
    summary_path = require_file(resolve(root, settings["official_metrics_summary"]))
    validation_config_path = require_file(
        resolve(root, settings["validation_config"])
    )

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    build = json.loads(build_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validation_config = json.loads(
        validation_config_path.read_text(encoding="utf-8")
    )
    selected = str(config["frozen_candidate"]["selected_variant"])
    expected_cases = int(config["frozen_candidate"]["expected_validation_cases"])

    if comparison.get("locked_test_used") is not False:
        raise RuntimeError("Revalidation comparison accessed the locked test")
    if comparison.get("ground_truth_used_during_candidate_build") is not False:
        raise RuntimeError("Ground truth was used while building the candidate")
    if comparison.get("decision") != settings["expected_decision"]:
        raise RuntimeError("Revalidated candidate did not receive a GO decision")
    if comparison.get("selected_variant") != selected:
        raise RuntimeError("Revalidation selected a different candidate")
    if comparison.get("selection_gate") != validation_config.get("selection_gate"):
        raise RuntimeError("Revalidation gate differs from the frozen gate")

    matches = [
        item
        for item in comparison.get("comparisons", [])
        if item.get("variant") == selected
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one selected revalidation comparison")
    result = matches[0]
    checks = result.get("checks", {})
    if result.get("passes_gate") is not True or not checks or not all(checks.values()):
        raise RuntimeError("Revalidated candidate failed one or more frozen checks")
    bootstrap = result.get("paired_bootstrap", {})
    macro_ci = bootstrap.get("macro_mean_delta_95ci", [])
    et_ci = bootstrap.get("et_mean_delta_95ci", [])
    if len(macro_ci) != 2 or float(macro_ci[0]) <= 0:
        raise RuntimeError("Revalidated macro gain is not positive at 95% CI")
    if len(et_ci) != 2 or float(et_ci[0]) <= 0:
        raise RuntimeError("Revalidated ET gain is not positive at 95% CI")

    if build.get("locked_test_used") is not False:
        raise RuntimeError("Revalidation build accessed the locked test")
    if build.get("ground_truth_used_during_build") is not False:
        raise RuntimeError("Revalidation build used ground truth")
    if build.get("sources", {}).get("baseline_checkpoint_sha256") != model_hashes[
        "baseline"
    ]:
        raise RuntimeError("Revalidation baseline checkpoint hash changed")
    if build.get("sources", {}).get("specialist_checkpoint_sha256") != model_hashes[
        "specialist"
    ]:
        raise RuntimeError("Revalidation specialist checkpoint hash changed")
    variant_build = build.get("variants", {}).get(selected)
    if not variant_build or int(variant_build.get("cases", -1)) != expected_cases:
        raise RuntimeError("Revalidation build case count is invalid")
    if variant_build.get("rc_topology_invariant_all_cases") is not True:
        raise RuntimeError("RC topology invariant failed during revalidation")

    if int(metadata.get("requested_cases", -1)) != expected_cases:
        raise RuntimeError("Official metric requested-case count is invalid")
    if int(metadata.get("successful_cases", -1)) != expected_cases:
        raise RuntimeError("Official metric validation is incomplete")
    if int(metadata.get("failed_cases", -1)) != 0:
        raise RuntimeError("Official metric validation contains failures")
    if metadata.get("metric_repository_commit") != settings[
        "expected_metric_repository_commit"
    ]:
        raise RuntimeError("Official metric repository commit changed")

    return {
        "status": "PASS_REVALIDATED",
        "mode": "independent_official_validation",
        "selected_variant": selected,
        "validation_cases": expected_cases,
        "candidate_mean": float(result["candidate_mean"]),
        "mean_gain": float(result["mean_gain"]),
        "regions": result["regions"],
        "non_degraded_region_count": int(result["non_degraded_region_count"]),
        "checks": checks,
        "paired_bootstrap": bootstrap,
        "artifacts": {
            "comparison": {"path": str(comparison_path), "sha256": sha256(comparison_path)},
            "build_manifest": {"path": str(build_path), "sha256": sha256(build_path)},
            "official_metrics_metadata": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
            "official_metrics_summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
            "validation_config": {"path": str(validation_config_path), "sha256": sha256(validation_config_path)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the final V2 test model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    args = parser.parse_args()

    root = project_root()
    config_path = args.config.resolve()
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_dir = resolve(root, config["paths"]["experiment_dir"])
    experiment_dir.mkdir(parents=True, exist_ok=True)
    lock_path = experiment_dir / "frozen_model_manifest.json"
    if (experiment_dir / "test_label_release_manifest.json").exists():
        raise RuntimeError("Test labels have already been released; model cannot change")

    audit_path = resolve(root, config["frozen_candidate"]["source_final_audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("locked_test_used") is not False:
        raise RuntimeError("Frozen V2 source audit is not valid")

    source_config = root / "research" / "grouped_et_topology_guard_v2" / (
        "config.json"
    )
    if sha256(source_config) != audit["config_sha256"]:
        raise RuntimeError("The source V2 configuration changed after its audit")

    models: dict[str, Any] = {}
    for name in ("baseline", "specialist"):
        checkpoint = resolve(root, config["models"][name]["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        models[name] = {
            "checkpoint": str(checkpoint),
            "sha256": sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "fold": int(config["models"][name]["fold"]),
            "checkpoint_name": config["models"][name]["checkpoint_name"],
        }
    if models["baseline"]["sha256"] != config["models"]["baseline"][
        "expected_sha256"
    ]:
        raise RuntimeError("Baseline checkpoint hash differs from frozen V2")

    verification_path = require_file(
        experiment_dir / "specialist_reproduction_verification.json"
    )
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if models["specialist"]["sha256"] != verification.get(
        "recreated_specialist_checkpoint_sha256"
    ):
        raise RuntimeError("Specialist checkpoint changed after verification")

    exact_reproduction = verification.get("status") == "PASS" and verification.get(
        "voxel_exact_reproduction"
    ) is True
    if exact_reproduction:
        selected = str(audit["selected_variant"])
        validation_acceptance = {
            "status": "PASS_EXACT",
            "mode": "voxel_exact_reproduction",
            "selected_variant": selected,
            "validation_cases": int(audit["validation_cases"]),
            "candidate_mean": float(audit["six_region_mean"]),
            "mean_gain": float(audit["six_region_mean_gain"]),
        }
    else:
        validation_acceptance = validate_revalidation(
            root,
            config,
            {
                "baseline": models["baseline"]["sha256"],
                "specialist": models["specialist"]["sha256"],
            },
        )
        selected = str(validation_acceptance["selected_variant"])

    tracked_code = [
        Path(__file__),
        Path(__file__).with_name("prepare_test_images.py"),
        Path(__file__).with_name("predict_model.py"),
        Path(__file__).with_name("apply_frozen_v2.py"),
        Path(__file__).with_name("build_reproduced_validation.py"),
        Path(__file__).with_name("reproduced_validation_config.json"),
        root / "research" / "grouped_et_topology_guard_v2" / (
            "compare_candidates.py"
        ),
        root / "research" / "grouped_et_topology_guard_v2" / "topology.py",
        root / "research" / "grouped_et_topology_guard_v2" / (
            "build_topology_guard.py"
        ),
    ]
    code_hashes = {str(path.resolve()): sha256(path.resolve()) for path in tracked_code}
    manifest = {
        "status": "FROZEN",
        "model_version": "grouped_et_topology_guard_v2_revalidated",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_completed_before_test": True,
        "test_labels_accessed": False,
        "post_test_model_selection_allowed": False,
        "selected_variant": selected,
        "fallback": config["frozen_candidate"]["fallback"],
        "validation_cases": int(validation_acceptance["validation_cases"]),
        "validation_six_region_mean": float(validation_acceptance["candidate_mean"]),
        "validation_six_region_mean_gain": float(validation_acceptance["mean_gain"]),
        "models": models,
        "specialist_reproduction": verification,
        "validation_acceptance": validation_acceptance,
        "source_v2_audit": {
            "path": str(audit_path),
            "sha256": sha256(audit_path),
            "config_sha256": audit["config_sha256"],
        },
        "protocol": {
            "locked_test_cases": int(config["protocol"]["expected_test_cases"]),
            "locked_test_subjects": int(
                config["protocol"]["expected_test_subjects"]
            ),
            "locked_test_csv_sha256": sha256(
                resolve(root, config["protocol"]["locked_test_cases"])
            ),
            "split_sha256": sha256(
                resolve(root, config["protocol"]["split_file"])
            ),
            "policy": config["protocol"]["test_policy"],
        },
        "inference": config["inference"],
        "config_sha256": sha256(config_path),
        "code_sha256": code_hashes,
    }

    if lock_path.exists():
        locked = json.loads(lock_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "model_version",
            "selected_variant",
            "fallback",
            "models",
            "validation_acceptance",
            "protocol",
            "inference",
            "config_sha256",
            "code_sha256",
        )
        if any(locked.get(key) != manifest.get(key) for key in comparable_keys):
            raise RuntimeError("Existing frozen model manifest differs; refusing overwrite")
        print(json.dumps(locked, indent=2))
        return

    lock_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
