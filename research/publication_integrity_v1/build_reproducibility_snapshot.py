from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze code, protocol, environment, and checkpoint provenance for publication."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/publication_integrity_v1/reproducibility_snapshot"),
    )
    parser.add_argument("--hash-checkpoints", action="store_true")
    parser.add_argument("--hash-large-data", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def file_record(root: Path, relative: str, hash_file: bool = True) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        return {"path": relative, "exists": False}
    stat = path.stat()
    return {
        "path": relative,
        "exists": True,
        "bytes": stat.st_size,
        "modified_at_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": sha256(path) if hash_file else None,
    }


def command_output(arguments: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        return {"available": False, "error": str(error)}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "nnunetv2",
        "dynamic-network-architectures",
        "batchgenerators",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "nibabel",
        "SimpleITK",
        "matplotlib",
    )
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def torch_environment() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        return {"available": False, "error": str(error)}
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    checks = payload["snapshot_completeness"]
    lines = [
        "# Reproducibility snapshot",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        f"- Python: {payload['runtime']['python_version']}",
        f"- Platform: {payload['runtime']['platform']}",
        f"- Git repository available: {payload['version_control']['available']}",
        f"- Key files present: {checks['key_files_present']}/{checks['key_files_expected']}",
        f"- Checkpoints present: {checks['checkpoints_present']}/{checks['checkpoints_expected']}",
        f"- Checkpoints hashed: {checks['checkpoints_hashed']}/{checks['checkpoints_present']}",
        f"- Source archive hashed: {checks['source_archive_hashed']}",
        "",
        "This manifest records provenance; it does not upgrade interim evidence to final evidence.",
        "The late-defined holdout remains closed and must not be reused.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    key_files = [
        "scripts/create_brats_subject_grouped_protocol.py",
        "scripts/prepare_nnunet_custom_split.py",
        "scripts/evaluate_brats2024_official_metrics.py",
        "processed/brats_subject_grouped_protocol_v1/case_assignments.csv",
        "processed/brats_subject_grouped_protocol_v1/development_cases.csv",
        "processed/brats_subject_grouped_protocol_v1/locked_test_cases.csv",
        "processed/brats_subject_grouped_protocol_v1/splits_final.json",
        "processed/brats_subject_grouped_protocol_v1/summary.json",
        "processed/nnunet_raw/Dataset503_BraTS2024Grouped4Ch/dataset.json",
        "processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json",
        "processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/nnUNetResEncUNet10p5GBPlans.json",
        "research/grouped_fivefold_cv_v11/trainer.py",
        "research/grouped_fivefold_cv_v11/run_fold.py",
        "research/grouped_fivefold_cv_v11/run_fold_stage.ps1",
        "research/publication_integrity_v1/audit_publication_integrity.py",
        "research/publication_integrity_v1/build_publication_statistics.py",
        "research/publication_integrity_v1/build_results_assets.py",
        "research/publication_integrity_v1/build_results_section.py",
        "research/publication_integrity_v1/inspect_checkpoint_metadata.py",
        "research/publication_integrity_v1/build_training_recipe.py",
        "research/publication_integrity_v1/audit_data_provenance.py",
        "research/publication_integrity_v1/build_submission_readiness.py",
        "research/publication_integrity_v1/test_integrity_tools.py",
        "research/publication_integrity_v1/METHODS_SOURCE_EVIDENCE.md",
        "research/vendor/BraTS-2024-Metrics/metrics_GLI.py",
        "research/final_locked_test_v1/config.json",
        "research/final_locked_test_v1/run_locked_test.ps1",
        "experiments/final_locked_test_v1/frozen_model_manifest.json",
        "experiments/final_locked_test_v1/test_summary.json",
        "experiments/final_locked_test_v1/LOCKED_TEST_CLOSED.txt",
        "experiments/publication_integrity_v1/audit.json",
        "experiments/publication_integrity_v1/statistics/publication_statistics.json",
        "experiments/publication_integrity_v1/checkpoint_metadata/checkpoint_metadata.json",
        "experiments/publication_integrity_v1/training_recipe/training_recipe.json",
        "experiments/publication_integrity_v1/data_provenance/data_provenance.json",
        "experiments/publication_integrity_v1/SCI_THESIS_READINESS.md",
        "paper_results_assets/results_section.tex",
        "paper_results_assets/publication_references.bib",
    ]
    checkpoint_files = [
        (
            "experiments/nnunet_results/Dataset503_BraTS2024Grouped4Ch/"
            "nnUNetTrainer_250epochs__nnUNetResEncUNet10p5GBPlans__3d_fullres/"
            "fold_0/checkpoint_best.pth"
        ),
        (
            "experiments/nnunet_results/Dataset503_BraTS2024Grouped4Ch/"
            "nnUNetTrainer_250epochs__nnUNetResEncUNet10p5GBPlans__3d_fullres/"
            "fold_0/checkpoint_final.pth"
        ),
    ]
    for fold in range(1, 5):
        for checkpoint in ("best", "final"):
            checkpoint_files.append(
                "experiments/nnunet_results/Dataset503_BraTS2024Grouped4Ch/"
                "nnUNetTrainer_GroupedCV250Duty80__nnUNetResEncUNet10p5GBPlans__3d_fullres/"
                f"fold_{fold}/checkpoint_{checkpoint}.pth"
            )
    checkpoint_files.append(
        "experiments/nnunet_results/Dataset503_BraTS2024Grouped4Ch/"
        "nnUNetTrainer_BraTSRegionAware120__nnUNetResEncUNet9GBPlans__3d_fullres/"
        "fold_0/checkpoint_best.pth"
    )

    key_records = [file_record(root, path) for path in key_files]
    checkpoint_records = [
        file_record(root, path, hash_file=args.hash_checkpoints)
        for path in checkpoint_files
    ]
    source_record = file_record(
        root,
        "BrainMRI_Project_Archives.zip",
        hash_file=args.hash_large_data,
    )
    pip_freeze = command_output([sys.executable, "-m", "pip", "freeze"])
    requirements = pip_freeze.get("stdout", "")
    (output_dir / "requirements_frozen.txt").write_text(
        requirements + ("\n" if requirements else ""),
        encoding="utf-8",
    )
    git = command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    nvidia = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": package_versions(),
            "torch": torch_environment(),
            "nvidia_smi": nvidia,
            "nnunet_environment": {
                key: os.environ.get(key)
                for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
            },
        },
        "version_control": {
            "available": bool(git.get("available")),
            "git_probe": git,
            "risk": (
                None
                if git.get("available")
                else "No valid Git commit is available; file hashes are the authoritative code snapshot."
            ),
        },
        "key_files": key_records,
        "checkpoints": checkpoint_records,
        "source_archive": source_record,
        "snapshot_completeness": {
            "key_files_expected": len(key_records),
            "key_files_present": sum(item["exists"] for item in key_records),
            "checkpoints_expected": len(checkpoint_records),
            "checkpoints_present": sum(item["exists"] for item in checkpoint_records),
            "checkpoints_hashed": sum(
                bool(item.get("sha256")) for item in checkpoint_records if item["exists"]
            ),
            "source_archive_hashed": bool(source_record.get("sha256")),
        },
        "claim_boundary": (
            "The grouped OOF analysis is primary. The late-defined retrospective holdout is secondary and permanently closed."
        ),
    }
    (output_dir / "snapshot.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "snapshot.md").write_text(
        render_markdown(payload),
        encoding="utf-8",
    )
    print(json.dumps(payload["snapshot_completeness"], indent=2))


if __name__ == "__main__":
    main()
