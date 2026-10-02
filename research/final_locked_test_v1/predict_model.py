from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from nnunetv2.inference import predict_from_raw_data as predict_module
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


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


def install_custom_trainer_resolver(root: Path) -> None:
    trainer_dir = root / "research" / "grouped_general_sota_v1"
    sys.path.insert(0, str(trainer_dir))
    from trainer import nnUNetTrainer_BraTSRegionAware120

    original = predict_module.recursive_find_python_class

    def resolver(folder: str, class_name: str, current_module: str):
        if class_name == "nnUNetTrainer_BraTSRegionAware120":
            return nnUNetTrainer_BraTSRegionAware120
        return original(folder, class_name, current_module)

    predict_module.recursive_find_python_class = resolver


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen nnU-Net model.")
    parser.add_argument("model", choices=("baseline", "specialist"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    args = parser.parse_args()

    root = project_root()
    config: dict[str, Any] = json.loads(
        args.config.resolve().read_text(encoding="utf-8")
    )
    lock_path = resolve(root, config["paths"]["experiment_dir"]) / (
        "frozen_model_manifest.json"
    )
    if not lock_path.is_file():
        raise FileNotFoundError(
            "Frozen model manifest is required before locked-test inference"
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN":
        raise RuntimeError("Frozen model manifest is not in FROZEN state")
    experiment_dir = resolve(root, config["paths"]["experiment_dir"])
    if (experiment_dir / "inference_lock.json").exists():
        raise RuntimeError("Predictions are already frozen and cannot be rerun")
    if (experiment_dir / "test_label_release_manifest.json").exists():
        raise RuntimeError("Test labels are released; inference is immutable")

    model = config["models"][args.model]
    checkpoint = resolve(root, model["checkpoint"])
    model_folder = resolve(root, model["model_folder"])
    input_dir = resolve(root, config["paths"]["input_images"])
    output_key = f"{args.model}_predictions"
    output_dir = resolve(root, config["paths"][output_key])
    output_dir.mkdir(parents=True, exist_ok=True)

    current_hash = sha256(checkpoint)
    locked_hash = lock["models"][args.model]["sha256"]
    if current_hash != locked_hash:
        raise RuntimeError(f"{args.model} checkpoint changed after model lock")

    expected = int(config["protocol"]["expected_test_cases"])
    if len(list(input_dir.glob("BraTS-GLI-*_0000.nii.gz"))) != expected:
        raise RuntimeError("Locked-test input staging is incomplete")

    install_custom_trainer_resolver(root)
    inference = config["inference"]
    predictor = nnUNetPredictor(
        tile_step_size=float(inference["tile_step_size"]),
        use_gaussian=bool(inference["use_gaussian"]),
        use_mirroring=bool(inference["use_mirroring"]),
        perform_everything_on_device=bool(
            inference["perform_everything_on_device"]
        ),
        device=torch.device("cuda"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(int(model["fold"]),),
        checkpoint_name=str(model["checkpoint_name"]),
    )
    predictor.predict_from_files(
        str(input_dir),
        str(output_dir),
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=int(
            inference["num_processes_preprocessing"]
        ),
        num_processes_segmentation_export=int(
            inference["num_processes_export"]
        ),
    )

    predictions = sorted(output_dir.glob("BraTS-GLI-*.nii.gz"))
    if len(predictions) != expected:
        raise AssertionError(
            f"Expected {expected} {args.model} predictions, found {len(predictions)}"
        )
    manifest = {
        "model": args.model,
        "status": "complete",
        "cases": expected,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": current_hash,
        "fold": int(model["fold"]),
        "checkpoint_name": model["checkpoint_name"],
        "tile_step_size": float(inference["tile_step_size"]),
        "mirroring_tta": bool(inference["use_mirroring"]),
        "save_probabilities": False,
        "test_labels_used": False,
        "output_dir": str(output_dir),
    }
    (output_dir / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
