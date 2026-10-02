from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("nnUNet_raw", str(_DEFAULT_ROOT / "processed/nnunet_raw"))
os.environ.setdefault(
    "nnUNet_preprocessed", str(_DEFAULT_ROOT / "processed/nnunet_preprocessed")
)
os.environ.setdefault("nnUNet_results", str(_DEFAULT_ROOT / "experiments/nnunet_results"))

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_250epochs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the exact publication training/inference recipe."
    )
    parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/publication_integrity_v1/training_recipe"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def source_record(obj: Any) -> dict[str, str]:
    path = Path(inspect.getsourcefile(obj) or "").resolve()
    return {"path": str(path), "sha256": sha256(path)}


def render_markdown(recipe: dict[str, Any]) -> str:
    plan = recipe["plan"]
    architecture = plan["architecture"]
    augmentation = recipe["augmentation"]
    lines = [
        "# Publication training and inference recipe",
        "",
        f"- Generated: {recipe['generated_at_utc']}",
        f"- nnU-Net: `{recipe['software']['nnunetv2']}`",
        f"- Dataset: `{recipe['dataset']['name']}` ({recipe['dataset']['num_training']} development scans)",
        f"- Input channels: `{recipe['dataset']['channels']}`",
        f"- Labels: `{recipe['dataset']['labels']}`",
        f"- Configuration: `{recipe['configuration']}`",
        f"- Patch/batch: `{plan['patch_size']}`, batch size `{plan['batch_size']}`",
        f"- Spacing: `{plan['spacing']}` mm",
        f"- Architecture: `{architecture['network_class_name']}`",
        f"- Features/stage: `{architecture['features_per_stage']}`",
        f"- Epochs: `{recipe['training']['epochs']}`; iterations/epoch: `{recipe['training']['iterations_per_epoch']}`",
        "",
        "## Optimization",
        "",
        f"- Optimizer: {recipe['training']['optimizer']}.",
        f"- Initial learning rate: {recipe['training']['initial_lr']}; scheduler: {recipe['training']['scheduler']}.",
        f"- Loss: {recipe['training']['loss']}.",
        f"- Foreground oversampling: {recipe['training']['oversample_foreground_percent']:.0%}.",
        f"- Deep supervision: {recipe['training']['deep_supervision']}.",
        "",
        "## Augmentation",
        "",
    ]
    for item in augmentation["transforms"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Full-volume inference",
            "",
            f"- Sliding-window step: {recipe['inference']['tile_step_size']}.",
            f"- Gaussian weighting: {recipe['inference']['use_gaussian']}.",
            f"- Mirroring/TTA axes: {recipe['inference']['mirroring_axes']}.",
            f"- Checkpoint: {recipe['inference']['checkpoint_selection']}.",
            "- Official reporting uses vendored BraTS 2024 lesion-wise metrics, not nnU-Net's patch pseudo-Dice.",
            "",
            "## Reproducibility boundary",
            "",
            recipe["reproducibility_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessed = root / "processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch"
    plans_path = preprocessed / "nnUNetResEncUNet10p5GBPlans.json"
    dataset_path = preprocessed / "dataset.json"
    plans = json.loads(plans_path.read_text(encoding="utf-8-sig"))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    config = plans["configurations"]["3d_fullres"]
    architecture = config["architecture"]
    arch_kwargs = architecture["arch_kwargs"]
    trainer_path = root / "research/publication_integrity_v1/../grouped_fivefold_cv_v11/trainer.py"
    trainer_path = trainer_path.resolve()

    recipe = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": "3d_fullres",
        "dataset": {
            "name": plans["dataset_name"],
            "num_training": dataset["numTraining"],
            "channels": dataset["channel_names"],
            "labels": dataset["labels"],
            "file_ending": dataset["file_ending"],
        },
        "plan": {
            "plans_name": plans["plans_name"],
            "batch_size": config["batch_size"],
            "patch_size": config["patch_size"],
            "spacing": config["spacing"],
            "median_image_size_in_voxels": config["median_image_size_in_voxels"],
            "normalization_schemes": config["normalization_schemes"],
            "use_mask_for_norm": config["use_mask_for_norm"],
            "batch_dice": config["batch_dice"],
            "architecture": {
                "network_class_name": architecture["network_class_name"],
                "n_stages": arch_kwargs["n_stages"],
                "features_per_stage": arch_kwargs["features_per_stage"],
                "kernel_sizes": arch_kwargs["kernel_sizes"],
                "strides": arch_kwargs["strides"],
                "n_blocks_per_stage": arch_kwargs["n_blocks_per_stage"],
                "n_conv_per_stage_decoder": arch_kwargs["n_conv_per_stage_decoder"],
                "normalization": arch_kwargs["norm_op"],
                "activation": arch_kwargs["nonlin"],
                "dropout": arch_kwargs["dropout_op"],
            },
        },
        "training": {
            "epochs": 250,
            "iterations_per_epoch": 250,
            "validation_iterations_per_epoch": 50,
            "batch_size": config["batch_size"],
            "optimizer": "SGD(momentum=0.99, Nesterov=True, weight_decay=3e-5)",
            "initial_lr": 0.01,
            "scheduler": "PolyLRScheduler over 250 epochs",
            "loss": (
                "equal-weight cross-entropy plus memory-efficient soft Dice "
                "(background excluded, smooth=1e-5), wrapped with normalized "
                "deep-supervision weights proportional to 1/2^i; lowest output weight 0"
            ),
            "oversample_foreground_percent": 0.33,
            "probabilistic_oversampling": False,
            "deep_supervision": True,
            "best_checkpoint_rule": "highest exponential-moving-average patch pseudo-Dice",
            "runtime_seeds": "20260726 + fold index",
            "fold0_trainer": "nnUNetTrainer_250epochs",
            "fold1_to_4_trainer": "nnUNetTrainer_GroupedCV250Duty80",
            "duty_wrapper_effect": (
                "sleep after each optimization step to target 80% average duty; "
                "no change to batches, gradients, optimizer, scheduler, or loss"
            ),
        },
        "augmentation": {
            "dummy_2d": False,
            "rotation_radians": [-0.5235987755982988, 0.5235987755982988],
            "mirror_axes": [0, 1, 2],
            "transforms": [
                "Spatial rotation +/-30 degrees (apply p=0.2) and scaling 0.7--1.4 (p=0.2; synchronized across axes); elastic deformation disabled.",
                "Gaussian noise variance 0--0.1 (p=0.1; synchronized channels).",
                "Gaussian blur sigma 0.5--1.0 (p=0.2; per-channel p=0.5).",
                "Multiplicative brightness 0.75--1.25 (p=0.15).",
                "Contrast 0.75--1.25 with range preservation (p=0.15).",
                "Simulated low resolution scale 0.5--1.0 (p=0.25; per-channel p=0.5).",
                "Inverted gamma 0.7--1.5 with retained statistics (p=0.1).",
                "Gamma 0.7--1.5 with retained statistics (p=0.3).",
                "Independent mirroring over all three spatial axes.",
                "Mask-based normalization for all four MRI channels; outside-mask values set to zero.",
            ],
        },
        "inference": {
            "checkpoint_selection": "checkpoint_best.pth per grouped fold",
            "tile_step_size": 0.5,
            "use_gaussian": True,
            "use_mirroring": True,
            "mirroring_axes": [0, 1, 2],
            "perform_everything_on_device": True,
            "save_probabilities": False,
        },
        "software": {
            "nnunetv2": importlib.metadata.version("nnunetv2"),
            "nnunet_trainer_source": source_record(nnUNetTrainer),
            "nnunet_250epoch_source": source_record(nnUNetTrainer_250epochs),
            "project_duty_trainer_source": {
                "path": str(trainer_path),
                "sha256": sha256(trainer_path),
            },
            "plans_source": {"path": str(plans_path), "sha256": sha256(plans_path)},
            "dataset_json_source": {
                "path": str(dataset_path),
                "sha256": sha256(dataset_path),
            },
        },
        "reproducibility_boundary": (
            "Python/NumPy/PyTorch seeds were set per fold, but cuDNN benchmarking and "
            "TF32 were enabled and deterministic algorithms were not enforced. The run "
            "is recipe-reproducible, not guaranteed bitwise deterministic."
        ),
    }
    (output_dir / "training_recipe.json").write_text(
        json.dumps(recipe, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_recipe.md").write_text(
        render_markdown(recipe),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
