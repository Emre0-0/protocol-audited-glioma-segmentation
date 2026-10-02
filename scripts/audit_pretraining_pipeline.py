from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_mri_project.data import (  # noqa: E402
    load_pretrain_npz,
    make_masked_input,
    pretrain_subject_id,
    random_crop,
)
from brain_mri_project.models import (  # noqa: E402
    UNet3D,
    load_encoder_state_with_channel_adaptation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit spatial consistency, masking weights, and encoder transfer for normal-T1 pretraining."
    )
    parser.add_argument("--pretrain-root", default="processed/pretrain_t1_npz")
    parser.add_argument("--brats-root", default="processed/brats2024_npz_full")
    parser.add_argument(
        "--encoder-checkpoint",
        default="experiments/normal_t1_masked_pretraining_sota_5070ti/best_encoder.pt",
    )
    parser.add_argument("--output-dir", default="experiments/pretraining_pipeline_audit_20260716")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--base-channels", type=int, default=48)
    return parser.parse_args()


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def load_image(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        return data["image"].astype(np.float32)


def save_midplane_montage(pretrain_root: Path, brats_root: Path, output_path: Path) -> None:
    rows: list[tuple[str, np.ndarray]] = []
    for dataset in ("ADNI", "OASIS1", "OASIS2"):
        path = next(iter(sorted((pretrain_root / dataset).glob("*.npz"))))
        rows.append((dataset, load_image(path)))

    brats_path = next(iter(sorted((brats_root / "train").glob("*.npz"))))
    brats_image = load_image(brats_path)
    rows.append(("BraTS T1n", brats_image[0]))

    figure, axes = plt.subplots(len(rows), 3, figsize=(10, 12), constrained_layout=True)
    for row_index, (name, image) in enumerate(rows):
        views = (
            image[image.shape[0] // 2, :, :],
            image[:, image.shape[1] // 2, :],
            image[:, :, image.shape[2] // 2],
        )
        for axis_index, view in enumerate(views):
            nonzero = view[view != 0]
            if nonzero.size:
                lower, upper = np.percentile(nonzero, (1, 99))
            else:
                lower, upper = float(view.min()), float(view.max())
            axes[row_index, axis_index].imshow(
                np.rot90(view), cmap="gray", vmin=lower, vmax=upper
            )
            axes[row_index, axis_index].axis("off")
            axes[row_index, axis_index].set_title(
                f"{name}: array axis {axis_index} | {tuple(image.shape)}", fontsize=9
            )
    figure.suptitle("Array-axis consistency audit for pretraining and BraTS T1n", fontsize=13)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def audit_masking(
    files: list[Path],
    samples: int,
    seed: int,
    patch_size: tuple[int, int, int],
    mask_ratio: float,
    noise_std: float,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(files), size=min(samples, len(files)), replace=False)
    selected = [files[int(index)] for index in selected_indices]

    rows: list[dict[str, float | str]] = []
    for path in selected:
        image, dataset, _ = load_pretrain_npz(path)
        image = image[None, ...]
        crop, _ = random_crop(image, None, patch_size)
        _, mask = make_masked_input(crop, mask_ratio=mask_ratio, noise_std=noise_std)
        brain = crop != 0
        masked = mask > 0
        weighted_mask = np.clip(mask + 0.1 * brain.astype(np.float32), 0.0, 1.0)
        rows.append(
            {
                "dataset": dataset,
                "brain_fraction": float(brain.mean()),
                "actual_mask_fraction": float(masked.mean()),
                "masked_voxels": float(masked.sum()),
                "masked_background_voxels": float((masked & ~brain).sum()),
                "masked_brain_voxels": float((masked & brain).sum()),
                "weighted_total": float(weighted_mask.sum()),
                "weighted_masked_background": float(weighted_mask[masked & ~brain].sum()),
                "weighted_masked_brain": float(weighted_mask[masked & brain].sum()),
                "weighted_unmasked_brain": float(weighted_mask[~masked & brain].sum()),
            }
        )

    total_weight = sum(float(row["weighted_total"]) for row in rows)
    total_masked = sum(float(row["masked_voxels"]) for row in rows)
    return {
        "sample_count": len(rows),
        "dataset_counts": dict(Counter(str(row["dataset"]) for row in rows)),
        "brain_fraction": quantiles([float(row["brain_fraction"]) for row in rows]),
        "requested_mask_ratio": mask_ratio,
        "actual_mask_fraction": quantiles(
            [float(row["actual_mask_fraction"]) for row in rows]
        ),
        "masked_region_content_share": {
            "background": sum(float(row["masked_background_voxels"]) for row in rows)
            / total_masked,
            "brain": sum(float(row["masked_brain_voxels"]) for row in rows) / total_masked,
        },
        "reconstruction_loss_weight_share": {
            "masked_background": sum(
                float(row["weighted_masked_background"]) for row in rows
            )
            / total_weight,
            "masked_brain": sum(float(row["weighted_masked_brain"]) for row in rows)
            / total_weight,
            "unmasked_brain_at_0p1": sum(
                float(row["weighted_unmasked_brain"]) for row in rows
            )
            / total_weight,
        },
    }


def audit_encoder(checkpoint_path: Path, base_channels: int) -> dict[str, object]:
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_state = (
        loaded["encoder_state"]
        if isinstance(loaded, dict) and "encoder_state" in loaded
        else loaded
    )
    model = UNet3D(
        in_channels=4,
        out_channels=5,
        base_channels=base_channels,
        dropout=0.15,
    )
    missing, unexpected = load_encoder_state_with_channel_adaptation(
        model, encoder_state, in_channels=4
    )
    source_weight = encoder_state["enc1.block.0.weight"]
    loaded_weight = model.encoder.state_dict()["enc1.block.0.weight"]
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "encoder_tensor_count": len(encoder_state),
        "source_first_conv_shape": list(source_weight.shape),
        "adapted_first_conv_shape": list(loaded_weight.shape),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "all_encoder_keys_loaded": not missing and not unexpected,
        "adaptation": "The single-channel filter is repeated across four modalities and divided by four.",
        "channel_sum_matches_source_max_abs_error": float(
            (loaded_weight.sum(dim=1, keepdim=True) - source_weight).abs().max()
        ),
    }


def main() -> None:
    args = parse_args()
    pretrain_root = Path(args.pretrain_root).resolve()
    brats_root = Path(args.brats_root).resolve()
    checkpoint_path = Path(args.encoder_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pretrain_root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No pretraining NPZ files found under {pretrain_root}")

    dataset_files = Counter(path.parent.name for path in files)
    subject_files = Counter(pretrain_subject_id(path) for path in files)
    dataset_subjects: dict[str, set[str]] = {}
    shape_counts: dict[str, Counter[str]] = {}
    for path in files:
        dataset_subjects.setdefault(path.parent.name, set()).add(pretrain_subject_id(path))
        shape_counts.setdefault(path.parent.name, Counter())[str(tuple(load_image(path).shape))] += 1

    montage_path = output_dir / "representative_array_midplanes.png"
    save_midplane_montage(pretrain_root, brats_root, montage_path)
    masking = audit_masking(
        files=files,
        samples=args.samples,
        seed=args.seed,
        patch_size=tuple(int(value) for value in args.patch_size),
        mask_ratio=args.mask_ratio,
        noise_std=args.noise_std,
    )
    encoder = audit_encoder(checkpoint_path, args.base_channels)

    report = {
        "status": "CRITICAL_PRETRAINING_PROTOCOL_WEAKNESSES",
        "pretraining_files": len(files),
        "unique_subjects": len(subject_files),
        "dataset_file_counts": dict(dataset_files),
        "dataset_subject_counts": {
            dataset: len(subjects) for dataset, subjects in dataset_subjects.items()
        },
        "multi_visit_subjects": sum(count > 1 for count in subject_files.values()),
        "max_files_per_subject": max(subject_files.values()),
        "shape_counts": {
            dataset: dict(counts) for dataset, counts in shape_counts.items()
        },
        "spatial_metadata": {
            "affine_present_in_npz": False,
            "canonical_orientation_recorded": False,
            "visual_evidence": str(montage_path),
            "observation": (
                "Representative ADNI and BraTS arrays use sagittal/coronal/axial array axes, "
                "whereas representative OASIS arrays use a different axis permutation."
            ),
        },
        "masking_objective": masking,
        "encoder_transfer": encoder,
        "interpretation": [
            "The old low validation reconstruction loss is not patient-independent because the legacy split leaked subjects.",
            "Most reconstruction-loss weight falls on masked background rather than masked brain tissue.",
            "OASIS dominates the file count and is not spatially harmonized to the BraTS array orientation.",
            "The encoder checkpoint loads without missing tensors; checkpoint loading itself is not the primary bug.",
            "Equal repetition of the T1 filter across four MRI modalities is a weak transfer adapter and discards modality-specific initialization.",
        ],
    }

    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    loss_share = masking["reconstruction_loss_weight_share"]
    actual_mask = masking["actual_mask_fraction"]
    lines = [
        "# Normal-T1 pretraining pipeline audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Files/subjects: {len(files)}/{len(subject_files)}",
        f"- Dataset files: {dict(dataset_files)}",
        f"- Dataset subjects: {report['dataset_subject_counts']}",
        f"- Multi-visit subjects: {report['multi_visit_subjects']}",
        "",
        "## Masking objective",
        f"- Requested mask ratio: {args.mask_ratio:.3f}",
        f"- Observed mask fraction: {actual_mask['mean']:.3f}",
        f"- Masked-background share of loss weight: {loss_share['masked_background']:.3f}",
        f"- Masked-brain share of loss weight: {loss_share['masked_brain']:.3f}",
        f"- Unmasked-brain share of loss weight: {loss_share['unmasked_brain_at_0p1']:.3f}",
        "",
        "## Spatial consistency",
        "- NPZ files do not contain affine/orientation metadata.",
        "- Representative midplanes show an OASIS array-axis permutation relative to ADNI and BraTS.",
        f"- Evidence image: `{montage_path.name}`",
        "",
        "## Encoder loading",
        f"- All encoder tensors loaded: {encoder['all_encoder_keys_loaded']}",
        f"- Missing/unexpected: {encoder['missing_keys']}/{encoder['unexpected_keys']}",
        f"- First convolution: {encoder['source_first_conv_shape']} -> {encoder['adapted_first_conv_shape']}",
        "- The T1 filter is copied equally into all four modality channels and scaled by 1/4.",
        "",
        "## Conclusion",
        "The checkpoint file is technically loadable, but the legacy pretraining evidence is weakened by subject leakage, spatial inconsistency, background-dominated reconstruction loss, and a weak multi-modal transfer adapter.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
