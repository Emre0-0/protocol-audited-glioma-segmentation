from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_mri_project.config import get_device, load_config, resolve_path, set_seed
from brain_mri_project.data import (
    as_tuple3,
    list_pretrain_files,
    load_pretrain_npz,
    make_masked_input,
    pad_image_and_seg,
    split_pretrain_files,
)
from brain_mri_project.models import UNet3D
from brain_mri_project.training import configure_torch_runtime


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Metrics file is empty: {path}")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                "epoch": int(float(row["epoch"])),
                "elapsed_sec": float(row.get("elapsed_sec", 0.0)),
                "lr": float(row.get("lr", 0.0)),
                "train_reconstruction_loss": float(row["train_reconstruction_loss"]),
                "val_reconstruction_loss": float(row["val_reconstruction_loss"]),
                "reconstruction_generalization_gap": float(row["reconstruction_generalization_gap"]),
            }
        )
    return parsed


def plot_loss_curves(metrics: list[dict[str, Any]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in metrics]
    train_loss = [row["train_reconstruction_loss"] for row in metrics]
    val_loss = [row["val_reconstruction_loss"] for row in metrics]
    gap = [row["reconstruction_generalization_gap"] for row in metrics]

    best_idx = int(np.argmin(val_loss))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="validation")
    axes[0].scatter([epochs[best_idx]], [val_loss[best_idx]], color="tab:red", zorder=3, label="best val")
    axes[0].set_title("Masked reconstruction loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, gap, color="tab:purple")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Validation - train gap")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss gap")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def center_crop(image: np.ndarray, patch_size: tuple[int, int, int]) -> np.ndarray:
    image, _ = pad_image_and_seg(image, None, patch_size)
    starts = [(size - patch) // 2 for size, patch in zip(image.shape[-3:], patch_size)]
    sx, sy, sz = starts
    px, py, pz = patch_size
    return image[..., sx : sx + px, sy : sy + py, sz : sz + pz]


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    brain = values[np.abs(values) > 1e-6]
    sample = brain if brain.size else values.reshape(-1)
    lo = float(np.percentile(sample, 1))
    hi = float(np.percentile(sample, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def safe_case_name(case_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in case_id)[:80]


def select_evenly(items: list[Path], count: int) -> list[Path]:
    if count >= len(items):
        return list(items)
    positions = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[int(pos)] for pos in positions]


def write_sample_figure(
    target: np.ndarray,
    masked: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    target_3d = target[0]
    masked_3d = masked[0]
    pred_3d = pred[0]
    mask_3d = mask[0]
    error_3d = np.abs(pred_3d - target_3d)

    brain_voxels = np.argwhere(np.abs(target_3d) > 1e-6)
    if len(brain_voxels) > 0:
        center = np.median(brain_voxels, axis=0).astype(int)
    else:
        center = np.asarray(target_3d.shape) // 2
    axial = int(center[2])
    coronal = int(center[1])
    sagittal = int(center[0])

    views = [
        ("axial", lambda x: np.rot90(x[:, :, axial])),
        ("coronal", lambda x: np.rot90(x[:, coronal, :])),
        ("sagittal", lambda x: np.rot90(x[sagittal, :, :])),
    ]
    columns = [
        ("target", target_3d, "gray"),
        ("masked input", masked_3d, "gray"),
        ("reconstruction", pred_3d, "gray"),
        ("abs error", error_3d, "magma"),
        ("mask", mask_3d, "gray"),
    ]

    lo, hi = robust_limits(target_3d)
    err_hi = float(np.percentile(error_3d[np.isfinite(error_3d)], 99))
    err_hi = max(err_hi, 1e-6)

    fig, axes = plt.subplots(len(views), len(columns), figsize=(14, 8.5), dpi=150)
    fig.suptitle(title, fontsize=11)
    for row_idx, (view_name, slicer) in enumerate(views):
        for col_idx, (col_name, volume, cmap) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            data = slicer(volume)
            if col_name == "abs error":
                ax.imshow(data, cmap=cmap, vmin=0.0, vmax=err_hi)
            elif col_name == "mask":
                ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0)
            else:
                ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_name, fontsize=9)
            if col_idx == 0:
                ax.text(
                    0.02,
                    0.98,
                    view_name,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    color="white",
                    fontsize=8,
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 2, "edgecolor": "none"},
                )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_summary_markdown(summary: dict[str, Any], output_path: Path) -> None:
    model = summary["model"]
    pretraining = summary["pretraining"]
    data = summary["data"]
    best = summary["best"]
    latest = summary["latest"]
    lines = [
        "# Normal T1 Pretraining Report",
        "",
        "## Data",
        f"- Total normal T1 cases: {data['total_cases']}",
        f"- Train cases: {data['train_cases']}",
        f"- Validation cases: {data['val_cases']}",
        f"- Source counts: {json.dumps(data['source_counts'], sort_keys=True)}",
        f"- All train cases covered per epoch: {data['all_train_cases_seen_each_epoch']}",
        "",
        "## Model",
        f"- Architecture: {model['architecture']}",
        f"- Input channels: {model['in_channels']}",
        f"- Output channels: {model['out_channels']}",
        f"- Base channels: {model['base_channels']}",
        f"- Dropout: {model['dropout']}",
        "",
        "## Pretraining settings",
        f"- Patch size: {pretraining['patch_size']}",
        f"- Batch size: {pretraining['batch_size']}",
        f"- Epoch budget: {pretraining['epochs']}",
        f"- Patches per epoch: {pretraining['patches_per_epoch']}",
        f"- Mask ratio: {pretraining['mask_ratio']}",
        f"- Noise std: {pretraining['noise_std']}",
        f"- Learning rate: {pretraining['lr']}",
        f"- Weight decay: {pretraining['weight_decay']}",
        f"- Early stopping patience: {pretraining['early_stopping_patience']}",
        "",
        "## Results",
        f"- Best epoch: {best['epoch']}",
        f"- Best validation reconstruction loss: {best['val_reconstruction_loss']:.6f}",
        f"- Train reconstruction loss at best epoch: {best['train_reconstruction_loss']:.6f}",
        f"- Generalization gap at best epoch: {best['reconstruction_generalization_gap']:.6f}",
        f"- Latest epoch in metrics: {latest['epoch']}",
        f"- Latest validation reconstruction loss: {latest['val_reconstruction_loss']:.6f}",
        "",
        "## Files",
        f"- Loss curves: {summary['files']['loss_curves']}",
        f"- JSON summary: {summary['files']['summary_json']}",
        f"- Sample figures: {', '.join(summary['files']['sample_figures'])}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create normal T1 pretraining diagnostics.")
    parser.add_argument("--config", default="configs/sota_5070ti.json")
    parser.add_argument("--experiment-name", default="normal_t1_masked_pretraining_sota_5070ti")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(cfg["project_root"]).resolve()
    exp_root = resolve_path(project_root, cfg["experiments_root"])
    exp_dir = exp_root / args.experiment_name
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "pretraining_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["seed"]))
    metrics = read_metrics(exp_dir / "metrics.csv")
    best_metric = min(metrics, key=lambda row: row["val_reconstruction_loss"])
    latest_metric = metrics[-1]

    checkpoint_path = exp_dir / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    device = get_device(args.device)
    configure_torch_runtime(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_cfg = checkpoint.get("config", cfg)
    model_cfg = ckpt_cfg["model"]
    train_cfg = ckpt_cfg["pretraining"]
    patch_size = as_tuple3(train_cfg["patch_size"])

    model = UNet3D(
        in_channels=int(model_cfg["pretrain_in_channels"]),
        out_channels=1,
        base_channels=int(model_cfg["base_channels"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pretrain_root = resolve_path(project_root, cfg["pretrain_npz_root"])
    files = list_pretrain_files(pretrain_root)
    train_files, val_files = split_pretrain_files(files, val_fraction=0.15, seed=int(cfg["seed"]))
    source_counts: dict[str, int] = {}
    for path in files:
        source_counts[path.parent.name] = source_counts.get(path.parent.name, 0) + 1

    plot_loss_curves(metrics, output_dir / "loss_curves.png")

    sample_paths = select_evenly(val_files, max(1, int(args.num_samples)))
    sample_figures: list[str] = []
    for sample_idx, sample_path in enumerate(sample_paths, start=1):
        image, dataset, case_id = load_pretrain_npz(sample_path)
        target = center_crop(image[None, ...], patch_size)
        np.random.seed(int(cfg["seed"]) + sample_idx)
        masked, mask = make_masked_input(
            target,
            mask_ratio=float(train_cfg["mask_ratio"]),
            noise_std=float(train_cfg["noise_std"]),
        )
        input_tensor = torch.from_numpy(masked[None, ...].astype(np.float32)).to(device)
        with torch.no_grad():
            if bool(ckpt_cfg.get("amp", True)) and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    pred_tensor = model(input_tensor)
            else:
                pred_tensor = model(input_tensor)
        pred = pred_tensor.detach().cpu().numpy()[0]
        sample_file = output_dir / f"sample_{sample_idx:02d}_{safe_case_name(case_id)}.png"
        title = f"{dataset} | {case_id} | checkpoint={args.checkpoint}"
        write_sample_figure(target, masked, pred, mask, title, sample_file)
        sample_figures.append(str(sample_file))

    summary = {
        "experiment": args.experiment_name,
        "checkpoint": str(checkpoint_path),
        "data": {
            "pretrain_root": str(pretrain_root),
            "total_cases": len(files),
            "train_cases": len(train_files),
            "val_cases": len(val_files),
            "source_counts": source_counts,
            "all_train_cases_seen_each_epoch": int(train_cfg["patches_per_epoch"]) >= len(train_files),
        },
        "model": {
            "architecture": "3D UNet masked reconstruction",
            "in_channels": int(model_cfg["pretrain_in_channels"]),
            "out_channels": 1,
            "base_channels": int(model_cfg["base_channels"]),
            "dropout": float(model_cfg.get("dropout", 0.0)),
        },
        "pretraining": {
            "patch_size": list(patch_size),
            "batch_size": int(train_cfg["batch_size"]),
            "epochs": int(train_cfg["epochs"]),
            "patches_per_epoch": int(train_cfg["patches_per_epoch"]),
            "mask_ratio": float(train_cfg["mask_ratio"]),
            "noise_std": float(train_cfg["noise_std"]),
            "lr": float(train_cfg["lr"]),
            "weight_decay": float(train_cfg["weight_decay"]),
            "early_stopping_patience": int(train_cfg["early_stopping_patience"]),
        },
        "best": best_metric,
        "latest": latest_metric,
        "files": {
            "loss_curves": str(output_dir / "loss_curves.png"),
            "summary_json": str(output_dir / "summary.json"),
            "summary_md": str(output_dir / "summary.md"),
            "sample_figures": sample_figures,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_summary_markdown(summary, output_dir / "summary.md")
    print(json.dumps(summary["files"], indent=2))


if __name__ == "__main__":
    main()
