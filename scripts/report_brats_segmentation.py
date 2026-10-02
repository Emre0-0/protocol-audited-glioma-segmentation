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
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_mri_project.config import get_device, load_config, resolve_path
from brain_mri_project.data import as_tuple3, list_npz_files, load_brats_npz, load_residual_npz
from brain_mri_project.inference import sliding_window_logits
from brain_mri_project.models import UNet3D
from brain_mri_project.training import configure_torch_runtime


LABEL_GROUPS = {
    "label_1": [1],
    "label_2": [2],
    "label_3": [3],
    "label_4": [4],
    "netc_label_1": [1],
    "snfh_label_2": [2],
    "et_label_3": [3],
    "rc_label_4_official": [4],
    "tumor_core_labels_1_3": [1, 3],
    "whole_tumor_labels_1_2_3": [1, 2, 3],
    "all_abnormal_labels_1_2_3_4": [1, 2, 3, 4],
    # Legacy aliases retained so earlier comparison scripts still load.
    "whole_foreground_labels_1_2_3_4": [1, 2, 3, 4],
    "non_rc_labels_1_2_3": [1, 2, 3],
    "rc_label_4": [4],
}


def read_training_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {}
        for key, value in row.items():
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                out[key] = value
        if "epoch" in out:
            out["epoch"] = int(out["epoch"])
        parsed.append(out)
    return parsed


def plot_training_curves(rows: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    if not rows:
        return {}
    epochs = [row["epoch"] for row in rows]
    val_fg = [row["val_mean_foreground_dice"] for row in rows]
    best_idx = int(np.nanargmax(np.asarray(val_fg, dtype=np.float64)))
    best = rows[best_idx]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_mean_foreground_dice"] for row in rows], label="train")
    axes[1].plot(epochs, val_fg, label="validation")
    axes[1].scatter([best["epoch"]], [best["val_mean_foreground_dice"]], color="tab:red", label="best val")
    axes[1].set_title("Mean Foreground Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    for label in range(1, 5):
        key = f"val_dice_label_{label}"
        if key in rows[0]:
            axes[2].plot(epochs, [row[key] for row in rows], label=f"label {label}")
    axes[2].set_title("Validation Dice by Label")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return {
        "best_epoch": int(best["epoch"]),
        "best_val_mean_foreground_dice": float(best["val_mean_foreground_dice"]),
        "best_train_mean_foreground_dice": float(best["train_mean_foreground_dice"]),
        "best_generalization_gap_fg_dice": float(best.get("generalization_gap_fg_dice", float("nan"))),
    }


def binary_confusion(pred: np.ndarray, target: np.ndarray, labels: list[int]) -> dict[str, float]:
    pred_mask = np.isin(pred, labels)
    target_mask = np.isin(target, labels)
    tp = float(np.logical_and(pred_mask, target_mask).sum())
    fp = float(np.logical_and(pred_mask, ~target_mask).sum())
    fn = float(np.logical_and(~pred_mask, target_mask).sum())
    return {
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "pred_voxels": float(pred_mask.sum()),
        "target_voxels": float(target_mask.sum()),
        "pred_present": float(pred_mask.any()),
        "target_present": float(target_mask.any()),
    }


def binary_metrics(pred: np.ndarray, target: np.ndarray, labels: list[int]) -> tuple[float, float]:
    counts = binary_confusion(pred, target, labels)
    tp, fp, fn = counts["tp_voxels"], counts["fp_voxels"], counts["fn_voxels"]
    if tp + fp + fn == 0:
        return float("nan"), float("nan")
    dice = (2.0 * tp) / max(2.0 * tp + fp + fn, 1e-8)
    iou = tp / max(tp + fp + fn, 1e-8)
    return float(dice), float(iou)


def summarize_case_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {"case_count": float(len(rows))}
    metric_keys = sorted(k for k in rows[0] if k.endswith("_dice") or k.endswith("_iou")) if rows else []
    for key in metric_keys:
        values = np.asarray([row[key] for row in rows if np.isfinite(row[key])], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean()) if values.size else float("nan")
        summary[f"{key}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
    return summary


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    brain = finite[np.abs(finite) > 1e-6]
    sample = brain if brain.size else finite
    lo = float(np.percentile(sample, 1))
    hi = float(np.percentile(sample, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def mask_rgb(mask: np.ndarray) -> np.ndarray:
    colors = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.86, 0.15, 0.15],
            [0.15, 0.65, 0.20],
            [0.15, 0.35, 0.95],
            [0.95, 0.78, 0.15],
        ],
        dtype=np.float32,
    )
    clipped = np.clip(mask.astype(np.int64), 0, len(colors) - 1)
    return colors[clipped]


def error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    err = np.zeros_like(target, dtype=np.int64)
    target_fg = target > 0
    pred_fg = pred > 0
    err[np.logical_and(~pred_fg, target_fg)] = 1  # false negative
    err[np.logical_and(pred_fg, ~target_fg)] = 2  # false positive
    err[np.logical_and(pred_fg, target_fg) & (pred != target)] = 3  # wrong tumor label
    return err


def save_case_figure(image: np.ndarray, seg: np.ndarray, pred: np.ndarray, case_id: str, score: float, output_path: Path) -> None:
    union = np.logical_or(seg > 0, pred > 0)
    if union.any():
        z = int(np.argmax(union.sum(axis=(0, 1))))
    else:
        z = image.shape[-1] // 2
    channels = min(4, image.shape[0])
    fig, axes = plt.subplots(1, channels + 3, figsize=(3.2 * (channels + 3), 3.4), dpi=150)
    for channel in range(channels):
        ax = axes[channel]
        sl = np.rot90(image[channel, :, :, z])
        lo, hi = robust_limits(sl)
        ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"MR ch{channel}", fontsize=9)
        ax.axis("off")

    target_rgb = np.rot90(mask_rgb(seg[:, :, z]), axes=(0, 1))
    pred_rgb = np.rot90(mask_rgb(pred[:, :, z]), axes=(0, 1))
    err = np.rot90(error_map(pred[:, :, z], seg[:, :, z]))
    err_cmap = ListedColormap(["black", "#f2c94c", "#eb5757", "#9b51e0"])
    panels = [
        ("GT mask", target_rgb),
        ("Pred mask", pred_rgb),
        ("Error", err),
    ]
    for offset, (title, data) in enumerate(panels, start=channels):
        ax = axes[offset]
        if title == "Error":
            ax.imshow(data, cmap=err_cmap, vmin=0, vmax=3)
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{case_id} | mean FG Dice={score:.4f} | axial z={z}", fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_split(
    cfg: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    output_dir: Path,
    anomaly_aux: bool,
    max_cases: int | None,
    num_examples: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    project_root = Path(cfg["project_root"]).resolve()
    brats_root = resolve_path(project_root, cfg["brats_npz_root"])
    residual_root = resolve_path(project_root, cfg["anomaly_residual_root"])
    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluation"]
    patch_size = as_tuple3(eval_cfg["patch_size"])
    num_classes = int(model_cfg["num_classes"])
    in_channels = int(model_cfg["brats_in_channels"]) + (1 if anomaly_aux else 0)

    model = UNet3D(
        in_channels=in_channels,
        out_channels=num_classes,
        base_channels=int(model_cfg["base_channels"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    files = list_npz_files(brats_root, split)
    if max_cases is not None:
        files = files[:max_cases]
    rows: list[dict[str, Any]] = []
    predictions: list[tuple[float, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for path in tqdm(files, desc=f"report-{split}"):
        image, seg, case_id = load_brats_npz(path)
        model_input = image
        if anomaly_aux:
            residual_path = residual_root / split / f"{case_id}.npz"
            model_input = np.concatenate([image, load_residual_npz(residual_path)], axis=0)
        logits = sliding_window_logits(
            model,
            model_input,
            patch_size=patch_size,
            num_classes=num_classes,
            overlap=float(eval_cfg["overlap"]),
            device=device,
            amp=bool(cfg.get("amp", True)),
        )
        pred = torch.argmax(logits, dim=0).cpu().numpy().astype(np.int64)
        row: dict[str, Any] = {"case_id": case_id}
        for name, labels in LABEL_GROUPS.items():
            dice, iou = binary_metrics(pred, seg, labels)
            row[f"{name}_dice"] = dice
            row[f"{name}_iou"] = iou
            for count_name, value in binary_confusion(pred, seg, labels).items():
                row[f"{name}_{count_name}"] = value
        label_scores = [row[f"label_{label}_dice"] for label in range(1, 5) if np.isfinite(row[f"label_{label}_dice"])]
        row["mean_foreground_dice"] = float(np.mean(label_scores)) if label_scores else float("nan")
        rows.append(row)
        predictions.append((float(row["mean_foreground_dice"]), case_id, image, seg, pred))

    pd.DataFrame(rows).to_csv(output_dir / f"{split}_case_metrics.csv", index=False)
    summary = summarize_case_rows(rows)
    pd.DataFrame([summary]).to_csv(output_dir / f"{split}_summary.csv", index=False)

    valid_predictions = [item for item in predictions if np.isfinite(item[0])]
    valid_predictions.sort(key=lambda item: item[0])
    failures = valid_predictions[:num_examples]
    successes = list(reversed(valid_predictions[-num_examples:]))
    for group_name, group in [("failure_cases", failures), ("success_cases", successes)]:
        for idx, (score, case_id, image, seg, pred) in enumerate(group, start=1):
            save_case_figure(
                image,
                seg,
                pred,
                case_id,
                score,
                output_dir / group_name / f"{idx:02d}_{case_id}.png",
            )
    return rows, summary


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    train_best = summary.get("training_best", {})
    val = summary.get("validation_summary", {})
    lines = [
        "# BraTS Segmentation Validation Report",
        "",
        "## Training Curves",
        f"- Best epoch: {train_best.get('best_epoch', 'pending')}",
        f"- Best validation mean foreground Dice: {train_best.get('best_val_mean_foreground_dice', float('nan')):.6f}",
        f"- Train mean foreground Dice at best epoch: {train_best.get('best_train_mean_foreground_dice', float('nan')):.6f}",
        f"- Generalization gap at best epoch: {train_best.get('best_generalization_gap_fg_dice', float('nan')):.6f}",
        "",
        "## Validation Sliding-Window Metrics",
        f"- Case count: {val.get('case_count', float('nan')):.0f}",
        f"- Mean foreground Dice: {val.get('mean_foreground_dice_mean', float('nan')):.6f}",
        f"- WT labels 1/2/3 Dice: {val.get('whole_tumor_labels_1_2_3_dice_mean', float('nan')):.6f}",
        f"- TC labels 1/3 Dice: {val.get('tumor_core_labels_1_3_dice_mean', float('nan')):.6f}",
        f"- RC label 4 Dice: {val.get('rc_label_4_official_dice_mean', float('nan')):.6f}",
        f"- All-abnormal labels 1/2/3/4 Dice: {val.get('all_abnormal_labels_1_2_3_4_dice_mean', float('nan')):.6f}",
        f"- All-abnormal labels 1/2/3/4 IoU: {val.get('all_abnormal_labels_1_2_3_4_iou_mean', float('nan')):.6f}",
        "",
        "## Files",
        f"- Training curves: {summary['files']['training_curves']}",
        f"- Per-case validation metrics: {summary['files']['case_metrics']}",
        f"- Success examples: {summary['files']['success_examples_dir']}",
        f"- Failure examples: {summary['files']['failure_examples_dir']}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create validation report for a BraTS segmentation experiment.")
    parser.add_argument("--config", default="configs/sota_5070ti.json")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--anomaly-aux", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(cfg["project_root"]).resolve()
    exp_root = resolve_path(project_root, cfg["experiments_root"])
    exp_dir = exp_root / args.experiment_name
    checkpoint_path = exp_dir / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / f"{args.split}_segmentation_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    configure_torch_runtime(cfg, device)
    training_best = plot_training_curves(read_training_metrics(exp_dir / "metrics.csv"), output_dir / "training_curves.png")
    _, validation_summary = evaluate_split(
        cfg,
        checkpoint_path,
        args.split,
        output_dir,
        anomaly_aux=args.anomaly_aux,
        max_cases=args.max_cases,
        num_examples=max(1, int(args.num_examples)),
        device=device,
    )
    summary = {
        "experiment": args.experiment_name,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "training_best": training_best,
        "validation_summary": validation_summary,
        "files": {
            "training_curves": str(output_dir / "training_curves.png"),
            "case_metrics": str(output_dir / f"{args.split}_case_metrics.csv"),
            "split_summary": str(output_dir / f"{args.split}_summary.csv"),
            "success_examples_dir": str(output_dir / "success_cases"),
            "failure_examples_dir": str(output_dir / "failure_cases"),
            "summary_json": str(output_dir / "summary.json"),
            "summary_md": str(output_dir / "summary.md"),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_markdown(summary, output_dir / "summary.md")
    print(json.dumps(summary["validation_summary"], indent=2))


if __name__ == "__main__":
    main()
