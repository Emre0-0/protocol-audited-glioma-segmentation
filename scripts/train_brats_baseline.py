from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_mri_project.config import get_device, load_config, resolve_path, set_seed
from brain_mri_project.data import BratsPatchDataset, as_tuple3, list_npz_files
from brain_mri_project.losses import DiceCELoss, dice_from_logits
from brain_mri_project.models import UNet3D
from brain_mri_project.training import (
    EarlyStopping,
    append_metrics_csv,
    configure_torch_runtime,
    dataloader_kwargs,
    prepare_experiment_dir,
    save_checkpoint,
)


def assert_split_integrity(train_files: list[Path], val_files: list[Path], test_files: list[Path]) -> None:
    train = {p.stem for p in train_files}
    val = {p.stem for p in val_files}
    test = {p.stem for p in test_files}
    if train & val or train & test or val & test:
        raise RuntimeError("Case leakage detected across train/val/test splits.")


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    loss_fn,
    device,
    use_amp: bool,
    num_classes: int,
    max_grad_norm: float,
    step_sleep_sec: float = 0.0,
    eval_modules: list[torch.nn.Module] | None = None,
) -> dict[str, float]:
    model.train()
    if eval_modules:
        for module in eval_modules:
            module.eval()
    totals = {"loss": 0.0, "dice_loss": 0.0, "ce_loss": 0.0}
    dice_rows = []
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        seg = batch["seg"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(image)
                loss, dice_loss, ce_loss = loss_fn(logits, seg)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(image)
            loss, dice_loss, ce_loss = loss_fn(logits, seg)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        totals["loss"] += float(loss.item())
        totals["dice_loss"] += float(dice_loss.item())
        totals["ce_loss"] += float(ce_loss.item())
        dice_rows.append(dice_from_logits(logits.detach(), seg, num_classes=num_classes))
        if step_sleep_sec > 0:
            time.sleep(step_sleep_sec)
    n = max(1, len(loader))
    metrics = {f"train_{k}": v / n for k, v in totals.items()}
    dices = np.vstack(dice_rows) if dice_rows else np.empty((0, num_classes))
    for cls in range(num_classes):
        values = dices[:, cls]
        values = values[~np.isnan(values)]
        metrics[f"train_dice_label_{cls}"] = float(values.mean()) if values.size else float("nan")
    fg = [metrics[f"train_dice_label_{cls}"] for cls in range(1, num_classes)]
    fg = [v for v in fg if not np.isnan(v)]
    metrics["train_mean_foreground_dice"] = float(np.mean(fg)) if fg else float("nan")
    return metrics


@torch.no_grad()
def validate(model, loader, loss_fn, device, num_classes: int, use_amp: bool, step_sleep_sec: float = 0.0) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "dice_loss": 0.0, "ce_loss": 0.0}
    dice_rows = []
    for batch in tqdm(loader, desc="val", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        seg = batch["seg"].to(device, non_blocking=True)
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(image)
                loss, dice_loss, ce_loss = loss_fn(logits, seg)
        else:
            logits = model(image)
            loss, dice_loss, ce_loss = loss_fn(logits, seg)
        totals["loss"] += float(loss.item())
        totals["dice_loss"] += float(dice_loss.item())
        totals["ce_loss"] += float(ce_loss.item())
        dice_rows.append(dice_from_logits(logits, seg, num_classes=num_classes))
        if step_sleep_sec > 0:
            time.sleep(step_sleep_sec)
    n = max(1, len(loader))
    metrics = {f"val_{k}": v / n for k, v in totals.items()}
    dices = np.vstack(dice_rows) if dice_rows else np.empty((0, num_classes))
    for cls in range(num_classes):
        values = dices[:, cls]
        values = values[~np.isnan(values)]
        metrics[f"val_dice_label_{cls}"] = float(values.mean()) if values.size else float("nan")
    fg = [metrics[f"val_dice_label_{cls}"] for cls in range(1, num_classes)]
    fg = [v for v in fg if not np.isnan(v)]
    metrics["val_mean_foreground_dice"] = float(np.mean(fg)) if fg else float("nan")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BraTS-only supervised 3D U-Net baseline.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--experiment-name", default="brats_baseline_unet")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt in the experiment directory if present.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(cfg["project_root"]).resolve()
    brats_root = resolve_path(project_root, cfg["brats_npz_root"])
    exp_root = resolve_path(project_root, cfg["experiments_root"])
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    configure_torch_runtime(cfg, device)
    use_amp = bool(cfg.get("amp", True))

    train_cfg = cfg["brats_training"]
    model_cfg = cfg["model"]
    patch_size = as_tuple3(train_cfg["patch_size"])
    num_classes = int(model_cfg["num_classes"])
    label_weights = {int(k): float(v) for k, v in train_cfg.get("foreground_label_weights", {}).items()}
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    step_sleep_sec = float(cfg.get("step_sleep_sec", train_cfg.get("step_sleep_sec", 0.0)))

    train_files = list_npz_files(brats_root, "train")
    val_files = list_npz_files(brats_root, "val")
    test_files = list_npz_files(brats_root, "test")
    assert_split_integrity(train_files, val_files, test_files)
    if not train_files or not val_files:
        raise FileNotFoundError(f"Expected extracted BraTS NPZ files under {brats_root}")
    print(
        {
            "device": str(device),
            "brats_root": str(brats_root),
            "train_cases": len(train_files),
            "val_cases": len(val_files),
            "test_cases_locked": len(test_files),
            "patch_size": patch_size,
            "batch_size": int(train_cfg["batch_size"]),
            "train_patches_per_epoch": int(train_cfg["train_patches_per_epoch"]),
            "val_patches_per_epoch": int(train_cfg["val_patches_per_epoch"]),
            "all_train_cases_seen_each_epoch": int(train_cfg["train_patches_per_epoch"]) >= len(train_files),
            "all_val_cases_seen_each_epoch": int(train_cfg["val_patches_per_epoch"]) >= len(val_files),
            "foreground_label_weights": label_weights,
            "max_grad_norm": max_grad_norm,
            "step_sleep_sec": step_sleep_sec,
        }
    )

    exp_dir = prepare_experiment_dir(exp_root, args.experiment_name, cfg)
    model = UNet3D(
        in_channels=int(model_cfg["brats_in_channels"]),
        out_channels=num_classes,
        base_channels=int(model_cfg["base_channels"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(train_cfg.get("scheduler_factor", 0.5)),
        patience=int(train_cfg.get("scheduler_patience", 8)),
        min_lr=float(train_cfg.get("min_lr", 1e-6)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    loss_fn = DiceCELoss(num_classes=num_classes)
    early = EarlyStopping(
        patience=int(train_cfg["early_stopping_patience"]),
        min_delta=float(train_cfg["min_delta"]),
    )

    start_epoch = 1
    best_score = -float("inf")
    if args.resume and (exp_dir / "last.pt").exists():
        checkpoint = torch.load(exp_dir / "last.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if (exp_dir / "best.pt").exists():
            best_checkpoint = torch.load(exp_dir / "best.pt", map_location="cpu")
            best_score = float(best_checkpoint.get("metrics", {}).get("val_mean_foreground_dice", best_score))
        print({"resume": True, "start_epoch": start_epoch, "best_score": best_score})

    train_ds = BratsPatchDataset(
        train_files,
        patches_per_epoch=int(train_cfg["train_patches_per_epoch"]),
        patch_size=patch_size,
        foreground_prob=float(train_cfg["foreground_patch_prob"]),
        augment=True,
        foreground_label_weights=label_weights,
        intensity_augment=bool(train_cfg.get("intensity_augment", True)),
    )
    val_ds = BratsPatchDataset(
        val_files,
        patches_per_epoch=int(train_cfg["val_patches_per_epoch"]),
        patch_size=patch_size,
        foreground_prob=float(train_cfg["foreground_patch_prob"]),
        augment=False,
        foreground_label_weights=label_weights,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        **dataloader_kwargs(cfg, device),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        **dataloader_kwargs(cfg, device),
    )

    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        start = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            loss_fn,
            device,
            use_amp,
            num_classes,
            max_grad_norm,
            step_sleep_sec,
        )
        val_metrics = validate(model, val_loader, loss_fn, device, num_classes, use_amp, step_sleep_sec)
        score = float(val_metrics["val_mean_foreground_dice"])
        train_fg = float(train_metrics["train_mean_foreground_dice"])
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "elapsed_sec": round(time.time() - start, 2),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **train_metrics,
            **val_metrics,
            "generalization_gap_fg_dice": train_fg - score,
        }
        append_metrics_csv(exp_dir / "metrics.csv", row)
        print(row)

        save_checkpoint(exp_dir / "last.pt", model, optimizer, epoch, row, cfg)
        if score > best_score:
            best_score = score
            save_checkpoint(exp_dir / "best.pt", model, optimizer, epoch, row, cfg)
        if early.step(score):
            print(f"Early stopping at epoch {epoch}. Best validation score: {early.best:.6f}")
            break


if __name__ == "__main__":
    main()
