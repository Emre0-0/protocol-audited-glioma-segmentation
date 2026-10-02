from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train_brats_baseline as baseline
from brain_mri_project.config import get_device, load_config, resolve_path, set_seed
from brain_mri_project.data import BratsPatchDataset, as_tuple3, list_npz_files, load_brats_seg_npz
from brain_mri_project.losses import DiceCELoss
from brain_mri_project.models import UNet3D, load_encoder_state_with_channel_adaptation
from brain_mri_project.training import (
    EarlyStopping,
    append_metrics_csv,
    configure_torch_runtime,
    dataloader_kwargs,
    prepare_experiment_dir,
    save_checkpoint,
)


def class_weight_tensor(
    train_cfg: dict,
    key: str,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor | None:
    value = train_cfg.get(key)
    if value is None:
        return None
    if isinstance(value, dict):
        weights = [1.0] * num_classes
        for label, weight in value.items():
            weights[int(label)] = float(weight)
    else:
        weights = [float(v) for v in value]
        if len(weights) != num_classes:
            raise ValueError(f"{key} must contain {num_classes} values, got {len(weights)}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_case_sampling_weights(files: list[Path], train_cfg: dict) -> tuple[list[float] | None, dict]:
    strategy = str(train_cfg.get("case_sampling_strategy", "uniform"))
    if strategy == "uniform":
        return None, {"case_sampling_strategy": "uniform"}
    if strategy != "small_foreground_boost":
        raise ValueError(f"Unknown case_sampling_strategy: {strategy}")

    fg_volumes: list[int] = []
    rc_volumes: list[int] = []
    for path in files:
        seg, _ = load_brats_seg_npz(path)
        fg_volumes.append(int(np.sum(seg > 0)))
        rc_volumes.append(int(np.sum(seg == 4)))

    fg = np.asarray(fg_volumes, dtype=np.float64)
    rc = np.asarray(rc_volumes, dtype=np.float64)
    q33, q66 = np.quantile(fg, [0.33, 0.66])
    weights = np.ones_like(fg, dtype=np.float64)
    weights[fg <= q33] *= float(train_cfg.get("small_foreground_sampling_factor", 1.0))
    weights[(fg > q33) & (fg <= q66)] *= float(train_cfg.get("medium_foreground_sampling_factor", 1.0))
    weights[fg > q66] *= float(train_cfg.get("large_foreground_sampling_factor", 1.0))
    weights[rc == 0] *= float(train_cfg.get("rc_negative_sampling_factor", 1.0))
    weights[rc > 0] *= float(train_cfg.get("rc_positive_sampling_factor", 1.0))
    weights = np.clip(weights, 0.0, None)

    summary = {
        "case_sampling_strategy": strategy,
        "foreground_volume_q33": float(q33),
        "foreground_volume_q66": float(q66),
        "small_foreground_cases": int(np.sum(fg <= q33)),
        "medium_foreground_cases": int(np.sum((fg > q33) & (fg <= q66))),
        "large_foreground_cases": int(np.sum(fg > q66)),
        "rc_negative_cases": int(np.sum(rc == 0)),
        "rc_positive_cases": int(np.sum(rc > 0)),
        "sampling_weight_min": float(weights.min()),
        "sampling_weight_max": float(weights.max()),
    }
    return weights.tolist(), summary


def set_encoder_trainable(model: UNet3D, trainable: bool) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = trainable


def build_transfer_optimizer(model: UNet3D, train_cfg: dict) -> torch.optim.Optimizer:
    base_lr = float(train_cfg["lr"])
    encoder_lr = float(train_cfg.get("encoder_lr", base_lr))
    decoder_lr = float(train_cfg.get("decoder_lr", base_lr))
    weight_decay = float(train_cfg["weight_decay"])
    param_groups = [
        {"params": model.encoder.parameters(), "lr": encoder_lr, "name": "encoder"},
        {"params": model.decoder.parameters(), "lr": decoder_lr, "name": "decoder"},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def compute_selection_score(
    train_metrics: dict,
    val_metrics: dict,
    train_cfg: dict,
) -> tuple[float, dict]:
    selection_metric = str(train_cfg.get("selection_metric", "val_mean_foreground_dice"))
    val_mean_fg = float(val_metrics["val_mean_foreground_dice"])
    train_mean_fg = float(train_metrics["train_mean_foreground_dice"])
    gap = train_mean_fg - val_mean_fg

    if selection_metric == "val_mean_foreground_dice":
        return val_mean_fg, {
            "selection_metric": selection_metric,
            "selection_score": val_mean_fg,
            "selection_penalty": 0.0,
        }
    if selection_metric != "balanced_patch_score":
        raise ValueError(f"Unknown selection_metric: {selection_metric}")

    components = {
        "mean_foreground": val_mean_fg,
        "label_1": float(val_metrics["val_dice_label_1"]),
        "label_2": float(val_metrics["val_dice_label_2"]),
        "label_3": float(val_metrics["val_dice_label_3"]),
        "label_4": float(val_metrics["val_dice_label_4"]),
    }
    weights = train_cfg.get(
        "selection_weights",
        {
            "mean_foreground": 0.35,
            "label_1": 0.05,
            "label_2": 0.20,
            "label_3": 0.20,
            "label_4": 0.20,
        },
    )
    raw_score = 0.0
    for name, value in components.items():
        raw_score += float(weights.get(name, 0.0)) * value

    penalty = 0.0
    gap_target = float(train_cfg.get("selection_gap_target", 0.06))
    gap_penalty_weight = float(train_cfg.get("selection_gap_penalty_weight", 0.0))
    penalty += gap_penalty_weight * max(0.0, gap - gap_target)

    floor_penalty_weight = float(train_cfg.get("selection_floor_penalty_weight", 0.0))
    for name, floor_value in train_cfg.get("selection_label_floors", {}).items():
        if name in components:
            penalty += floor_penalty_weight * max(0.0, float(floor_value) - components[name])

    score = raw_score - penalty
    return score, {
        "selection_metric": selection_metric,
        "selection_score": score,
        "selection_raw_score": raw_score,
        "selection_penalty": penalty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BraTS segmentation from normal-T1 pretrained encoder.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--pretrained-encoder", required=True, help="Path to best_encoder.pt or a checkpoint with encoder_state.")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional full segmentation checkpoint whose model_state initializes the run before training.",
    )
    parser.add_argument("--experiment-name", default="brats_transfer_from_normal_t1")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt in the experiment directory if present.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(cfg["project_root"]).resolve()
    brats_root = resolve_path(project_root, cfg["brats_npz_root"])
    exp_root = resolve_path(project_root, cfg["experiments_root"])
    pretrained_path = resolve_path(project_root, args.pretrained_encoder)
    init_checkpoint_path = None if args.init_checkpoint is None else resolve_path(project_root, args.init_checkpoint)
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
    freeze_encoder_epochs = int(train_cfg.get("freeze_encoder_epochs", 0))

    train_files = list_npz_files(brats_root, "train")
    val_files = list_npz_files(brats_root, "val")
    test_files = list_npz_files(brats_root, "test")
    baseline.assert_split_integrity(train_files, val_files, test_files)
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
            "case_sampling_strategy": train_cfg.get("case_sampling_strategy", "uniform"),
            "freeze_encoder_epochs": freeze_encoder_epochs,
            "encoder_lr": float(train_cfg.get("encoder_lr", train_cfg["lr"])),
            "decoder_lr": float(train_cfg.get("decoder_lr", train_cfg["lr"])),
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

    loaded = torch.load(pretrained_path, map_location="cpu")
    encoder_state = loaded["encoder_state"] if isinstance(loaded, dict) and "encoder_state" in loaded else loaded
    missing, unexpected = load_encoder_state_with_channel_adaptation(
        model,
        encoder_state,
        in_channels=int(model_cfg["brats_in_channels"]),
    )
    print({"loaded_pretrained_encoder": str(pretrained_path), "missing": missing, "unexpected": unexpected})

    if init_checkpoint_path is not None:
        init_checkpoint = torch.load(init_checkpoint_path, map_location=device)
        model.load_state_dict(init_checkpoint["model_state"])
        print(
            {
                "loaded_init_checkpoint": str(init_checkpoint_path),
                "init_epoch": init_checkpoint.get("epoch"),
                "init_val_mean_foreground_dice": init_checkpoint.get("metrics", {}).get("val_mean_foreground_dice"),
            }
        )

    optimizer = build_transfer_optimizer(model, train_cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(train_cfg.get("scheduler_factor", 0.5)),
        patience=int(train_cfg.get("scheduler_patience", 8)),
        min_lr=float(train_cfg.get("min_lr", 1e-6)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    ce_weight = class_weight_tensor(train_cfg, "ce_class_weights", num_classes, device)
    dice_weight = class_weight_tensor(train_cfg, "dice_class_weights", num_classes, device)
    loss_fn = DiceCELoss(
        num_classes=num_classes,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        dice_loss_weight=float(train_cfg.get("dice_loss_weight", 1.0)),
        ce_loss_weight=float(train_cfg.get("ce_loss_weight", 1.0)),
    )
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
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            except ValueError as exc:
                print({"optimizer_resume_skipped": str(exc)})
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if (exp_dir / "best.pt").exists():
            best_checkpoint = torch.load(exp_dir / "best.pt", map_location="cpu")
            best_metrics = best_checkpoint.get("metrics", {})
            best_score = float(
                best_metrics.get("selection_score", best_metrics.get("val_mean_foreground_dice", best_score))
            )
        print({"resume": True, "start_epoch": start_epoch, "best_score": best_score})

    case_sampling_weights, case_sampling_summary = compute_case_sampling_weights(train_files, train_cfg)
    print(case_sampling_summary)

    train_ds = BratsPatchDataset(
        train_files,
        patches_per_epoch=int(train_cfg["train_patches_per_epoch"]),
        patch_size=patch_size,
        foreground_prob=float(train_cfg["foreground_patch_prob"]),
        augment=True,
        foreground_label_weights=label_weights,
        intensity_augment=bool(train_cfg.get("intensity_augment", True)),
        case_sampling_weights=case_sampling_weights,
    )
    val_ds = BratsPatchDataset(
        val_files,
        patches_per_epoch=int(train_cfg["val_patches_per_epoch"]),
        patch_size=patch_size,
        foreground_prob=float(train_cfg["foreground_patch_prob"]),
        augment=False,
        foreground_label_weights=label_weights,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        **dataloader_kwargs(cfg, device),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        **dataloader_kwargs(cfg, device),
    )

    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        encoder_frozen = epoch <= freeze_encoder_epochs
        set_encoder_trainable(model, not encoder_frozen)
        train_metrics = baseline.train_one_epoch(
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
            eval_modules=[model.encoder] if encoder_frozen else None,
        )
        val_metrics = baseline.validate(model, val_loader, loss_fn, device, num_classes, use_amp, step_sleep_sec)
        val_mean_fg = float(val_metrics["val_mean_foreground_dice"])
        score, selection_metrics = compute_selection_score(train_metrics, val_metrics, train_cfg)
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[-1]["lr"]),
            "encoder_lr": float(optimizer.param_groups[0]["lr"]),
            "decoder_lr": float(optimizer.param_groups[-1]["lr"]),
            "encoder_frozen": int(encoder_frozen),
            **train_metrics,
            **val_metrics,
            "generalization_gap_fg_dice": float(train_metrics["train_mean_foreground_dice"]) - val_mean_fg,
            **selection_metrics,
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
