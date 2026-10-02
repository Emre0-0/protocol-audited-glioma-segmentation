from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_mri_project.config import get_device, load_config, resolve_path, set_seed
from brain_mri_project.data import NormalT1PatchDataset, as_tuple3, list_pretrain_files, split_pretrain_files
from brain_mri_project.losses import masked_reconstruction_loss
from brain_mri_project.models import UNet3D
from brain_mri_project.training import (
    EarlyStopping,
    append_metrics_csv,
    configure_torch_runtime,
    dataloader_kwargs,
    prepare_experiment_dir,
    save_checkpoint,
)


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    use_amp: bool,
    train: bool,
    max_grad_norm: float,
    step_sleep_sec: float = 0.0,
) -> float:
    model.train(train)
    total = 0.0
    for batch in tqdm(loader, desc="pretrain" if train else "pretrain-val", leave=False):
        image = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                pred = model(image)
                loss = masked_reconstruction_loss(pred, target, mask)
        else:
            pred = model(image)
            loss = masked_reconstruction_loss(pred, target, mask)
        if train:
            if use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        total += float(loss.item())
        if step_sleep_sec > 0:
            time.sleep(step_sleep_sec)
    return total / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-supervised normal T1 masked reconstruction pretraining.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--experiment-name", default="normal_t1_masked_pretraining")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt in the experiment directory if present.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(cfg["project_root"]).resolve()
    pretrain_root = resolve_path(project_root, cfg["pretrain_npz_root"])
    exp_root = resolve_path(project_root, cfg["experiments_root"])
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    configure_torch_runtime(cfg, device)
    use_amp = bool(cfg.get("amp", True))
    train_cfg = cfg["pretraining"]
    model_cfg = cfg["model"]
    patch_size = as_tuple3(train_cfg["patch_size"])
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    step_sleep_sec = float(cfg.get("step_sleep_sec", train_cfg.get("step_sleep_sec", 0.0)))

    files = list_pretrain_files(pretrain_root)
    if not files:
        raise FileNotFoundError(f"Expected extracted normal T1 NPZ files under {pretrain_root}")
    train_files, val_files = split_pretrain_files(files, val_fraction=0.15, seed=int(cfg["seed"]))
    exp_dir = prepare_experiment_dir(exp_root, args.experiment_name, cfg)
    dataset_counts: dict[str, int] = {}
    for path in files:
        dataset_counts[path.parent.name] = dataset_counts.get(path.parent.name, 0) + 1
    print(
        {
            "device": str(device),
            "pretrain_root": str(pretrain_root),
            "total_cases": len(files),
            "train_cases": len(train_files),
            "val_cases": len(val_files),
            "dataset_counts": dataset_counts,
            "patch_size": patch_size,
            "batch_size": int(train_cfg["batch_size"]),
            "patches_per_epoch": int(train_cfg["patches_per_epoch"]),
            "all_train_cases_seen_each_epoch": int(train_cfg["patches_per_epoch"]) >= len(train_files),
            "max_grad_norm": max_grad_norm,
            "step_sleep_sec": step_sleep_sec,
        }
    )

    model = UNet3D(
        in_channels=int(model_cfg["pretrain_in_channels"]),
        out_channels=1,
        base_channels=int(model_cfg["base_channels"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg.get("scheduler_factor", 0.5)),
        patience=int(train_cfg.get("scheduler_patience", 6)),
        min_lr=float(train_cfg.get("min_lr", 1e-6)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
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
            best_val_loss = float(best_checkpoint.get("metrics", {}).get("val_reconstruction_loss", float("inf")))
            best_score = -best_val_loss
        print({"resume": True, "start_epoch": start_epoch, "best_score": best_score})

    train_ds = NormalT1PatchDataset(
        train_files,
        patches_per_epoch=int(train_cfg["patches_per_epoch"]),
        patch_size=patch_size,
        mask_ratio=float(train_cfg["mask_ratio"]),
        noise_std=float(train_cfg["noise_std"]),
        augment=True,
    )
    val_ds = NormalT1PatchDataset(
        val_files,
        patches_per_epoch=max(16, int(train_cfg["patches_per_epoch"]) // 5),
        patch_size=patch_size,
        mask_ratio=float(train_cfg["mask_ratio"]),
        noise_std=float(train_cfg["noise_std"]),
        augment=False,
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
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            train=True,
            max_grad_norm=max_grad_norm,
            step_sleep_sec=step_sleep_sec,
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model,
                val_loader,
                optimizer,
                scaler,
                device,
                use_amp,
                train=False,
                max_grad_norm=0.0,
                step_sleep_sec=step_sleep_sec,
            )
        score = -val_loss
        scheduler.step(val_loss)
        row = {
            "epoch": epoch,
            "elapsed_sec": round(time.time() - start, 2),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_reconstruction_loss": train_loss,
            "val_reconstruction_loss": val_loss,
            "reconstruction_generalization_gap": val_loss - train_loss,
        }
        append_metrics_csv(exp_dir / "metrics.csv", row)
        print(row)
        extra = {"encoder_state": model.encoder.state_dict()}
        save_checkpoint(exp_dir / "last.pt", model, optimizer, epoch, row, cfg, extra=extra)
        if score > best_score:
            best_score = score
            save_checkpoint(exp_dir / "best.pt", model, optimizer, epoch, row, cfg, extra=extra)
            torch.save(model.encoder.state_dict(), exp_dir / "best_encoder.pt")
        if early.step(score):
            print(f"Early stopping at epoch {epoch}. Best validation loss: {-early.best:.6f}")
            break


if __name__ == "__main__":
    main()
