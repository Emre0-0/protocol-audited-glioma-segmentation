from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch

from .config import save_json


def configure_torch_runtime(config: dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", True))
    torch.backends.cuda.matmul.allow_tf32 = bool(config.get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(config.get("allow_tf32", True))
    try:
        torch.set_float32_matmul_precision(str(config.get("matmul_precision", "high")))
    except Exception:
        pass


def dataloader_kwargs(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    workers = int(config.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return kwargs


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None or value > self.best + self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def append_metrics_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "saved_at_unix": time.time(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def prepare_experiment_dir(root: str | Path, name: str, config: dict[str, Any]) -> Path:
    exp_dir = Path(root) / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, exp_dir / "config.json")
    return exp_dir
