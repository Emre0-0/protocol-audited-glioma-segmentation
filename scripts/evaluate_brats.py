from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_mri_project.config import get_device, load_config, resolve_path
from brain_mri_project.data import as_tuple3, list_npz_files, load_brats_npz, load_residual_npz
from brain_mri_project.inference import sliding_window_logits
from brain_mri_project.losses import dice_from_prediction
from brain_mri_project.metrics import summarize_dice
from brain_mri_project.models import UNet3D
from brain_mri_project.training import configure_torch_runtime, prepare_experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BraTS segmentation checkpoint on a locked split.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--experiment-name", default="brats_evaluation")
    parser.add_argument("--anomaly-aux", action="store_true", help="Append precomputed anomaly residual as a fifth input channel.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(cfg["project_root"]).resolve()
    brats_root = resolve_path(project_root, cfg["brats_npz_root"])
    residual_root = resolve_path(project_root, cfg["anomaly_residual_root"])
    exp_root = resolve_path(project_root, cfg["experiments_root"])
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    device = get_device(cfg.get("device", "auto"))
    configure_torch_runtime(cfg, device)
    use_amp = bool(cfg.get("amp", True))
    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluation"]
    patch_size = as_tuple3(eval_cfg["patch_size"])
    num_classes = int(model_cfg["num_classes"])

    files = list_npz_files(brats_root, args.split)
    if eval_cfg.get("max_cases") is not None:
        files = files[: int(eval_cfg["max_cases"])]
    if not files:
        raise FileNotFoundError(f"No NPZ files found under {brats_root / args.split}")
    print(
        {
            "device": str(device),
            "split": args.split,
            "case_count": len(files),
            "checkpoint": str(checkpoint_path),
            "patch_size": patch_size,
            "anomaly_aux": bool(args.anomaly_aux),
        }
    )

    exp_dir = prepare_experiment_dir(exp_root, f"{args.experiment_name}_{args.split}", cfg)
    in_channels = int(model_cfg["brats_in_channels"]) + (1 if args.anomaly_aux else 0)
    model = UNet3D(
        in_channels=in_channels,
        out_channels=num_classes,
        base_channels=int(model_cfg["base_channels"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows: list[dict[str, float | str]] = []
    for path in tqdm(files, desc=f"evaluate-{args.split}"):
        image, seg, case_id = load_brats_npz(path)
        if args.anomaly_aux:
            residual_path = residual_root / args.split / f"{case_id}.npz"
            residual = load_residual_npz(residual_path)
            image = np.concatenate([image, residual], axis=0)
        logits = sliding_window_logits(
            model,
            image,
            patch_size=patch_size,
            num_classes=num_classes,
            overlap=float(eval_cfg["overlap"]),
            device=device,
            amp=use_amp,
        )
        pred = torch.argmax(logits, dim=0).cpu()[None]
        target = torch.from_numpy(seg[None]).long()
        dices = dice_from_prediction(pred, target, num_classes=num_classes)
        row: dict[str, float | str] = {"case_id": case_id}
        for cls, value in enumerate(dices):
            row[f"dice_label_{cls}"] = float(value)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(exp_dir / f"{args.split}_case_metrics.csv", index=False)
    summary = summarize_dice(rows, num_classes=num_classes)  # type: ignore[arg-type]
    pd.DataFrame([summary]).to_csv(exp_dir / f"{args.split}_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
