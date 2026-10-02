from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import (
    maybe_convert_to_dataset_name,
)

from trainer import (
    nnUNetTrainer_GroupedCV250Duty80,
    nnUNetTrainer_GroupedCVSmoke,
)


TRAINERS = {
    "full": nnUNetTrainer_GroupedCV250Duty80,
    "smoke": nnUNetTrainer_GroupedCVSmoke,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "validate", "show-output"))
    parser.add_argument("--variant", choices=tuple(TRAINERS), default="full")
    parser.add_argument("--dataset-id", type=int, default=503)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--plans", default="nnUNetResEncUNet10p5GBPlans")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--save-probabilities", action="store_true")
    return parser.parse_args()


def build_trainer(args: argparse.Namespace):
    dataset_name = maybe_convert_to_dataset_name(args.dataset_id)
    preprocessed = Path(nnUNet_preprocessed) / dataset_name
    plans = load_json(str(preprocessed / f"{args.plans}.json"))
    plans["continue_training"] = bool(args.continue_training)
    dataset_json = load_json(str(preprocessed / "dataset.json"))
    return TRAINERS[args.variant](
        plans=plans,
        configuration=args.configuration,
        fold=args.fold,
        dataset_json=dataset_json,
        device=torch.device(args.device),
    )


def initialize_or_resume(trainer, continue_training: bool) -> None:
    if not continue_training:
        trainer.initialize()
        return
    candidates = [
        Path(trainer.output_folder) / "checkpoint_latest.pth",
        Path(trainer.output_folder) / "checkpoint_best.pth",
    ]
    candidates = [item for item in candidates if item.is_file()]
    if not candidates:
        raise FileNotFoundError("No checkpoint is available to continue.")

    def checkpoint_epoch(path: Path) -> int:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return int(payload.get("current_epoch", -1))

    checkpoint = max(
        candidates,
        key=lambda item: (checkpoint_epoch(item), item.stat().st_mtime_ns),
    )
    epoch = checkpoint_epoch(checkpoint)
    trainer.load_checkpoint(str(checkpoint))
    print(
        f"Continuing from {checkpoint} (current_epoch={epoch})",
        flush=True,
    )


def configure_runtime(fold: int) -> None:
    seed = 20260726 + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)


def main() -> None:
    args = parse_args()
    if args.fold not in range(5):
        raise ValueError("fold must be in [0, 4]")
    configure_runtime(args.fold)
    trainer = build_trainer(args)

    if args.mode == "show-output":
        print(trainer.output_folder)
        return

    if args.mode == "train":
        initialize_or_resume(trainer, args.continue_training)
        trainer.run_training()
        return

    checkpoint = Path(trainer.output_folder) / f"checkpoint_{args.checkpoint}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    trainer.load_checkpoint(str(checkpoint))
    trainer.perform_actual_validation(
        save_probabilities=bool(args.save_probabilities)
    )


if __name__ == "__main__":
    main()
