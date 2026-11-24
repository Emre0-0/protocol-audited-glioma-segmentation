from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.utilities.dataset_name_id_conversion import (
    maybe_convert_to_dataset_name,
)
from torch._dynamo.eval_frame import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from trainer import (
    nnUNetTrainer_BaselineStress,
    nnUNetTrainer_BraTSRegionAware120,
    nnUNetTrainer_BraTSRegionAwareCorrected80,
    nnUNetTrainer_BraTSRegionAwareCorrectedProbe1,
    nnUNetTrainer_BraTSRegionAwareCorrectedSmoke,
    nnUNetTrainer_BraTSRegionAwareSmoke,
    nnUNetTrainer_BraTSRegionAwareStress,
)


TRAINERS = {
    "proxy": nnUNetTrainer_BraTSRegionAware120,
    "smoke": nnUNetTrainer_BraTSRegionAwareSmoke,
    "stress": nnUNetTrainer_BraTSRegionAwareStress,
    "baseline-stress": nnUNetTrainer_BaselineStress,
    "corrected-full": nnUNetTrainer_BraTSRegionAwareCorrected80,
    "corrected-probe": nnUNetTrainer_BraTSRegionAwareCorrectedProbe1,
    "corrected-smoke": nnUNetTrainer_BraTSRegionAwareCorrectedSmoke,
}


def build_trainer(args: argparse.Namespace):
    dataset_name = maybe_convert_to_dataset_name(int(args.dataset_id))
    preprocessed = Path(nnUNet_preprocessed) / dataset_name
    plans = load_json(str(preprocessed / f"{args.plans}.json"))
    plans["continue_training"] = bool(args.continue_training)
    dataset_json = load_json(str(preprocessed / "dataset.json"))
    trainer_class = TRAINERS[args.variant]
    return trainer_class(
        plans=plans,
        configuration=args.configuration,
        fold=args.fold,
        dataset_json=dataset_json,
        device=torch.device(args.device),
    )


def load_for_training(trainer, args: argparse.Namespace) -> None:
    if args.continue_training:
        candidates = (
            Path(trainer.output_folder) / "checkpoint_latest.pth",
            Path(trainer.output_folder) / "checkpoint_best.pth",
        )
        checkpoint = next((path for path in candidates if path.exists()), None)
        if checkpoint is None:
            raise FileNotFoundError(
                "Continuation requested but no latest/best checkpoint exists."
            )
        trainer.load_checkpoint(str(checkpoint))
        print(f"Continuing from {checkpoint}")
        return

    if not args.pretrained_weights:
        raise ValueError(
            "A locked baseline checkpoint is required for a fresh candidate run."
        )
    checkpoint = Path(args.pretrained_weights).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    trainer.initialize()
    if args.variant.startswith("corrected-"):
        saved_model = torch.load(
            str(checkpoint), map_location="cpu", weights_only=False
        )
        pretrained_dict = saved_model["network_weights"]
        network = trainer.network
        if isinstance(network, DDP):
            network = network.module
        if isinstance(network, OptimizedModule):
            network = network._orig_mod
        incompatible = network.load_state_dict(pretrained_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("Corrected warm start was not a strict match.")
        head_keys = [
            key for key in pretrained_dict if ".seg_layers." in key
        ]
        if not head_keys:
            raise RuntimeError("Segmentation-head tensors are missing.")
        print(
            "Strict full-network warm start: "
            f"{len(pretrained_dict)} tensors, including "
            f"{len(head_keys)} segmentation-head tensors."
        )
        if args.variant == "corrected-full":
            baseline_ema = saved_model.get("_best_ema")
            if baseline_ema is not None:
                trainer._best_ema = float(baseline_ema)
            output = Path(trainer.output_folder)
            output.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(output / "checkpoint_best.pth"))
            print(
                "Seeded corrected checkpoint_best.pth with the exact "
                f"warm start (reference EMA={trainer._best_ema})."
            )
        del pretrained_dict
        del saved_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        load_pretrained_weights(
            trainer.network, str(checkpoint), verbose=False
        )
    print(f"Warm-started from {checkpoint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "validate", "show-output"))
    parser.add_argument("--variant", choices=tuple(TRAINERS), default="proxy")
    parser.add_argument("--dataset-id", type=int, default=503)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--plans", default="nnUNetResEncUNet10p5GBPlans")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--pretrained-weights")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--disable-checkpointing", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--validation-output-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    trainer = build_trainer(args)

    if args.mode == "show-output":
        print(trainer.output_folder)
        return

    if args.mode == "train":
        trainer.disable_checkpointing = args.disable_checkpointing
        load_for_training(trainer, args)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        trainer.run_training()
        return

    checkpoint = Path(trainer.output_folder) / f"checkpoint_{args.checkpoint}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    trainer.load_checkpoint(str(checkpoint))
    if args.validation_output_root:
        validation_output_root = Path(args.validation_output_root).resolve()
        validation_output_root.mkdir(parents=True, exist_ok=True)
        trainer.output_folder = str(validation_output_root)
        print(
            "Validation outputs redirected to "
            f"{validation_output_root / 'validation'}"
        )
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    trainer.perform_actual_validation(
        save_probabilities=bool(args.save_probabilities)
    )


if __name__ == "__main__":
    main()
