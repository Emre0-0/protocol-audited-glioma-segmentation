from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DATASET = "Dataset503_BraTS2024Grouped4Ch"
PLANS = "nnUNetResEncUNet10p5GBPlans"
CONFIGURATION = "3d_fullres"
EXPECTED_EPOCHS = 250
EXPECTED_TRAINERS = {
    0: "nnUNetTrainer_250epochs",
    1: "nnUNetTrainer_GroupedCV250Duty80",
    2: "nnUNetTrainer_GroupedCV250Duty80",
    3: "nnUNetTrainer_GroupedCV250Duty80",
    4: "nnUNetTrainer_GroupedCV250Duty80",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect publication-critical nnU-Net checkpoint metadata."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/publication_integrity_v1/checkpoint_metadata"),
    )
    return parser.parse_args()


def checkpoint_path(root: Path, fold: int, kind: str) -> Path:
    trainer = (
        "nnUNetTrainer_250epochs"
        if fold == 0
        else "nnUNetTrainer_GroupedCV250Duty80"
    )
    return (
        root
        / "experiments"
        / "nnunet_results"
        / DATASET
        / f"{trainer}__{PLANS}__{CONFIGURATION}"
        / f"fold_{fold}"
        / f"checkpoint_{kind}.pth"
    )


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def state_signature(weights: dict[str, torch.Tensor]) -> tuple[str, int, int]:
    records = [
        (name, list(tensor.shape), str(tensor.dtype))
        for name, tensor in weights.items()
    ]
    return (
        canonical_hash(records),
        len(records),
        sum(int(tensor.numel()) for tensor in weights.values()),
    )


def selected_plan_signature(plans: dict[str, Any]) -> str:
    selected = {
        "dataset_name": plans.get("dataset_name"),
        "plans_name": plans.get("plans_name"),
        "transpose_forward": plans.get("transpose_forward"),
        "transpose_backward": plans.get("transpose_backward"),
        "configuration": plans.get("configurations", {}).get(CONFIGURATION),
        "label_manager": plans.get("label_manager"),
    }
    return canonical_hash(selected)


def inspect_one(path: Path, fold: int, kind: str) -> dict[str, Any]:
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    required = {
        "network_weights",
        "optimizer_state",
        "logging",
        "current_epoch",
        "init_args",
        "trainer_name",
    }
    missing_keys = sorted(required.difference(payload))
    init_args = payload.get("init_args", {})
    plans = init_args.get("plans", {})
    dataset_json = init_args.get("dataset_json", {})
    weights = payload.get("network_weights", {})
    architecture_hash, tensor_count, state_numel = state_signature(weights)
    logging = payload.get("logging", {})
    optimizer_groups = payload.get("optimizer_state", {}).get("param_groups", [])
    optimizer_hyperparameters = [
        {
            key: group.get(key)
            for key in (
                "initial_lr",
                "lr",
                "momentum",
                "weight_decay",
                "nesterov",
                "dampening",
            )
        }
        for group in optimizer_groups
    ]
    checks = {
        "required_keys_present": not missing_keys,
        "fold_matches": int(init_args.get("fold", -1)) == fold,
        "dataset_matches": plans.get("dataset_name") == DATASET,
        "plans_match": plans.get("plans_name") == PLANS,
        "configuration_matches": init_args.get("configuration") == CONFIGURATION,
        "trainer_matches": payload.get("trainer_name") == EXPECTED_TRAINERS[fold],
        "four_input_channels": len(dataset_json.get("channel_names", {})) == 4,
        "expected_label_schema": set(dataset_json.get("labels", {}).values())
        == {0, 1, 2, 3, 4},
        "epoch_budget_respected": int(payload.get("current_epoch", -1)) <= EXPECTED_EPOCHS,
        "final_epoch_complete": (
            int(payload.get("current_epoch", -1)) == EXPECTED_EPOCHS
            if kind == "final"
            else True
        ),
        "optimizer_recipe_matches": all(
            group.get("initial_lr") == 0.01
            and group.get("momentum") == 0.99
            and group.get("weight_decay") == 3e-5
            and group.get("nesterov") is True
            for group in optimizer_groups
        )
        and len(optimizer_groups) == 1,
        "mirroring_axes_match": tuple(
            payload.get("inference_allowed_mirroring_axes", ())
        )
        == (0, 1, 2),
    }
    record = {
        "fold": fold,
        "kind": kind,
        "path": str(path),
        "bytes": path.stat().st_size,
        "trainer_name": payload.get("trainer_name"),
        "current_epoch": int(payload.get("current_epoch", -1)),
        "best_ema": float(payload.get("_best_ema", float("nan"))),
        "configuration": init_args.get("configuration"),
        "dataset_name": plans.get("dataset_name"),
        "plans_name": plans.get("plans_name"),
        "fold_in_checkpoint": int(init_args.get("fold", -1)),
        "network_tensor_count": tensor_count,
        "network_state_numel_including_registered_aliases": state_numel,
        "architecture_signature_sha256": architecture_hash,
        "plan_signature_sha256": selected_plan_signature(plans),
        "dataset_json_signature_sha256": canonical_hash(dataset_json),
        "logging_lengths": {
            key: len(value) if isinstance(value, list) else None
            for key, value in logging.items()
        },
        "optimizer_hyperparameters": optimizer_hyperparameters,
        "missing_keys": missing_keys,
        "checks": checks,
        "valid": all(checks.values()),
    }
    del payload
    return record


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Checkpoint metadata audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Completed folds: {report['completed_folds']}/5",
        f"- Inspected checkpoints: {len(report['checkpoints'])}",
        f"- Generated: {report['generated_at_utc']}",
        "",
        "| Fold | Kind | Epoch | Trainer | Valid |",
        "|---:|---|---:|---|---|",
    ]
    for item in report["checkpoints"]:
        lines.append(
            f"| {item['fold']} | {item['kind']} | {item['current_epoch']} | "
            f"`{item['trainer_name']}` | {item['valid']} |"
        )
    lines.extend(
        [
            "",
            "Fold 0 uses the stock 250-epoch trainer. Folds 1--4 use a subclass "
            "that changes only operational duty-cycle timing and checkpoint cadence; "
            "the optimization recipe and network state signature are required to match.",
            "Exact bitwise reruns are not claimed because cuDNN benchmarking is enabled "
            "and deterministic algorithms are not enforced.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for fold in range(5):
        for kind in ("best", "final"):
            path = checkpoint_path(root, fold, kind)
            if not path.is_file():
                missing.append(str(path))
                continue
            records.append(inspect_one(path, fold, kind))

    architecture_signatures = {
        item["architecture_signature_sha256"] for item in records
    }
    plan_signatures = {item["plan_signature_sha256"] for item in records}
    dataset_signatures = {
        item["dataset_json_signature_sha256"] for item in records
    }
    complete_folds = sum(
        checkpoint_path(root, fold, "final").is_file() for fold in range(5)
    )
    cross_checkpoint_checks = {
        "one_architecture_signature": len(architecture_signatures) == 1,
        "one_plan_signature": len(plan_signatures) == 1,
        "one_dataset_json_signature": len(dataset_signatures) == 1,
    }
    invalid = [item["path"] for item in records if not item["valid"]]
    if invalid or not all(cross_checkpoint_checks.values()):
        status = "FAIL"
    elif complete_folds < 5:
        status = "PENDING"
    else:
        status = "PASS"
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "completed_folds": complete_folds,
        "expected_folds": 5,
        "expected_epochs": EXPECTED_EPOCHS,
        "checkpoints": records,
        "missing_checkpoints": missing,
        "invalid_checkpoints": invalid,
        "cross_checkpoint_checks": cross_checkpoint_checks,
        "reproducibility_boundary": (
            "Seeds and environment are recorded, but bitwise determinism is not claimed "
            "because cuDNN benchmarking/TF32 are enabled and deterministic algorithms "
            "are not enforced."
        ),
    }
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "checkpoint_metadata.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "completed_folds": complete_folds}))
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
