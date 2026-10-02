from __future__ import annotations

import numpy as np
import torch
from time import perf_counter, sleep

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from losses import BraTSRegionAwareCompoundLoss


class _BraTSRegionAwareTrainer(nnUNetTrainer):
    region_auxiliary_weight = 0.35
    absent_region_false_positive_weight = 0.25
    target_guard_batches = 32
    gpu_duty_cycle_target = 0.80

    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._target_guard_batches_seen = 0
        self._memory_log_steps = 0

    def _guard_target_range(self, target) -> None:
        if self._target_guard_batches_seen >= self.target_guard_batches:
            return
        targets = target if isinstance(target, (tuple, list)) else (target,)
        for level, item in enumerate(targets):
            finite = torch.isfinite(item).all().item()
            minimum = float(item.min().item())
            maximum = float(item.max().item())
            if not finite or minimum < 0 or maximum >= 5:
                invalid = item[
                    (~torch.isfinite(item)) | (item < 0) | (item >= 5)
                ]
                sample = torch.unique(invalid)[:20].cpu().tolist()
                raise RuntimeError(
                    "Invalid segmentation target detected before CUDA loss: "
                    f"deep_supervision_level={level}, min={minimum}, "
                    f"max={maximum}, finite={finite}, invalid_values={sample}"
                )
        self._target_guard_batches_seen += 1

    def train_step(self, batch: dict) -> dict:
        self._guard_target_range(batch["target"])
        step_started = perf_counter()
        result = super().train_step(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        active_seconds = perf_counter() - step_started
        if 0 < self.gpu_duty_cycle_target < 1:
            sleep(
                active_seconds
                * ((1.0 / self.gpu_duty_cycle_target) - 1.0)
            )
        self._memory_log_steps += 1
        if self.device.type == "cuda" and (
            self._memory_log_steps <= 10 or self._memory_log_steps % 16 == 0
        ):
            self.print_to_log_file(
                "CUDA memory after step "
                f"{self._memory_log_steps}: "
                f"allocated={torch.cuda.memory_allocated(self.device) / 2**30:.3f} GiB, "
                f"reserved={torch.cuda.memory_reserved(self.device) / 2**30:.3f} GiB, "
                f"peak={torch.cuda.max_memory_allocated(self.device) / 2**30:.3f} GiB"
            )
        return result

    def validation_step(self, batch: dict) -> dict:
        self._guard_target_range(batch["target"])
        return super().validation_step(batch)

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise RuntimeError(
                "This trainer expects mutually exclusive labels 0-4, not a "
                "region-based dataset.json."
            )
        if tuple(self.label_manager.all_labels) != (0, 1, 2, 3, 4):
            raise RuntimeError(
                f"Expected BraTS labels (0, 1, 2, 3, 4), got "
                f"{tuple(self.label_manager.all_labels)}."
            )

        base_loss = DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        loss = BraTSRegionAwareCompoundLoss(
            base_loss=base_loss,
            region_auxiliary_weight=self.region_auxiliary_weight,
            absent_region_false_positive_weight=(
                self.absent_region_false_positive_weight
            ),
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array(
                [1 / (2**i) for i in range(len(deep_supervision_scales))],
                dtype=np.float64,
            )
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainer_BraTSRegionAware120(_BraTSRegionAwareTrainer):
    """Warm-started, metric-aligned grouped fold-0 refinement candidate."""

    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 120
        self.initial_lr = 1e-3
        self.save_every = 20


class nnUNetTrainer_BraTSRegionAwareSmoke(_BraTSRegionAwareTrainer):
    """Two-batch integration test; checkpointing is disabled by the runner."""

    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1
        self.initial_lr = 1e-3
        self.save_every = 1


class nnUNetTrainer_BraTSRegionAwareStress(_BraTSRegionAwareTrainer):
    """Short target-range and sustained-GPU diagnostic."""

    target_guard_batches = 128

    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 64
        self.num_val_iterations_per_epoch = 2
        self.initial_lr = 1e-3
        self.save_every = 1


class nnUNetTrainer_BaselineStress(nnUNetTrainer):
    """Standard nnU-Net loss control under the same sustained GPU load."""

    gpu_duty_cycle_target = 0.80

    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 32
        self.num_val_iterations_per_epoch = 2
        self.initial_lr = 1e-3
        self.save_every = 1
        self._memory_log_steps = 0

    def train_step(self, batch: dict) -> dict:
        step_started = perf_counter()
        result = super().train_step(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        active_seconds = perf_counter() - step_started
        sleep(
            active_seconds
            * ((1.0 / self.gpu_duty_cycle_target) - 1.0)
        )
        self._memory_log_steps += 1
        if self.device.type == "cuda" and (
            self._memory_log_steps <= 10 or self._memory_log_steps % 16 == 0
        ):
            self.print_to_log_file(
                "CUDA memory after step "
                f"{self._memory_log_steps}: "
                f"allocated={torch.cuda.memory_allocated(self.device) / 2**30:.3f} GiB, "
                f"reserved={torch.cuda.memory_reserved(self.device) / 2**30:.3f} GiB, "
                f"peak={torch.cuda.max_memory_allocated(self.device) / 2**30:.3f} GiB"
            )
        return result


class _BraTSRegionAwareCorrected(_BraTSRegionAwareTrainer):
    """Conservative region alignment that preserves the mature label heads."""

    region_auxiliary_weight = 0.15
    absent_region_false_positive_weight = 0.10


class nnUNetTrainer_BraTSRegionAwareCorrected80(
    _BraTSRegionAwareCorrected
):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 80
        self.initial_lr = 2e-4
        self.save_every = 10


class nnUNetTrainer_BraTSRegionAwareCorrectedProbe1(
    _BraTSRegionAwareCorrected
):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.initial_lr = 2e-4
        self.save_every = 1


class nnUNetTrainer_BraTSRegionAwareCorrectedSmoke(
    _BraTSRegionAwareCorrected
):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1
        self.initial_lr = 2e-4
        self.save_every = 1
