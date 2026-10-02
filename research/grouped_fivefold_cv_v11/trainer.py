from __future__ import annotations

from time import perf_counter, sleep

import torch
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_250epochs,
)


class _Duty80Mixin:
    """Limit average training-step GPU duty cycle without changing optimization."""

    gpu_duty_cycle_target = 0.80

    def on_train_start(self) -> None:
        super().on_train_start()
        self.print_to_log_file(
            f"GPU duty-cycle target: {self.gpu_duty_cycle_target:.0%}"
        )

    def train_step(self, batch: dict) -> dict:
        started = perf_counter()
        result = super().train_step(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        active_seconds = perf_counter() - started
        if 0 < self.gpu_duty_cycle_target < 1:
            sleep(
                active_seconds
                * ((1.0 / self.gpu_duty_cycle_target) - 1.0)
            )
        self._duty_step += 1
        if self.device.type == "cuda" and (
            self._duty_step <= 3 or self._duty_step % 250 == 0
        ):
            self.print_to_log_file(
                "CUDA memory after step "
                f"{self._duty_step}: "
                f"allocated={torch.cuda.memory_allocated(self.device) / 2**30:.3f} GiB, "
                f"reserved={torch.cuda.memory_reserved(self.device) / 2**30:.3f} GiB, "
                f"peak={torch.cuda.max_memory_allocated(self.device) / 2**30:.3f} GiB"
            )
        return result


class nnUNetTrainer_GroupedCV250Duty80(
    _Duty80Mixin, nnUNetTrainer_250epochs
):
    """Exact 250-epoch baseline recipe with operational duty-cycle limiting."""

    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(
            plans, configuration, fold, dataset_json, device
        )
        self._duty_step = 0
        self.save_every = 25


class nnUNetTrainer_GroupedCVSmoke(_Duty80Mixin, nnUNetTrainer_250epochs):
    """Two-step integration check before a long fold run."""

    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(
            plans, configuration, fold, dataset_json, device
        )
        self._duty_step = 0
        self.num_epochs = 1
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1
        self.save_every = 1
