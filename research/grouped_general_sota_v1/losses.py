from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


BRATS_REGIONS: Mapping[str, tuple[int, ...]] = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "NETC": (1,),
    "SNFH": (2,),
    "ET": (3,),
    "RC": (4,),
}


class BraTSRegionAuxiliaryLoss(nn.Module):
    """Soft region Dice aligned with the six official BraTS GLI regions."""

    def __init__(
        self,
        regions: Mapping[str, Sequence[int]] = BRATS_REGIONS,
        smooth: float = 1.0,
        absent_region_false_positive_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if not regions:
            raise ValueError("At least one BraTS region is required.")
        if absent_region_false_positive_weight < 0:
            raise ValueError("Absent-region false-positive weight must be non-negative.")
        self.regions = {
            name: tuple(int(label) for label in labels)
            for name, labels in regions.items()
        }
        self.smooth = float(smooth)
        self.absent_region_false_positive_weight = float(
            absent_region_false_positive_weight
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim < 4 or logits.shape[1] != 5:
            raise ValueError(
                f"Expected logits shaped (B, 5, ...), received {tuple(logits.shape)}."
            )
        if target.ndim == logits.ndim - 1:
            target = target[:, None]
        if target.ndim != logits.ndim or target.shape[1] != 1:
            raise ValueError(
                f"Expected target shaped (B, 1, ...), received {tuple(target.shape)}."
            )

        probabilities = torch.softmax(logits, dim=1)
        labels = target[:, 0].long()
        spatial_axes = tuple(range(1, labels.ndim))
        region_losses: list[torch.Tensor] = []

        for region_labels in self.regions.values():
            prediction = probabilities[:, region_labels].sum(dim=1)
            reference = torch.zeros_like(prediction, dtype=torch.bool)
            for label in region_labels:
                reference |= labels == label
            reference_float = reference.to(dtype=prediction.dtype)

            intersection = (prediction * reference_float).sum(
                dim=spatial_axes, dtype=torch.float32
            )
            prediction_sum = prediction.sum(
                dim=spatial_axes, dtype=torch.float32
            )
            reference_sum = reference_float.sum(
                dim=spatial_axes, dtype=torch.float32
            )
            present = reference_sum > 0

            present_loss = 1.0 - (
                (2.0 * intersection + self.smooth)
                / (prediction_sum + reference_sum + self.smooth).clamp_min(1e-8)
            )

            # A Dice term is ill-conditioned when a region is absent from a patch.
            # Penalize only its mean probability in that case, which suppresses
            # false positives without overwhelming positive-region gradients.
            absent_loss = (
                prediction.mean(dim=spatial_axes, dtype=torch.float32)
                * self.absent_region_false_positive_weight
            )
            region_losses.append(torch.where(present, present_loss, absent_loss).mean())

        return torch.stack(region_losses).mean()


class BraTSRegionAwareCompoundLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,
        region_auxiliary_weight: float = 0.35,
        absent_region_false_positive_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if region_auxiliary_weight < 0:
            raise ValueError("Region auxiliary weight must be non-negative.")
        self.base_loss = base_loss
        self.region_auxiliary_weight = float(region_auxiliary_weight)
        self.region_loss = BraTSRegionAuxiliaryLoss(
            absent_region_false_positive_weight=(
                absent_region_false_positive_weight
            )
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.base_loss(logits, target) + (
            self.region_auxiliary_weight * self.region_loss(logits, target)
        )
