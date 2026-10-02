from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLossMultiClass(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        include_background: bool = False,
        eps: float = 1e-6,
        class_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.eps = eps
        self.register_buffer("class_weight", None if class_weight is None else class_weight.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target.long(), num_classes=self.num_classes)
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target_oh, dim=dims)
        denominator = torch.sum(probs + target_oh, dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        weight = self.class_weight
        if not self.include_background:
            dice = dice[1:]
            weight = None if weight is None else weight[1:]
        if weight is not None:
            weight = torch.clamp(weight.to(dice.device), min=0.0)
            denom = torch.clamp(weight.sum(), min=self.eps)
            return 1.0 - torch.sum(dice * weight) / denom
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        ce_weight: torch.Tensor | None = None,
        dice_weight: torch.Tensor | None = None,
        dice_loss_weight: float = 1.0,
        ce_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice = DiceLossMultiClass(num_classes=num_classes, include_background=False, class_weight=dice_weight)
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_loss_weight = float(dice_loss_weight)
        self.ce_loss_weight = float(ce_loss_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dice_loss = self.dice(logits, target)
        ce_loss = self.ce(logits, target.long())
        loss = self.dice_loss_weight * dice_loss + self.ce_loss_weight * ce_loss
        return loss, dice_loss.detach(), ce_loss.detach()


def masked_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    brain = (target != 0).float()
    weighted_mask = torch.clamp(mask + 0.1 * brain, 0.0, 1.0)
    denom = torch.clamp(weighted_mask.sum(), min=1.0)
    return torch.sum(torch.abs(pred - target) * weighted_mask) / denom


@torch.no_grad()
def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> np.ndarray:
    pred = torch.argmax(logits, dim=1)
    return dice_from_prediction(pred, target, num_classes=num_classes, eps=eps)


@torch.no_grad()
def dice_from_prediction(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> np.ndarray:
    values: list[float] = []
    for cls in range(num_classes):
        pred_c = pred == cls
        target_c = target == cls
        denom = pred_c.sum().float() + target_c.sum().float()
        if denom.item() == 0:
            values.append(float("nan"))
        else:
            inter = (pred_c & target_c).sum().float()
            values.append(float(((2.0 * inter + eps) / (denom + eps)).item()))
    return np.asarray(values, dtype=np.float64)
