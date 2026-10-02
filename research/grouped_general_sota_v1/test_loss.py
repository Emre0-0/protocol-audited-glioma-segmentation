from __future__ import annotations

import torch
from torch import nn

from losses import BRATS_REGIONS, BraTSRegionAwareCompoundLoss


class SqueezedCrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, target[:, 0].long())


def logits_for_target(target: torch.Tensor) -> torch.Tensor:
    logits = torch.full(
        (target.shape[0], 5, *target.shape[2:]),
        -8.0,
        dtype=torch.float32,
    )
    logits.scatter_(1, target.long(), 8.0)
    return logits


def main() -> None:
    target = torch.zeros((2, 1, 12, 14, 10), dtype=torch.long)
    target[0, :, 2:9, 2:10, 2:8] = 2
    target[0, :, 4:8, 4:9, 3:7] = 1
    target[0, :, 5:7, 5:8, 4:6] = 3
    target[1, :, 3:10, 3:11, 2:9] = 2
    target[1, :, 5:8, 5:9, 4:7] = 4

    compound = BraTSRegionAwareCompoundLoss(
        base_loss=SqueezedCrossEntropyLoss(),
        region_auxiliary_weight=0.35,
        absent_region_false_positive_weight=0.25,
    )
    perfect = logits_for_target(target)
    wrong = torch.zeros_like(perfect)

    perfect_loss = compound(perfect, target)
    wrong_loss = compound(wrong, target)
    if not torch.isfinite(perfect_loss) or not torch.isfinite(wrong_loss):
        raise AssertionError("Region-aware loss returned a non-finite value.")
    if not perfect_loss < wrong_loss:
        raise AssertionError(
            f"Perfect prediction loss {perfect_loss.item():.6f} is not below "
            f"uninformative prediction loss {wrong_loss.item():.6f}."
        )

    train_logits = torch.randn_like(perfect, requires_grad=True)
    loss = compound(train_logits, target)
    loss.backward()
    if train_logits.grad is None or not torch.isfinite(train_logits.grad).all():
        raise AssertionError("Region-aware loss produced invalid gradients.")
    print(
        "PASS",
        {
            "regions": list(BRATS_REGIONS),
            "perfect_loss": round(perfect_loss.item(), 6),
            "uninformative_loss": round(wrong_loss.item(), 6),
            "gradient_norm": round(train_logits.grad.norm().item(), 6),
        },
    )


if __name__ == "__main__":
    main()
