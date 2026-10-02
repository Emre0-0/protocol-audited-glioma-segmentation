from __future__ import annotations

from itertools import product

import numpy as np
import torch

from .data import pad_image_and_seg


def _starts(size: int, patch: int, overlap: float) -> list[int]:
    if size <= patch:
        return [0]
    stride = max(1, int(round(patch * (1.0 - overlap))))
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


@torch.no_grad()
def sliding_window_logits(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: tuple[int, int, int],
    num_classes: int,
    overlap: float,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    original_shape = image.shape[-3:]
    image, _ = pad_image_and_seg(image, None, patch_size)
    _, x, y, z = image.shape
    px, py, pz = patch_size
    xs = _starts(x, px, overlap)
    ys = _starts(y, py, overlap)
    zs = _starts(z, pz, overlap)

    logits_sum = torch.zeros((num_classes, x, y, z), dtype=torch.float32, device=device)
    count = torch.zeros((1, x, y, z), dtype=torch.float32, device=device)

    for sx, sy, sz in product(xs, ys, zs):
        patch = image[:, sx : sx + px, sy : sy + py, sz : sz + pz]
        inp = torch.from_numpy(np.ascontiguousarray(patch[None], dtype=np.float32)).to(device)
        if amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(inp)[0].float()
        else:
            logits = model(inp)[0].float()
        logits_sum[:, sx : sx + px, sy : sy + py, sz : sz + pz] += logits
        count[:, sx : sx + px, sy : sy + py, sz : sz + pz] += 1.0

    logits_mean = logits_sum / torch.clamp(count, min=1.0)
    ox, oy, oz = original_shape
    return logits_mean[:, :ox, :oy, :oz]
