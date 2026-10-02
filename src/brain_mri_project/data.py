from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


def as_tuple3(value: Iterable[int]) -> tuple[int, int, int]:
    items = tuple(int(v) for v in value)
    if len(items) != 3:
        raise ValueError(f"Expected a 3D size, got {items}")
    return items


def list_npz_files(root: str | Path, split: str | None = None) -> list[Path]:
    root = Path(root)
    search_root = root / split if split else root
    files = sorted(search_root.glob("*.npz"))
    if not files and split is None:
        files = sorted(root.rglob("*.npz"))
    return files


def list_pretrain_files(root: str | Path) -> list[Path]:
    return sorted(Path(root).rglob("*.npz"))


def pretrain_subject_id(path: str | Path) -> str:
    """Extract a stable person id from ADNI/OASIS longitudinal scan names."""
    path = Path(path)
    name = path.stem
    adni = re.search(r"(\d{3}_S_\d{4})", name)
    if adni:
        return f"ADNI:{adni.group(1)}"
    oasis = re.match(r"(OAS[12]_\d{4})_", name)
    if oasis:
        return f"{path.parent.name}:{oasis.group(1)}"
    return f"{path.parent.name}:{name}"


def load_brats_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=True) as data:
        image = data["image"].astype(np.float32)
        seg = data["seg"].astype(np.int64)
        case_id = str(data["case_id"].item()) if np.asarray(data["case_id"]).ndim == 0 else Path(path).stem
    return image, seg, case_id


def load_brats_seg_npz(path: str | Path) -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=True) as data:
        seg = data["seg"].astype(np.int64)
        case_id = str(data["case_id"].item()) if np.asarray(data["case_id"]).ndim == 0 else Path(path).stem
    return seg, case_id


def load_residual_npz(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if "residual" not in data:
            raise KeyError(f"Missing 'residual' key in {path}")
        residual = data["residual"].astype(np.float32)
    if residual.ndim == 3:
        residual = residual[None, ...]
    return residual


def load_pretrain_npz(path: str | Path) -> tuple[np.ndarray, str, str]:
    with np.load(path, allow_pickle=True) as data:
        image = data["image"].astype(np.float32)
        dataset = str(data["dataset"].item()) if "dataset" in data and np.asarray(data["dataset"]).ndim == 0 else "unknown"
        case_id = str(data["case_id"].item()) if "case_id" in data and np.asarray(data["case_id"]).ndim == 0 else Path(path).stem
    return image, dataset, case_id


def pad_image_and_seg(
    image: np.ndarray,
    seg: np.ndarray | None,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    spatial_shape = image.shape[-3:]
    pads = [max(p - s, 0) for s, p in zip(spatial_shape, patch_size)]
    if all(p == 0 for p in pads):
        return image, seg

    before = [p // 2 for p in pads]
    after = [p - b for p, b in zip(pads, before)]
    image_pad = [(0, 0)] * (image.ndim - 3) + list(zip(before, after))
    image = np.pad(image, image_pad, mode="constant", constant_values=0)

    if seg is not None:
        seg = np.pad(seg, list(zip(before, after)), mode="constant", constant_values=0)

    return image, seg


def random_crop(
    image: np.ndarray,
    seg: np.ndarray | None,
    patch_size: tuple[int, int, int],
    foreground_prob: float = 0.0,
    foreground_label_weights: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    image, seg = pad_image_and_seg(image, seg, patch_size)
    spatial_shape = image.shape[-3:]
    starts: list[int] = []

    use_foreground = seg is not None and np.any(seg > 0) and random.random() < foreground_prob
    if use_foreground:
        if foreground_label_weights:
            present = [label for label in foreground_label_weights if np.any(seg == label)]
            if present:
                weights = np.asarray([foreground_label_weights[label] for label in present], dtype=np.float64)
                weights = weights / weights.sum()
                label = int(np.random.choice(present, p=weights))
                candidates = np.argwhere(seg == label)
            else:
                candidates = np.argwhere(seg > 0)
        else:
            candidates = np.argwhere(seg > 0)
        center = candidates[np.random.randint(len(candidates))]
        starts = [int(c - p // 2) for c, p in zip(center, patch_size)]
    else:
        for size, patch in zip(spatial_shape, patch_size):
            starts.append(np.random.randint(0, size - patch + 1) if size > patch else 0)

    starts = [max(0, min(start, size - patch)) for start, size, patch in zip(starts, spatial_shape, patch_size)]
    sx, sy, sz = starts
    px, py, pz = patch_size
    cropped_image = image[..., sx : sx + px, sy : sy + py, sz : sz + pz]
    cropped_seg = None if seg is None else seg[sx : sx + px, sy : sy + py, sz : sz + pz]
    return cropped_image, cropped_seg


def random_flips(image: np.ndarray, seg: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    for axis in range(-3, 0):
        if random.random() < 0.5:
            image = np.flip(image, axis=axis)
            if seg is not None:
                seg = np.flip(seg, axis=axis)
    return image.copy(), None if seg is None else seg.copy()


def random_intensity_augment(
    image: np.ndarray,
    noise_std: float = 0.03,
    scale_range: tuple[float, float] = (0.9, 1.1),
    shift_range: tuple[float, float] = (-0.1, 0.1),
    gamma_range: tuple[float, float] = (0.85, 1.15),
    channel_dropout_prob: float = 0.05,
) -> np.ndarray:
    augmented = image.copy().astype(np.float32)
    channels = augmented.shape[0]
    for channel in range(channels):
        brain = augmented[channel] != 0
        if not np.any(brain):
            continue

        if random.random() < channel_dropout_prob and channels > 1:
            augmented[channel, brain] = 0.0
            continue

        scale = np.random.uniform(*scale_range)
        shift = np.random.uniform(*shift_range)
        augmented[channel, brain] = augmented[channel, brain] * scale + shift

        if random.random() < 0.3:
            values = augmented[channel, brain]
            min_v = float(values.min())
            max_v = float(values.max())
            if max_v > min_v:
                gamma = np.random.uniform(*gamma_range)
                normalized = (values - min_v) / (max_v - min_v + 1e-8)
                augmented[channel, brain] = np.power(normalized, gamma) * (max_v - min_v) + min_v

        if noise_std > 0:
            noise = np.random.normal(0.0, noise_std, size=int(brain.sum())).astype(np.float32)
            augmented[channel, brain] = augmented[channel, brain] + noise

    return augmented.astype(np.float32)


def make_masked_input(
    image: np.ndarray,
    mask_ratio: float,
    noise_std: float,
    block_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    masked = image.copy()
    mask = np.zeros_like(image, dtype=np.float32)

    _, x, y, z = image.shape
    total_blocks = max(1, int((x * y * z / (block_size**3)) * mask_ratio))
    for _ in range(total_blocks):
        sx = np.random.randint(0, max(1, x - block_size + 1))
        sy = np.random.randint(0, max(1, y - block_size + 1))
        sz = np.random.randint(0, max(1, z - block_size + 1))
        masked[:, sx : sx + block_size, sy : sy + block_size, sz : sz + block_size] = 0.0
        mask[:, sx : sx + block_size, sy : sy + block_size, sz : sz + block_size] = 1.0

    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, size=masked.shape).astype(np.float32)
        brain = image != 0
        masked = np.where(brain, masked + noise, masked).astype(np.float32)

    return masked.astype(np.float32), mask.astype(np.float32)


class BratsPatchDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        patches_per_epoch: int,
        patch_size: tuple[int, int, int],
        foreground_prob: float,
        augment: bool,
        extra_channel_root: str | Path | None = None,
        foreground_label_weights: dict[int, float] | None = None,
        intensity_augment: bool = True,
        case_sampling_weights: list[float] | np.ndarray | None = None,
    ) -> None:
        if not files:
            raise ValueError("No BraTS NPZ files were found.")
        self.files = list(files)
        self.patches_per_epoch = int(patches_per_epoch)
        self.patch_size = patch_size
        self.foreground_prob = float(foreground_prob)
        self.augment = augment
        self.extra_channel_root = None if extra_channel_root is None else Path(extra_channel_root)
        self.foreground_label_weights = foreground_label_weights
        self.intensity_augment = intensity_augment
        self.case_sampling_probs: np.ndarray | None = None
        if case_sampling_weights is not None:
            weights = np.asarray(case_sampling_weights, dtype=np.float64)
            if weights.shape != (len(self.files),):
                raise ValueError(f"Expected {len(self.files)} case weights, got {weights.shape}")
            weights = np.clip(weights, 0.0, None)
            if float(weights.sum()) <= 0.0:
                raise ValueError("Case sampling weights must contain at least one positive value.")
            self.case_sampling_probs = weights / weights.sum()

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if self.case_sampling_probs is None:
            path = self.files[index % len(self.files)]
        else:
            path = self.files[int(np.random.choice(len(self.files), p=self.case_sampling_probs))]
        image, seg, case_id = load_brats_npz(path)
        if self.extra_channel_root is not None:
            split = path.parent.name
            residual_path = self.extra_channel_root / split / f"{case_id}.npz"
            residual = load_residual_npz(residual_path)
            if residual.shape[-3:] != image.shape[-3:]:
                raise ValueError(f"Residual shape mismatch for {case_id}: {residual.shape} vs {image.shape}")
            image = np.concatenate([image, residual], axis=0)
        image, seg = random_crop(
            image,
            seg,
            self.patch_size,
            self.foreground_prob,
            foreground_label_weights=self.foreground_label_weights,
        )
        if self.augment:
            image, seg = random_flips(image, seg)
            if self.intensity_augment:
                image = random_intensity_augment(image)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            "seg": torch.from_numpy(np.ascontiguousarray(seg, dtype=np.int64)),
            "case_id": case_id,
        }


class NormalT1PatchDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        patches_per_epoch: int,
        patch_size: tuple[int, int, int],
        mask_ratio: float,
        noise_std: float,
        augment: bool,
    ) -> None:
        if not files:
            raise ValueError("No normal T1 NPZ files were found.")
        self.files = list(files)
        self.patches_per_epoch = int(patches_per_epoch)
        self.patch_size = patch_size
        self.mask_ratio = float(mask_ratio)
        self.noise_std = float(noise_std)
        self.augment = augment

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index % len(self.files)]
        image, dataset, case_id = load_pretrain_npz(path)
        image = image[None, ...]
        image, _ = random_crop(image, None, self.patch_size)
        if self.augment:
            image, _ = random_flips(image)
            image = random_intensity_augment(
                image,
                noise_std=0.02,
                scale_range=(0.95, 1.05),
                shift_range=(-0.05, 0.05),
                gamma_range=(0.9, 1.1),
                channel_dropout_prob=0.0,
            )
        masked, mask = make_masked_input(image, self.mask_ratio, self.noise_std)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(masked, dtype=np.float32)),
            "target": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)),
            "dataset": dataset,
            "case_id": case_id,
        }


def split_pretrain_files(files: list[Path], val_fraction: float = 0.15, seed: int = 42) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[Path]] = {}
    for path in files:
        by_dataset.setdefault(path.parent.name, []).append(path)

    train_files: list[Path] = []
    val_files: list[Path] = []
    for dataset_files in by_dataset.values():
        by_subject: dict[str, list[Path]] = defaultdict(list)
        for path in dataset_files:
            by_subject[pretrain_subject_id(path)].append(path)
        shuffled_subjects = sorted(by_subject)
        rng.shuffle(shuffled_subjects)
        val_subject_count = max(1, int(round(len(shuffled_subjects) * val_fraction)))
        val_subjects = set(shuffled_subjects[:val_subject_count])
        for subject, subject_files in by_subject.items():
            destination = val_files if subject in val_subjects else train_files
            destination.extend(sorted(subject_files))

    rng.shuffle(train_files)
    rng.shuffle(val_files)
    return train_files, val_files
