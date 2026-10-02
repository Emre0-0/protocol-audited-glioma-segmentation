from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np


SUBJECT_PATTERN = re.compile(r"^(BraTS-GLI-\d+)-\d+$")
EXPECTED_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
EXPECTED_LABELS = {0, 1, 2, 3, 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream every BraTS NPZ file and audit numeric, label, modality, and image-mask integrity."
    )
    parser.add_argument("--root", default="processed/brats2024_npz_full")
    parser.add_argument("--output-dir", default="experiments/brats_npz_content_audit_20260716")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def scalar_text(value: np.ndarray | str) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    return str(value)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        (split, path)
        for split in ("train", "val", "test")
        for path in sorted((root / split).glob("*.npz"))
    ]
    if not paths:
        raise FileNotFoundError(f"No BraTS NPZ files found under {root}")

    rows: list[dict[str, object]] = []
    errors: list[str] = []
    warnings: list[str] = []
    label_presence: Counter[int] = Counter()
    shape_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    started = time.perf_counter()

    for index, (split, path) in enumerate(paths, start=1):
        case_errors: list[str] = []
        case_warnings: list[str] = []
        try:
            with np.load(path, allow_pickle=True) as data:
                required = {"image", "seg", "case_id", "modalities"}
                missing = sorted(required - set(data.files))
                if missing:
                    raise KeyError(f"missing keys: {missing}")
                image = data["image"].astype(np.float32, copy=False)
                seg = data["seg"].astype(np.uint8, copy=False)
                embedded_case_id = scalar_text(data["case_id"])
                modalities = tuple(str(value) for value in np.asarray(data["modalities"]).tolist())

            if embedded_case_id != path.stem:
                case_errors.append(
                    f"embedded case_id {embedded_case_id!r} differs from filename {path.stem!r}"
                )
            if modalities != EXPECTED_MODALITIES:
                case_errors.append(f"modalities are {modalities}, expected {EXPECTED_MODALITIES}")
            modality_counts[str(modalities)] += 1
            if image.ndim != 4 or image.shape[0] != 4:
                case_errors.append(f"image shape is {image.shape}, expected four channels")
            if seg.shape != image.shape[-3:]:
                case_errors.append(f"seg shape {seg.shape} does not match image {image.shape[-3:]}")
            if not np.isfinite(image).all():
                case_errors.append("image contains NaN or infinity")

            labels = {int(value) for value in np.unique(seg)}
            invalid_labels = sorted(labels - EXPECTED_LABELS)
            if invalid_labels:
                case_errors.append(f"unexpected labels: {invalid_labels}")
            for label in labels - {0}:
                label_presence[label] += 1
            if not labels - {0}:
                case_errors.append("segmentation contains no foreground")

            channel_nonzero_counts: list[int] = []
            channel_means: list[float] = []
            channel_stds: list[float] = []
            for channel in range(min(4, image.shape[0])):
                values = image[channel][image[channel] != 0]
                channel_nonzero_counts.append(int(values.size))
                if not values.size:
                    channel_means.append(float("nan"))
                    channel_stds.append(float("nan"))
                    case_errors.append(f"channel {channel} has no nonzero voxels")
                    continue
                mean = float(values.mean())
                std = float(values.std())
                channel_means.append(mean)
                channel_stds.append(std)
                if abs(mean) > 1e-4:
                    case_warnings.append(f"channel {channel} nonzero mean is {mean:.6g}")
                if not (0.999 <= std <= 1.001):
                    case_warnings.append(f"channel {channel} nonzero std is {std:.6g}")

            foreground = seg > 0
            foreground_voxels = int(foreground.sum())
            all_channel_support = np.all(image != 0, axis=0)
            any_channel_support = np.any(image != 0, axis=0)
            outside_all_channels = int((foreground & ~all_channel_support).sum())
            outside_any_channel = int((foreground & ~any_channel_support).sum())
            if outside_any_channel:
                case_errors.append(
                    f"{outside_any_channel} foreground voxels lie outside every modality support"
                )
            elif outside_all_channels:
                case_warnings.append(
                    f"{outside_all_channels} foreground voxels are zero in at least one modality"
                )

            pairwise_identical = []
            if image.shape[0] == 4:
                for left in range(4):
                    for right in range(left + 1, 4):
                        if np.array_equal(image[left], image[right]):
                            pairwise_identical.append(f"{left}-{right}")
            if pairwise_identical:
                case_errors.append(f"identical modality arrays: {pairwise_identical}")

            row = {
                "case_id": path.stem,
                "subject_id": (
                    SUBJECT_PATTERN.fullmatch(path.stem).group(1)
                    if SUBJECT_PATTERN.fullmatch(path.stem)
                    else path.stem
                ),
                "split": split,
                "image_shape": str(tuple(image.shape)),
                "seg_shape": str(tuple(seg.shape)),
                "modalities": ";".join(modalities),
                "labels": ";".join(str(value) for value in sorted(labels)),
                "foreground_voxels": foreground_voxels,
                "foreground_outside_all_channel_support": outside_all_channels,
                "foreground_outside_any_channel_support": outside_any_channel,
                "channel_nonzero_counts": ";".join(str(value) for value in channel_nonzero_counts),
                "channel_nonzero_means": ";".join(f"{value:.8g}" for value in channel_means),
                "channel_nonzero_stds": ";".join(f"{value:.8g}" for value in channel_stds),
                "errors": " | ".join(case_errors),
                "warnings": " | ".join(case_warnings),
            }
            rows.append(row)
            shape_counts[str(tuple(image.shape))] += 1
            split_counts[split] += 1
            errors.extend(f"{path.stem}: {message}" for message in case_errors)
            warnings.extend(f"{path.stem}: {message}" for message in case_warnings)
        except Exception as exc:
            message = f"{path.stem}: failed to read/audit: {exc!r}"
            errors.append(message)
            rows.append(
                {
                    "case_id": path.stem,
                    "subject_id": path.stem,
                    "split": split,
                    "errors": message,
                    "warnings": "",
                }
            )

        if index == 1 or index % max(1, args.progress_every) == 0 or index == len(paths):
            elapsed = time.perf_counter() - started
            rate = index / max(elapsed, 1e-8)
            eta = (len(paths) - index) / max(rate, 1e-8)
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(paths),
                        "errors": len(errors),
                        "warnings": len(warnings),
                        "elapsed_minutes": round(elapsed / 60.0, 2),
                        "eta_minutes": round(eta / 60.0, 2),
                    }
                ),
                flush=True,
            )

    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "case_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "PASS" if not errors else "FAIL",
        "root": str(root),
        "cases": len(paths),
        "split_counts": dict(split_counts),
        "shape_counts": dict(shape_counts),
        "modality_metadata_counts": dict(modality_counts),
        "label_presence_case_counts": {str(key): value for key, value in sorted(label_presence.items())},
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:200],
        "warnings": warnings[:200],
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# BraTS NPZ content audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Cases: {report['cases']}",
        f"- Splits: {report['split_counts']}",
        f"- Image shapes: {report['shape_counts']}",
        f"- Label presence: {report['label_presence_case_counts']}",
        f"- Numeric/content errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        f"- Elapsed: {report['elapsed_minutes']:.2f} minutes",
        "",
        "This audit checks every NPZ payload for finite values, z-score statistics, label range, modality metadata, image-mask support, and duplicated modality arrays. Subject-level leakage is reported separately.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
