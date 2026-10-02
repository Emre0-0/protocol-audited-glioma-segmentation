from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd


REGIONS = ("WT", "TC", "NETC", "SNFH", "ET", "RC")
CASE_PATTERN = re.compile(r"^(BraTS-GLI-\d+)-(\d+)$")
COLORS = {
    "ink": "#202428",
    "blue": "#277DA1",
    "green": "#4D908E",
    "gold": "#F9C74F",
    "red": "#E45756",
    "gray": "#8A939B",
    "light": "#E8ECEF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build corrected publication figures.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--statistics",
        type=Path,
        default=Path(
            "experiments/publication_integrity_v1/statistics/publication_statistics.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_results_assets/corrected_figures"),
    )
    return parser.parse_args()


def subject_id(case_id: str) -> str:
    match = CASE_PATTERN.fullmatch(case_id)
    if match is None:
        raise ValueError(case_id)
    return match.group(1)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(colors=COLORS["ink"], labelsize=9)
    axis.xaxis.label.set_color(COLORS["ink"])
    axis.yaxis.label.set_color(COLORS["ink"])
    axis.title.set_color(COLORS["ink"])


def signed_value(value: float, digits: int = 3) -> str:
    """Format signed effects without displaying a misleading negative zero."""
    rounded = round(float(value), digits)
    if rounded == 0:
        return f"{0:.{digits}f}"
    return f"{rounded:+.{digits}f}"


def compact_effect(value: float, digits: int = 4, *, zero_label: bool = True) -> str:
    """Format effects without hiding real near-zero values."""
    value = float(value)
    if value == 0:
        return "0 (unchanged)" if zero_label else f"{0:.{digits}f}"
    if abs(value) < 10 ** (-digits):
        mantissa, exponent = f"{value:+.2e}".split("e")
        return f"{mantissa}e{int(exponent):+d}"
    return f"{value:+.{digits}f}"


def compact_interval(lower: float, upper: float, digits: int = 4) -> str:
    return (
        f"[{compact_effect(lower, digits, zero_label=False)}, "
        f"{compact_effect(upper, digits, zero_label=False)}]"
    )


def style_value_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_xticks([])
    axis.tick_params(left=False, labelleft=False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def validate_interval(point: float, lower: float, upper: float, label: str) -> None:
    values = np.asarray([point, lower, upper], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite value in {label}: {values.tolist()}")
    if lower > point or point > upper:
        raise ValueError(
            f"Point estimate outside interval for {label}: "
            f"{lower} <= {point} <= {upper}"
        )


def protocol_overlap_figure(root: Path, output_dir: Path) -> None:
    legacy = pd.read_csv(root / "_archive_inspect/eda_outputs/eda_outputs/brats2024_split.csv")
    legacy["subject_id"] = legacy["case_id"].astype(str).map(subject_id)
    subject_sets = {
        split: set(group["subject_id"].astype(str))
        for split, group in legacy.groupby("split")
    }
    labels = ["Train-val", "Train-test", "Val-test"]
    legacy_values = [
        len(subject_sets["train"] & subject_sets["val"]),
        len(subject_sets["train"] & subject_sets["test"]),
        len(subject_sets["val"] & subject_sets["test"]),
    ]
    if legacy_values != [129, 132, 36]:
        raise ValueError(f"Unexpected legacy overlap counts: {legacy_values}")

    y = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(7.2, 3.45))
    axis.hlines(
        y,
        xmin=0,
        xmax=legacy_values,
        color=COLORS["light"],
        linewidth=3,
        zorder=1,
    )
    axis.scatter(
        legacy_values,
        y,
        s=70,
        marker="o",
        color=COLORS["red"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    axis.scatter(
        np.zeros(len(labels)),
        y,
        s=78,
        marker="D",
        color=COLORS["green"],
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        clip_on=False,
    )
    for position, legacy_value in zip(y, legacy_values, strict=True):
        axis.text(
            legacy_value + 2.4,
            position,
            str(legacy_value),
            va="center",
            ha="left",
            fontsize=9,
            color=COLORS["ink"],
        )
        axis.text(
            3.2,
            position,
            "0 (none)",
            va="center",
            ha="left",
            fontsize=8,
            color=COLORS["green"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
            zorder=5,
        )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(-6, max(legacy_values) + 18)
    axis.set_xlabel("Number of shared subjects")
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLORS["red"],
                markeredgecolor="white",
                markersize=8,
                label="Legacy case-level split",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=COLORS["green"],
                markeredgecolor="white",
                markersize=8,
                label="Corrected protocol (zero overlap)",
            ),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        fontsize=8,
    )
    axis.grid(axis="x", color=COLORS["light"], linewidth=0.8)
    style_axis(axis)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_dir, "protocol_subject_overlap")


def grouped_oof_figure(statistics: dict[str, Any], output_dir: Path) -> None:
    oof = statistics["grouped_oof"]
    bootstrap = oof["case_weighted_cluster_bootstrap"]
    keys = [f"{region}_LesionWise_Score_Dice" for region in REGIONS] + [
        "six_region_mean"
    ]
    labels = list(REGIONS) + ["Mean"]
    means = np.array([bootstrap[key]["mean"] for key in keys])
    lower = np.array([bootstrap[key]["ci95_lower"] for key in keys])
    upper = np.array([bootstrap[key]["ci95_upper"] for key in keys])
    for label, point, low, high in zip(labels, means, lower, upper, strict=True):
        validate_interval(point, low, high, f"OOF Dice {label}")
    y = np.arange(len(labels))
    fig = plt.figure(figsize=(7.4, 4.85))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.5, 2.25], wspace=0.06)
    axis = fig.add_subplot(grid[0, 0])
    value_axis = fig.add_subplot(grid[0, 1], sharey=axis)
    axis.axhspan(len(labels) - 1.5, len(labels) - 0.5, color=COLORS["light"], alpha=0.45)
    axis.errorbar(
        means,
        y,
        xerr=np.vstack([means - lower, upper - means]),
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["gray"],
        capsize=3,
        markersize=6,
    )
    axis.set_ylim(len(labels) - 0.5, -0.72)
    axis.set_yticks(y, labels)
    axis.set_xlim(0.64, 0.85)
    axis.set_xlabel("Official lesion-wise Dice")
    axis.grid(axis="x", color=COLORS["light"], linewidth=0.8)
    style_axis(axis)
    style_value_axis(value_axis)
    value_axis.text(
        0.5,
        1.015,
        "Estimate [subject-cluster 95% CI]",
        transform=value_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color=COLORS["ink"],
    )
    for position, value, low, high in zip(y, means, lower, upper, strict=True):
        value_axis.text(
            0.5,
            position,
            f"{value:.3f} [{low:.3f}, {high:.3f}]",
            ha="center",
            va="center",
            fontsize=8.2,
            color=COLORS["ink"],
        )
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.11, right=0.99)
    save_figure(fig, output_dir, "grouped_oof_dice")


def locked_test_forest(statistics: dict[str, Any], output_dir: Path) -> None:
    comparison = statistics["locked_test"]["case_weighted_subject_cluster_bootstrap"]
    labels = list(REGIONS) + ["Six-region mean"]
    deltas = np.array([comparison[label]["delta"] for label in labels])
    lower = np.array([comparison[label]["delta_ci95_lower"] for label in labels])
    upper = np.array([comparison[label]["delta_ci95_upper"] for label in labels])
    for label, point, low, high in zip(labels, deltas, lower, upper, strict=True):
        validate_interval(point, low, high, f"holdout delta {label}")
    y = np.arange(len(labels))
    fig = plt.figure(figsize=(7.4, 4.9))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.3, 3.15], wspace=0.05)
    axis = fig.add_subplot(grid[0, 0])
    value_axis = fig.add_subplot(grid[0, 1], sharey=axis)
    axis.axhspan(len(labels) - 1.5, len(labels) - 0.5, color=COLORS["light"], alpha=0.45)
    for index in range(len(labels)):
        practically_unchanged = abs(deltas[index]) < 0.0001
        axis.errorbar(
            deltas[index],
            y[index],
            xerr=np.array(
                [[deltas[index] - lower[index]], [upper[index] - deltas[index]]]
            ),
            fmt="D" if practically_unchanged else "o",
            color=COLORS["gray"] if practically_unchanged else COLORS["blue"],
            ecolor=COLORS["gray"],
            capsize=3,
            markersize=6,
        )
    axis.axvline(0, color="#AAB2B9", linewidth=1, linestyle=(0, (4, 3)), zorder=0)
    axis.text(
        0,
        -0.55,
        "No change",
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLORS["gray"],
    )
    axis.set_yticks(y, labels)
    axis.set_ylim(len(labels) - 0.5, -0.72)
    axis.set_xlim(-0.020, 0.040)
    axis.set_xlabel("Frozen V2 minus baseline lesion-wise Dice")
    axis.grid(axis="x", color=COLORS["light"], linewidth=0.8)
    style_axis(axis)
    style_value_axis(value_axis)
    value_axis.text(
        0.5,
        1.015,
        "Delta [subject-cluster 95% CI]",
        transform=value_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color=COLORS["ink"],
    )
    for position, value, low, high in zip(y, deltas, lower, upper, strict=True):
        value_text = (
            "0 (exactly unchanged)"
            if value == 0 and low == 0 and high == 0
            else f"{compact_effect(value)} {compact_interval(low, high)}"
        )
        value_axis.text(
            0.5,
            position,
            value_text,
            ha="center",
            va="center",
            fontsize=8.0,
            color=COLORS["ink"],
        )
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.12, right=0.995)
    save_figure(fig, output_dir, "locked_test_delta_forest")


def small_lesion_tradeoff(statistics: dict[str, Any], output_dir: Path) -> None:
    subgroups = statistics["locked_test"]["exploratory_post_hoc_subgroups"]
    wanted = [
        "ET-positive bottom 10%",
        "ET-positive bottom 25%",
        "ET-negative",
        "RC-positive bottom 10%",
    ]
    selected = []
    for prefix in wanted:
        selected.append(next(item for item in subgroups if item["subgroup"].startswith(prefix)))
    labels = ["Tiny ET 10%", "Tiny ET 25%", "ET-negative", "Tiny RC 10%"]
    deltas = np.array([item["paired_delta"] for item in selected])
    lower = np.array([item["delta_ci95_lower"] for item in selected])
    upper = np.array([item["delta_ci95_upper"] for item in selected])
    for label, point, low, high in zip(labels, deltas, lower, upper, strict=True):
        validate_interval(point, low, high, f"small-lesion delta {label}")
    colors = [COLORS["green"] if value > 0 else COLORS["red"] for value in deltas]
    y = np.arange(len(labels))
    fig = plt.figure(figsize=(7.4, 4.45))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.1, 2.9], wspace=0.05)
    axis = fig.add_subplot(grid[0, 0])
    value_axis = fig.add_subplot(grid[0, 1], sharey=axis)
    for index in range(len(labels)):
        unchanged = deltas[index] == 0 and lower[index] == 0 and upper[index] == 0
        axis.errorbar(
            deltas[index],
            y[index],
            xerr=np.array(
                [[deltas[index] - lower[index]], [upper[index] - deltas[index]]]
            ),
            fmt="D" if unchanged else "o",
            color=COLORS["gray"] if unchanged else colors[index],
            ecolor=COLORS["gray"],
            capsize=3,
            markersize=6,
            linewidth=1.2,
        )
    axis.axvline(0, color="#AAB2B9", linewidth=1, linestyle=(0, (4, 3)), zorder=0)
    axis.text(
        0,
        -0.55,
        "No change",
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLORS["gray"],
    )
    axis.set_yticks(y, labels)
    axis.set_ylim(len(labels) - 0.5, -0.72)
    axis.set_xlim(-0.30, 0.30)
    axis.set_xlabel("Paired lesion-wise Dice change")
    axis.grid(axis="x", color=COLORS["light"], linewidth=0.8)
    style_axis(axis)
    style_value_axis(value_axis)
    value_axis.text(
        0.5,
        1.015,
        "Delta [subject-cluster 95% CI]",
        transform=value_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color=COLORS["ink"],
    )
    for position, value, low, high in zip(y, deltas, lower, upper, strict=True):
        value_text = (
            "0 (exactly unchanged)"
            if value == 0 and low == 0 and high == 0
            else f"{compact_effect(value, 3)} {compact_interval(low, high, 3)}"
        )
        value_axis.text(
            0.5,
            position,
            value_text,
            ha="center",
            va="center",
            fontsize=8.2,
            color=COLORS["ink"],
        )
    fig.subplots_adjust(top=0.94, bottom=0.17, left=0.13, right=0.995)
    save_figure(fig, output_dir, "locked_test_small_lesion_tradeoff")


def grouped_lesion_detection(statistics: dict[str, Any], output_dir: Path) -> None:
    detection = statistics["grouped_oof"]["lesion_detection"]
    labels = list(REGIONS)
    sensitivity = np.array([detection[label]["sensitivity"] for label in labels])
    precision = np.array([detection[label]["precision"] for label in labels])
    sensitivity_error = np.vstack(
        [
            sensitivity
            - np.array(
                [detection[label]["sensitivity_ci95_lower"] for label in labels]
            ),
            np.array(
                [detection[label]["sensitivity_ci95_upper"] for label in labels]
            )
            - sensitivity,
        ]
    )
    precision_error = np.vstack(
        [
            precision
            - np.array([detection[label]["precision_ci95_lower"] for label in labels]),
            np.array([detection[label]["precision_ci95_upper"] for label in labels])
            - precision,
        ]
    )
    for index, label in enumerate(labels):
        validate_interval(
            sensitivity[index],
            sensitivity[index] - sensitivity_error[0, index],
            sensitivity[index] + sensitivity_error[1, index],
            f"lesion sensitivity {label}",
        )
        validate_interval(
            precision[index],
            precision[index] - precision_error[0, index],
            precision[index] + precision_error[1, index],
            f"lesion precision {label}",
        )

    fig = plt.figure(figsize=(7.4, 5.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.8, 1.45], wspace=0.04)
    axis = fig.add_subplot(grid[0, 0])
    value_axis = fig.add_subplot(grid[0, 1], sharey=axis)
    y = np.arange(len(labels))
    offset = 0.13
    axis.errorbar(
        sensitivity,
        y - offset,
        xerr=sensitivity_error,
        fmt="o",
        color=COLORS["green"],
        ecolor=COLORS["green"],
        capsize=2.5,
        markersize=5.5,
        linewidth=1.2,
        label="Lesion sensitivity (95% CI)",
    )
    axis.errorbar(
        precision,
        y + offset,
        xerr=precision_error,
        fmt="s",
        color=COLORS["blue"],
        ecolor=COLORS["blue"],
        capsize=2.5,
        markersize=5.2,
        linewidth=1.2,
        label="Lesion precision (95% CI)",
    )
    axis.set_yticks(y, labels)
    axis.set_xlim(0.75, 1.0)
    axis.set_xlabel("Detection metric")
    axis.invert_yaxis()
    status = statistics["grouped_oof"]["status"].lower()
    if status != "complete":
        raise ValueError(f"OOF lesion-detection status is not complete: {status}")
    fig.legend(
        handles=axis.get_legend_handles_labels()[0],
        labels=axis.get_legend_handles_labels()[1],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        fontsize=9,
    )
    axis.grid(axis="x", color=COLORS["light"], linewidth=0.8)
    style_axis(axis)
    value_axis.set_xlim(0, 1)
    value_axis.set_xticks([])
    value_axis.tick_params(left=False, labelleft=False)
    for spine in value_axis.spines.values():
        spine.set_visible(False)
    value_axis.text(
        0.24,
        1.015,
        "Sensitivity",
        transform=value_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COLORS["green"],
    )
    value_axis.text(
        0.76,
        1.015,
        "Precision",
        transform=value_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COLORS["blue"],
    )
    for position, sens_value, precision_value in zip(
        y, sensitivity, precision, strict=True
    ):
        value_axis.text(
            0.24,
            position,
            f"{sens_value:.3f}",
            ha="center",
            va="center",
            fontsize=9.5,
            color=COLORS["ink"],
        )
        value_axis.text(
            0.76,
            position,
            f"{precision_value:.3f}",
            ha="center",
            va="center",
            fontsize=9.5,
            color=COLORS["ink"],
        )
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.11, right=0.98)
    save_figure(fig, output_dir, "grouped_oof_lesion_detection")


def robust_slice(image: np.ndarray, z_index: int) -> np.ndarray:
    plane = np.asarray(image[:, :, z_index], dtype=np.float32)
    nonzero = plane[np.isfinite(plane) & (plane != 0)]
    if nonzero.size:
        low, high = np.percentile(nonzero, [1, 99])
    else:
        low, high = float(np.nanmin(plane)), float(np.nanmax(plane))
    if high <= low:
        normalized = np.zeros_like(plane)
    else:
        normalized = np.clip((plane - low) / (high - low), 0, 1)
    return np.rot90(normalized)


def mask_slice(mask: np.ndarray, z_index: int) -> np.ndarray:
    return np.rot90(np.asarray(mask[:, :, z_index], dtype=np.uint8))


def choose_qualitative_cases(root: Path) -> list[dict[str, Any]]:
    metrics_path = root / "experiments/grouped_delta_roi_v1/official_metrics/baseline/case_metrics_official.csv"
    metrics = pd.read_csv(metrics_path)
    dice_columns = [f"{region}_LesionWise_Score_Dice" for region in REGIONS]
    metrics["macro"] = metrics[dice_columns].mean(axis=1)
    volumes = pd.read_csv(
        root / "processed/brats_subject_grouped_protocol_v1/development_cases.csv"
    ).set_index("case_id")
    high = metrics.sort_values("macro", ascending=False).iloc[0]
    low = metrics.sort_values("macro", ascending=True).iloc[0]
    et_positive = metrics[
        metrics["case_id"].map(volumes["label_3_voxels"]).astype(float) > 0
    ].copy()
    et_threshold = et_positive["case_id"].map(volumes["label_3_voxels"]).quantile(0.25)
    tiny_et = et_positive[
        et_positive["case_id"].map(volumes["label_3_voxels"]) <= et_threshold
    ].sort_values("ET_LesionWise_Score_Dice").iloc[0]
    rc_positive = metrics[
        metrics["case_id"].map(volumes["label_4_voxels"]).astype(float) > 0
    ].copy()
    rc_threshold = rc_positive["case_id"].map(volumes["label_4_voxels"]).quantile(0.25)
    tiny_rc = rc_positive[
        rc_positive["case_id"].map(volumes["label_4_voxels"]) <= rc_threshold
    ].sort_values("RC_LesionWise_Score_Dice").iloc[0]
    return [
        {"label": "High overlap", "row": high, "target": 0},
        {"label": "Low overlap", "row": low, "target": 0},
        {"label": "Tiny ET failure", "row": tiny_et, "target": 3},
        {"label": "Tiny RC failure", "row": tiny_rc, "target": 4},
    ]


def qualitative_figure(root: Path, output_dir: Path) -> None:
    image_dir = root / "processed/nnunet_raw/Dataset503_BraTS2024Grouped4Ch/imagesTr"
    label_dir = root / "processed/nnunet_raw/Dataset503_BraTS2024Grouped4Ch/labelsTr"
    prediction_dir = root / (
        "experiments/nnunet_results/Dataset503_BraTS2024Grouped4Ch/"
        "nnUNetTrainer_250epochs__nnUNetResEncUNet10p5GBPlans__3d_fullres/"
        "fold_0/validation"
    )
    cases = choose_qualitative_cases(root)
    label_cmap = ListedColormap(["#000000", "#E45756", "#4D908E", "#F9C74F", "#277DA1"])
    error_cmap = ListedColormap(["#000000", "#E45756", "#277DA1", "#F9C74F"])
    fig, axes = plt.subplots(len(cases), 5, figsize=(12.0, 9.2))
    column_titles = [
        "T1 contrast",
        "T2-FLAIR",
        "Reference standard",
        "Prediction",
        "Error",
    ]
    for axis, title in zip(axes[0], column_titles, strict=True):
        axis.set_title(title, fontsize=10, color=COLORS["ink"], pad=6)
    for row_index, item in enumerate(cases):
        case_id = str(item["row"]["case_id"])
        gt = np.asanyarray(nib.load(str(label_dir / f"{case_id}.nii.gz")).dataobj).astype(np.uint8)
        pred = np.asanyarray(nib.load(str(prediction_dir / f"{case_id}.nii.gz")).dataobj).astype(np.uint8)
        target = int(item["target"])
        target_mask = gt == target if target else gt > 0
        z_counts = target_mask.sum(axis=(0, 1))
        if z_counts.max() == 0:
            z_counts = (gt > 0).sum(axis=(0, 1))
        z_index = int(np.argmax(z_counts))
        t1c = np.asanyarray(
            nib.load(str(image_dir / f"{case_id}_0001.nii.gz")).dataobj
        )
        t2f = np.asanyarray(
            nib.load(str(image_dir / f"{case_id}_0003.nii.gz")).dataobj
        )
        gt_plane = mask_slice(gt, z_index)
        pred_plane = mask_slice(pred, z_index)
        error = np.zeros_like(gt_plane, dtype=np.uint8)
        error[(gt_plane == 0) & (pred_plane > 0)] = 1
        error[(gt_plane > 0) & (pred_plane == 0)] = 2
        error[(gt_plane > 0) & (pred_plane > 0) & (gt_plane != pred_plane)] = 3
        axes[row_index, 0].imshow(robust_slice(t1c, z_index), cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].imshow(robust_slice(t2f, z_index), cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].imshow(gt_plane, cmap=label_cmap, vmin=0, vmax=4, interpolation="nearest")
        axes[row_index, 3].imshow(pred_plane, cmap=label_cmap, vmin=0, vmax=4, interpolation="nearest")
        axes[row_index, 4].imshow(error, cmap=error_cmap, vmin=0, vmax=3, interpolation="nearest")
        macro = float(item["row"]["macro"])
        metric = (
            macro
            if target == 0
            else float(item["row"][f"{'ET' if target == 3 else 'RC'}_LesionWise_Score_Dice"])
        )
        metric_name = "mean" if target == 0 else ("ET" if target == 3 else "RC")
        axes[row_index, 0].set_ylabel(
            f"{item['label']}\n{metric_name} Dice={metric:.3f}",
            fontsize=9,
            color=COLORS["ink"],
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    fig.suptitle(
        "Subject-disjoint fold-0 examples (error: red FP, blue FN, yellow wrong label)",
        fontsize=12,
        color=COLORS["ink"],
        y=0.995,
    )
    legend_items = [
        Patch(facecolor="#E45756", label="NETC (1)"),
        Patch(facecolor="#4D908E", label="SNFH (2)"),
        Patch(facecolor="#F9C74F", label="ET (3)"),
        Patch(facecolor="#277DA1", label="RC (4)"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.975), h_pad=1.0, w_pad=0.4)
    save_figure(fig, output_dir, "grouped_fold0_qualitative_examples")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    statistics_path = args.statistics
    if not statistics_path.is_absolute():
        statistics_path = root / statistics_path
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    protocol_overlap_figure(root, output_dir)
    grouped_oof_figure(statistics, output_dir)
    grouped_lesion_detection(statistics, output_dir)
    locked_test_forest(statistics, output_dir)
    small_lesion_tradeoff(statistics, output_dir)
    qualitative_figure(root, output_dir)
    print(json.dumps({
        "output_dir": str(output_dir),
        "figures": sorted(path.name for path in output_dir.glob("*.png")),
    }, indent=2))


if __name__ == "__main__":
    main()
