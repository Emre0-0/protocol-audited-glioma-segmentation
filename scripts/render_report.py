"""Render stored aggregate estimates, not new inference or confidence intervals."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verify_release import ROOT, REGIONS, read_csv, verify_statistics


def format_estimate(value, effect=False):
    if value == 0:
        return "0"
    if abs(value) < .00005:
        return f"{value:+.2e}" if effect else f"{value:.2e}"
    return f"{value:+.4f}" if effect else f"{value:.4f}"


def forest(ax, rows, names, xlabel, mean="mean", low="ci95_lower", high="ci95_upper", zero=False):
    centers = np.array([float(r[mean]) for r in rows])
    lower = np.array([float(r[low]) for r in rows])
    upper = np.array([float(r[high]) for r in rows])
    y = np.arange(len(rows))
    ax.errorbar(centers, y, xerr=np.vstack((centers - lower, upper - centers)), fmt="o", color="#287d98", ecolor="#666666", capsize=3, markersize=5, linewidth=1.2)
    ax.set_yticks(y, names)
    ax.set_ylim(len(rows) - .5, -.8)
    ax.set_xlabel(xlabel, fontsize=10)
    span = max(float(upper.max() - lower.min()), .01)
    ax.set_xlim(float(lower.min()) - span * .12, float(upper.max()) + span * .12)
    ax.grid(axis="x", color="#e7e7e7")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if zero:
        ax.axvline(0, color="#333333", linewidth=.8)
    transform = ax.get_yaxis_transform()
    ax.text(1.05, -.65, "Estimate [95% CI]", transform=transform, fontsize=9, va="center")
    for position, m, l, u in zip(y, centers, lower, upper):
        label = f"{format_estimate(m, zero)} [{format_estimate(l, zero)}, {format_estimate(u, zero)}]"
        ax.text(1.05, position, label, transform=transform, fontsize=9, va="center")


def save(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)


def markdown_table(rows):
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        cells = []
        for key in keys:
            value = row[key]
            try:
                value = f"{float(value):.6g}"
            except ValueError:
                pass
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/review")
    args = parser.parse_args()
    verify_statistics()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "experiments/publication_integrity_v1/statistics"
    oof = read_csv(source / "grouped_oof_summary.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    for name, rows, labels, axis in (
        ("oof_dice", oof[:6] + [oof[-1]], list(REGIONS) + ["Six-region mean"], "Lesion-wise Dice (higher is better)"),
        ("oof_hd95", read_csv(source / "grouped_oof_hd95.csv"), REGIONS, "Lesion-wise HD95, mm (lower is better)"),
    ):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.subplots_adjust(left=.16, right=.56, top=.83, bottom=.16)
        forest(ax, rows, labels, axis)
        fig.suptitle("Subject-disjoint OOF: 1,155 scans / 521 subjects", y=.98, fontsize=12)
        fig.text(.16, .88, "Horizontal whiskers: 95% subject-cluster confidence intervals", fontsize=9)
        save(fig, output, name)
    holdout = read_csv(source / "locked_test_comparison.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.subplots_adjust(left=.16, right=.56, top=.83, bottom=.16)
    forest(ax, holdout, [r["metric"].replace("six_region_mean", "Six-region mean") for r in holdout], "Frozen V2 minus baseline Dice", "delta", "delta_ci95_lower", "delta_ci95_upper", zero=True)
    fig.suptitle("Retrospective holdout: paired effect estimates", y=.98, fontsize=12)
    fig.text(.16, .88, "Intervals crossing zero do not establish improvement; holdout is not external.", fontsize=9)
    save(fig, output, "holdout_paired_dice")
    report = ["# Aggregate study results", "Stored estimates; no MRI inference or statistical resampling performed.", "Intervals are 95% subject-cluster CIs. HD95 includes the official unmatched-lesion penalty."]
    for path in sorted(source.glob("*.csv")):
        report += ["## " + path.stem.replace("_", " "), markdown_table(read_csv(path))]
    (output / "results.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    print(f"Created three PDF/PNG figures and all aggregate tables in {output}")


if __name__ == "__main__":
    main()
