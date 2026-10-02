from __future__ import annotations

import numpy as np


def summarize_dice(rows: list[dict[str, float]], num_classes: int) -> dict[str, float]:
    summary: dict[str, float] = {}
    for cls in range(num_classes):
        key = f"dice_label_{cls}"
        values = np.asarray([row[key] for row in rows if not np.isnan(row[key])], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean()) if values.size else float("nan")
        summary[f"{key}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
    fg_values = []
    for row in rows:
        vals = [row[f"dice_label_{cls}"] for cls in range(1, num_classes)]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            fg_values.append(float(np.mean(vals)))
    summary["mean_foreground_dice"] = float(np.mean(fg_values)) if fg_values else float("nan")
    summary["case_count"] = float(len(rows))
    return summary

