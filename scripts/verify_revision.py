"""Recalculate the three primary revision contrasts from score-only extracts."""
import csv
from pathlib import Path
from statistics import mean

D = Path(__file__).resolve().parents[1] / "research/reviewer_revision"


def rows(name):
    with (D / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def paired(name, columns, n, expected):
    samples = rows(name)
    assert len(samples) == n and tuple(samples[0]) == columns
    a, b = [[float(r[k]) for r in samples] for k in columns]
    assert all(0 <= x <= 1 for x in a + b)
    values = (mean(a), mean(b), mean(y - x for x, y in zip(a, b)))
    assert tuple(round(v, 4) for v in values) == expected, values
    return " -> ".join(f"{v:.4f}" for v in values[:2]) + f" (delta {values[2]:+.4f})"


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Run without -O")
    print("OOF (1155 scans):", paired("oof_scan_scores.csv", ("baseline", "v2"), 1155, (.7634, .7712, .0078)))
    counts = {(r["volume_bin_voxels"], int(r["detected"])): int(r["lesions"]) for r in rows("et_detection_counts.csv")}
    missed, n = counts["11-61", 0], counts["11-61", 0] + counts["11-61", 1]
    assert (missed, n, sum(counts.values())) == (68, 109, 1082)
    print(f"ET, 11-61 voxels: {missed}/{n} missed ({100 * missed / n:.1f}%)")
    print("Exposure (40 index scans):", paired("overlap_index_scores.csv", ("clean", "exposed"), 40, (.7760, .7831, .0071)))
