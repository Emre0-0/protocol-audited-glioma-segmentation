"""Build the public score-only extracts from the authors' private scoring records."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
REGIONS = "WT TC NETC SNFH ET RC".split()


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    result = {r["case_id"]: mean(float(r[k + "_LesionWise_Score_Dice"]) for k in REGIONS) for r in rows}
    assert len(result) == len(rows)
    return result


def write(name, fields, rows):
    with (HERE / name).open("w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(sorted(rows))


def main(work, project):
    splits = json.loads((HERE.parents[1] / "processed/brats_subject_grouped_protocol_v1/splits_final.json").read_bytes())
    all_cases = set()
    pairs = []
    for fold, split in enumerate(splits):
        cases = set(split["val"])
        assert not all_cases.intersection(cases)
        all_cases.update(cases)
        if fold == 0:
            paths = ["grouped_delta_roi_v1/official_metrics/baseline", "final_locked_test_v1/reproduced_validation/official_metrics/et_cc_topology_guard_specialist_nonrc_reproduced"]
            a, b = [table(project / "experiments" / p / "case_metrics_official.csv") for p in paths]
            assert a.keys() == b.keys() == cases
            pairs.extend((a[c], b[c]) for c in cases)
        else:
            folder = work / "specialist_crossfit/paired_fusion" / f"fold_{fold}" / "official_scores"
            assert {p.stem for p in folder.glob("*.json")} == cases
            for c in cases:
                d = json.loads((folder / (c + ".json")).read_bytes())
                assert (d["provenance"]["case"], d["provenance"]["fold"]) == (c, fold)
                pairs.append(tuple(mean(d[arm][k]["LesionWise_Score_Dice"] for k in REGIONS) for arm in ("baseline", "v2")))
    assert len(pairs) == len(all_cases) == 1155
    # Omit scan order, IDs, fold, volumes and regional label information.
    write("oof_scan_scores.csv", ["baseline", "v2"], [(round(a, 6), round(b, 6)) for a, b in pairs])
    design = json.loads((HERE / "matched_overlap_design.json").read_bytes())
    a, b = [table(work / "fold0_overlap_single_run" / (arm + "_official.csv")) for arm in ("clean", "exposed")]
    cohort = json.loads((HERE / "fold0_overlap_single_run/cohort.json").read_bytes())
    assert a.keys() == b.keys() == set(cohort["validation_monitor"])
    write("overlap_index_scores.csv", ["clean", "exposed"], [(round(a[c], 6), round(b[c], 6)) for c in design["exposed_evaluation"]])
    counts = Counter()
    for c in all_cases:
        copies = [json.loads(p.read_bytes()) for folder in ("et_rows", "et_native_reverse") if (p := work / folder / (c + ".json")).exists()]
        assert copies and all(sorted(v) == sorted(copies[0]) for v in copies)
        for size, hit in copies[0]:
            assert size > 10 and hit in (0, 1)
            counts[(size <= 61, hit)] += 1
    write("et_detection_counts.csv", ["volume_bin_voxels", "detected", "lesions"], [("11-61" if small else "62+", hit, n) for (small, hit), n in counts.items()])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("revision_workspace", type=Path)
    p.add_argument("project_archive_root", type=Path)
    args = p.parse_args()
    main(args.revision_workspace, args.project_archive_root)
