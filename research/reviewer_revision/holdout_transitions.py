"""Paired holdout audit: do not interpret counts as independent external evidence."""
import csv
import io
import sys
import zipfile
from collections import Counter
from statistics import median

with zipfile.ZipFile(sys.argv[1]) as z:
    path, = [n for n in z.namelist() if n.endswith("/final_locked_test_v1/test_case_metrics_paired.csv")]
    rows = list(csv.DictReader(io.StringIO(z.read(path).decode("utf-8-sig"))))
assert len(rows) == len({r["case_id"] for r in rows}) == 195
assert len({r["subject_id"] for r in rows}) == 92
transitions = Counter()
by_scan = []
for r in rows:
    def value(field, arm):
        return float(r[f"ET_{field}_{arm}"])
    assert value("GT_Complete_Volume","baseline") == value("GT_Complete_Volume","final")
    n = [value("Num_TP",a) + value("Num_FN",a) for a in ("baseline","final")]
    assert n[0] == n[1]
    by_scan.append(int(n[0]))
    if value("GT_Complete_Volume","baseline") == 0:
        scores = tuple(value("LesionWise_Score_Dice",a) for a in ("baseline","final"))
        assert set(scores) <= {0, 1}
        transitions[scores] += 1
print("Absent ET paired (baseline score, V2 score):", sorted(transitions.items()))
print("Reference lesions per scan:", sorted(Counter(by_scan).items()), "median", median(by_scan))
for a in ("baseline","final"):
    print(a, {k:int(sum(float(r[f"ET_Num_{k}_{a}"]) for r in rows)) for k in ("TP","FN","FP")})
