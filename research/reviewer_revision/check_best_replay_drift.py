"""QC only: compare re-scored best with archived best on identical completed scans."""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

fold=int(sys.argv[2]) if len(sys.argv)>2 else 0
assert fold in (0,2,3,4)
root=Path(__file__).resolve().parent/f"fixed_epoch_fold{fold}"
with zipfile.ZipFile(sys.argv[1]) as z:
    def read(rel):
        n,=[n for n in z.namelist() if n.endswith("/"+rel)]
        return z.read(n)
    prefix=f"grouped_fivefold_cv_v11/fold_{fold}/official_metrics" if fold else "grouped_delta_roi_v1/official_metrics/baseline"
    metrics=list(csv.DictReader(io.StringIO(read(prefix+"/case_metrics_official.csv").decode("utf-8-sig"))))
    val=json.loads(read("processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json"))[fold]["val"]
arch={r["case_id"]:r for r in metrics}
assert len(metrics)==len(arch)==len(val)==[223,247,243,225,217][fold] and arch.keys()==set(val)
regions="WT TC NETC SNFH ET RC".split()
rows=[]
for p in sorted((root/"scores").glob("BraTS-*.json")):
    c=p.stem
    rec=json.loads(p.read_bytes())
    marker=json.loads((root/p.name).read_bytes())
    assert marker["case_id"]==rec["case_id"]==c and rec["hashes"]==marker["output_hashes"]
    for r in regions:
        def diff(k): return rec["best"][r][k]-float(arch[c][r+"_"+k])
        row={"case":c,"region":r,"dice_delta":diff("LesionWise_Score_Dice"),"hd95_delta_mm":diff("LesionWise_Score_HD95")}
        for k in ("TP","FN","FP"):
            v=diff("Num_"+k)
            assert v==int(v)
            row[k+"_delta"]=int(v)
        assert row["TP_delta"]+row["FN_delta"]==0
        rows.append(row)
assert rows
def writecsv(path,rows):
    stage=path.with_suffix(".tmp")
    with stage.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    stage.replace(path)
writecsv(root/"best_replay_drift_cases.csv",rows)
stats=[]
for r in regions:
    group=[a for a in rows if a["region"]==r]
    row={"region":r,"completed_scans":len(group),"total_fold_scans":len(val)}
    for k in ("dice_delta","hd95_delta_mm"):
        v=[a[k] for a in group]
        assert np.isfinite(v).all()
        row["mean_"+k]=np.mean(v)
        row["max_abs_"+k]=np.max(np.abs(v))
    row["scans_with_any_count_change"]=sum(any(a[k+"_delta"]!=0 for k in ("TP","FN","FP")) for a in group)
    stats.append(row)
writecsv(root/"best_replay_drift_summary.csv",stats)
print("Complete paired scans:",len(rows)//6,"/",len(val))
print("Maximum absolute regional Dice and HD95 differences:",max(s["max_abs_dice_delta"] for s in stats),max(s["max_abs_hd95_delta_mm"] for s in stats))
print("Scans with TP/FN/FP differences:",len({a["case"] for a in rows if any(a[k+"_delta"] for k in ("TP","FN","FP"))}))
