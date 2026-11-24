"""Rescore a new pilot mask against its development GT, compare to archived case metrics."""
import csv
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent/"fixed_epoch_pilot"
case=json.loads((root/"manifest.json").read_text())["case_id"]
kind=sys.argv[1]
assert kind in {"best","final","reference"}
project=Path(sys.argv[2])
prediction=root/("reference" if kind=="reference" else "replay_"+kind)/(case+".nii.gz")
with zipfile.ZipFile(next((Path.home()/"Downloads").glob("BrainMRI*004.zip"))) as z:
    key,=[k for k in z.namelist() if k.endswith("/grouped_delta_roi_v1/official_metrics/baseline/case_metrics_official.csv")]
    r,=[v for v in csv.DictReader(io.StringIO(z.read(key).decode("utf-8-sig"))) if v["case_id"]==case]
np.Inf=np.inf
sys.path.insert(0,str(project/"research/vendor/BraTS-2024-Metrics"))
from metrics_GLI import get_LesionWiseResults
mask=project/"processed/nnunet_raw/Dataset503_BraTS2024Grouped4Ch/labelsTr"/(case+".nii.gz")
cwd=Path.cwd()
with tempfile.TemporaryDirectory() as tmp:
    try:
        os.chdir(tmp)
        scores,_=get_LesionWiseResults(prediction.as_posix(),mask.as_posix(),"BraTS-GLI")
    finally:
        os.chdir(cwd)
scores=scores.set_index("Labels")
fields=("Num_TP","Num_FP","Num_FN","LesionWise_Score_Dice","LesionWise_Score_HD95")
rows=[]
for region in ("WT","TC","NETC","SNFH","ET","RC"):
    for metric in fields:
        old=float(r[region+"_"+metric])
        new=float(scores.loc[region,metric])
        rows.append((region,metric,old,new,new-old))
with (root/("official_"+kind+".csv")).open("w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["region","metric","archived_best",kind,"difference"])
    writer.writerows(rows)
print("Maximum absolute delta by metric:",{m:max(abs(v[4]) for v in rows if v[1]==m) for m in fields})
