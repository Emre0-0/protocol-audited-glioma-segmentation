"""Native-volume ET-only replay using the released scorer's own grouping and matcher."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile
import numpy as np
import psutil
from stage_revision_input import index_inputs,stage

p=argparse.ArgumentParser()
p.add_argument("project",type=Path);p.add_argument("archive",type=Path);p.add_argument("--continue-missing",action="store_true")
a=p.parse_args()
here=Path(__file__).resolve().parent
np.Inf=np.inf
rel="research/vendor/BraTS-2024-Metrics/metrics_GLI.py"
sys.path.insert(0,str((a.project/rel).parent))
import metrics_GLI as vendor
records={}
with zipfile.ZipFile(a.archive) as z:
    def read(s):
        key,=[k for k in z.namelist() if k.endswith("/"+s)]
        return z.read(key)
    assert Path(vendor.__file__).read_bytes()==read(rel)
    for fold in range(5):
        prefix=f"grouped_fivefold_cv_v11/fold_{fold}/official_metrics" if fold else "grouped_delta_roi_v1/official_metrics/baseline"
        meta=json.loads(read(prefix+"/metadata.json"))
        source=Path(meta["prediction_dir"]).relative_to(r"C:\Users\Emre\Desktop\BR")
        for r in csv.DictReader(io.StringIO(read(prefix+"/case_metrics_official.csv").decode("utf-8-sig"))):
            c=r["case_id"]
            assert c not in records
            records[c]=(fold,source,r)
assert len(records)==1155
local=index_inputs(tuple({v[1].as_posix()+"/" for v in records.values()}))
out=here/"et_native_reverse";out.mkdir(exist_ok=True)
primary=here/"et_rows"
checks=here/"specialist_crossfit/reference_volume_checks"
gtdir=here/"specialist_crossfit/preprocessed/Dataset503_BraTS2024Grouped4Ch/gt_segmentations"
def recover(c):
    while psutil.virtual_memory().available<6*2**30: time.sleep(15)
    fold,source,r=records[c]
    gt=gtdir/(c+".nii.gz")
    assert hashlib.sha256(gt.read_bytes()).hexdigest()==json.loads((checks/(c+".json")).read_bytes())["reference_sha256"]
    cwd=Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        pred=folder/(c+".nii.gz")
        staged=here/"specialist_crossfit/paired_fusion"/f"fold_{fold}"/"archived_baseline"/pred.name
        if fold and staged.is_file():
            expected=json.loads((staged.parent.parent/"baseline_sha256.json").read_bytes())[pred.name]
            assert hashlib.sha256(staged.read_bytes()).hexdigest()==expected
            pred=staged
        else: stage(a.project,source/pred.name,pred,local)
        try:
            os.chdir(folder)
            vendor.save_tmp_files(pred.as_posix(),gt.as_posix(),3)
            for side in ("gt","pred"):
                base=Path("tmp_"+side)/c
                raw=lambda reg:(base/f"{c}_{reg}_cc.nii.gz").as_posix()
                vendor.combine_lesions_tissues(raw("NETC"),raw("SNFH"),raw("ET"),raw("RC"))
                et=(base/f"{c}_ET_cc_combined.nii.gz").as_posix()
                vendor.reorder_labels_nifti(et,et)
            scores=vendor.get_LesionWiseScores(pred.as_posix(),gt.as_posix(),"ET",3)
        finally: os.chdir(cwd)
    details=scores[4]
    assert len(details)==int(r["ET_lesion_detail_rows"]) and scores[9]==float(r["ET_GT_Complete_Volume"])
    rows=[[int(v),int(len(ids)>0)] for ids,_,v,*_ in details if v>10]
    tp=sum(h for v,h in rows)
    assert (tp,len(rows)-tp)==tuple(int(float(r["ET_Num_"+k])) for k in ("TP","FN")),c
    return rows

# Check actual lesion assignments, including within-scan mixtures of hits and misses.
seen=sorted(p.stem for p in primary.glob("*.json"))
def mixed(c):
    rows=json.loads((primary/(c+".json")).read_bytes())
    return len(rows)>1 and {h for v,h in rows}=={0,1}
pilots=[next(c for c in seen if records[c][0]==fold and mixed(c)) for fold in (0,1)]
for c in pilots:
    assert sorted(recover(c))==sorted(json.loads((primary/(c+".json")).read_bytes())),c
    print("Exact lesion-pair match",c,flush=True)
if a.continue_missing:
    for c in sorted(records,reverse=True):
        dst=out/(c+".json")
        if dst.exists() or (primary/dst.name).exists(): continue
        rows=recover(c)
        tmp=dst.with_suffix(".tmp");tmp.write_text(json.dumps(rows));tmp.replace(dst)
        print("Recovered",c,flush=True)
