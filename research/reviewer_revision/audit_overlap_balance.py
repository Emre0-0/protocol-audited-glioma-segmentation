"""Describe training-arm composition before the fold-0 exposure experiment is fitted."""
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
import numpy as np

here=Path(__file__).resolve().parent
folder=here/"fold0_overlap_single_run"
def locked(p,hash_):
    data=p.read_bytes()
    assert hashlib.sha256(data).hexdigest()==hash_
    return json.loads(data)
design=locked(here/"matched_overlap_design.json","e5c521c7702c1dce29b944bea21d539b3c4482cad1743c46a805359d084491bd")
cohort=locked(folder/"cohort.json","a92848d00f95c5a2b6eb6d63358c8449bf513162e6b0219794ba072ccffac288")
with zipfile.ZipFile(sys.argv[1]) as z:
    def raw(suffix,hash_):
        key,=[n for n in z.namelist() if n.endswith("/"+suffix)]
        data=z.read(key);assert hashlib.sha256(data).hexdigest()==hash_
        return data
    split=json.loads(raw("processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json",design["split_sha256"]))[0]
    data=raw("brats_subject_grouped_protocol_v1/case_assignments.csv",design["assignments_sha256"])
meta={r["case_id"]:r for r in csv.DictReader(io.StringIO(data.decode("utf-8-sig"))) if r["protocol_partition"]=="development"}
clean,mixed=set(split["train"]),set(cohort["train"])
removed,added=clean-mixed,mixed-clean
assert removed=={s["removed"] for s in design["swaps"]} and added=={s["added"] for s in design["swaps"]}
def patients(ids): return {meta[c]["subject_id"] for c in ids}
evaluation=set(cohort["validation_monitor"])
assert (len(clean),len(mixed),len(clean&mixed),len(removed),len(added))==(932,932,892,40,40)
assert len(patients(clean))==len(patients(mixed))==418 and not (clean|mixed)&evaluation
assert not patients(clean)&patients(evaluation)
assert patients(mixed)&patients(evaluation)==patients(cohort["primary_patients"]) and len(patients(cohort["primary_patients"]))==40
rows=[]
for scope,left,right in [("all_training",clean,mixed),("swapped_scans",removed,added)]:
    for region,labels in {"WT":[1,2,3],"TC":[1,3],"NETC":[1],"SNFH":[2],"ET":[3],"RC":[4]}.items():
        def vol(ids):
            return np.array([sum(int(meta[c][f"label_{l}_voxels"]) for l in labels) for c in sorted(ids)])
        a,b=vol(left),vol(right)
        x,y=np.log1p(a),np.log1p(b)
        sd=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
        row=dict(scope=scope,region=region,scans_per_arm=len(a),clean_positive=int((a>0).sum()),exposed_positive=int((b>0).sum()),
                 prevalence_delta_pp=100*((b>0).mean()-(a>0).mean()),clean_median_voxels=np.median(a),exposed_median_voxels=np.median(b),
                 clean_total_voxels=int(a.sum()),exposed_total_voxels=int(b.sum()),smd_log1p_voxels=(y.mean()-x.mean())/sd if sd else None)
        rows.append(row)
out=folder/"training_balance.csv"
tmp=out.with_suffix(".tmp")
with tmp.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
tmp.replace(out)
for scope in ("all_training","swapped_scans"):
    g=[r for r in rows if r["scope"]==scope]
    print(scope,"max absolute prevalence difference (pp)",max(abs(r["prevalence_delta_pp"]) for r in g),
          "max |SMD| of log(1+volume)",max(abs(r["smd_log1p_voxels"]) for r in g if r["smd_log1p_voxels"] is not None))
