"""Score the one-run fold-0 exposure ablation; never substitute best-checkpoint metrics."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
import numpy as np

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("project",type=Path)
    p.add_argument("archive",type=Path)
    p.add_argument("--preflight",action="store_true")
    a=p.parse_args()
    here=Path(__file__).resolve().parent
    run=here/"fold0_overlap_single_run"
    cohort=run/"cohort.json"
    assert sha(cohort)=="a92848d00f95c5a2b6eb6d63358c8449bf513162e6b0219794ba072ccffac288"
    frozen=json.loads(cohort.read_bytes())
    ids=sorted(frozen["validation_monitor"])
    assert len(ids)==183 and not set(ids)&set(frozen["train"])
    controls=json.loads((run/"clean_final_manifest.json").read_bytes())
    assert controls["epoch"]==250 and set(controls["scans"])==set(ids)
    refs=here/"specialist_crossfit"
    refhash=json.loads((refs/"validation_reference_hashes.json").read_bytes())
    gtdir=refs/"preprocessed/Dataset503_BraTS2024Grouped4Ch/gt_segmentations"
    rel="research/vendor/BraTS-2024-Metrics/metrics_GLI.py"
    vendor=a.project/rel
    with zipfile.ZipFile(a.archive) as z:
        key,=[k for k in z.namelist() if k.endswith("/"+rel)]
        scorer_hash=hashlib.sha256(z.read(key)).hexdigest()
    assert sha(vendor)==scorer_hash
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS","1")
    np.Inf=np.inf
    sys.path.insert(0,str(vendor.parent))
    from metrics_GLI import get_LesionWiseResults
    regions="WT TC NETC SNFH ET RC".split()
    keys=["LesionWise_Score_Dice","LesionWise_Score_HD95","Num_TP","Num_FN","Num_FP"]
    def official(pred,gt):
        cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                df,_=get_LesionWiseResults(pred.as_posix(),gt.as_posix(),"BraTS-GLI")
            finally: os.chdir(cwd)
        df=df.set_index("Labels")[keys]
        assert set(df.index)==set(regions) and len(df)==6 and np.isfinite(df.to_numpy()).all()
        return df.to_dict(orient="index")
    records=run/"paired_scores"
    if not a.preflight:
        assert {p.stem for p in (run/"exposed_records").glob("*.json")}==set(ids),"Incomplete exposed inference"
        records.mkdir(exist_ok=True)
    manifest=controls["scans"]
    reusable=0
    for c in ids:
        pred=here/"fixed_epoch_fold0/final"/(c+".nii.gz")
        gt=gtdir/pred.name
        assert sha(gt)==refhash[c] and sha(pred)==manifest[c]["final_sha256"]
        oldfile=here/"fixed_epoch_fold0/scores"/(c+".json")
        old=json.loads(oldfile.read_bytes()) if oldfile.exists() else None
        if old:
            assert old["case_id"]==c and old["hashes"]["final"]==manifest[c]["final_sha256"]
            reusable+=1
        if a.preflight: continue
        exp=run/"exposed_predictions"/pred.name
        marker=json.loads((run/"exposed_records"/(c+".json")).read_bytes())
        assert marker["case"]==c and marker["clean_reference"]==manifest[c] and sha(exp)==marker["output"]
        provenance=dict(case=c,clean=manifest[c]["final_sha256"],exposed=marker["output"],gt=refhash[c],scorer=scorer_hash)
        path=records/(c+".json")
        if path.exists():
            assert json.loads(path.read_bytes())["provenance"]==provenance
            continue
        both=dict(provenance=provenance,clean=old["final"] if old else official(pred,gt),exposed=official(exp,gt))
        tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(both,allow_nan=False));tmp.replace(path)
        print("Paired scores",c,flush=True)
    if a.preflight:
        print("Verified scoring cohort, references, and scorer; available clean final scores:",reusable,"/ 183")
        return
    assert {p.stem for p in records.glob("*.json")}==set(ids)
    tables=[]
    for arm in ("clean","exposed"):
        dest=run/(arm+"_official.csv")
        tmp=dest.with_suffix(".tmp")
        with tmp.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["case_id"]+[r+"_"+k for r in regions for k in keys])
            w.writeheader()
            for c in ids:
                m=json.loads((records/(c+".json")).read_bytes())[arm]
                w.writerow(dict(case_id=c,**{r+"_"+k:m[r][k] for r in regions for k in keys}))
        tmp.replace(dest);tables.append(str(dest))
    subprocess.run([sys.executable,str(here/"score_overlap_contrast.py"),"--clean",tables[0],"--exposed",tables[1],"--single-fold0-release",str(a.archive)],check=True)

if __name__=="__main__": main()
