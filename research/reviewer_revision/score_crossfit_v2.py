"""Score the frozen cross-fitted fusion on development evaluation scans only."""
import argparse
import csv
import hashlib
import io
import json
import os
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
    p.add_argument("fold",type=int,choices=range(1,5))
    a=p.parse_args()
    root=Path(__file__).resolve().parent/"specialist_crossfit"
    folder=root/"paired_fusion"/f"fold_{a.fold}"
    regions="WT TC NETC SNFH ET RC".split()
    fields="LesionWise_Score_Dice LesionWise_Score_HD95 Num_TP Num_FN Num_FP".split()
    with zipfile.ZipFile(a.archive) as z:
        def raw(suffix):
            k,=[x for x in z.namelist() if x.endswith("/"+suffix)]
            return z.read(k)
        split=json.loads(raw("processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json"))[a.fold]
        patients={r["case_id"]:r["subject_id"] for r in csv.DictReader(io.StringIO(raw("brats_subject_grouped_protocol_v1/case_assignments.csv").decode("utf-8-sig"))) if r["protocol_partition"]=="development"}
        source=f"grouped_fivefold_cv_v11/fold_{a.fold}/official_metrics/case_metrics_official.csv"
        rows=list(csv.DictReader(io.StringIO(raw(source).decode("utf-8-sig"))))
    cases=set(split["val"])
    byid={r["case_id"]:r for r in rows}
    assert len(rows)==len(cases) and set(byid)==cases
    assert {patients[c] for c in cases}.isdisjoint(patients[c] for c in split["train"])
    baseline=json.loads((folder/"baseline_sha256.json").read_bytes())
    truth=json.loads((root/"validation_reference_hashes.json").read_bytes())
    manifest=folder/"reproduced_validation/build_manifest.json"
    m=json.loads(manifest.read_bytes())
    assert not m["ground_truth_used_during_build"] and not m["locked_test_used"]
    preds=list(manifest.parent.rglob("*.nii.gz"))
    assert len(preds)==len(cases) and {f.name for f in preds}==set(baseline)=={c+".nii.gz" for c in cases}
    scorer=a.project/"research/vendor/BraTS-2024-Metrics/metrics_GLI.py"
    assert sha(scorer)=="3826004ac62cd13ee9c910c8be683702c9716051da1595acccbd96595f970cc2"
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS","1")
    np.Inf=np.inf
    sys.path.insert(0,str(scorer.parent))
    from metrics_GLI import get_LesionWiseResults
    out=folder/"official_scores"
    out.mkdir(exist_ok=True)
    mh=sha(manifest)
    for f in sorted(preds):
        case=f.name[:-7]
        gt=root/"preprocessed/Dataset503_BraTS2024Grouped4Ch/gt_segmentations"/f.name
        orig=folder/"archived_baseline"/f.name
        assert sha(gt)==truth[case] and sha(orig)==baseline[f.name]
        provenance=dict(case=case,fold=a.fold,subject=patients[case],gt=truth[case],baseline=baseline[f.name],v2=sha(f),manifest=mh)
        dst=out/(case+".json")
        if dst.exists():
            assert json.loads(dst.read_bytes())["provenance"]==provenance
            continue
        clean={r:{k:float(byid[case][r+"_"+k]) for k in fields} for r in regions}
        result=clean
        if provenance["baseline"]!=provenance["v2"]:
            cwd=Path.cwd()
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    os.chdir(tmp)
                    scores,_=get_LesionWiseResults(f.as_posix(),gt.as_posix(),"BraTS-GLI")
                finally: os.chdir(cwd)
            frame=scores.set_index("Labels")[fields]
            assert len(frame)==6 and set(frame.index)==set(regions) and np.isfinite(frame.to_numpy()).all()
            result=frame.to_dict(orient="index")
        tmp=dst.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(provenance=provenance,baseline=clean,v2=result),allow_nan=False))
        tmp.replace(dst)
    assert {f.stem for f in out.glob("*.json")}==cases
    print(f"Complete: fold {a.fold}, {len(cases)} paired development scans",flush=True)

if __name__=="__main__": main()
