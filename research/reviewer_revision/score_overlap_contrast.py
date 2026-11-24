"""Prespecified, paired contrasts for the frozen longitudinal exposure experiment."""
import argparse
import csv
import hashlib
import json
import io
import zipfile
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
REGIONS=("WT","TC","NETC","SNFH","ET","RC")
DESIGN_HASH="e5c521c7702c1dce29b944bea21d539b3c4482cad1743c46a805359d084491bd"

def read_scores(path,ids):
    with path.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    by_id={r["case_id"]:r for r in rows}
    if len(rows)!=len(ids) or set(by_id)!=set(ids):
        raise ValueError("Require exactly one official score per frozen evaluation scan")
    a=np.array([[float(by_id[c][r+"_LesionWise_Score_Dice"]) for r in REGIONS] for c in ids])
    if not np.isfinite(a).all() or np.any((a<0)|(a>1)):
        raise ValueError("Invalid Dice")
    return np.column_stack([a.mean(axis=1),a])

def contrast(clean,exposed,index,rng,patient_ids=None):
    diff=(exposed-clean)[index]
    if patient_ids is not None:
        subjects,cluster=np.unique(np.asarray(patient_ids)[index],return_inverse=True)
        n=np.bincount(cluster)
        totals=np.zeros((len(n),diff.shape[1]))
        np.add.at(totals,cluster,diff)
        draws=rng.integers(len(subjects),size=(10000,len(subjects)))
        return diff.mean(axis=0),totals[draws].sum(axis=1)/n[draws].sum(axis=1)[:,None]
    draws=rng.integers(len(index),size=(10000,len(index)))
    return diff.mean(axis=0),diff[draws].mean(axis=1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean",type=Path,required=True)
    p.add_argument("--exposed",type=Path,required=True)
    p.add_argument("--single-fold0-release",type=Path,help="Use the revised 183-scan comparison and archived subject mapping")
    args=p.parse_args()
    raw=(HERE/"matched_overlap_design.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==DESIGN_HASH
    design=json.loads(raw)
    ids=design["evaluation"]
    assert len(ids)==len(set(ids))==103
    index_cases=ids.copy()
    patient_ids=None
    if args.single_fold0_release:
        raw=(HERE/"fold0_overlap_single_run/cohort.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest()=="a92848d00f95c5a2b6eb6d63358c8449bf513162e6b0219794ba072ccffac288"
        cohort=json.loads(raw)
        ids=cohort["validation_monitor"]
        assert len(ids)==len(set(ids))==183 and set(ids).isdisjoint(cohort["train"])
        with zipfile.ZipFile(args.single_fold0_release) as z:
            key,=[k for k in z.namelist() if k.endswith("/brats_subject_grouped_protocol_v1/case_assignments.csv")]
            assignments=z.read(key)
        assert hashlib.sha256(assignments).hexdigest()==design["assignments_sha256"]
        subj={r["case_id"]:r["subject_id"] for r in csv.DictReader(io.StringIO(assignments.decode("utf-8-sig"))) if r["protocol_partition"]=="development"}
        patient_ids=[subj[c] for c in ids]
        assert len(set(patient_ids))==len({subj[c] for c in index_cases})==103
    if args.clean.resolve()==args.exposed.resolve():
        raise ValueError("Both arms refer to the same file")
    a,b=(read_scores(path,ids) for path in (args.clean,args.exposed))
    groups={"primary_exposed":design["exposed_evaluation"],
            "unexposed_longitudinal":design["unexposed_longitudinal_evaluation"],
            "all_index_scans":index_cases}
    assert set(groups["primary_exposed"]).isdisjoint(groups["unexposed_longitudinal"])
    assert [len(v) for v in groups.values()]==[40,39,103]
    if patient_ids is not None: groups["all_eligible_scans"]=ids
    rng=np.random.default_rng(20261121)
    rows=[]
    stats={}
    for group,cs in groups.items():
        idx=[ids.index(c) for c in cs]
        d,boot=contrast(a,b,idx,rng,patient_ids)
        stats[group]=(d,boot)
        lo,hi=np.percentile(boot,[2.5,97.5],axis=0)
        for j,r in enumerate(("six_region_mean",*REGIONS)):
            n=len(cs) if patient_ids is None else len({patient_ids[i] for i in idx})
            rows.append([group,r,len(cs),n,a[idx,j].mean(),b[idx,j].mean(),d[j],lo[j],hi[j]])
    # Independent resampling of the two disjoint longitudinal patient groups.
    d,boot=[x-y for x,y in zip(stats["primary_exposed"],stats["unexposed_longitudinal"])]
    lo,hi=np.percentile(boot,[2.5,97.5],axis=0)
    for j,r in enumerate(("six_region_mean",*REGIONS)):
        rows.append(["exposed_minus_unexposed_longitudinal_delta",r,79,79,"","",d[j],lo[j],hi[j]])
    dest=(HERE/"fold0_overlap_single_run" if args.single_fold0_release else HERE)/"matched_overlap_results.csv"
    tmp=dest.with_suffix(".tmp")
    with tmp.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["stratum","metric","scans","patients","clean","exposure_arm","contrast","lower95","upper95"]);w.writerows(rows)
    tmp.replace(dest)

if __name__=="__main__": main()
