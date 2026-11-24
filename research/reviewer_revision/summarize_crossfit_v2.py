"""Paired official V2 vs baseline; never pool an incomplete set of folds."""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
regions="WT TC NETC SNFH ET RC".split()
keys="LesionWise_Score_Dice LesionWise_Score_HD95 Num_TP Num_FN Num_FP".split()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("--project",type=Path,help="Original archived experiment tree for the fold-0 fusion score table")
    p.add_argument("scope",choices=["pooled","nonselection",*map(str,range(5))])
    a=p.parse_args()
    folds=range(5) if a.scope=="pooled" else range(1,5) if a.scope=="nonselection" else [int(a.scope)]
    ids=[]; data=[]
    with zipfile.ZipFile(a.archive) as z:
        def raw(s):
            n,=[n for n in z.namelist() if n.endswith("/"+s)]
            return z.read(n)
        def table(s):
            rows=list(csv.DictReader(io.StringIO(raw(s).decode("utf-8-sig"))))
            d={r["case_id"]:r for r in rows}
            assert len(rows)==len(d)
            return d
        patients={c:r["subject_id"] for c,r in table("brats_subject_grouped_protocol_v1/case_assignments.csv").items() if r["protocol_partition"]=="development"}
        splits=json.loads(raw("processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json"))
        for f in folds:
            cases=sorted(splits[f]["val"])
            assert len(set(cases))==len(cases)==[223,247,243,225,217][f]
            seen={patients[c] for c in cases}
            assert seen.isdisjoint(patients[c] for c in ids+splits[f]["train"])
            if f==0:
                dirs=["grouped_delta_roi_v1/official_metrics/baseline","final_locked_test_v1/reproduced_validation/official_metrics/et_cc_topology_guard_specialist_nonrc_reproduced"]
                tables=[table(dirs[0]+"/case_metrics_official.csv")]
                if a.project is None: p.error("--project required for fold 0")
                with (a.project/"experiments"/dirs[1]/"case_metrics_official.csv").open(encoding="utf-8-sig",newline="") as s:
                    rows=list(csv.DictReader(s));assert len(rows)==len(cases)
                    tables.append({r["case_id"]:r for r in rows})
                assert all(set(t)==set(cases) for t in tables)
                for c in cases:
                    data.append([{r:{k:float(t[c][r+"_"+k]) for k in keys} for r in regions} for t in tables])
            else:
                folder=root/"specialist_crossfit/paired_fusion"/f"fold_{f}"/"official_scores"
                assert {p.stem for p in folder.glob("*.json")}==set(cases),f"Incomplete fold {f}"
                for c in cases:
                    record=json.loads((folder/(c+".json")).read_bytes())
                    m=record["provenance"]
                    assert (m["case"],m["fold"],m["subject"])==(c,f,patients[c])
                    data.append([record["baseline"],record["v2"]])
            ids.extend(cases)
    subjects,idx=np.unique([patients[c] for c in ids],return_inverse=True)
    if a.scope=="pooled": assert set(ids)==patients.keys() and len(subjects)==521
    if a.scope=="nonselection": assert len(ids)==932 and len(subjects)==418
    rng=np.random.default_rng(20260808)
    n=len(subjects)
    weights=rng.multinomial(n,np.full(n,1/n),size=10000)
    x=np.array([[[[arm[r][k] for k in keys] for r in regions] for arm in pair] for pair in data])
    assert np.isfinite(x).all()
    stats=[]
    def add(region,metric,num,den):
        arms=[];boots=[]
        for arm in range(2):
            a,b=num[:,arm],den[:,arm]
            group=lambda v:np.bincount(idx,weights=v,minlength=n)
            bd=weights@group(b)
            assert bd.min()>0
            arms.append(a.sum()/b.sum())
            boots.append((weights@group(a))/bd)
        lo,hi=np.percentile(boots[1]-boots[0],[2.5,97.5])
        stats.append([region,metric,*arms,arms[1]-arms[0],lo,hi])
    ones=np.ones((len(ids),2))
    add("six-region mean","Dice",x[:,:,:,0].mean(axis=2),ones)
    for i,r in enumerate(regions):
        dice,hd,tp,fn,fp=x[:,:,i,:].transpose(2,0,1)
        add(r,"Dice",dice,ones);add(r,"HD95",hd,ones)
        add(r,"sensitivity",tp,tp+fn);add(r,"precision",tp,tp+fp)
        for name,arr in (("TP",tp),("FN",fn),("FP",fp)):
            l,h=arr.sum(axis=0)
            stats.append([r,name,int(l),int(h),int(h-l),"",""])
    out=root/"specialist_crossfit"/("paired_"+a.scope+".csv")
    tmp=out.with_suffix(".tmp")
    with tmp.open("w",newline="") as s:
        w=csv.writer(s);w.writerow(["scans","patients","region","metric","baseline","v2","delta","paired_lower95","paired_upper95"])
        w.writerows([[len(ids),n,*row] for row in stats])
    tmp.replace(out)
    print(out,len(ids),n)

if __name__=="__main__": main()
