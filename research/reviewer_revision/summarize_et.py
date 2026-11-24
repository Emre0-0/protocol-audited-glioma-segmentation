"""Size-stratified OOF ET sensitivity: require exact scan-level reconciliation."""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path
import numpy as np

def summarize(records, patient_count, reps=10000):
    """Rows are (patient index, scan index, eligible reference volume, hit)."""
    s,c,v,h=np.asarray(records,dtype=np.int64).T
    assert np.all(v>10) and np.isin(h,[0,1]).all() and np.all((s>=0)&(s<patient_count))
    draws=np.random.default_rng(20260808).integers(patient_count,size=(reps,patient_count))
    results=[]
    for name,q in (("lowest_decile",0.1),("lowest_quartile",0.25),("all_eligible",1.0)):
        cutoff=float(np.quantile(v,q))
        use=v<=cutoff
        n=np.bincount(s[use],minlength=patient_count)
        tp=np.bincount(s[use],weights=h[use],minlength=patient_count)
        den=n[draws].sum(axis=1)
        valid=den>0
        ci=np.percentile(tp[draws[valid]].sum(axis=1)/den[valid],[2.5,97.5])
        results.append(dict(stratum=name,cutoff_voxels=cutoff,lesions=int(n.sum()),
                            scans=len(set(c[use])),patients=int((n>0).sum()),
                            detected=int(tp.sum()),missed=int(n.sum()-tp.sum()),
                            sensitivity=tp.sum()/n.sum(),lower95=ci[0],upper95=ci[1],
                            missed_fraction=1-tp.sum()/n.sum(),
                            missed_lower95=1-ci[1],missed_upper95=1-ci[0],
                            bootstrap_draws_with_nonzero_denominator=int(valid.sum())))
    return results

def latex_table(rows):
    names={"lowest_decile":"Lowest 10\\%", "lowest_quartile":"Lowest 25\\%", "all_eligible":"All eligible"}
    assert [r["stratum"] for r in rows]==list(names)
    lines=[r"\begin{table}[htbp]", r"\centering\small",
           r"\caption{Missed enhancing-tumor (ET) lesions in patient-disjoint out-of-fold predictions. Strata use individual reference-lesion volumes in voxels, not total ET burden per scan.}",
           r"\label{tab:oof_et_individual}",r"\begin{tabular}{@{}lrrrr@{}}",r"\toprule",
           r"Stratum & Max. volume & Lesions & Missed & Missed \% (95\% CI) \\",r"\midrule"]
    for r in rows:
        assert r["missed"]+r["detected"]==r["lesions"] and r["lesions"]>0
        pct,lo,hi=[100*r[k] for k in ("missed_fraction","missed_lower95","missed_upper95")]
        # Integer volumes: floor of the interpolated quantile is the inclusive ceiling.
        lines.append(f'{names[r["stratum"]]} & {int(r["cutoff_voxels"]):,} & {r["lesions"]:,} & {r["missed"]} & {pct:.1f} ({lo:.1f}--{hi:.1f}) '+r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\par\smallskip",
                  r"\parbox{0.95\linewidth}{\footnotesize Only scorer-eligible lesions ($>10$ voxels) are included. Groups are nested and ties are retained. Intervals resample patients (10,000 draws) with the observed volume cutoffs fixed.}",
                  r"\end{table}"])
    return "\n".join(lines)+"\n"

def main(archive):
    root=Path(__file__).resolve().parent
    with zipfile.ZipFile(archive) as z:
        def rows(suffix):
            key,=[k for k in z.namelist() if k.endswith("/"+suffix)]
            return list(csv.DictReader(io.StringIO(z.read(key).decode("utf-8-sig"))))
        dev=[r for r in rows("brats_subject_grouped_protocol_v1/case_assignments.csv") if r["protocol_partition"]=="development"]
        cases={r["case_id"]:r["subject_id"] for r in dev}
        expected={}
        volumes={}
        split_key,=[k for k in z.namelist() if k.endswith("/processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json")]
        splits=json.loads(z.read(split_key))
        assert len(splits)==5
        for f in range(5):
            prefix=f"grouped_fivefold_cv_v11/fold_{f}/official_metrics" if f else "grouped_delta_roi_v1/official_metrics/baseline"
            official=rows(prefix+"/case_metrics_official.csv")
            val=splits[f]["val"]
            assert len(official)==len(val)==len(set(val))
            assert {r["case_id"] for r in official}==set(val),f
            assert {cases[c] for c in val}.isdisjoint(cases[c] for c in splits[f]["train"]),f
            for r in official:
                k=r["case_id"]
                assert k not in expected
                expected[k]=tuple(int(float(r["ET_Num_"+field])) for field in ("TP","FN"))
                volumes[k]=(float(r["ET_GT_Complete_Volume"]),int(r["ET_lesion_detail_rows"]))
    patients={p:i for i,p in enumerate(sorted(set(cases.values())))}
    assert len(dev)==len(cases)==1155 and len(patients)==521 and cases.keys()==expected.keys()
    sources=(root/"et_rows",root/"et_native_reverse")
    available={p.stem for d in sources for p in d.glob("*.json")}
    assert not available-set(cases),"Unexpected scan records"
    records=[]
    for idx,(cid,p) in enumerate(sorted(cases.items())):
        if cid not in available: continue
        copies=[json.loads(path.read_bytes()) for d in sources if (path:=d/(cid+".json")).exists()]
        data=copies[0]
        assert all(sorted(x)==sorted(data) for x in copies),f"Lesion assignment disagreement: {cid}"
        assert all(isinstance(v,int) and v>10 and h in (0,1) for v,h in data),cid
        hits=sum(h for v,h in data)
        assert (hits,len(data)-hits)==expected[cid],cid
        total,n_components=volumes[cid]
        omitted=n_components-len(data)
        remainder=total-sum(v for v,h in data)
        assert omitted>=0 and omitted<=remainder<=10*omitted,(cid,total,omitted,remainder)
        records.extend((patients[p],idx,v,h) for v,h in data)
    if available!=set(cases):
        raise SystemExit(f"Counts and volume bounds verified for {len(available)} scans; {len(set(cases)-available)} missing. No table produced.")
    results=summarize(records,len(patients))
    all_et=results[-1]
    assert (all_et["lesions"],all_et["detected"],all_et["missed"])==(1082,953,129)
    out=root/"oof_individual_et.csv"
    temp=out.with_suffix(".tmp")
    with temp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(all_et));w.writeheader();w.writerows(results)
    temp.replace(out)
    tex=out.with_suffix(".tex")
    tmp=tex.with_suffix(".tex.tmp")
    tmp.write_text(latex_table(results),encoding="ascii")
    tmp.replace(tex)
    print(json.dumps(results,indent=2))

if __name__=="__main__": main(sys.argv[1])
