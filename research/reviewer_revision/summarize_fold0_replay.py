"""Paired scan-weighted Dice/HD95 and count-pooled sensitivity, subject-cluster percentile CIs."""
import csv
import io
import json
import sys
import subprocess
import zipfile
from pathlib import Path
import numpy as np

selection=sys.argv[2] if len(sys.argv)>2 else "0"
folds=list(range(5)) if selection=="pooled" else [int(selection)]
assert all(f in range(5) for f in folds)
base=Path(__file__).resolve().parent
root=base/("fixed_epoch_pooled" if selection=="pooled" else f"fixed_epoch_fold{folds[0]}")
records=[]
cases=[]
regions=["WT","TC","NETC","SNFH","ET","RC"]
with zipfile.ZipFile(sys.argv[1]) as z:
    def read(suffix):
        k,=[k for k in z.namelist() if k.endswith("/"+suffix)]
        return z.read(k)
    splits=json.loads(read("processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json"))
    ids={r["case_id"]:r["subject_id"] for r in csv.DictReader(io.StringIO(read("brats_subject_grouped_protocol_v1/case_assignments.csv").decode("utf-8-sig"))) if r["protocol_partition"]=="development"}
    for fold in folds:
        members=sorted(splits[fold]["val"])
        patients={ids[c] for c in members}
        assert len(members)==len(set(members))==[223,247,243,225,217][fold]
        assert len(patients)==[103,107,105,105,101][fold] and patients.isdisjoint(ids[c] for c in cases)
        assert patients.isdisjoint(ids[c] for c in splits[fold]["train"])
        cases.extend(members)
        if fold==1:
            identity=json.loads((base/"fold1_network_identity.json").read_bytes())
            assert (identity["fold"],identity["keys"],identity["different_entries"],identity["best_epoch"],identity["final_epoch"])==("fold_1",956,0,250,250)
            assert identity["best_network_signature"]==identity["final_network_signature"]=="4cf273995c2a3159a4c732e4878efffd7b5f466a33d021c911db80d933bde411"
            raw=list(csv.DictReader(io.StringIO(read("grouped_fivefold_cv_v11/fold_1/official_metrics/case_metrics_official.csv").decode("utf-8-sig"))))
            by_id={r["case_id"]:r for r in raw}
            assert len(raw)==len(members) and set(by_id)==set(members)
            for c in members:
                values={r:{k:float(by_id[c][r+"_"+k]) for k in ("Num_TP","Num_FN","LesionWise_Score_Dice","LesionWise_Score_HD95")} for r in regions}
                records.append({"best":values,"final":values})
            continue
        folder=base/f"fixed_epoch_fold{fold}"
        scores=folder/"scores"
        present={p.stem for p in scores.glob("BraTS-*.json")}
        if set(members)!=present:
            sys.exit(f"Fold {fold}: {len(set(members)-present)} missing, {len(present-set(members))} unexpected. No summary generated.")
        subprocess.run([sys.executable,str(base/"check_best_replay_drift.py"),sys.argv[1],str(fold)],check=True)
        for c in members:
            v=json.loads((scores/(c+".json")).read_text())
            meta=json.loads((folder/(c+".json")).read_text())
            assert meta["fold"]==fold and v["case_id"]==meta["case_id"]==c and v["hashes"]==meta["output_hashes"]
            records.append(v)
subjects,index=np.unique([ids[c] for c in cases],return_inverse=True)
assert len(records)==len(cases)==len(set(cases))
if selection=="pooled": assert set(cases)==ids.keys() and len(cases)==1155 and len(subjects)==521
draws=np.random.default_rng(20260808).integers(len(subjects),size=(10000,len(subjects)))
rows=[]
for metric in ["LesionWise_Score_Dice","LesionWise_Score_HD95","sensitivity"]:
    for region in (regions+["six-region mean"] if metric=="LesionWise_Score_Dice" else regions):
        arms=[]
        for arm in ["best","final"]:
            def vals(r,k):
                return np.array([v[arm][r][k] for v in records])
            if metric=="sensitivity":
                num=vals(region,"Num_TP")
                den=num+vals(region,"Num_FN")
            else:
                num=(np.mean([vals(r,metric) for r in regions],axis=0) if region=="six-region mean" else vals(region,metric))
                den=np.ones(len(cases))
            b=lambda x: np.bincount(index,weights=x,minlength=len(subjects))[draws].sum(axis=1)
            denominators=b(den)
            assert denominators.min()>0
            arms.append((num.sum()/den.sum(),b(num)/denominators))
        lo,hi=np.percentile(arms[1][1]-arms[0][1],[2.5,97.5])
        rows.append((region,metric,arms[0][0],arms[1][0],arms[1][0]-arms[0][0],lo,hi))
assert np.isfinite(np.array([r[2:] for r in rows])).all()
root.mkdir(exist_ok=True)
with (root/"paired_summary.tmp").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["region","metric","best","final","final_minus_best","paired_lower95","paired_upper95"])
    w.writerows(rows)
(root/"paired_summary.tmp").replace(root/"paired_summary.csv")
