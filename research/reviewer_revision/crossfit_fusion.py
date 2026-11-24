"""Stage archived OOF baselines; apply the archived fold-0 V2 builder per new fold."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from stage_revision_input import index_inputs,stage

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""): h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("project",type=Path)
p.add_argument("archive",type=Path)
p.add_argument("fold",type=int,choices=range(1,5))
p.add_argument("--stage-only",action="store_true")
a=p.parse_args()
run=Path(__file__).resolve().parent/"specialist_crossfit"
recipe=run/"retrospective_recipe.json"
assert sha(recipe)=="f9bd9199cd488b9a2fa61c4219306bd724bbbc14536eb03fc54929eb6c82264d"
r=json.loads(recipe.read_bytes())
entry,=[e for e in r["inputs"] if e["fold"]==a.fold]
ds="Dataset503_BraTS2024Grouped4Ch"
split=run/"preprocessed"/ds/"splits_final.json"
assert sha(split)==r["sha256"][f"processed/nnunet_preprocessed/{ds}/splits_final.json"]
ids=set(json.loads(split.read_bytes())[a.fold]["val"])
assert len(ids)==[223,247,243,225,217][a.fold]
with zipfile.ZipFile(a.archive) as z:
    suffix=f"grouped_fivefold_cv_v11/fold_{a.fold}/official_metrics/metadata.json"
    k,=[k for k in z.namelist() if k.endswith("/"+suffix)]
    rel=Path(json.loads(z.read(k))["prediction_dir"]).relative_to(r"C:\Users\Emre\Desktop\BR")
assert rel==Path(entry["path"]).parent/"validation"
folder=run/"paired_fusion"/f"fold_{a.fold}"
baseline=folder/"archived_baseline"
baseline.mkdir(parents=True,exist_ok=True)
names={cid+".nii.gz" for cid in ids}
manifest=folder/"baseline_sha256.json"
if not manifest.exists():
    local=index_inputs((rel.as_posix()+"/",))
    hashes={}
    for name in sorted(names):
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            candidate=Path(tmp)/name
            stage(a.project,rel/name,candidate,local)
            digest=sha(candidate)
            dest=baseline/name
            if dest.exists(): assert sha(dest)==digest,name
            else: candidate.rename(dest)
            hashes[name]=digest
    tmp=manifest.with_suffix(".tmp")
    tmp.write_text(json.dumps(hashes,sort_keys=True));tmp.replace(manifest)
hashes=json.loads(manifest.read_bytes())
assert set(hashes)==names=={f.name for f in baseline.glob("*.nii.gz")}
assert all(sha(baseline/n)==h for n,h in hashes.items())
print("Archived baseline verified:",a.fold,len(names),flush=True)
if a.stage_only: sys.exit(0)
specialist=run/"best_validation"/f"fold_{a.fold}"/"validation"
assert {f.name for f in specialist.glob("*.nii.gz")}==names,"Incomplete specialist predictions"
assert (specialist/"summary.json").is_file(),"Native validation has not finished"
trainer="nnUNetTrainer_BraTSRegionAware120__nnUNetResEncUNet9GBPlans__3d_fullres"
best=run/"results"/ds/trainer/f"fold_{a.fold}"/"checkpoint_best.pth"
pred=json.loads((specialist.parent/"prediction_manifest.json").read_bytes())
assert pred["fold"]==a.fold and pred["checkpoint_best_sha256"]==sha(best)
assert set(pred["predictions"])==names and all(sha(specialist/n)==h for n,h in pred["predictions"].items())
base_weights=run/"warmstarts"/f"fold_{a.fold}"/"checkpoint_best.pth"
if not base_weights.exists(): base_weights=a.project/entry["path"]
audit=json.loads((run/"weight_checks"/f"fold_{a.fold}.json").read_bytes())
assert sha(base_weights)==audit["sha256"] and best.is_file()
script="research/final_locked_test_v1/build_reproduced_validation.py"
for k in (script,"research/grouped_et_topology_guard_v2/build_topology_guard.py","research/grouped_et_topology_guard_v2/topology.py"):
    assert sha(a.project/k)==r["sha256"][k],k
assert not (folder/"reproduced_validation").exists(),"Inspect previous fusion before retry"
config=dict(frozen_candidate=dict(baseline_validation_dir=str(baseline),specialist_validation_dir=str(specialist),expected_validation_cases=len(ids)),
            paths=dict(experiment_dir=str(folder)),models=dict(baseline=dict(checkpoint=str(base_weights)),specialist=dict(checkpoint=str(best))))
cfg=folder/"build_config.json"
cfg.write_text(json.dumps(config,indent=2))
subprocess.run([sys.executable,str(a.project/script),"--config",str(cfg)],check=True)
report=json.loads((folder/"reproduced_validation/build_manifest.json").read_bytes())
assert report["locked_test_used"] is False and report["ground_truth_used_during_build"] is False
assert report["sources"]["specialist_checkpoint_sha256"].lower()==sha(best)
assert report["sources"]["baseline_checkpoint_sha256"].lower()==audit["sha256"]
