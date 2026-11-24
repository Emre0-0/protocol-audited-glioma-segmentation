"""Exercise frozen fold-0 development fusion on selected scans, without reference masks."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
from stage_revision_input import index_inputs,stage

root=Path(sys.argv[1])
full=sys.argv[2:]==["--all"]
assert full or len(sys.argv)==2
here=Path(__file__).resolve().parent
recipe=(here/"specialist_crossfit/retrospective_recipe.json").read_bytes()
assert hashlib.sha256(recipe).hexdigest()=="f9bd9199cd488b9a2fa61c4219306bd724bbbc14536eb03fc54929eb6c82264d"
hashes=json.loads(recipe)["sha256"]
for rel in (k for k in hashes if k.endswith(("/topology.py","/build_topology_guard.py"))):
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==hashes[rel]
sys.path.insert(0,str(root/"research/grouped_et_topology_guard_v2"))
from build_topology_guard import read_array,assert_geometry,component_filter,rc_topology_labels

c=json.loads((root/"research/final_locked_test_v1/config.json").read_bytes())
fc=c["frozen_candidate"]
exp=root/c["paths"]["experiment_dir"]/"reproduced_validation"
variant=fc["selected_variant"]
baseline=root/fc["baseline_validation_dir"]
specialist=root/fc["specialist_validation_dir"]
target=exp/"predictions"/variant
stats=json.loads((exp/"build_manifest.json").read_bytes())["variants"][variant]
split=here/"specialist_crossfit/preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json"
assert hashlib.sha256(split.read_bytes()).hexdigest()==hashes["processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json"]
ids=sorted(json.loads(split.read_bytes())[0]["val"])
assert len(ids)==223 and fc["fallback"]=="specialist_nonrc"
rollback_ids=set(stats["topology_rollback_case_ids"])
assert rollback_ids<=set(ids)
samples={}
case_stats={}
local=index_inputs(tuple(d.relative_to(root).as_posix()+"/" for d in (baseline,specialist,target)))
for cid in sorted(ids,key=lambda cid:(cid not in rollback_ids,cid)):
    if not full and cid in rollback_ids and "rollback" in samples: continue
    name=cid+".nii.gz"
    with tempfile.TemporaryDirectory() as tmp:
        data=[]
        for i,d in enumerate((baseline,specialist,target)):
            path=Path(tmp)/f"mask_{i}.nii.gz"
            stage(root,(d/name).relative_to(root),path,local)
            data.append(read_array(path))
        (im,a),(sm,b),(rm,ref)=data
    assert_geometry(im,sm,cid)
    assert_geometry(im,rm,cid)
    prop,details=component_filter(a,b,"specialist_nonrc")
    changed=not np.array_equal(prop,a)
    rollback=changed and not np.array_equal(rc_topology_labels(prop),rc_topology_labels(a))
    assert rollback==(cid in rollback_ids),cid
    result=a if rollback else prop
    assert np.array_equal(result,ref),cid
    branch="rollback" if rollback else "edit_applied" if changed else "unchanged"
    vox=int(np.count_nonzero(result!=a))
    samples.setdefault(branch,dict(case=cid,changed_voxels=vox))
    case_stats[cid]=dict(branch=branch,edited_voxels=vox,removed_components=0 if rollback else details["removed_et_components"])
    if full and len(case_stats)%25==0: print("Voxel matches",len(case_stats),flush=True)
    if not full and {"edit_applied","unchanged"}<=samples.keys() and (not rollback_ids or "rollback" in samples): break
assert {"edit_applied","unchanged"}<=samples.keys()
assert not rollback_ids or "rollback" in samples
report=samples
if full:
    assert set(case_stats)==set(ids)
    totals=dict(cases=len(case_stats),final_changed_cases=sum(v["edited_voxels"]>0 for v in case_stats.values()),
                final_removed_et_voxels=sum(v["edited_voxels"] for v in case_stats.values()),
                final_removed_et_components=sum(v["removed_components"] for v in case_stats.values()),
                topology_rollback_cases=len(rollback_ids))
    assert all(stats[k]==v for k,v in totals.items()),(stats,totals)
    report=dict(totals=totals,cases=case_stats)
out=here/"specialist_crossfit"/("fold0_full_mask_parity.json" if full else "fold0_mask_parity.json")
tmp=out.with_suffix(".tmp")
tmp.write_text(json.dumps(report,indent=2));tmp.replace(out)
print(json.dumps(report.get("totals",samples)))
