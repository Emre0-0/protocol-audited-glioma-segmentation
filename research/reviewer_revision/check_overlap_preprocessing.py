"""Test decoded cache equivalence against fold-0 preprocessing on actual scans."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from stage_revision_input import index_inputs,stage

here=Path(__file__).resolve().parent
sp=here/"specialist_crossfit"
ds="Dataset503_BraTS2024Grouped4Ch"
dsdir=sp/"preprocessed"/ds
project=Path(sys.argv[1])
os.environ.update(nnUNet_preprocessed=str(dsdir.parent),nnUNet_raw=str(sp/"raw"),nnUNet_results=str(sp/"results"),
                  OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1")
import numpy as np
import torch
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
torch.set_num_threads(1)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

folder=here/"fold0_overlap_single_run"
cohort=folder/"cohort.json"
assert sha(cohort)=="a92848d00f95c5a2b6eb6d63358c8449bf513162e6b0219794ba072ccffac288"
f=json.loads(cohort.read_bytes())
old=set(json.loads((dsdir/"splits_final.json").read_bytes())[0]["train"])
groups=dict(shared_training=old&set(f["train"]),inserted_sibling=set(f["excluded_sibling_scans"]),fixed_evaluation=set(f["fixed_index_scans"]))
complete={p.stem for p in (sp/"records").glob("*.json")}
plans=here/"fixed_epoch_pilot/model/plans.json"
assert sha(plans)=="8a5ff20900da81149466ce793a9ac6474ea33644f53784ee197d72851829687e"
pm=PlansManager(str(plans))
cfg=pm.get_configuration("3d_fullres")
dataset=json.loads((dsdir/"dataset.json").read_bytes())
prefix=f"processed/nnunet_raw/{ds}/"
local=index_inputs((prefix,))
checks=[]
for name,ids in groups.items():
    c=min(ids&complete)
    rec=json.loads((sp/"records"/(c+".json")).read_bytes())
    cache=dsdir/cfg.data_identifier
    assert all(sha(cache/k)==v for k,v in rec["outputs"].items())
    with tempfile.TemporaryDirectory() as tmp:
        t=Path(tmp);paths=[]
        for i in range(5):
            rel=(f"imagesTr/{c}_{i:04d}.nii.gz" if i<4 else f"labelsTr/{c}.nii.gz")
            path=t/Path(rel).name
            stage(project,prefix+rel,path,local)
            assert sha(path)==rec["source_sha256"][path.name]
            paths.append(str(path))
        cfg.preprocessor_class(verbose=False).run_case_save(str(t/c),paths[:4],paths[4],pm,cfg,dataset)
        a,s,_,props=nnUNetDatasetBlosc2(str(t),[c]).load_case(c)
        b,v,_,prior=nnUNetDatasetBlosc2(str(cache),[c]).load_case(c)
        assert a.shape==b.shape and s.shape==v.shape and a.dtype==b.dtype and s.dtype==v.dtype
        for j in range(a.shape[0]): assert np.array_equal(a[j:j+1],b[j:j+1]),(name,"MRI",j)
        assert np.array_equal(s[:],v[:]),(name,"mask")
        def equal(x,y):
            if isinstance(x,dict):
                return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
            return np.array_equal(x,y)
        assert equal(props,prior),(name,"sampling or image geometry")
        checks.append(dict(stratum=name,case=c,image_shape=list(a.shape),geometry_and_sampling_fields=sorted(props)))
        del a,b,s,v
out=folder/"preprocessing_spot_check.json"
out.write_text(json.dumps(checks,indent=2))
print([(r["stratum"],r["case"]) for r in checks],"identical decoded arrays and metadata")
