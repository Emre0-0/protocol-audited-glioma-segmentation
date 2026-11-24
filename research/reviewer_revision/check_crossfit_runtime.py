"""Resolve the archived specialist on CPU; do not initialize a network or train."""
import hashlib
import importlib.metadata as versions
import json
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

base=Path(sys.argv[1])
run=Path(__file__).resolve().parent/"specialist_crossfit"
recipe_file=run/"retrospective_recipe.json"
recipe=json.loads(recipe_file.read_bytes())
def sha(b): return hashlib.sha256(b).hexdigest()
assert sha(recipe_file.read_bytes())=="f9bd9199cd488b9a2fa61c4219306bd724bbbc14536eb03fc54929eb6c82264d"
code=run/"trainer_source"
code.mkdir(exist_ok=True)
deps={}
with zipfile.ZipFile(next((Path.home()/"Downloads").glob("BrainMRI*004.zip"))) as z:
    for name in ("run_custom_trainer.py","trainer.py","losses.py"):
        rel="research/grouped_general_sota_v1/"+name
        key,=[k for k in z.namelist() if k.endswith("/"+rel)]
        data=z.read(key)
        assert data==(base/rel).read_bytes()
        deps[name]=sha(data)
        if rel in recipe["sha256"]: assert deps[name]==recipe["sha256"][rel]
        out=code/name
        if out.exists(): assert sha(out.read_bytes())==deps[name]
        else: out.write_bytes(data)
# Use an isolated output root, including for logger creation by constructors.
temp_results=run/"runtime_check"
for folder in (temp_results,run/"raw"): folder.mkdir(exist_ok=True)
os.environ.update(nnUNet_raw=str(run/"raw"),nnUNet_preprocessed=str(run/"preprocessed"),
                  nnUNet_results=str(temp_results),nnUNet_compile="false",OPENBLAS_NUM_THREADS="1")
sys.path.insert(0,str(code))
from run_custom_trainer import build_trainer
checked=[]
for entry in recipe["inputs"]:
    fold=entry["fold"]
    trainer=build_trainer(SimpleNamespace(dataset_id=503,plans="nnUNetResEncUNet9GBPlans",
                         configuration="3d_fullres",fold=fold,device="cpu",variant="proxy",continue_training=False))
    dest=Path(trainer.output_folder)
    assert temp_results in dest.parents and dest.name==f"fold_{fold}"
    assert trainer.num_epochs==120 and trainer.initial_lr==0.001 and trainer.save_every==20
    assert trainer.region_auxiliary_weight==0.35 and trainer.gpu_duty_cycle_target==0.80
    assert trainer.network is None and trainer.device.type=="cpu"
    assert Path(trainer.preprocessed_dataset_folder).is_dir()
    checked.append({"fold":fold,"trainer":type(trainer).__name__,"output":str(dest),
                    "train_batches":trainer.num_iterations_per_epoch,"validation_batches":trainer.num_val_iterations_per_epoch})
record=dict(source_hashes=deps,folds=checked,versions={v:versions.version(v) for v in ("torch","nnunetv2","dynamic-network-architectures")})
(run/"runtime_check.json").write_text(json.dumps(record,indent=2))
print("Verified four CPU trainer constructors:",[r["fold"] for r in checked])
