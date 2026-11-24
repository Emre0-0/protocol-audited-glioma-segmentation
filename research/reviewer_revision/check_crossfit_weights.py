"""Exercise nnU-Net's actual warm-start on CPU, comparing all post-load tensors."""
import gc
import hashlib
import json
import os
import sys
import shutil
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

project=Path(sys.argv[1])
run=Path(__file__).resolve().parent/"specialist_crossfit"
recipe_file=run/"retrospective_recipe.json"
assert hashlib.sha256(recipe_file.read_bytes()).hexdigest()=="f9bd9199cd488b9a2fa61c4219306bd724bbbc14536eb03fc54929eb6c82264d"
recipe=json.loads(recipe_file.read_bytes())
os.environ.update(nnUNet_raw=str(run/"raw"),nnUNet_preprocessed=str(run/"preprocessed"),
                  nnUNet_results=str(run/"runtime_check"),nnUNet_compile="false",OPENBLAS_NUM_THREADS="1")
sys.path.insert(0,str(run/"trainer_source"))
import torch
from run_custom_trainer import build_trainer, load_pretrained_weights

torch.set_num_threads(1)
def check(entry,local=None):
    fold=entry["fold"]
    path=local or project/entry["path"]
    assert path.stat().st_size==entry["expected_bytes"]
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""): h.update(block)
    cp=torch.load(path,map_location="cpu",weights_only=False)
    assert cp["current_epoch"]==entry["epoch"] and int(cp["init_args"]["fold"])==fold
    weights=cp["network_weights"]
    torch.manual_seed(743)
    t=build_trainer(SimpleNamespace(dataset_id=503,plans="nnUNetResEncUNet9GBPlans",
                    configuration="3d_fullres",fold=fold,device="cpu",variant="proxy",continue_training=False))
    t.initialize()
    st=t.network.state_dict()
    heads={k:v.clone() for k,v in st.items() if ".seg_layers." in k}
    assert heads and set(st)==set(weights) and any(not torch.equal(v,weights[k]) for k,v in heads.items())
    # Remap deserialization to CPU without changing the library's transfer logic.
    with patch.object(torch,"load",return_value=cp) as loader:
        load_pretrained_weights(t.network,str(path))
        loader.assert_called_once()
    st=t.network.state_dict()
    for k,v in st.items():
        assert torch.equal(v,heads[k] if k in heads else weights[k]),(fold,k)
    result=dict(fold=fold,epoch=cp["current_epoch"],sha256=h.hexdigest(),
                transferred_entries=len(st)-len(heads),retained_head_entries=len(heads))
    print(result,flush=True)
    return result

out=run/"weight_checks"
out.mkdir(exist_ok=True)
for entry in recipe["inputs"]:
    if len(sys.argv)>2 and entry["fold"]!=int(sys.argv[2]): continue
    if len(sys.argv)>3:
        with zipfile.ZipFile(sys.argv[3]) as z,tempfile.TemporaryDirectory(dir=out) as tmp:
            key,=[k for k in z.namelist() if k.endswith("/"+entry["path"])]
            assert z.getinfo(key).file_size==entry["expected_bytes"]
            local=Path(tmp)/"checkpoint_best.pth"
            with z.open(key) as src,local.open("wb") as dst: shutil.copyfileobj(src,dst)
            result=check(entry,local)
            result["verified_copy"]=Path(sys.argv[3]).name
    else: result=check(entry)
    (out/f"fold_{entry['fold']}.json").write_text(json.dumps(result,indent=2))
    gc.collect()
