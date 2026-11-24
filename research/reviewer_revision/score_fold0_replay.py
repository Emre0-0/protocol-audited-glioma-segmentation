"""Score completed pairs only. Resume from atomically saved per-scan results."""
import hashlib
import json
import os
import sys
import tempfile
import time
import psutil
from pathlib import Path
import numpy as np
from stage_revision_input import index_inputs,stage

fold=int(sys.argv[3]) if len(sys.argv)>3 else 0
assert fold in (0,2,3,4)
root=Path(__file__).resolve().parent/f"fixed_epoch_fold{fold}"
project=Path(sys.argv[1]).resolve()
splits=json.loads((project/"processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json").read_bytes())
expected=set(splits[fold]["val"])
assert len(expected)==[223,247,243,225,217][fold]
gt_dir="processed/nnunet_raw/Dataset503_BraTS2024Grouped4Ch/labelsTr/"
local=index_inputs((gt_dir,))
sys.path.insert(0,str(project/"research/vendor/BraTS-2024-Metrics"))
np.Inf=np.inf
from metrics_GLI import get_LesionWiseResults

dest=root/"scores"
dest.mkdir(exist_ok=True)
keys=["Num_TP","Num_FP","Num_FN","LesionWise_Score_Dice","LesionWise_Score_HD95"]
def ready():
    seen=set()
    producer=psutil.Process(int(sys.argv[2])) if len(sys.argv)>2 and int(sys.argv[2])>0 else None
    limit=int(sys.argv[4]) if len(sys.argv)>4 else None
    while True:
        batch=sorted(set(root.glob("BraTS-*.json"))-seen)
        for path in batch:
            while psutil.virtual_memory().available < 4*(1<<30):
                time.sleep(10)
            yield path
            seen.add(path)
            if limit is not None and len(seen)>=limit:
                return
        if len(seen)==len(expected) or producer is None:
            return
        if not producer.is_running():
            if set(root.glob("BraTS-*.json"))-seen:
                continue
            raise RuntimeError(f"Inference stopped; scored {len(seen)} of {len(expected)} pairs.")
        time.sleep(5)

for marker in ready():
    m=json.loads(marker.read_text())
    cid=m["case_id"]
    assert cid in expected and m["fold"]==fold
    output=dest/(cid+".json")
    if output.exists():
        assert json.loads(output.read_text())["hashes"]==m["output_hashes"]
        continue
    item={"case_id":cid,"hashes":m["output_hashes"]}
    for kind in ["best","final"]:
        pred=root/kind/(cid+".nii.gz")
        assert hashlib.sha256(pred.read_bytes()).hexdigest()==item["hashes"][kind]
        cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            gt=Path(tmp)/(cid+".nii.gz")
            stage(project,gt_dir+gt.name,gt,local)
            try:
                os.chdir(tmp)
                scores,_=get_LesionWiseResults(pred.as_posix(),gt.as_posix(),"BraTS-GLI")
            finally:
                os.chdir(cwd)
        df=scores.set_index("Labels")[keys]
        assert set(df.index)=={"WT","TC","NETC","SNFH","ET","RC"} and np.isfinite(df.to_numpy()).all()
        item[kind]=df.to_dict(orient="index")
    partial=output.with_suffix(".partial")
    partial.write_text(json.dumps(item,allow_nan=False))
    partial.replace(output)
    print("SCORED",cid,flush=True)
