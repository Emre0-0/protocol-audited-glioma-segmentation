"""Build isolated development cache for the archived specialist plan."""
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

drive = Path(sys.argv[1])
root = Path(__file__).resolve().parent
run = root / "specialist_crossfit"
ds = "Dataset503_BraTS2024Grouped4Ch"
dsdir = run / "preprocessed" / ds
dsdir.mkdir(parents=True,exist_ok=True)
os.environ.update(nnUNet_preprocessed=str(dsdir.parent),nnUNet_raw=str(run/"raw"),
                  nnUNet_results=str(run/"results"),OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1",ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1")
import numpy as np
import torch
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

torch.set_num_threads(1)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
source = drive/"processed/nnunet_preprocessed"/ds
audit = json.loads((root/"specialist_input_audit.json").read_bytes())
plan = "nnUNetResEncUNet9GBPlans.json"
for name in (plan,"splits_final.json","dataset.json","dataset_fingerprint.json"):
    src = source/name
    dst = dsdir/name
    if not dst.exists(): shutil.copyfile(src,dst)
    assert digest(dst) == digest(src)
assert digest(dsdir/plan) == audit["plans_sha256"]
assert digest(dsdir/"splits_final.json") == audit["splits_sha256"]
rows = list(csv.DictReader((drive/"processed/brats_subject_grouped_protocol_v1/case_assignments.csv").open(encoding="utf-8-sig")))
dev = {r["case_id"] for r in rows if r["protocol_partition"]=="development"}
assert len(dev)==1155
for split in json.loads((dsdir/"splits_final.json").read_bytes()):
    tr,val = set(split["train"]),set(split["val"])
    assert tr|val == dev and not tr&val
pm = PlansManager(str(dsdir/plan))
config = pm.get_configuration("3d_fullres")
prep = config.preprocessor_class(verbose=False)
dataset = json.loads((dsdir/"dataset.json").read_bytes())
target = dsdir/config.data_identifier
done = run/"records"
for d in (target,done): d.mkdir(exist_ok=True)
prefix = "processed/nnunet_raw/"+ds+"/"
local = {}
for arc in sorted((Path.home()/"Downloads").glob("BR-*.zip")):
    with zipfile.ZipFile(arc) as z:
        for entry in z.infolist():
            key = entry.filename.partition("/")[2]
            if key.startswith(prefix): local.setdefault(key,(arc,entry.filename))
def inputs(c):
    return [prefix+f"imagesTr/{c}_{i:04d}.nii.gz" for i in range(4)]+[prefix+f"labelsTr/{c}.nii.gz"]
for i,c in enumerate(sorted(dev,key=lambda c:(-sum(k in local for k in inputs(c)),c)),1):
    rec = done/(c+".json")
    if rec.exists():
        r = json.loads(rec.read_bytes())
        assert all(digest(target/k)==v for k,v in r["outputs"].items())
        continue
    if shutil.disk_usage(run).free < 18*2**30: sys.exit("Stopped before next scan: 18 GiB reserve")
    with tempfile.TemporaryDirectory(dir=run) as tmp:
        p=Path(tmp)
        files=[]
        for key in inputs(c):
            f=p/Path(key).name
            if key in local:
                arc,name=local[key]
                with zipfile.ZipFile(arc) as z,z.open(name) as src,f.open("wb") as out: shutil.copyfileobj(src,out)
            else: shutil.copyfile(drive/key,f)
            files.append(f)
        assert all(f.stat().st_size for f in files)
        prep.run_case_save(str(p/c),list(map(str,files[:4])),str(files[4]),pm,config,dataset)
        a,s,_,_=nnUNetDatasetBlosc2(str(p),[c]).load_case(c)
        assert a.shape[0]==4 and s.shape[0]==1 and a.shape[1:]==s.shape[1:]
        assert np.isfinite(a[:]).all() and set(np.unique(s[:]))<=set(range(-1,5))
        del a,s
        output={}
        for suffix in (".b2nd","_seg.b2nd",".pkl"):
            f=p/(c+suffix)
            output[f.name]=digest(f)
            f.replace(target/f.name)
        record=dict(case=c,source_sha256={f.name:digest(f) for f in files},outputs=output)
        interim=rec.with_suffix(".tmp")
        interim.write_text(json.dumps(record))
        interim.replace(rec)
    print(i,c,flush=True)
assert {p.stem for p in done.glob("*.json")} == dev
print("COMPLETE: development cache",len(dev),flush=True)
