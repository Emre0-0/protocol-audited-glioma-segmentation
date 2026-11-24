"""Stage one outcome-independent fold-0 replay control from read-only ZIPs."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

downloads = Path.home() / "Downloads"
root = Path(__file__).resolve().parent / "fixed_epoch_pilot"
ds = "Dataset503_BraTS2024Grouped4Ch"
model = f"experiments/nnunet_results/{ds}/nnUNetTrainer_250epochs__nnUNetResEncUNet10p5GBPlans__3d_fullres"
raw = f"processed/nnunet_raw/{ds}/imagesTr/"
pre = f"processed/nnunet_preprocessed/{ds}/splits_final.json"
files = {}
for arc in sorted(downloads.glob("BR-*.zip"))+sorted(downloads.glob("BrainMRI*004.zip")):
    with zipfile.ZipFile(arc) as z:
        for entry in z.infolist():
            _,_,key = entry.filename.partition("/")
            if not entry.is_dir() and (key.startswith(raw) or key.startswith(model+"/") or key==pre):
                files[key]=(arc,entry.filename)
def member(key):
    arc,path=files[key]
    with zipfile.ZipFile(arc) as z:
        return z.read(path)
val=sorted(json.loads(member(pre))[0]["val"])
assert len(val)==223
def inputs(case):
    return [raw+f"{case}_{i:04d}.nii.gz" for i in range(4)]
def ref(case):
    return model+"/fold_0/validation/"+case+".nii.gz"
case=next(c for c in val if all(k in files for k in inputs(c)+[ref(c)]))
records={}
def stage(key, dest):
    dest=root/dest
    dest.parent.mkdir(exist_ok=True,parents=True)
    arc,path=files[key]
    tmp=dest.with_suffix(".partial")
    with zipfile.ZipFile(arc) as z,z.open(path) as src,tmp.open("wb") as dst:
        shutil.copyfileobj(src,dst,1<<20)
    with tmp.open("rb") as f:
        digest=hashlib.file_digest(f,"sha256").hexdigest()
    if key.endswith("checkpoint_best.pth"):
        assert digest.upper()=="D75038EB1FE563D04FB19B955BC7B3380ECA09ABACBD81DFC9C238505CAFAA70"
    tmp.replace(dest)
    records[str(dest.relative_to(root))]=dict(archive=arc.name,member=path,sha256=digest)
for name in ["plans.json","dataset.json","fold_0/checkpoint_best.pth","fold_0/checkpoint_final.pth"]:
    stage(model+"/"+name,"model/"+name)
for key in inputs(case):
    stage(key,"images/"+Path(key).name)
stage(ref(case),"reference/"+case+".nii.gz")
(root/"manifest.json").write_text(json.dumps(dict(case_id=case,selection="first sorted fold-0 validation ID with all channels and archived prediction locally available",files=records),indent=2))
print(case,len(records),"files")
