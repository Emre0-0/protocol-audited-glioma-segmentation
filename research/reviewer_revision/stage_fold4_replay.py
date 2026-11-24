"""Restore fold 4 checkpoint files for replay without using the GPU."""
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

here=Path(__file__).resolve().parent
downloads=Path.home()/"Downloads"
release=next(downloads.glob("BrainMRI*004.zip"))
model=here/"fixed_epoch_fold4/model"
sub=model/"fold_4"
sub.mkdir(parents=True,exist_ok=True)
def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()
with zipfile.ZipFile(release) as z:
    def read(suffix):
        n,=[n for n in z.namelist() if n.endswith("/"+suffix)]
        return z.read(n)
    caps={c["kind"]:c for c in json.loads(read("publication_integrity_v1/checkpoint_metadata/checkpoint_metadata.json"))["checkpoints"] if c["fold"]==4}
    rel=Path(caps["final"]["path"]).relative_to(r"C:\Users\Emre\Desktop\BR").as_posix()
    for name in ("plans.json","dataset.json"):
        data=read(rel.rsplit("/",2)[0]+"/"+name)
        p=model/name
        if not p.exists(): p.write_bytes(data)
        assert p.read_bytes()==data
best=here/"specialist_crossfit/warmstarts/fold_4/checkpoint_best.pth"
if not (sub/best.name).exists(): os.link(best,sub/best.name)
assert sha(sub/best.name)==json.loads((here/"specialist_crossfit/weight_checks/fold_4.json").read_bytes())["sha256"]
final=sub/"checkpoint_final.pth"
arc=downloads/"BR-20261121T131708Z-1-019.zip"
with zipfile.ZipFile(arc) as z:
    n,=[n for n in z.namelist() if n.endswith("/"+rel)]
    assert z.getinfo(n).file_size==caps["final"]["bytes"]
    if not final.exists():
        assert shutil.disk_usage(sub).free>caps["final"]["bytes"]+18*2**30
        with tempfile.TemporaryDirectory(dir=sub) as d:
            p=Path(d)/final.name
            with z.open(n) as src,p.open("xb") as dst: shutil.copyfileobj(src,dst,4<<20)
            p.rename(final)
    with z.open(n) as src: expected=hashlib.file_digest(src,"sha256").hexdigest()
assert sha(final)==expected
print("Verified fold 4 backup:",final.stat().st_size,expected)
