"""Record a reproducible retrospective comparison before new specialists train."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PureWindowsPath

project = Path(sys.argv[1])
out = Path(__file__).resolve().parent/"specialist_crossfit"
archive = next((Path.home()/"Downloads").glob("BrainMRI*004.zip"))
def sha(data): return hashlib.sha256(data).hexdigest()
with zipfile.ZipFile(archive) as z:
    def read(rel):
        n, = [k for k in z.namelist() if k.endswith("/"+rel)]
        return z.read(n)
    frozen = json.loads(read("experiments/final_locked_test_v1/frozen_model_manifest.json"))
    meta = json.loads(read("experiments/publication_integrity_v1/checkpoint_metadata/checkpoint_metadata.json"))
    common = "research/grouped_general_sota_v1/"
    files = [common+"run_custom_trainer.py",common+"trainer.py","research/final_locked_test_v1/recreate_specialist.ps1"]
    hashes = {}
    for f in files:
        hashes[f]=sha(read(f))
        assert hashes[f]==sha((project/f).read_bytes())
    for path, recorded in frozen["code_sha256"].items():
        rel = PureWindowsPath(path).relative_to(r"C:\Users\Emre\Desktop\BR").as_posix()
        hashes[rel]=sha(read(rel))
        assert hashes[rel]==recorded.lower()
    starts = []
    for c in meta["checkpoints"]:
        if c["kind"]!="best" or c["fold"]==0: continue
        fold=c["fold"]
        rel=PureWindowsPath(c["path"]).relative_to(r"C:\Users\Emre\Desktop\BR").as_posix()
        assert c["fold_in_checkpoint"]==fold and all(c["checks"].values())
        assert (project/rel).stat().st_size==c["bytes"]
        starts.append(dict(fold=fold,path=rel,expected_bytes=c["bytes"],epoch=c["current_epoch"],
                           architecture_signature=c["architecture_signature_sha256"]))
    assert sorted(c["fold"] for c in starts)==[1,2,3,4]
    prefix="processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/"
    for f in ("splits_final.json","nnUNetResEncUNet9GBPlans.json"):
        if f=="splits_final.json":
            payload=read(prefix+f)
        else:
            payload=(project/prefix/f).read_bytes()
            old=project/"experiments/nnunet_results/Dataset503_BraTS2024Grouped4Ch/nnUNetTrainer_BraTSRegionAware120__nnUNetResEncUNet9GBPlans__3d_fullres/plans.json"
            historical=json.loads(old.read_bytes())
            historical.pop("continue_training",None)
            assert historical==json.loads(payload)
        hashes[prefix+f]=sha(payload)
        assert hashes[prefix+f]==sha((out/"preprocessed/Dataset503_BraTS2024Grouped4Ch"/f).read_bytes())
record = dict(
    purpose="Exploratory retrospective patient-grouped crossfit; historical design exposure and fold-level best-checkpoint selection remain.",
    inputs=starts,sha256=hashes,
    train="For each additional fold 1-4 initialize from that fold's baseline best, using proxy variant of original launcher, 9GB 3d_fullres, 120 epochs, selecting best EMA. Do not train on the 195 holdout scans.",
    rule=dict(name=frozen["selected_variant"],fallback=frozen["fallback"],threshold_search=False),
    comparison="Reproduced fold-0 rule plus four newly fitted specialists; compare paired baseline and V2 full-volume predictions on all 1155 OOF scans; each scan appears once.",
    primary_summary="V2 minus baseline mean six-region official lesion-wise Dice with paired patient-cluster percentile 95% CI, 10000 replicates; no configuration selection on these scores.",
    secondary="Report six regional Dice and HD95, ET sensitivity/precision and paired changes, ET-positive low scan-burden strata, and per-fold results including losses. No claim of independent external validation.")
path=out/"retrospective_recipe.json"
data=(json.dumps(record,indent=2,sort_keys=True)+"\n").encode()
if path.exists(): assert path.read_bytes()==data
else:
    with path.open("xb") as f: f.write(data)
print("recipe SHA256",sha(data),"folds",len(starts),"verified source/config hashes",len(hashes))
