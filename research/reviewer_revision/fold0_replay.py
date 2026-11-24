"""Paired best/final OOF inference, isolated and resumable by scan."""
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from unittest.mock import patch
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from pilot_checkpoint_inference import PacedPredictor
from stage_revision_input import stage

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

if __name__=="__main__":
    project=Path(sys.argv[1])
    root=Path(__file__).resolve().parent
    pilot=root/"fixed_epoch_pilot"
    fold=int(sys.argv[2]) if len(sys.argv)>2 else 0
    assert fold in (0,2,3,4)
    dest=root/f"fixed_epoch_fold{fold}"
    dest.mkdir(exist_ok=True)
    downloads=Path.home()/"Downloads"
    archive=next(downloads.glob("BrainMRI*004.zip"))
    ds="Dataset503_BraTS2024Grouped4Ch"
    inputs=f"processed/nnunet_raw/{ds}/imagesTr/"
    with zipfile.ZipFile(archive) as z:
        def read(suffix):
            key,=[k for k in z.namelist() if k.endswith("/"+suffix)]
            return z.read(key)
        split=json.loads(read(f"processed/nnunet_preprocessed/{ds}/splits_final.json"))[fold]
        checkpoints=json.loads(read("publication_integrity_v1/checkpoint_metadata/checkpoint_metadata.json"))["checkpoints"]
        cases=sorted(split["val"])
        dev={r["case_id"] for r in csv.DictReader(io.StringIO(read("brats_subject_grouped_protocol_v1/case_assignments.csv").decode("utf-8-sig"))) if r["protocol_partition"]=="development"}
    assert len(set(cases))==len(cases)==[223,247,243,225,217][fold] and set(cases)<=dev and not set(cases)&set(split["train"])
    files={}
    for arc in sorted(downloads.glob("BR-*.zip")):
        with zipfile.ZipFile(arc) as z:
            for k in z.namelist():
                key=k.partition("/")[2]
                if key.startswith(inputs) and k.endswith(".nii.gz"):
                    files[key]=arc,k
    manifest=json.loads((pilot/"manifest.json").read_text())
    os.environ["nnUNet_compile"]="false"
    torch.set_num_threads(2)
    model=pilot/"model"
    for rel,entry in manifest["files"].items():
        if Path(rel).parts[0]=="model": assert digest(pilot/rel)==entry["sha256"]
    predictor=PacedPredictor(allow_tqdm=False)
    if fold:
        src=project/"research/grouped_fivefold_cv_v11"
        with zipfile.ZipFile(archive) as z:
            assert (src/"trainer.py").read_bytes()==read("research/grouped_fivefold_cv_v11/trainer.py")
        sys.path.insert(0,str(src))
        from trainer import nnUNetTrainer_GroupedCV250Duty80 as cls
        import nnunetv2.inference.predict_from_raw_data as inference
        original=inference.recursive_find_python_class
        def resolve(folder,name,*args,**kwargs):
            return cls if name==cls.__name__ else original(folder,name,*args,**kwargs)
        caps={c["kind"]:c for c in checkpoints if c["fold"]==fold}
        model=project/Path(caps["best"]["path"]).relative_to(r"C:\Users\Emre\Desktop\BR").parent.parent
        if fold==4 and (dest/"model/fold_4/checkpoint_final.pth").is_file():
            model=dest/"model"
            assert digest(model/"fold_4/checkpoint_final.pth")=="c8b9179bd8c27f7ea2d4e68bd5a731825c496d78c359f5367da6199097fd5733"
        audit=json.loads((root/"specialist_crossfit/weight_checks"/f"fold_{fold}.json").read_bytes())
        for kind in ("best","final"):
            ck=model/f"fold_{fold}"/f"checkpoint_{kind}.pth"
            assert ck.stat().st_size==caps[kind]["bytes"]
            if kind=="best": assert digest(ck)==audit["sha256"]
        with patch.object(inference,"recursive_find_python_class",resolve):
            predictor.initialize_from_trained_model_folder(str(model),(fold,),"checkpoint_best.pth")
    else:
        predictor.initialize_from_trained_model_folder(str(model),(0,),"checkpoint_best.pth")
    assert tuple(predictor.allowed_mirroring_axes)==(0,1,2)
    final=torch.load(model/f"fold_{fold}/checkpoint_final.pth",map_location="cpu",weights_only=False)
    assert int(final["init_args"]["fold"])==fold and final["current_epoch"]==250
    assert tuple(final["inference_allowed_mirroring_axes"])==tuple(predictor.allowed_mirroring_axes)
    states={"best":predictor.list_of_parameters,"final":[final.pop("network_weights")]}
    del final
    for i,case in enumerate(cases):
        done=dest/(case+".json")
        outputs={kind:dest/kind/(case+".nii.gz") for kind in states}
        if done.exists():
            prior=json.loads(done.read_text())
            assert all(digest(p)==prior["output_hashes"][kind] for kind,p in outputs.items())
            continue
        if shutil.disk_usage(dest).free<10*(1<<30): raise RuntimeError("Disk safety margin reached")
        with tempfile.TemporaryDirectory(dir=dest) as tmp:
            tmp=Path(tmp)
            paths=[]
            for channel in range(4):
                key=inputs+f"{case}_{channel:04d}.nii.gz"
                target=tmp/Path(key).name
                if key in files:
                    arc,k=files[key]
                    with zipfile.ZipFile(arc) as z,z.open(k) as src,target.open("wb") as out:
                        shutil.copyfileobj(src,out,1<<20)
                else: stage(project,key,target,files)
                paths.append(str(target))
            img=nib.load(paths[0])
            for kind,out in outputs.items():
                predictor.list_of_parameters=states[kind]
                out.parent.mkdir(exist_ok=True)
                predictor.predict_from_files_sequential([paths],[str(tmp/(kind+"_"+case))])
                pred=tmp/(kind+"_"+case+".nii.gz")
                seg=nib.load(pred)
                assert seg.shape==img.shape and np.allclose(seg.affine,img.affine,atol=1e-5)
                assert set(np.unique(np.asarray(seg.dataobj)))<=set(range(5))
                pred.replace(out)
            marker=tmp/"done.json"
            marker.write_text(json.dumps({"case_id":case,"fold":fold,"input_hashes":[digest(Path(p)) for p in paths],"output_hashes":{k:digest(p) for k,p in outputs.items()}}))
            marker.replace(done)
            print(f"COMPLETE {i+1}/{len(cases)} {case}",flush=True)
