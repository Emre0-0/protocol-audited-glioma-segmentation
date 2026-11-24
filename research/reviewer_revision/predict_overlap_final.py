"""Pair exposed final-epoch predictions with historic clean final-epoch replay."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from stage_revision_input import index_inputs,stage

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(4<<20),b""): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("project",type=Path)
    p.add_argument("--verify-clean",action="store_true")
    args=p.parse_args()
    here=Path(__file__).resolve().parent
    run=here/"fold0_overlap_single_run"
    cohort=run/"cohort.json"
    assert sha(cohort)=="a92848d00f95c5a2b6eb6d63358c8449bf513162e6b0219794ba072ccffac288"
    split=json.loads(cohort.read_bytes())
    ids=sorted(split["validation_monitor"])
    assert len(set(ids))==len(ids)==183 and not set(ids)&set(split["train"])
    pilot=here/"fixed_epoch_pilot"
    model=pilot/"model"
    anchor=json.loads((pilot/"manifest.json").read_bytes())["files"]
    cp=model/"fold_0/checkpoint_final.pth"
    assert sha(cp)==anchor[str(cp.relative_to(pilot))]["sha256"]
    records={}
    clean=here/"fixed_epoch_fold0"
    for c in ids:
        r=json.loads((clean/(c+".json")).read_bytes())
        assert r["fold"]==0 and r["case_id"]==c and len(r["input_hashes"])==4
        digest=sha(clean/"final"/(c+".nii.gz"))
        assert digest==r["output_hashes"]["final"]
        records[c]=dict(input_hashes=r["input_hashes"],final_sha256=digest)
    path=run/"clean_final_manifest.json"
    blob=json.dumps(dict(checkpoint_sha256=anchor[str(cp.relative_to(pilot))]["sha256"],epoch=250,scans=records),sort_keys=True,indent=2).encode()
    if path.exists(): assert path.read_bytes()==blob
    else:
        tmp=path.with_suffix(".tmp");tmp.write_bytes(blob);tmp.replace(path)
    print("Verified 183 clean final predictions and their four input digests each.",flush=True)
    if args.verify_clean: return
    done=run/"training/exposed/completed.json"
    assert done.is_file(),"No completed exposed training; refusing inference"
    d=json.loads(done.read_bytes())
    exposed=Path(d["checkpoint"]).resolve()
    assert exposed.is_relative_to((run/"training/exposed/results").resolve()) and d["epoch"]==250 and exposed.name=="checkpoint_final.pth"
    os.environ["nnUNet_compile"]="false"
    import torch
    import nibabel as nib
    import numpy as np
    from pilot_checkpoint_inference import PacedPredictor
    torch.set_num_threads(2)
    predictor=PacedPredictor(allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model),(0,),"checkpoint_final.pth")
    learned=torch.load(exposed,map_location="cpu",weights_only=False)
    assert learned["current_epoch"]==250 and learned["init_args"]["fold"]==0
    assert tuple(learned["inference_allowed_mirroring_axes"])==tuple(predictor.allowed_mirroring_axes)==(0,1,2)
    state=learned.pop("network_weights")
    assert state.keys()==predictor.list_of_parameters[0].keys()
    assert all(v.shape==predictor.list_of_parameters[0][k].shape for k,v in state.items())
    predictor.list_of_parameters=[state]; del learned
    source=sha(exposed)
    dest=run/"exposed_predictions";dest.mkdir(exist_ok=True)
    markers=run/"exposed_records";markers.mkdir(exist_ok=True)
    prefix="processed/nnunet_raw/Dataset503_BraTS2024Grouped4Ch/imagesTr/"
    local=index_inputs((prefix,))
    for i,c in enumerate(ids,1):
        pred=dest/(c+".nii.gz")
        marker=markers/(c+".json")
        expected=dict(case=c,exposed_checkpoint=source,clean_reference=records[c])
        if marker.exists():
            assert json.loads(marker.read_bytes())==dict(**expected,output=sha(pred))
            continue
        assert shutil.disk_usage(run).free>10*2**30
        with tempfile.TemporaryDirectory(dir=run) as temp:
            t=Path(temp);files=[]
            for j,h in enumerate(records[c]["input_hashes"]):
                f=t/f"{c}_{j:04d}.nii.gz"
                stage(args.project,prefix+f.name,f,local)
                assert sha(f)==h
                files.append(str(f))
            predictor.predict_from_files_sequential([files],[str(t/c)])
            staged=t/pred.name
            img,ref=nib.load(staged),nib.load(clean/"final"/pred.name)
            assert img.shape==ref.shape and np.allclose(img.affine,ref.affine,atol=1e-5,rtol=0)
            assert set(np.unique(np.asanyarray(img.dataobj)))<=set(range(5))
            staged.replace(pred)
            m=t/"marker.json";m.write_text(json.dumps(dict(**expected,output=sha(pred))));m.replace(marker)
        print(f"Paired {i}/{len(ids)}: {c}",flush=True)
    assert {p.stem for p in markers.glob("*.json")}==set(ids)

if __name__=="__main__": main()
