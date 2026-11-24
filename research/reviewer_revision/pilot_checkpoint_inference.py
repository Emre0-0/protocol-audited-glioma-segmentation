import gc
import argparse
import json
import os
import time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class PacedPredictor(nnUNetPredictor):
    def _internal_maybe_mirror_and_predict(self,x):
        torch.cuda.synchronize()
        start=time.monotonic()
        y=super()._internal_maybe_mirror_and_predict(x)
        torch.cuda.synchronize()
        time.sleep((time.monotonic()-start)*0.25)
        return y

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--paired-final",action="store_true",help="Use preserved same-runtime best as control; never overwrite it.")
    args=parser.parse_args()
    root=Path(__file__).resolve().parent/"fixed_epoch_pilot"
    cid=json.loads((root/"manifest.json").read_text())["case_id"]
    os.environ["nnUNet_compile"]="false"
    torch.set_num_threads(2)
    ref=nib.load(root/"reference"/(cid+".nii.gz"))
    reference=np.asarray(ref.dataobj)
    report=root/"pilot_comparison.json"
    summary=json.loads(report.read_text()) if args.paired_final else {}
    if args.paired_final:
        best_img=nib.load(root/"replay_best"/(cid+".nii.gz"))
        assert best_img.shape==ref.shape and np.allclose(best_img.affine,ref.affine,atol=1e-5)
        best_arr=np.asarray(best_img.dataobj)
        assert int(np.count_nonzero(best_arr!=reference))==summary["best"]["voxels_different_from_archived_best"]
    for kind in (["final"] if args.paired_final else ["best","final"]):
        predictor=PacedPredictor(allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(str(root/"model"),(0,),"checkpoint_"+kind+".pth")
        assert tuple(predictor.allowed_mirroring_axes)==(0,1,2)
        out=root/("replay_"+kind)
        out.mkdir(exist_ok=True)
        predictor.predict_from_files_sequential([[str(p) for p in sorted((root/"images").glob(cid+"_*.nii.gz"))]],[str(out/cid)])
        img=nib.load(out/(cid+".nii.gz"))
        arr=np.asarray(img.dataobj)
        assert arr.shape==reference.shape and np.allclose(img.affine,ref.affine,atol=1e-5)
        summary[kind]=dict(voxels_different_from_archived_best=int(np.count_nonzero(arr!=reference)),per_label_changed=[int(np.count_nonzero((arr==i)!=(reference==i))) for i in range(5)])
        if args.paired_final:
            summary[kind]["voxels_different_from_replayed_best"]=int(np.count_nonzero(best_arr!=arr))
        report.write_text(json.dumps(summary,indent=2))
        print(kind,summary[kind],flush=True)
        if kind=="best" and summary[kind]["voxels_different_from_archived_best"]:
            break
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
