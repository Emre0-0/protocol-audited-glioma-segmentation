"""Compare archived best and replayed best OOF masks, without using outcomes."""
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np

r=Path(__file__).resolve().parent
replay=r/"fixed_epoch_fold3"
base=r/"specialist_crossfit/paired_fusion/fold_3"
expected=set(json.loads((r/"specialist_crossfit/preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json").read_bytes())[3]["val"])
arch=json.loads((base/"baseline_sha256.json").read_bytes())
assert len(expected)==225 and set(arch)=={c+".nii.gz" for c in expected}
counts=[]; identical_bytes=0; transitions=np.zeros((5,5),dtype=np.int64)
for c in sorted(expected):
    name=c+".nii.gz"
    pair=[base/"archived_baseline"/name,replay/"best"/name]
    ref=json.loads((replay/(c+".json")).read_bytes())
    assert c==ref["case_id"] and ref["fold"]==3
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in pair]
    assert hashes==[arch[name],ref["output_hashes"]["best"]]
    identical_bytes+=hashes[0]==hashes[1]
    if hashes[0]==hashes[1]:
        counts.append(0);continue
    a,b=map(nib.load,pair)
    assert a.shape==b.shape and np.allclose(a.affine,b.affine,rtol=0,atol=1e-5)
    x,y=np.asanyarray(a.dataobj),np.asanyarray(b.dataobj)
    m=x!=y
    old,new=x[m].astype(int),y[m].astype(int)
    assert np.isin(old,range(5)).all() and np.isin(new,range(5)).all()
    np.add.at(transitions,(old,new),1)
    counts.append(len(old))
assert transitions.sum()==sum(counts)
result=dict(fold=3,checkpoint="best",scans=len(counts),identical_files=identical_bytes,
            identical_masks=counts.count(0),changed_voxels_total=sum(counts),
            changed_voxels_per_scan_median=float(np.median(counts)),changed_voxels_per_scan_max=max(counts),
            transitions_archive_rows_replay_columns=transitions.tolist())
out=replay/"voxel_reproducibility.json"
out.write_text(json.dumps(result,indent=2))
print(result)
