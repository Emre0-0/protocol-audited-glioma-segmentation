"""Independently reconcile 6 region volumes with archived OOF scoring inputs."""
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
import nibabel as nib
import numpy as np

root=Path(__file__).resolve().parent/"specialist_crossfit"
ds=root/"preprocessed/Dataset503_BraTS2024Grouped4Ch"
manifest=json.loads((root/"validation_reference_hashes.json").read_bytes())
splits=json.loads((ds/"splits_final.json").read_bytes())
regions={"WT":[1,2,3],"TC":[1,3],"NETC":[1],"SNFH":[2],"ET":[3],"RC":[4]}
out=root/"reference_volume_checks"
out.mkdir(exist_ok=True)
with zipfile.ZipFile(sys.argv[1]) as z:
    seen=set()
    for fold,split in enumerate(splits):
        prefix=f"grouped_fivefold_cv_v11/fold_{fold}/official_metrics" if fold else "grouped_delta_roi_v1/official_metrics/baseline"
        key,=[k for k in z.namelist() if k.endswith("/"+prefix+"/case_metrics_official.csv")]
        rows=list(csv.DictReader(io.StringIO(z.read(key).decode("utf-8-sig"))))
        ids={r["case_id"] for r in rows}
        assert ids==set(split["val"]) and len(ids)==len(rows) and not ids&seen
        seen.update(ids)
        for row in rows:
            c=row["case_id"]
            name=c+".nii.gz"
            p=ds/"gt_segmentations"/name
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            assert digest==manifest[c],c
            img=nib.load(p)
            assert np.allclose(img.header.get_zooms()[:3],1,rtol=0,atol=1e-6) and len(img.shape)==3,c
            data=np.asanyarray(img.dataobj)
            assert data.dtype.kind in "iu" and data.max()<=4 and data.min()>=0,c
            counts=np.bincount(data.ravel(order="K"),minlength=5)
            vols={k:int(counts[v].sum()) for k,v in regions.items()}
            assert all(v==float(row[k+"_GT_Complete_Volume"]) for k,v in vols.items()),(c,vols)
            record=dict(case=c,fold=fold,reference_sha256=digest,volumes_mm3=vols)
            tmp=out/(c+".tmp")
            tmp.write_text(json.dumps(record));tmp.replace(out/(c+".json"))
        print("Verified fold",fold,len(rows),flush=True)
    assert seen==manifest.keys() and len(seen)==1155
print("Complete: 1155 reference masks, 6930 official volumes.",flush=True)
