"""Recover ET lesion size/hit pairs; replay the official scorer where aggregation was lossy."""
import argparse
import csv
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from stage_revision_input import index_inputs,stage

def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("--replay", type=int, default=0)
    a = p.parse_args()
    cache = Path(__file__).resolve().parent / "et_rows"
    cache.mkdir(exist_ok=True)
    replay = known = pending = 0
    local=index_inputs(("processed/nnunet_raw/Dataset503_", "experiments/nnunet_results/Dataset503_", "experiments/grouped_"))
    with zipfile.ZipFile(a.archive) as z:
        def read(suffix):
            key, = [k for k in z.namelist() if k.endswith("/"+suffix)]
            return z.read(key).decode("utf-8-sig")
        for fold in range(5):
            base = "experiments/" + (f"grouped_fivefold_cv_v11/fold_{fold}/official_metrics" if fold else "grouped_delta_roi_v1/official_metrics/baseline")
            md = json.loads(read(base+"/metadata.json"))
            for r in csv.DictReader(io.StringIO(read(base+"/case_metrics_official.csv"))):
                cid = r["case_id"]
                get = lambda k: int(float(r["ET_"+k]))
                tp, fn = get("Num_TP"), get("Num_FN")
                dest = cache / (cid+".json")
                if dest.exists():
                    rows = json.loads(dest.read_text())
                elif tp+fn == 0:
                    rows = []
                elif get("lesion_detail_rows") == 1 and tp+fn == 1:
                    rows = [[get("GT_Complete_Volume"), tp]]
                elif replay < a.replay:
                    import numpy as np
                    np.Inf = np.inf
                    sys.path.insert(0,str(a.root/"research/vendor/BraTS-2024-Metrics"))
                    from metrics_GLI import get_LesionWiseResults
                    paths = [Path(md[k]).relative_to(r"C:\Users\Emre\Desktop\BR") / (cid+".nii.gz") for k in ("prediction_dir","ground_truth_dir")]
                    cwd = Path.cwd()
                    with tempfile.TemporaryDirectory() as tmp:
                        try:
                            os.chdir(tmp)
                            staged=[]
                            for sub,rel in zip(("pred","gt"),paths):
                                target=Path(tmp)/sub/(cid+".nii.gz")
                                target.parent.mkdir()
                                stage(a.root,rel,target,local)
                                staged.append(target.as_posix())
                            _, detail = get_LesionWiseResults(*staged,"BraTS-GLI")
                        finally:
                            os.chdir(cwd)
                    rows = [[int(t.gt_lesion_vol),int(t._asdict()["detected"])] for t in detail.query("Label == 'ET' and gt_lesion_vol > 10").assign(detected=lambda x:x.predicted_lesion_numbers.map(len)>0).itertuples()]
                    replay += 1
                else:
                    pending += 1
                    continue
                assert all(vol > 10 and hit in (0,1) for vol,hit in rows),cid
                assert (len(rows),sum(h for _,h in rows)) == (tp+fn,tp),cid
                if not dest.exists():
                    partial = dest.with_suffix(".tmp")
                    partial.write_text(json.dumps(rows))
                    partial.replace(dest)
                known += len(rows)
    print(f"validated lesions={known}; pending scans={pending}; replays this run={replay}")

if __name__ == "__main__":
    main()
