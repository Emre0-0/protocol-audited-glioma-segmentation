"""Check development-only raw file inventory and patient-disjoint specialist folds."""
import csv
import hashlib
import json
import sys
from pathlib import Path

base = Path(sys.argv[1]) / "processed"
ds = "Dataset503_BraTS2024Grouped4Ch"
prep = base / "nnunet_preprocessed" / ds
raw = base / "nnunet_raw" / ds
assign = base / "brats_subject_grouped_protocol_v1/case_assignments.csv"
rows = list(csv.DictReader(assign.open(encoding="utf-8-sig")))
dev = {r["case_id"]: r["subject_id"] for r in rows if r["protocol_partition"] == "development"}
holdout = {r["case_id"]: r["subject_id"] for r in rows if r["protocol_partition"] != "development"}
assert len(dev) == 1155 and len(holdout) == 195 and len(rows) == 1350
assert not (set(dev.values()) & set(holdout.values()))
planpath = prep / "nnUNetResEncUNet9GBPlans.json"
plan = json.loads(planpath.read_bytes())["configurations"]["3d_fullres"]
splitpath = prep / "splits_final.json"
splits = json.loads(splitpath.read_bytes())
seen = set()
summary = []
for fold, parts in enumerate(splits):
    tr, val = (set(parts[k]) for k in ("train","val"))
    assert len(tr) == len(parts["train"]) and len(val) == len(parts["val"])
    assert tr | val == set(dev) and not tr & val and not seen & val
    st, sv = ({dev[c] for c in partition} for partition in (tr,val))
    assert not st & sv
    seen.update(val)
    summary.append(dict(fold=fold, train_scans=len(tr), validation_scans=len(val),
                        train_patients=len(st), validation_patients=len(sv)))
assert len(splits) == 5 and seen == set(dev)
inventory = {}
for folder, names in (("imagesTr",{f"{c}_{i:04d}.nii.gz" for c in dev for i in range(4)}),
                      ("labelsTr",{f"{c}.nii.gz" for c in dev})):
    found = {p.name:p.stat().st_size for p in (raw / folder).iterdir()
             if p.name in names}
    inventory[folder] = dict(expected=len(names),nonzero=sum(x>0 for x in found.values()),
                             missing=sorted(names-found.keys()), zero_byte=sorted(k for k,v in found.items() if not v))
result = dict(folds=summary, raw_directory_inventory=inventory,
              plans_sha256=hashlib.sha256(planpath.read_bytes()).hexdigest(),
              splits_sha256=hashlib.sha256(splitpath.read_bytes()).hexdigest(),
              expected_cache=plan["data_identifier"],
              cache_dir_exists=(prep/plan["data_identifier"]).is_dir())
Path(__file__).with_name("specialist_input_audit.json").write_text(json.dumps(result,indent=2))
print(summary)
for name, r in inventory.items(): print(name,r["nonzero"],"/",r["expected"])
print("Preprocessed cache directory present:",result["cache_dir_exists"])
