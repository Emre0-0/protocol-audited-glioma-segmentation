"""Audit retained epoch timings, not elapsed wall time including interruptions."""
import csv
import json
import re
import sys
from pathlib import Path

root=Path(sys.argv[1])
meta=json.loads((root/"experiments/publication_integrity_v1/checkpoint_metadata/checkpoint_metadata.json").read_text())
result=[]
for rec in meta["checkpoints"]:
    if rec["kind"]!="best":
        continue
    folder=root/Path(rec["path"]).relative_to(r"C:\Users\Emre\Desktop\BR").parent
    times={}
    count=0
    best_updates=[]
    for path in folder.glob("training_log*.txt"):
        epoch=None
        for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
            match=re.search(r": Epoch (\d+)\s*$",line)
            if match:
                epoch=int(match[1])
            duration=re.search(r"Epoch time: ([\d.]+) s",line)
            if duration and epoch is not None:
                count+=1
                # Keep latest completed execution of each zero-indexed epoch.
                times[epoch]=max(times.get(epoch,("",0)),(line[:26],float(duration[1])))
            if "New best EMA pseudo Dice" in line and epoch is not None:
                best_updates.append((line[:26],epoch+1))
    assert set(times)==set(range(250)),(rec["fold"],len(times))
    last_best=max(best_updates)[1]
    gpu=json.loads((folder/"debug.json").read_text())["gpu_name"]
    result.append([rec["fold"],rec["current_epoch"],last_best,count,count-250,sum(t for _,t in times.values())/3600,gpu])
with (Path(__file__).parent/"training_log_audit.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow("fold,best_epoch_metadata,last_best_log_epoch,completed_epoch_records,repeated_records,hours_latest_250_epoch_records,gpu".split(","))
    w.writerows(result)
