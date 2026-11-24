"""Pair official columns for the preserved same-runtime best and final predictions."""
import csv
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent/"fixed_epoch_pilot"
def read(kind):
    with (root/f"official_{kind}.csv").open(newline="") as f:
        return {(r["region"],r["metric"]):(float(r["archived_best"]),float(r[kind])) for r in csv.DictReader(f)}
best,final=read("best"),read("final")
assert len(best)==30 and best.keys()==final.keys()
rows=[]
for key,(old,b) in best.items():
    assert final[key][0]==old
    v=final[key][1]
    rows.append((*key,old,b,v,b-old,v-b))
with (root/"paired_pilot.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["region","metric","archived_best","replay_best","replay_final","replay_drift","paired_final_minus_best"])
    w.writerows(rows)
print("six-region mean Dice (archived best, replay best, replay final, drift, paired delta):",
      [statistics.mean(r[j] for r in rows if r[1]=="LesionWise_Score_Dice") for j in range(2,7)])
for r in rows:
    if r[0]=="ET": print(r)
