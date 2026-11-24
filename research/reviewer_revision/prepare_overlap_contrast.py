"""Freeze matched development-only exposure lists before training. No prediction scores."""
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

def rank(tag, value):
    return hashlib.sha256(f"overlap-20261121:{tag}:{value}".encode()).hexdigest()

with zipfile.ZipFile(sys.argv[1]) as z:
    def read(end):
        key,=[k for k in z.namelist() if k.endswith("/"+end)]
        return z.read(key)
    splits=read("processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/splits_final.json")
    assert hashlib.sha256(splits).hexdigest()=="6545c6c03469753a0bf6c470e6532cfb5a57a78ab3786b10d45bbc7df57e7eac"
    split=json.loads(splits)[0]
    raw=read("brats_subject_grouped_protocol_v1/case_assignments.csv")
rows={r["case_id"]:r for r in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))) if r["protocol_partition"]=="development"}
train=set(split["train"])
val=set(split["val"])
assert len(rows)==1155 and train|val==rows.keys() and not train&val and len(val)==223
patient=lambda c: rows[c]["subject_id"]
def group(ids):
    groups=defaultdict(list)
    for c in sorted(ids): groups[patient(c)].append(c)
    return groups
tr,ev=group(train),group(val)
assert tr.keys().isdisjoint(ev) and len(ev)==103
probe={p:min(cs,key=lambda c:rank("index",c)) for p,cs in ev.items()}
multi=sorted((p for p,cs in ev.items() if len(cs)>1),key=lambda p:rank("exposure",p))
exposed=multi[:(len(multi)+1)//2]
siblings=[min(set(ev[p])-{probe[p]},key=lambda c:rank("other",c)) for p in exposed]
ordered=sorted(tr,key=lambda p:rank("monitor",p))
monitor={c for p in ordered[:max(1,len(tr)//10)] for c in tr[p]}
clean=train-monitor
donors=sorted(cs[0] for cs in tr.values() if len(cs)==1 and cs[0] in clean)
assert len(donors)>=len(siblings)>0
def volumes(ids):
    return np.array([[int(rows[c][f"label_{j}_voxels"]) for j in range(1,5)] for c in ids])
a,b=volumes(siblings),volumes(donors)
# Prioritize matching presence/absence, then match log regional volumes.
cost=(np.not_equal((a>0)[:,None,:],(b>0)[None,:,:]).sum(2)*100+
      np.abs(np.log1p(a)[:,None,:]-np.log1p(b)[None,:,:]).sum(2))
i,j=linear_sum_assignment(cost)
removed=[donors[k] for k in j]
assert list(i)==list(range(len(siblings)))
mixed=(clean-set(removed))|set(siblings)
eval_ids=set(probe.values())
assert len(mixed)==len(clean) and len(group(mixed))==len(group(clean))
assert not eval_ids&(clean|mixed|monitor) and not (group(mixed).keys()&group(monitor).keys())
assert set(group(mixed))&set(ev)==set(exposed) and not (set(group(clean))&set(ev))
data=dict(description="Matched longitudinal exposure experiment, not an exact replay of the historical split.",
          split_sha256=hashlib.sha256(splits).hexdigest(),assignments_sha256=hashlib.sha256(raw).hexdigest(),
          train_clean=sorted(clean),train_exposed=sorted(mixed),monitor=sorted(monitor),
          evaluation=sorted(eval_ids),exposed_evaluation=sorted(probe[p] for p in exposed),
          unexposed_longitudinal_evaluation=sorted(probe[p] for p in multi if p not in exposed),
          swaps=[dict(added=s,removed=d,matching_cost=float(cost[k,j[k]])) for k,(s,d) in enumerate(zip(siblings,removed))])
out=Path(__file__).resolve().parent/"matched_overlap_design.json"
payload=json.dumps(data,indent=2,sort_keys=True).encode()
if out.exists(): assert out.read_bytes()==payload,"Refusing to replace a frozen design"
else: out.write_bytes(payload)
print("Design SHA256",hashlib.sha256(payload).hexdigest(),"train scans/subjects",len(clean),len(group(clean)),
      "monitor",len(monitor),"probes",len(eval_ids),"exposed",len(exposed),"unexposed longitudinal",len(multi)-len(exposed),
      "same label-presence profile",int(((a>0)==(b[j]>0)).all(1).sum()))
