"""Cross-check raw input provenance before the controlled training run."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def verify(allow_partial=False):
    dataset=ROOT/"specialist_crossfit"
    cohort=json.loads((ROOT/"fold0_overlap_single_run/cohort.json").read_bytes())
    need=set(cohort["train"])|set(cohort["validation_monitor"])
    anchored=set(cohort["validation_monitor"])|set(cohort["excluded_sibling_scans"])
    assert len(need)==1115 and len(anchored)==223 and anchored<=need
    checked=paired=0
    train=set(cohort["train"])
    fingerprints=[{},{}]
    for c in sorted(need):
        rec=dataset/"records"/(c+".json")
        if allow_partial and not rec.exists(): continue
        cache=json.loads(rec.read_bytes())
        ref=json.loads((dataset/"reference_volume_checks"/rec.name).read_bytes())
        assert cache["case"]==ref["case"]==c
        src=cache["source_sha256"]
        vols=[f"{c}_{i:04d}.nii.gz" for i in range(4)]
        assert set(src)==set(vols)|{c+".nii.gz"} and src[c+".nii.gz"]==ref["reference_sha256"],c
        fingerprint=tuple(src[v] for v in vols)
        fingerprints[int(c not in train)].setdefault(fingerprint,[]).append(c)
        if c in anchored:
            replay=json.loads((ROOT/"fixed_epoch_fold0"/rec.name).read_bytes())
            assert ref["fold"]==replay["fold"]==0 and replay["case_id"]==c
            assert [src[v] for v in vols]==replay["input_hashes"],c
            paired+=1
        checked+=1
    if not allow_partial: assert (checked,paired)==(1115,223)
    collisions=fingerprints[0].keys() & fingerprints[1].keys()
    if collisions:
        pairs=[(fingerprints[0][h],fingerprints[1][h]) for h in sorted(collisions)]
        raise ValueError(f"Identical four-file MRI input across arms' exposed train/evaluation split: {pairs}")
    nt,nv=(sum(map(len,by_hash.values())) for by_hash in fingerprints)
    print(f"No byte-identical four-channel training/evaluation pair among {nt} x {nv} scans.",flush=True)
    print(f"Reference hashes matched: {checked}/1115; fold-0 four-channel matches: {paired}/223.",flush=True)
    return checked,paired

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--partial",action="store_true")
    verify(p.parse_args().partial)
