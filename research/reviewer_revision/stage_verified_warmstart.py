"""Restore an audited warm start from its verified local backup, never from holdout."""
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

run = Path(__file__).resolve().parent / "specialist_crossfit"
fold = int(sys.argv[1])
assert fold in (1, 2, 3, 4)
recipe = (run / "retrospective_recipe.json").read_bytes()
assert hashlib.sha256(recipe).hexdigest() == "f9bd9199cd488b9a2fa61c4219306bd724bbbc14536eb03fc54929eb6c82264d"
entry, = [e for e in json.loads(recipe)["inputs"] if e["fold"] == fold]
audit = json.loads((run / "weight_checks" / f"fold_{fold}.json").read_bytes())
assert audit["fold"] == fold and audit["epoch"] == entry["epoch"]
out = run / "warmstarts" / f"fold_{fold}" / "checkpoint_best.pth"
out.parent.mkdir(parents=True, exist_ok=True)

def verify(p):
    assert p.stat().st_size == entry["expected_bytes"]
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == audit["sha256"], "Checkpoint hash mismatch"

if not out.exists():
    assert shutil.disk_usage(out.parent).free > entry["expected_bytes"] + 18 * 2**30
    source = Path.home() / "Downloads" / audit["verified_copy"]
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp, zipfile.ZipFile(source) as z:
        key, = [n for n in z.namelist() if n.endswith("/" + entry["path"])]
        assert z.getinfo(key).file_size == entry["expected_bytes"]
        candidate = Path(tmp) / out.name
        with z.open(key) as src, candidate.open("xb") as dst:
            shutil.copyfileobj(src, dst, 4 << 20)
        verify(candidate)
        candidate.rename(out)
verify(out)
print(f"Fold {fold}: verified {out.stat().st_size} bytes; SHA256 {audit['sha256']}")
