"""Stage a single release-relative input; prefer CRC-checked local ZIP entries."""
from pathlib import Path
import shutil
import time
import zipfile

def index_inputs(prefixes):
    matches={}
    for archive in sorted((Path.home()/"Downloads").glob("*.zip")):
        if not archive.name.startswith(("BR-","BrainMRI_Latest_SubjectDisjoint_Release_")): continue
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                rel=n.partition("/")[2]
                if rel.startswith(prefixes) and rel.endswith(".nii.gz"):
                    matches.setdefault(rel,(archive,n))
    return matches

def stage(project,relative,target,matches):
    rel=Path(relative).as_posix()
    if rel in matches:
        archive,n=matches[rel]
        with zipfile.ZipFile(archive) as z,z.open(n) as src,target.open("wb") as dst:
            shutil.copyfileobj(src,dst,1<<20)
            assert target.stat().st_size <= z.getinfo(n).file_size
        assert target.stat().st_size==z.getinfo(n).file_size
        return
    for attempt in range(3):
        try:
            source=project/rel
            shutil.copyfile(source,target)
            assert target.stat().st_size==source.stat().st_size>0
            return
        except OSError:
            if attempt==2: raise
            time.sleep(2**attempt)
