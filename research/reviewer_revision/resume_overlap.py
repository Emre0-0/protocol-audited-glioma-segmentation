"""Recover the interrupted, frozen overlap training run from its last saved epoch."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import types
from importlib.metadata import version


def main():
    here = Path(__file__).resolve().parent
    project, archive = map(Path, sys.argv[1:3])
    run = here / "fold0_overlap_single_run"
    arm = run / "training/exposed"
    init = json.loads((arm / "initialization.json").read_bytes())
    def check():
        for n, h in init["pipeline_sha256"].items():
            assert hashlib.sha256((here / n).read_bytes()).hexdigest() == h, n
    check()
    assert all(version(k) == v for k, v in init["training_versions"].items())
    assert not (arm / "completed.json").exists()
    src = project / "research/grouped_fivefold_cv_v11/trainer.py"
    assert hashlib.sha256(src.read_bytes()).hexdigest() == init["trainer_source_sha256"]
    os.environ.update(nnUNet_preprocessed=str(here/"specialist_crossfit/preprocessed"),
                      nnUNet_raw=str(here/"specialist_crossfit/raw"), nnUNet_results=str(arm/"results"),
                      nnUNet_compile="false", nnUNet_n_proc_DA="4", OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    sys.path.insert(0, str(src.parent))
    from trainer import nnUNetTrainer_GroupedCV250Duty80 as Trainer
    import numpy as np
    import torch
    import psutil
    for p in psutil.process_iter(["pid","name","cmdline"]):
        if p.pid != os.getpid() and "python" in (p.info["name"] or "").lower():
            assert not any(Path(s).name in ("resume_overlap.py","train_overlap_arm.py","fold0_replay.py","run_custom_trainer.py") for s in p.info["cmdline"] or []),p.pid
    cohort = json.loads((run/"cohort.json").read_bytes())
    def split(self):
        return cohort["train"].copy(), cohort["validation_monitor"].copy()
    candidates = list((arm/"results").rglob("checkpoint_*.pth"))
    assert candidates and not any(p.name=="checkpoint_final.pth" for p in candidates)
    # Select by completed epoch, never by validation score.
    latest = max((torch.load(p,map_location="cpu",weights_only=False)["current_epoch"],str(p)) for p in candidates)
    epoch, path = latest
    state = torch.load(path,map_location="cpu",weights_only=False)
    assert state["trainer_name"]==Trainer.__name__ and state["optimizer_state"]["state"] and 0<epoch<250
    assert state["current_epoch"]==len(state["logging"]["train_losses"])==epoch
    seed = init["seed"] + epoch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    args = dict(state["init_args"],device=torch.device("cuda"))
    args["plans"] = dict(args["plans"], continue_training=True)
    t = Trainer(**args)
    assert Path(t.output_folder).resolve()==Path(path).parent.resolve()
    t.do_split=types.MethodType(split,t)
    t.initialize()
    del state
    t.load_checkpoint(path)
    assert t.current_epoch==epoch and t.num_epochs==250 and t.num_iterations_per_epoch==250
    t._duty_step=epoch*t.num_iterations_per_epoch
    record=dict(epoch=epoch,checkpoint=path,sha256=hashlib.file_digest(open(path,"rb"),"sha256").hexdigest()
                if hasattr(hashlib,"file_digest") else hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                seed=seed,reason="OS restart; optimizer restored; augmentation RNG stream not recoverable")
    with (arm/"restart_log.jsonl").open("a") as f: f.write(json.dumps(record)+"\n")
    print(json.dumps(record),flush=True)
    t.run_training()
    final=Path(t.output_folder)/"checkpoint_final.pth"
    assert final.is_file() and t.current_epoch==250
    (arm/"completed.json").write_text(json.dumps({"checkpoint":str(final),"epoch":250}))
    check()
    del t
    import gc
    gc.collect(); torch.cuda.empty_cache()
    subprocess.run([sys.executable,str(here/"predict_overlap_final.py"),str(project)],check=True)
    subprocess.run([str(here/".venv/Scripts/python.exe"),"-u",str(here/"score_overlap_predictions.py"),str(project),str(archive)],check=True)


if __name__=="__main__": main()
