"""Matched longitudinal-exposure ablation. CPU check does not train or access holdout."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import shutil
import sys
import types
import zipfile
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project",type=Path)
    parser.add_argument("archive",type=Path)
    parser.add_argument("arm",choices=["clean","exposed"])
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--reuse-fold0",action="store_true",help="One exposed run against archived fold-0 final")
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    run=here/("fold0_overlap_single_run" if args.reuse_fold0 else "matched_overlap")
    if args.reuse_fold0: assert args.arm=="exposed"
    spec=here/"matched_overlap_design.json"
    assert hashlib.sha256(spec.read_bytes()).hexdigest()=="e5c521c7702c1dce29b944bea21d539b3c4482cad1743c46a805359d084491bd"
    design=json.loads(spec.read_bytes())
    ds="Dataset503_BraTS2024Grouped4Ch"
    pre=here/"specialist_crossfit/preprocessed"
    paths=here/"fixed_epoch_pilot/model"
    mode="cpu_check" if args.check else "training"
    arm=run/mode/args.arm
    os.environ.update(nnUNet_preprocessed=str(pre),nnUNet_raw=str(here/"specialist_crossfit/raw"),
                      nnUNet_results=str(arm/"results"),nnUNet_compile="false",nnUNet_n_proc_DA="4",
                      OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    src="research/grouped_fivefold_cv_v11/trainer.py"
    with zipfile.ZipFile(args.archive) as z:
        key,=[n for n in z.namelist() if n.endswith("/"+src)]
        assert z.read(key)==(args.project/src).read_bytes()
    sys.path.insert(0,str((args.project/src).parent))
    from trainer import nnUNetTrainer_GroupedCV250Duty80 as Trainer
    import numpy as np
    import torch
    import psutil
    dev="cpu" if args.check else "cuda"
    plans=json.loads((paths/"plans.json").read_bytes())
    cache=json.loads((pre/ds/"nnUNetResEncUNet9GBPlans.json").read_bytes())
    assert plans["plans_name"]=="nnUNetResEncUNet10p5GBPlans"
    assert all(plans[k]==cache[k] for k in plans if k not in ("configurations","plans_name"))
    a,b=[v["configurations"]["3d_fullres"] for v in (plans,cache)]
    assert {k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}=={"patch_size"}
    dataset=json.loads((paths/"dataset.json").read_bytes())
    assert dataset==json.loads((pre/ds/"dataset.json").read_bytes())
    tr=design["train_"+args.arm];val=design["monitor"]
    assert len(set(tr))==len(tr)==835 and len(set(val))==len(val)==97 and not set(tr)&set(val)
    assert not set(design["evaluation"])&(set(tr)|set(val))
    if args.reuse_fold0:
        split_path=pre/ds/"splits_final.json"
        assert hashlib.sha256(split_path.read_bytes()).hexdigest()==design["split_sha256"]
        orig=json.loads(split_path.read_bytes())[0]
        assert set(orig["train"])==set(design["train_clean"])|set(val)
        tr=sorted(set(tr)|set(val))
        swapped_in=set(tr)-set(orig["train"])
        assert swapped_in=={s["added"] for s in design["swaps"]}
        val=sorted(set(orig["val"])-swapped_in)
        assert len(tr)==len(orig["train"])==932 and len(val)==183 and len(swapped_in)==40
        assert set(val).isdisjoint(tr) and set(design["evaluation"])<=set(val)
        audit=json.loads((here/"fixed_epoch_pilot/manifest.json").read_bytes())["files"]
        for k in ("plans.json","dataset.json","fold_0/checkpoint_final.pth"):
            model_file=paths/k
            h=hashlib.sha256()
            with model_file.open("rb") as stream:
                for block in iter(lambda:stream.read(4<<20),b""): h.update(block)
            assert h.hexdigest()==audit[str(Path("model")/k)]["sha256"]
        cohort=dict(train=tr,validation_monitor=val,primary_patients=design["exposed_evaluation"],
                    fixed_index_scans=design["evaluation"],excluded_sibling_scans=sorted(swapped_in),
                    reference_checkpoint="archived fold0 final epoch 250",
                    limitation="Original initialization not paired; one run per arm.")
        run.mkdir(exist_ok=True)
        frozen=run/"cohort.json"
        payload=json.dumps(cohort,indent=2,sort_keys=True).encode()
        if frozen.exists(): assert frozen.read_bytes()==payload
        else: frozen.write_bytes(payload)
        if not args.check:
            audit=json.loads((run/"decoded_sibling_comparison.json").read_bytes())
            assert audit["patients"]==40 and audit["identical_channel_counts"]["4"]==0
            pairs=audit["comparisons"]
            assert len(pairs)==audit["pairs"] and {p["added"] for p in pairs}==swapped_in
            assert all(p["evaluation"] in val and p["identical_channels"]<4 for p in pairs)
    if not args.check:
        if args.reuse_fold0:
            from verify_overlap_inputs import verify
            verify()
        for p in psutil.process_iter(["pid","name","cmdline"]):
            if p.pid==os.getpid() or "python" not in (p.info["name"] or "").lower(): continue
            if any(Path(t).name in ("train_overlap_arm.py","fold0_replay.py","run_custom_trainer.py") for t in (p.info["cmdline"] or [])):
                raise RuntimeError("Another GPU process is active")
        if args.arm=="exposed" and not args.reuse_fold0:
            assert (run/"training/clean").joinpath("completed.json").is_file(),"Train clean arm first"
        assert shutil.disk_usage(here).free>18*2**30 and torch.cuda.is_available()
        assert not arm.exists(),"Inspect existing output; never overwrite a training run"
        for c in set(tr)|set(val):
            assert (here/"specialist_crossfit/records"/(c+".json")).is_file(),c
            assert all((pre/ds/a["data_identifier"]/(c+s)).is_file() for s in (".pkl",".b2nd","_seg.b2nd")),c
    seed=20261121
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if dev=="cuda": torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    plans["continue_training"]=False
    t=Trainer(plans,"3d_fullres",0,dataset,torch.device(dev))
    t.do_split=types.MethodType(lambda self:(tr.copy(),val.copy()),t)
    assert t.do_split()==(tr,val)
    t.initialize()
    assert (t.num_epochs,t.num_iterations_per_epoch,t.num_val_iterations_per_epoch)==(250,250,50)
    recipe_check={}
    if args.reuse_fold0:
        suffix=f"/{ds}/nnUNetTrainer_250epochs__nnUNetResEncUNet10p5GBPlans__3d_fullres/fold_0/debug.json"
        with zipfile.ZipFile(args.archive) as z:
            name,=[n for n in z.namelist() if n.endswith(suffix)]
            baseline=json.loads(z.read(name))
        for k in ("num_epochs","num_iterations_per_epoch","num_val_iterations_per_epoch",
                  "batch_size","num_input_channels","initial_lr","weight_decay",
                  "oversample_foreground_percent","probabilistic_oversampling",
                  "enable_deep_supervision","loss","optimizer"):
            recipe_check[k]=str(getattr(t,k))
        recipe_check["network"]=type(t.network).__name__
        assert type(t.lr_scheduler).__name__ in baseline["lr_scheduler"]
        recipe_check["dataloader_train.num_processes"]=os.environ["nnUNet_n_proc_DA"]
        for k,v in recipe_check.items():
            assert v==baseline[k],f"Archived recipe mismatch for {k}: {v!r} vs {baseline[k]!r}"
    h=hashlib.sha256()
    for name,tensor in sorted(t.network.state_dict().items()):
        h.update(name.encode()+b"\0")
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    evidence=dict(seed=seed,device=dev,initial_network_sha256=h.hexdigest(),epochs=t.num_epochs,
                  updates_per_epoch=t.num_iterations_per_epoch,monitor_iterations=t.num_val_iterations_per_epoch,
                  patch_size=a["patch_size"],initial_lr=t.initial_lr,trainer_source_sha256=hashlib.sha256((args.project/src).read_bytes()).hexdigest())
    if recipe_check: evidence["verified_against_original_debug"]=recipe_check
    if args.reuse_fold0:
        from importlib.metadata import version
        pipeline=("train_overlap_arm.py","predict_overlap_final.py","score_overlap_predictions.py",
                  "score_overlap_contrast.py","verify_overlap_inputs.py","matched_overlap_design.json",
                  "fold0_overlap_single_run/cohort.json","fold0_overlap_single_run/analysis_plan.txt",
                  "pilot_checkpoint_inference.py","stage_revision_input.py")
        def code_hashes():
            return {name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in pipeline}
        evidence["pipeline_sha256"]=code_hashes()
        evidence["training_versions"]={k:version(k) for k in ("torch","nnunetv2","numpy","batchgenerators","blosc2")}
        evidence["python_version"]=sys.version
    if args.arm=="exposed" and not args.reuse_fold0:
        first=json.loads((run/mode/"clean/initialization.json").read_bytes())
        assert first==evidence,"Training arms do not have identical initialization/recipe"
    arm.mkdir(parents=True,exist_ok=True)
    (arm/"initialization.json").write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence),flush=True)
    if not args.check:
        t.run_training()
        final=Path(t.output_folder)/"checkpoint_final.pth"
        assert final.is_file() and t.current_epoch==250
        (arm/"completed.json").write_text(json.dumps({"checkpoint":str(final),"epoch":250}))
        if args.reuse_fold0:
            if code_hashes()!=evidence["pipeline_sha256"]:
                raise RuntimeError("Training completed; pipeline changed. Review before prediction/scoring.")
            del t
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            subprocess.run([sys.executable,str(here/"predict_overlap_final.py"),str(args.project)],check=True)
            subprocess.run([str(here/".venv/Scripts/python.exe"),"-u",str(here/"score_overlap_predictions.py"),
                            str(args.project),str(args.archive)],check=True)

if __name__=="__main__": main()
