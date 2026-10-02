# Reproduction guide

Run commands from the repository root. This release does not download data,
start training automatically, or open the retrospective holdout.

## 1. Review without data

Python 3.10 was used for training; the analysis tools also support Python 3.11.
Create and activate an isolated environment:

```bash
python -m venv .venv
```

PowerShell: `.\.venv\Scripts\Activate.ps1`; Linux: `source .venv/bin/activate`.

```bash
python -m pip install -r requirements-analysis.txt
python scripts/verify_release.py
python -m unittest discover -s tests -v
python scripts/render_report.py --output-dir outputs/review
```

The verifier checks subject separation, exact OOF coverage, region order,
HD95 intervals, upstream metric hashes, and release file hashes. The tests
exercise conversion and official evaluation with **synthetic arrays**, not
patient data. Figures read the stored aggregate estimates; their whiskers are
95% subject-cluster CIs. Values are printed in a separate column to avoid
overlapping markers and intervals.

## 2. Training environment

Install a PyTorch 2.10.0 build compatible with your GPU/driver before installing
`requirements.txt`. The original machine used PyTorch 2.10.0+cu130 on an NVIDIA
RTX 5070 Ti, with nnU-Net v2.7.0. The code does not require that exact GPU.
Do not replace a working GPU build with a CPU-only wheel.

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The baseline uses batch 2 and patch 128 x 192 x 128. Resource settings and a
250-epoch training run are substantial; CPU smoke tests are not a performance
benchmark. Full training was not repeated to prepare this repository.

## 3. Prepare the development dataset

Obtain authorized inputs separately, as described in [DATA_ACCESS.md](DATA_ACCESS.md).
The converter expects the original, **trusted**, processed NPZ schema (`image`,
`seg`, `case_id`). Its historical loader uses NumPy pickle support; do not feed
it untrusted archives. NPZ identity-affine conversion does not reconstruct raw
DICOM orientation or replace skull stripping/registration.

```bash
python scripts/prepare_publication_data.py --npz-root /path/to/authorized_npz --workers 2
python scripts/run_grouped.py preprocess --workers 2 --dry-run
python scripts/run_grouped.py preprocess --workers 2
```

Only the 1,155 development cases are converted. The fixed, identifier-only
protocol excludes all 195 holdout scans. The wrapper keeps the supplied
10.5-GB plans and splits; it extracts the fingerprint and preprocesses without
replanning. Do not run split generation again for the reported study.

Conversion refuses a nonempty destination unless `--resume` is supplied. Resume
is only safe with the identical source archive. nnU-Net arrays and checkpoints
are ignored by Git.

## 4. Train and evaluate the five baseline folds

```bash
python scripts/run_grouped.py train --fold 0 --dry-run
python scripts/run_grouped.py train --fold 0
python scripts/run_grouped.py validate --fold 0
python scripts/run_grouped.py metrics --fold 0 --workers 2
```

Repeat the last three commands for folds 1, 2, 3, and 4. To continue an
interrupted training run, use `train --fold N --resume`. This does not imply
bitwise-identical continuation of data-loader RNG state.

Fold 0 uses the standard `nnUNetTrainer_250epochs`. Folds 1-4 use
`nnUNetTrainer_GroupedCV250Duty80`, which adds operational sleeping to reduce
device duty; it is not a different segmentation objective. Best-checkpoint
full-volume validation uses Gaussian weighting, step 0.5, and mirroring.
These are held-out fold predictions, **not** an ensemble on development scans.

Checkpoints are written under:

```text
experiments/nnunet_results/Dataset503_BraTS2024Grouped4Ch/
  <trainer>__nnUNetResEncUNet10p5GBPlans__3d_fullres/fold_N/
    checkpoint_best.pth
    validation/<case_id>.nii.gz
```

The metrics wrapper requires exactly the expected validation IDs. The historical
official evaluator caches each case; therefore the wrapper rejects an existing
cache. Use a new `--output-dir` for changed predictions. Never reuse caches
between checkpoints or models.

## 5. Region-aware specialist and frozen V2

This is a **secondary** fold-0/retrospective-holdout experiment, not the source
of the five-fold OOF baseline. The specialist is the historical `proxy` variant
(`nnUNetTrainer_BraTSRegionAware120`) with the 9-GB plan. `corrected-*` variants
in the source are different experiments and must not be substituted silently.

For direct historical scripts, set the nnU-Net directories first. In PowerShell:

```powershell
$env:nnUNet_raw = "$PWD/processed/nnunet_raw"
$env:nnUNet_preprocessed = "$PWD/processed/nnunet_preprocessed"
$env:nnUNet_results = "$PWD/experiments/nnunet_results"
$env:nnUNet_n_proc_DA = "1"
$env:nnUNet_compile = "false"
```

On Linux, use `export NAME="$PWD/..."` for the corresponding variables.
Preprocess the 9-GB plan as well, then train from the fold-0 baseline:

```bash
nnUNetv2_preprocess -d 503 -plans_name nnUNetResEncUNet9GBPlans -c 3d_fullres -np 2
python research/grouped_general_sota_v1/run_custom_trainer.py train --variant proxy --fold 0 --plans nnUNetResEncUNet9GBPlans --pretrained-weights experiments/nnunet_results/Dataset503_BraTS2024Grouped4Ch/nnUNetTrainer_250epochs__nnUNetResEncUNet10p5GBPlans__3d_fullres/fold_0/checkpoint_best.pth
python research/grouped_general_sota_v1/run_custom_trainer.py validate --variant proxy --fold 0 --plans nnUNetResEncUNet9GBPlans
```

The component-filter implementation and its RC-topology guard are in
`research/grouped_et_topology_guard_v2/`. The historical frozen inference
workflow is in `research/final_locked_test_v1/`: revalidate the reconstructed
specialist, freeze model hashes, stage images, predict, freeze the output masks,
and only then release reference masks and evaluate. These scripts intentionally
require lock manifests and original checkpoint identities that are **not**
bundled in this code release. They are an audit trail, not an instruction to
reopen the existing holdout for new tuning.

The manuscript's frozen specialist was not byte-identical to the earlier
specialist reconstruction and was independently revalidated. New training
produces a new model and must not be labelled as the original frozen V2.

## 6. Recompute statistics from case-level outputs

Aggregate CSVs alone are insufficient to reconstruct subject-cluster CIs.
Authorized reproductions need these per-case files:

```text
experiments/grouped_delta_roi_v1/official_metrics/baseline/case_metrics_official.csv
experiments/grouped_fivefold_cv_v11/fold_1/official_metrics/case_metrics_official.csv
experiments/grouped_fivefold_cv_v11/fold_2/official_metrics/case_metrics_official.csv
experiments/grouped_fivefold_cv_v11/fold_3/official_metrics/case_metrics_official.csv
experiments/grouped_fivefold_cv_v11/fold_4/official_metrics/case_metrics_official.csv
experiments/final_locked_test_v1/test_case_metrics_paired.csv
```

The last file is built by `research/final_locked_test_v1/analyze_results.py`
from paired baseline/V2 metrics and reference-volume information. Each row has
`case_id`, `subject_id`, region metric/count columns with `_baseline` and
`_final` suffixes, and the subgroup reference volumes. Missing files are not
fabricated from published means.

```bash
python research/publication_integrity_v1/build_publication_statistics.py --bootstrap-samples 10000 --permutation-samples 100000 --seed 20260808 --output-dir outputs/recomputed_statistics
```

Use a new output directory; retain the reference aggregate tables unchanged.
This historical script can summarize partial folds, so inspect
`completed_fold_count == 5` before calling any output a five-fold result.
The exact six-region order is WT, TC, NETC, SNFH, ET, RC. HD95 includes the
official 337-mm unmatched-lesion penalty. Primary means are case-weighted;
subject-weighted sensitivity estimates are a different estimand.

## 7. Legacy code

The custom U-Net pretraining and transfer code under `src/`, `configs/`, and
`scripts/` is retained as exploratory history. Its case-level split is not the
publication's corrected grouped protocol. Do not compare those exploratory
numbers against primary OOF estimates as a controlled pretraining ablation.
Legacy report scripts require their original outputs, which are not bundled.

See [MANUSCRIPT_MAP.md](MANUSCRIPT_MAP.md) for the mapping and methodological
limitations, including internal checkpoint selection and the retrospective
holdout definition. All code is research-only, not a clinical deployment tool.
