# Manuscript-to-code map

Checked against the authors' Overleaf source downloaded on 2026-10-03.
Its SHA-256 is recorded in `source_provenance.json`. The unpublished source,
author contact details, and review submission files are not distributed.

| Manuscript component | Source / evidence |
| --- | --- |
| Subject-level protocol audit | `scripts/audit_brats_longitudinal_integrity.py`; `research/publication_integrity_v1/audit_publication_integrity.py` |
| Subject-grouped split construction | `scripts/create_brats_subject_grouped_protocol.py`; `processed/brats_subject_grouped_protocol_v1/` |
| Full-resolution 3D baseline | Fixed 10.5-GB plan; `research/grouped_fivefold_cv_v11/` |
| Fold-0 baseline vs folds 1-4 | Standard `nnUNetTrainer_250epochs` vs `nnUNetTrainer_GroupedCV250Duty80`; see `TRAINING_RECIPE.md` |
| Warm-started 120-epoch specialist | `research/grouped_general_sota_v1/run_custom_trainer.py`, `trainer.py`, `losses.py`; fixed 9-GB plan |
| Frozen V2 rule | `research/grouped_et_topology_guard_v2/build_topology_guard.py`, `topology.py`; `research/final_locked_test_v1/apply_frozen_v2.py` |
| Resolved specialist reconstruction | `research/final_locked_test_v1/verify_recreated_specialist.py`, `build_reproduced_validation.py`, `freeze_model.py` |
| Official Dice / HD95 / lesion counts | `scripts/evaluate_brats2024_official_metrics.py`; pinned vendor `metrics_GLI.py` |
| Pooled OOF, fold-level HD95, holdout CIs | `research/publication_integrity_v1/build_publication_statistics.py`; aggregate CSVs |
| Subject-cluster sign-flip tests / Holm correction | `build_publication_statistics.py` |
| Lesion size / detection analysis | `research/final_locked_test_v1/analyze_results.py`; `build_publication_statistics.py` |
| Historical reconstruction pretraining | `scripts/pretrain_normal_anatomy.py`; `src/brain_mri_project/`; legacy `configs/` |

## Interpretation boundaries

- Primary OOF results are five held-out fold predictions pooled across 1,155 scans. They are not a five-fold inference ensemble.
- Best checkpoints were selected using each fold's internal validation monitoring. This is not a nested-CV estimate with a separate checkpoint-selection cohort.
- Frozen V2 is a baseline-plus-specialist post-processing rule, not a second architecture.
- The specialist reconstruction was not byte-identical. The paper reports independent fold-0 revalidation before holdout opening, including differences in 78/223 masks (1,400 voxels).
- The legacy pretraining and case-level results are exploratory. They do not demonstrate a patient-independent pretraining benefit.
- Case-weighted means are primary. Subject-cluster resampling retains all scans from a sampled subject. The subject-weighted means are a distinct sensitivity estimand.
- Published CIs use 10,000 bootstrap draws; sign-flip tests use 100,000 draws, with Holm correction across seven Dice endpoints.
- HD95 includes the official unmatched-lesion penalty of 337 mm. It is not a pure boundary error restricted to matched lesions.

## Export changes

The copied source hashes in `source_provenance.json` record the local experiment
files before export. The release adds portable entry points and documentation.
The evaluator's upstream-version lookup was adapted to verify `UPSTREAM.json`
instead of accidentally returning this enclosing repository's Git commit.
The official metric implementation and historical training/loss code are
unchanged. `scripts/render_report.py` creates reviewer-facing plots from the
stored aggregate tables; these are not claimed to be byte-identical to the
manuscript's externally edited figure layouts.
