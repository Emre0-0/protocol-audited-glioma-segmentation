# Revision analyses

All three comparisons use the original development folds. None requires a new inference run on the 195-scan holdout.

| Comparison | Design or inputs | Analysis | Published result |
| --- | --- | --- | --- |
| Baseline vs cross-fitted V2 | Original five folds; `specialist_crossfit/retrospective_recipe.json`; fold-specific warm-start SHA-256 in `weight_checks` under that directory | `crossfit_fusion.py`, `score_crossfit_v2.py`, `summarize_crossfit_v2.py` | `specialist_crossfit/paired_pooled.csv`; `paired_nonselection.csv` omits fold 0 |
| Longitudinal exposure | `matched_overlap_design.json` and `fold0_overlap_single_run/cohort.json`; one exposed-arm fit vs archived clean fold-0 final checkpoint | `train_overlap_arm.py`, `predict_overlap_final.py`, `score_overlap_contrast.py` | `fold0_overlap_single_run/matched_overlap_results.csv`; primary comparison uses 40 index scans, not all 183 common scans |
| Individual ET lesions | Baseline reference components eligible under the official scorer (>10 voxels) | `extract_et.py`, `recover_et_reverse.py`, `summarize_et.py` | `oof_individual_et.csv`; case burden and individual lesion size are different denominators |
| Checkpoint sensitivity | Best vs epoch 250, same fold and evaluation scans | `fold0_replay.py`, `summarize_fold0_replay.py` | `fixed_epoch_pooled/paired_summary.csv` |

For the public point-estimate checks, use `python scripts/verify_revision.py` from the repository root. The scan-level extracts report only paired six-region means. Neither their row order nor the pooled ET counts provide patient clusters for bootstrap resampling.

The original scoring scripts work with the private experiment archive, not a fresh BraTS download. For example, after staging the original files in the layout used by each script, run the following from this directory in a working copy:

```bash
python summarize_crossfit_v2.py "$ARCHIVE" pooled --project "$PROJECT"
python summarize_et.py "$ARCHIVE"
python score_overlap_contrast.py --clean "$CLEAN_CSV" --exposed "$EXPOSED_CSV" --single-fold0-release "$ARCHIVE"
```

`ARCHIVE` is the experiment ZIP containing the original split and official metrics. `PROJECT` is the archived experiment tree containing the reproduced fold-0 V2 score table. The first command also reads the per-scan records under `specialist_crossfit/paired_fusion/fold_{1,2,3,4}/official_scores/`; the second needs the reconciled ET records under `et_rows/` and `et_native_reverse/`. The overlap inputs are the official score CSVs for the two arms on the same 183 scans. Run on copies: the scripts write summary files. Historical training launchers target Windows and additionally require local caches and model checkpoints. No private archive or pretrained weight is downloaded by these commands.
