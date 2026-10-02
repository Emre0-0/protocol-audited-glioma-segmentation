# Release validation

Release checks were performed on 2026-10-03. These are software/export checks,
not new clinical validation or a repeat of the long training experiments.

- Fresh isolated Python 3.11 analysis environment installed from
  `requirements-analysis.txt`; dependency consistency checked with `pip check`.
- Fixed partition checked: 1,155 development scans / 521 subjects, 195 holdout
  scans / 92 subjects, zero subject overlap and exactly one OOF assignment per
  development scan. Validation sizes: 223, 247, 243, 225, 217 scans.
- 22 release tests cover subject leakage rejection, official metric source
  identity, synthetic NPZ conversion, official Dice/HD95, RC topology, cluster
  statistics, and training-command routing.
- The synthetic perfect prediction yields Dice 1 and HD95 0 for all six regions.
  A deliberately missed ET lesion yields the official 337-mm HD95 penalty.
- 12 existing publication-integrity tests passed in the original Python 3.10
  environment. The specialist loss/gradient smoke test also passed.
- Python sources passed syntax compilation. Aggregate reports rendered without
  MRI inputs. Figure values were kept in separate aligned columns.
- No patient scan arrays, masks, model weights, personal machine paths,
  authentication material, or unpublished manuscript source are included.

The CI workflow repeats the portable checks on Linux/Python 3.10. Its status is
available in GitHub Actions. The source manifest checks bytes, not scientific
validity; data-level reanalysis requires authorized inputs and the original
case-level outputs. No claim of error-free code, independent external
validation, or guaranteed model retraining equivalence is made.
