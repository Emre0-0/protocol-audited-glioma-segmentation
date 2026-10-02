# Protocol-audited post-treatment glioma segmentation

Code accompanying **Protocol-Audited Subject-Disjoint Evaluation Reveals Small-Lesion Failure Modes in Post-Treatment Glioma Segmentation** by Emre Karabaca and Evin Sahin Sadik.

[![Release checks](https://github.com/Emre0-0/protocol-audited-glioma-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/Emre0-0/protocol-audited-glioma-segmentation/actions/workflows/ci.yml)

This release was checked against the authors' manuscript source on **3 October 2026**. It provides the subject-grouped partition, nnU-Net training configuration, frozen ET-component filtering, official BraTS 2024 evaluation, and subject-cluster statistical analysis. The study concerns evaluation reliability and small-lesion failure modes; it does not introduce a new architecture or establish clinical readiness.

## Reported study

The source cohort contains 1,350 scans from 613 subjects. Development comprises 1,155 scans from 521 subjects, evaluated across five subject-disjoint folds. The secondary retrospective holdout contains 195 scans from 92 subjects. All time points from a subject stay together.

| Region | Pooled OOF lesion-wise Dice | Lesion-wise HD95 (mm) |
| --- | ---: | ---: |
| WT | 0.7979 | 36.19 |
| TC | 0.7313 | 50.75 |
| NETC | 0.8130 | 32.13 |
| SNFH | 0.8019 | 33.27 |
| ET | 0.7315 | 50.80 |
| RC | 0.7048 | 50.56 |

The six-region mean Dice is **0.7634 (95% subject-cluster CI 0.7519-0.7746)**. These are pooled out-of-fold predictions, not a five-model ensemble evaluated on its own training subjects. Full-precision estimates, confidence intervals, fold tables, detection metrics, and holdout comparisons are in [the statistics directory](experiments/publication_integrity_v1/statistics).

The retrospective holdout comparison is 0.7592 for the fold-0 baseline and 0.7623 for frozen V2: difference +0.0031, paired 95% CI [-0.0029, +0.0100]. This does not establish superiority. The earlier 945/202/203 case-level split had longitudinal subject overlap and is excluded from confirmatory claims. The holdout was defined after legacy exploration and is not an untouched external test set.

## Quick start: inspect the release without MRI data

Use Python 3.10. The release verifier only needs the standard library:

```bash
git clone https://github.com/Emre0-0/protocol-audited-glioma-segmentation.git
cd protocol-audited-glioma-segmentation
python scripts/verify_release.py
```

For statistics tests and plotting:

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
python scripts/render_report.py --output-dir outputs/review
```

The report command renders the published aggregate tables and plots without accessing scans. It does not recompute the confidence intervals from individual cases. See [reproduction instructions](docs/REPRODUCING.md) for the separate case-level statistics workflow and training commands.

See [release validation](docs/VALIDATION.md) for the checks performed and what they do not establish.

## Contents

| Location | Purpose |
| --- | --- |
| `research/grouped_fivefold_cv_v11/` | Grouped 250-epoch training and fold validation |
| `research/grouped_general_sota_v1/` | Region-aware specialist, loss, and warm-start runner |
| `research/grouped_et_topology_guard_v2/` | Frozen ET filtering and RC-topology guard |
| `research/final_locked_test_v1/` | Historical model/inference locks and paired holdout analysis |
| `research/publication_integrity_v1/` | Integrity checks, subject-cluster CIs, sign-flip tests, tables |
| `research/vendor/BraTS-2024-Metrics/` | Pinned official evaluation implementation and upstream license |
| `processed/brats_subject_grouped_protocol_v1/` | Fixed splits and pseudonymous identifier-only assignments |
| `processed/nnunet_preprocessed/Dataset503_BraTS2024Grouped4Ch/` | Fixed dataset and 10.5-GB/9-GB plans, no image arrays |
| `src/`, legacy training scripts, `configs/` | Exploratory pretraining/custom U-Net code retained for audit |
| `docs/` | Training recipe, data requirements, manuscript-to-code map, provenance |

The legacy names containing `sota` are historical experiment identifiers, not claims of state-of-the-art performance. The exploratory pretraining code is not part of the primary grouped nnU-Net result.

## Data and model availability

MRI volumes, masks, individual-case metric tables, checkpoint weights, private logs, and the unpublished manuscript are not distributed here. Dataset access must be obtained from the respective providers. Published partition files contain only pseudonymous BraTS identifiers and assignments. [DATA_ACCESS.md](docs/DATA_ACCESS.md) describes the required inputs and the limits of reproducing the original processed archive.

Training from authorized input data is supported. Exact reproduction of historical masks additionally requires the authors' original checkpoints; these are maintained separately, and no public download is claimed. Hardware-dependent kernels mean retraining is not guaranteed to be bitwise identical.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this code release. A journal/DOI will be added after publication; none is asserted for the current manuscript. Project-authored code is provided under the [MIT License](LICENSE). Vendored code keeps its own license and attribution; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset permissions and model-weight redistribution are not granted by the code license.
