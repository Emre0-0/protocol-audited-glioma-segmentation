# Data and provenance

Request the BraTS 2024 post-treatment glioma cohort through the dataset provider.
ADNI and OASIS are relevant only to the retained exploratory pretraining code.
No MRI, mask, DICOM, per-case clinical measurement, or weight file is included here.

The source used in the study was a supplied **processed NPZ archive**, not a
fresh conversion of a new public download. Each file contains `image` shaped
`(4, X, Y, Z)`, `seg` shaped `(X, Y, Z)`, and scalar `case_id`. Channel order is
T1n, T1c, T2w, T2f. Labels are 0 background, 1 NETC, 2 SNFH, 3 ET, 4 RC.
WT is labels 1+2+3; TC is labels 1+3. RC is not included in WT or TC.

The audited conversion preserves the archive arrays and uses an identity
affine with 1-mm spacing. It cannot recover scanner orientation or spatial
metadata discarded upstream. Do not use that conversion for arbitrary
unprocessed MRI or claim it reconstructs the raw acquisition pipeline.
Original archive availability and preprocessing provenance remain dependencies
for reproducing the historical experiment.

The fixed partition is published as pseudonymous BraTS case identifiers. All
time points with the same `BraTS-GLI-xxxxx` subject prefix belong together.
Per-case label-presence flags and volumes have been removed from the public
assignment CSVs. For analysis of size subgroups, compute those measurements
from authorized reference masks using the provided evaluation/analysis code.

`scripts/create_brats_subject_grouped_protocol.py` records the original split
construction, including candidate balancing by label presence and volume.
Its source feature table is not included. Use the preserved split files to
reproduce this study rather than selecting a new favorable partition.

Public aggregate tables can be inspected immediately. To regenerate their
confidence intervals, supply the original case-level official metric CSVs
privately in the documented directory structure. Missing case-level inputs
cannot be replaced by aggregate means to reconstruct a valid bootstrap CI.

The study's 195-scan holdout was opened once and has already informed the
reported retrospective analysis. It is not available for further tuning or
for relabeling as a new independent test cohort.
