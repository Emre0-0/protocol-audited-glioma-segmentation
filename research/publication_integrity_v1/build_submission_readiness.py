from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLAIM_URL = "https://pubs.rsna.org/doi/10.1148/ryai.240300"
METRICS_URL = "https://www.nature.com/articles/s41592-023-02151-z"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an evidence-backed SCI/thesis submission readiness checklist."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/publication_integrity_v1/SCI_THESIS_READINESS.md"),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checklist(completed_folds: int) -> list[tuple[int, str, str, str]]:
    final_state = "READY" if completed_folds == 5 else "PENDING"
    uncertainty_evidence = (
        "Final five-fold OOF values and subject-cluster 95% CIs are synchronized "
        "with the Results draft."
        if completed_folds == 5
        else "Interim values must be replaced after all five folds finish."
    )
    return [
        (1, "AI methodology identified", "ACTION", "Use deep learning/3D nnU-Net and a reliability-focused title; do not claim SOTA."),
        (2, "Structured abstract", final_state, "Include retrospective design, 613 subjects/1,350 scans, patient-level partitioning, OOF uncertainty, and resource availability."),
        (3, "Scientific/clinical background", "ACTION", "Complete focused post-treatment glioma and small-lesion reliability literature review."),
        (4, "Aims and hypotheses", "ACTION", "Distinguish pre-specified aims from post-hoc exploratory analyses; do not rewrite exploration as a priori."),
        (5, "Prospective or retrospective", "READY", "Retrospective secondary analysis of the supplied BraTS-derived archive."),
        (6, "Study goal", "READY", "Methodological reliability/evaluation study, not a clinical deployment trial."),
        (7, "Data sources", "READY", "BraTS 2024, CLAIM, Metrics Reloaded, nnU-Net, and HD-BET source evidence plus BibTeX entries are archived; ADNI/OASIS remains a limited exploratory ablation."),
        (8, "Inclusion/exclusion criteria", "PARTIAL", "The source cohort is post-treatment diffuse glioma, but exact local inclusion/exclusion and case-removal reasons are not recoverable from the archive."),
        (9, "Preprocessing", "PARTIAL", "All 565 normal T1 inputs trace to successful HD-BET outputs. BraTS inherited challenge skull stripping/registration; NPZ conversion used nonzero z-score and discarded affines. OASIS command parameters remain unavailable."),
        (10, "Selection of data subsets", "READY", "All 1,350 supplied cases are inventoried; assignment manifests record every case exactly once."),
        (11, "De-identification", "AUTHOR", "Add source-dataset de-identification/licensing statement and institutional confirmation."),
        (12, "Missing data", "ACTION", "State modality completeness checks, exclusions, and that no imputation was performed if confirmed."),
        (13, "Image acquisition protocol", "LIMITATION", "Scanner/site/acquisition metadata are absent from the archive; cite source documentation and disclose this limitation."),
        (14, "Reference standard definition", "READY", "Exact label/region mapping is audited and the challenge annotation source is archived; use 'reference standard'."),
        (15, "Reference standard rationale", "PARTIAL", "Challenge labels are the only supplied reference standard; explain their use and possible errors/biases in the manuscript."),
        (16, "Annotation source", "PARTIAL", "The source protocol and expert groups are documented, but local annotator identities and inter-rater data are unavailable."),
        (17, "Internal-test annotation", "READY", "All folds inherit one source reference standard, and the exact label-to-region mapping is recorded against the vendored official metric code."),
        (18, "Inter-/intrarater variability", "LIMITATION", "Not available in the supplied archive; explicitly report as unavailable."),
        (19, "Partition assignment", "READY", "Five subject-grouped development folds plus a documented late-defined retrospective holdout."),
        (20, "Disjointness level", "READY", "Patient/subject-disjoint; every development scan appears in OOF testing exactly once."),
        (21, "Intended test sample size", "PARTIAL", "All eligible retrospective cases are used; justify no a priori power calculation and report confidence-interval precision."),
        (22, "Detailed model description", "READY", "nnU-Net plans, architecture state signatures, channels, labels, and scripts are archived."),
        (23, "Libraries/software/hardware", "READY", "Frozen package/environment manifest and RTX 5070 Ti details are generated."),
        (24, "Parameter initialization", "READY", "Document random nnU-Net initialization and separately disclose the flawed/limited pretraining ablation."),
        (25, "Training approach", "PARTIAL", "Code and checkpoint metadata exist; manuscript/supplement still needs a compact augmentation, loss, optimizer, LR, batch, and stopping table."),
        (26, "Final model selection", "PARTIAL", "Describe best-EMA checkpoint selection. The already opened holdout cannot select or tune any later model."),
        (27, "Ensembling", "PARTIAL", "Describe the frozen V2 combination only where reported; do not reuse the invalid legacy five-seed result as five-fold evidence."),
        (28, "Performance metrics", "READY", "Official BraTS lesion-wise metrics plus lesion sensitivity, precision, F1, and failure analysis."),
        (29, "Uncertainty/significance", final_state, "Subject-cluster bootstrap CIs, paired tests, Holm correction, and explicit estimands are implemented."),
        (30, "Robustness/sensitivity", final_state, "Report fold heterogeneity and clearly label small-lesion subgroup analyses as post-hoc."),
        (31, "Explainability", "NA", "No validated explainability method was used; qualitative error maps are failure analysis, not explainability."),
        (32, "Internal testing", final_state, "Primary evidence is patient-disjoint grouped OOF internal testing."),
        (33, "External testing", "LIMITATION", "No independent institution/site dataset is available. The retrospective holdout is not external testing."),
        (34, "Clinical trial registration", "NA", "Not a clinical trial; state retrospective secondary-data design and any ethics/waiver requirements."),
        (35, "Included/excluded counts", "PARTIAL", "Counts are audited; add a cohort flow diagram and exclusion reasons."),
        (36, "Demographic/clinical characteristics", "BLOCKER", "Age, sex, site, scanner, treatment and other demographics are unavailable locally; obtain source metadata or state the limitation prominently."),
        (37, "Performance with uncertainty", final_state, uncertainty_evidence),
        (38, "Diagnostic estimates/precision", "PARTIAL", "Lesion detection CIs are available; ROC/AUC is not appropriate for hard segmentation masks and must not be fabricated."),
        (39, "Failure analysis", "READY", "False-positive/false-negative lesions, tiny-ET trade-off, and qualitative examples are reported."),
        (40, "Limitations", "ACTION", "Include legacy leakage, late holdout definition, no external test, metadata gaps, pretraining leakage/orientation issues, and nondeterministic GPU boundary."),
        (41, "Implications for practice", "AUTHOR", "Clinical co-author should frame intended use; no autonomous or clinical-readiness claim is supported."),
        (42, "Protocol/technical details", "READY", "Audit, split manifests, metric vendor hash, training scripts, and reproducibility snapshot are available."),
        (43, "Software/model/data availability", "BLOCKER", "Create a valid version-controlled public repository/DOI or document controlled access; current .git directory is not a valid repository."),
        (44, "Funding/support", "AUTHOR", "Authors must provide funding, funder role, conflicts, ethics, licenses, and AI-assistance disclosures."),
    ]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    audit_path = root / "experiments/publication_integrity_v1/audit.json"
    audit = load_json(audit_path)
    completion = next(
        item
        for item in audit["checks"]
        if item["check_id"] == "grouped_fivefold_completion"
    )
    completed_folds = len(completion["evidence"].get("completed_folds", []))
    rows = checklist(completed_folds)
    counts: dict[str, int] = {}
    for _, _, status, _ in rows:
        counts[status] = counts.get(status, 0) + 1

    submission_ready = (
        audit["overall_status"] in {"READY", "READY_WITH_MANDATORY_LIMITATIONS"}
        and completed_folds == 5
        and counts.get("BLOCKER", 0) == 0
    )
    blockers: list[str] = []
    if completed_folds != 5:
        blockers.append(
            "Finish all five grouped folds, then regenerate OOF statistics, "
            "confidence intervals, lesion detection, figures, and Results text."
        )
    if audit.get("array_check") != "full":
        blockers.append(
            "Complete the full NPZ-to-NIfTI array-fidelity audit."
        )
    blockers.extend(
        [
            "Obtain or explicitly document unavailable demographic, scanner/site, "
            "acquisition, eligibility, and reference-standard metadata.",
            "Create a valid version-controlled code release with a permanent "
            "archive/DOI or a controlled-access statement.",
            "Add author/institution-controlled ethics, data-license, funding, "
            "conflict-of-interest, author-contribution, and AI-assistance statements.",
            "Have a neuroradiology/clinical co-author review label definitions, "
            "qualitative examples, intended use, and clinical interpretation.",
        ]
    )
    lines = [
        "# SCI journal and master's thesis readiness",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Automated integrity status: **{audit['overall_status']}**",
        f"- Subject-grouped folds with official metrics: **{completed_folds}/5**",
        f"- Submission ready now: **{'YES' if submission_ready else 'NO'}**",
        f"- Checklist counts: `{json.dumps(counts, sort_keys=True)}`",
        "",
        "## Defensible study identity",
        "",
        "Recommended framing: a retrospective methodological study of leakage-aware, "
        "patient-disjoint evaluation and small-lesion reliability in post-treatment "
        "glioma segmentation. The contribution is the corrected evaluation and failure "
        "analysis, not a proven SOTA architecture or clinical product.",
        "",
        "Recommended working title: **Beyond Mean Dice: Patient-Disjoint Evaluation "
        "and Small-Lesion Reliability in Post-Treatment Glioma Segmentation**.",
        "",
        "## Non-negotiable evidence boundaries",
        "",
        "1. The legacy 945/202/203 case split is excluded because longitudinal subjects crossed partitions.",
        "2. Primary generalization evidence is five-fold patient-disjoint OOF internal testing.",
        "3. The 195-scan holdout is late-defined retrospective secondary evidence, not pristine or external testing.",
        "4. That holdout was opened once and is permanently closed; no later checkpoint may be tested on it.",
        "5. The frozen V2 change is not statistically conclusive and worsens tiny ET sensitivity; superiority is not supported.",
        "6. ADNI/OASIS pretraining is a limited negative/exploratory ablation because its original split leaked subjects and provenance is incomplete.",
        "7. No clinical-readiness, deployment, external-validity, or absolute-SOTA claim is currently defensible.",
        "",
        "## Remaining submission blockers",
        "",
        *[f"- {item}" for item in blockers],
        "",
        "## CLAIM 2024 working checklist",
        "",
        "This is a working readiness map, not a substitute for the journal's official checklist. "
        f"Official source: {CLAIM_URL}. Metric-selection framework: {METRICS_URL}.",
        "",
        "| Item | Requirement | Status | Project action/evidence |",
        "|---:|---|---|---|",
    ]
    for number, requirement, status, evidence in rows:
        lines.append(f"| {number} | {requirement} | **{status}** | {evidence} |")
    lines.extend(
        [
            "",
            "## Final release gate",
            "",
            "Journal submission should occur only after the automated audit has no FAIL or "
            "PENDING checks, all mandatory limitations are present in the manuscript, and "
            "the human-controlled BLOCKER/AUTHOR items above are resolved. A technically "
            "correct thesis can still report negative or inconclusive findings; hiding them "
            "would make the work weaker, not stronger.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
