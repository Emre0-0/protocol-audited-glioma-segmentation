from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REGIONS = ("WT", "TC", "NETC", "SNFH", "ET", "RC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize the corrected Results OOF sections with statistics."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def f4(value: float) -> str:
    return f"{float(value):.4f}"


def f2(value: float) -> str:
    return f"{float(value):.2f}"


def signed4(value: float) -> str:
    rounded = round(float(value), 4)
    return "0.0000" if rounded == 0 else f"{rounded:+.4f}"


def metric_key(region: str) -> str:
    return f"{region}_LesionWise_Score_Dice"


def hd95_key(region: str) -> str:
    return f"{region}_LesionWise_Score_HD95"


def estimate_ci(values: dict[str, float], prefix: str = "") -> str:
    mean_key = prefix[:-1] if prefix.endswith("_") else prefix
    if prefix:
        mean = values[mean_key]
        lower = values[f"{prefix}ci95_lower"]
        upper = values[f"{prefix}ci95_upper"]
    else:
        mean = values["mean"]
        lower = values["ci95_lower"]
        upper = values["ci95_upper"]
    return f"{f2(mean)} [{f2(lower)}, {f2(upper)}]"


def development_section(oof: dict[str, Any]) -> str:
    count = int(oof["completed_fold_count"])
    expected = int(oof["expected_fold_count"])
    final = count == expected
    pooled = oof["case_weighted_cluster_bootstrap"]
    six = pooled["six_region_mean"]
    completed = ", ".join(str(item) for item in oof["completed_folds"])
    if final:
        opening = (
            "All five pre-specified subject-grouped folds completed training, "
            "full-volume inference, and official internal testing."
        )
        status_word = "final"
        pool_label = "Pooled five-fold OOF"
        caption_status = "Final"
    else:
        opening = (
            f"At the current audit cutoff, {count} of the {expected} pre-specified "
            "subject-grouped folds had completed training, full-volume inference, "
            "and official internal testing."
        )
        status_word = "interim"
        pool_label = "Pooled interim OOF"
        caption_status = "Interim"

    regions_sentence = (
        f"Region-wise {status_word} Dice was {f4(pooled[metric_key('WT')]['mean'])} "
        "for whole tumor (WT), "
        f"{f4(pooled[metric_key('TC')]['mean'])} for tumor core (TC), "
        f"{f4(pooled[metric_key('NETC')]['mean'])} for non-enhancing tumor core "
        f"(NETC), {f4(pooled[metric_key('SNFH')]['mean'])} for surrounding "
        "non-enhancing FLAIR hyperintensity (SNFH), "
        f"{f4(pooled[metric_key('ET')]['mean'])} for enhancing tumor (ET), and "
        f"{f4(pooled[metric_key('RC')]['mean'])} for resection cavity (RC)"
    )
    fold_means = ", ".join(f4(item["six_region_mean"]) for item in oof["folds"])
    fold_range = (
        f"The fold-specific six-region means were {fold_means} for folds "
        f"{completed}, respectively."
    )
    if final:
        interpretation = (
            "This heterogeneity, particularly for TC, ET, and RC, supports the "
            "pooled OOF estimate and subject-cluster uncertainty rather than "
            "selection of the strongest fold."
        )
    else:
        interpretation = (
            "This variation, particularly for TC, ET, and RC, supports reporting "
            "OOF uncertainty rather than selecting the strongest fold. Because "
            "folds 3 and 4 are still running, these pooled values are explicitly "
            "interim and are not final five-fold estimates."
        )

    rows = []
    for item in oof["folds"]:
        values = " & ".join(f4(item[region]) for region in REGIONS)
        rows.append(
            f"Fold {item['fold']} & {item['cases']} & {item['subjects']} & "
            f"{values} & {f4(item['six_region_mean'])} \\\\"
        )
    pooled_values = " & ".join(
        f4(pooled[metric_key(region)]["mean"]) for region in REGIONS
    )
    fold_hd95 = {
        (int(item["fold"]), item["region"]): item for item in oof["fold_hd95"]
    }
    hd95_rows = []
    for region in REGIONS:
        cells = [estimate_ci(pooled[hd95_key(region)])]
        cells.extend(
            estimate_ci(fold_hd95[(int(item["fold"]), region)])
            for item in oof["folds"]
        )
        hd95_rows.append(f"{region} & " + " & ".join(cells) + r" \\")
    hd95_rows_text = "\n".join(hd95_rows)
    pooled_hd95_sentence = ", ".join(
        f"{region} {f2(pooled[hd95_key(region)]['mean'])}"
        for region in REGIONS
    )
    rows_text = "\n".join(rows)
    return rf"""\subsection{{Subject-disjoint development performance}}
\label{{subsec:grouped_development}}

{opening} These folds contained {oof['cases']} non-duplicated OOF scans from
{oof['subjects']} subjects. The pooled {status_word} six-region lesion-wise
Dice was {f4(six['mean'])} (subject-cluster 95\% CI,
{f4(six['ci95_lower'])}--{f4(six['ci95_upper'])}), and the corresponding
subject-weighted mean Dice was
{f4(oof['subject_weighted_alternative_estimand']['six_region_mean'])}.
{regions_sentence}
(Table~\ref{{tab:interim_grouped_folds}}; Fig.~\ref{{fig:grouped_oof}}).

{fold_range} {interpretation} Folds completed after the one-time retrospective
holdout opening were used only to finish the pre-specified development OOF
analysis; those later checkpoints were never applied to the closed holdout.

Complementary pooled lesion-wise HD95 values were {pooled_hd95_sentence} mm
(Table~\ref{{tab:grouped_hd95}}). Lower HD95 indicates better boundary
agreement. Under the official GLI implementation, unmatched false-positive or
false-negative lesions receive a 337-mm penalty; consequently, lesion-wise
HD95 summarizes both boundary localization and lesion-detection failures.

\begin{{table*}}[t]
\centering
\caption{{Official lesion-wise Dice on the completed subject-disjoint folds.
The pooled row is {status_word}{' until all folds complete' if not final else ''}.}}
\label{{tab:interim_grouped_folds}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccccccc}}
\toprule
Analysis & Cases & Subjects & WT & TC & NETC & SNFH & ET & RC & Six-region mean \\
\midrule
{rows_text}
\midrule
{pool_label} & {oof['cases']} & {oof['subjects']} & {pooled_values} & {f4(six['mean'])} \\
\bottomrule
\end{{tabular}}}}
\end{{table*}}

\begin{{table*}}[t]
\centering
\caption{{Official lesion-wise HD95 in millimetres for the pooled OOF analysis
and each subject-disjoint fold. Entries are case-weighted mean [95\% subject-cluster
bootstrap CI]; lower values are better. Fold intervals resample subjects within
the corresponding fold.}}
\label{{tab:grouped_hd95}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccc}}
\toprule
Region & Pooled OOF & Fold 0 & Fold 1 & Fold 2 & Fold 3 & Fold 4 \\
\midrule
{hd95_rows_text}
\bottomrule
\end{{tabular}}}}
\end{{table*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\linewidth]{{corrected_figures/grouped_oof_dice.pdf}}
    \caption{{{caption_status} subject-disjoint OOF performance from {count} completed
    folds. Points show case-weighted means, horizontal whiskers show 95\%
    subject-cluster bootstrap intervals, and the right column reports each
    estimate and interval numerically.}}
\label{{fig:grouped_oof}}
\end{{figure*}}
"""


def holdout_hd95_block(locked: dict[str, Any]) -> str:
    hd95 = locked["hd95_case_weighted_subject_cluster_bootstrap"]
    rows = []
    for region in REGIONS:
        values = hd95[region]
        baseline = estimate_ci(values, "baseline_")
        final = estimate_ci(values, "final_")
        delta = (
            f"{signed4(values['delta'])} "
            f"[{f4(values['delta_ci95_lower'])}, {f4(values['delta_ci95_upper'])}]"
        )
        rows.append(f"{region} & {baseline} & {final} & {delta} " + r"\\")
    rows_text = "\n".join(rows)
    et = hd95["ET"]
    return rf"""% BEGIN GENERATED HOLDOUT HD95
Boundary-distance estimates did not show a uniform benefit for frozen V2.
The largest point-estimate change was for ET, where lesion-wise HD95 increased
from {f2(et['baseline'])} to {f2(et['final'])} mm (paired difference
{f2(et['delta'])} mm). Table~\ref{{tab:locked_test_hd95}} reports uncertainty
for each model and for the paired change. For HD95, lower values are better and
a negative difference favors frozen V2.

\begin{{table*}}[t]
\centering
\caption{{One-time retrospective holdout lesion-wise HD95 in millimetres.
Baseline and frozen-V2 entries are case-weighted mean [95\% subject-cluster
bootstrap CI]; differences are frozen V2 minus baseline with paired 95\%
subject-cluster bootstrap CIs. Lower values and negative differences are better.}}
\label{{tab:locked_test_hd95}}
\begin{{tabular}}{{lccc}}
\toprule
Region & Baseline [95\% CI] & Frozen V2 [95\% CI] & Difference [95\% CI] \\
\midrule
{rows_text}
\bottomrule
\end{{tabular}}
\end{{table*}}
% END GENERATED HOLDOUT HD95"""


def replace_or_insert_holdout_hd95(text: str, block: str) -> str:
    marker_pattern = (
        r"% BEGIN GENERATED HOLDOUT HD95.*?% END GENERATED HOLDOUT HD95"
    )
    if re.search(marker_pattern, text, flags=re.S):
        return re.sub(marker_pattern, lambda _: block, text, count=1, flags=re.S)
    table_pattern = (
        r"(\\begin\{table\*\}\[t\].*?"
        r"\\label\{tab:locked_test\}.*?\\end\{table\*\})"
    )
    updated, count = re.subn(
        table_pattern,
        lambda match: match.group(1) + "\n\n" + block,
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError(f"Could not insert holdout HD95 block; matches={count}")
    return updated


def detection_opening(oof: dict[str, Any]) -> str:
    count = int(oof["completed_fold_count"])
    final = count == int(oof["expected_fold_count"])
    status = "final" if final else "interim"
    et = oof["lesion_detection"]["ET"]
    rc = oof["lesion_detection"]["RC"]
    return rf"""\subsection{{Small-lesion reliability and failure modes}}
\label{{subsec:small_lesion_reliability}}

The {status} grouped OOF predictions detected
{int(et['true_positive_lesions'])} of {int(et['reference_lesions'])} reference
ET lesions, with {int(et['false_positive_lesions'])} false-positive and
{int(et['false_negative_lesions'])} false-negative lesions. This corresponded
to lesion sensitivity of {f4(et['sensitivity'])} (subject-cluster 95\% CI,
{f4(et['sensitivity_ci95_lower'])}--{f4(et['sensitivity_ci95_upper'])}),
precision of {f4(et['precision'])}
({f4(et['precision_ci95_lower'])}--{f4(et['precision_ci95_upper'])}), and F1 of
{f4(et['f1'])} ({f4(et['f1_ci95_lower'])}--{f4(et['f1_ci95_upper'])}). RC
detection had higher sensitivity ({f4(rc['sensitivity'])}; 95\% CI,
{f4(rc['sensitivity_ci95_lower'])}--{f4(rc['sensitivity_ci95_upper'])}) but
lower precision ({f4(rc['precision'])}; {f4(rc['precision_ci95_lower'])}--
{f4(rc['precision_ci95_upper'])}), indicating a different false-positive
burden (Fig.~\ref{{fig:grouped_detection}}). These object-level results
complement overlap metrics by separating missed lesions from spurious
detections.

\begin{{figure*}}[t]
\centering
\includegraphics[width=\linewidth]{{corrected_figures/grouped_oof_lesion_detection.pdf}}
    \caption{{{status.capitalize()} subject-disjoint lesion detection sensitivity and precision
    from {count} completed folds. Circles denote sensitivity, squares denote
    precision, and horizontal whiskers show 95\% subject-cluster bootstrap
    intervals.}}
\label{{fig:grouped_detection}}
\end{{figure*}}

"""


def replace_once(text: str, pattern: str, replacement: str, name: str) -> str:
    updated, count = re.subn(pattern, lambda _: replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"Could not replace {name}; matches={count}")
    return updated


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    statistics = json.loads(
        (root / "experiments/publication_integrity_v1/statistics/publication_statistics.json")
        .read_text(encoding="utf-8-sig")
    )
    oof = statistics["grouped_oof"]
    locked = statistics["locked_test"]
    results_path = root / "paper_results_assets/results_section.tex"
    text = results_path.read_text(encoding="utf-8")
    final = int(oof["completed_fold_count"]) == int(oof["expected_fold_count"])
    header = (
        "% Corrected Results section generated from the subject-disjoint protocol.\n"
        + (
            "% Final five-fold OOF statistics are synchronized from the publication evidence bundle.\n"
            if final
            else "% IMPORTANT: This is an interim manuscript draft until all five grouped folds complete.\n"
        )
        + "% Do not manually edit numeric OOF values; rerun build_results_section.py.\n"
        + "% Required packages: graphicx, booktabs."
    )
    text = replace_once(
        text,
        r"\A% Corrected Results section.*?% Required packages: graphicx, booktabs\.",
        header,
        "header",
    )
    text = replace_once(
        text,
        r"\\subsection\{Subject-disjoint development performance\}.*?(?=\\subsection\{Retrospective one-time holdout evaluation\})",
        development_section(oof) + "\n",
        "development subsection",
    )
    opening = detection_opening(oof)
    text = replace_once(
        text,
        r"\\subsection\{Small-lesion reliability and failure modes\}.*?(?=The retrospective holdout comparison)",
        opening,
        "small-lesion OOF opening",
    )
    text = replace_or_insert_holdout_hd95(text, holdout_hd95_block(locked))
    results_path.write_text(text, encoding="utf-8")
    print(results_path)


if __name__ == "__main__":
    main()
