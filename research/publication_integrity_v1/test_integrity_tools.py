from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Support both `python test_integrity_tools.py` and package-style unittest discovery.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_publication_integrity import subject_id
from build_submission_readiness import checklist
from build_publication_statistics import (
    cluster_bootstrap_means,
    cluster_bootstrap_paired,
    cluster_case_weighted_sign_flip_p_value,
    detection_metrics_from_counts,
    holm_adjust,
    sign_flip_p_value,
    subject_weighted_mean,
)
from inspect_checkpoint_metadata import canonical_hash
from build_results_assets import compact_effect, compact_interval


class PublicationStatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "subject_id": ["S1", "S1", "S2"],
                "metric": [0.2, 0.4, 0.9],
                "baseline": [0.5, 0.6, 0.7],
                "final": [0.5, 0.6, 0.7],
            }
        )

    def test_case_and_subject_weighted_estimands_are_distinct(self) -> None:
        case_weighted = float(self.frame["metric"].mean())
        subject_weighted = subject_weighted_mean(self.frame, ["metric"])["metric"]
        self.assertAlmostEqual(case_weighted, 0.5)
        self.assertAlmostEqual(subject_weighted, 0.6)

    def test_cluster_bootstrap_preserves_observed_case_mean(self) -> None:
        result = cluster_bootstrap_means(
            self.frame,
            ["metric"],
            samples=200,
            seed=17,
        )["metric"]
        self.assertAlmostEqual(result["mean"], 0.5)
        self.assertLessEqual(result["ci95_lower"], result["mean"])
        self.assertGreaterEqual(result["ci95_upper"], result["mean"])

    def test_zero_paired_difference_has_unit_p_value(self) -> None:
        self.assertEqual(sign_flip_p_value(np.zeros(4), samples=100, seed=7), 1.0)
        self.assertEqual(
            cluster_case_weighted_sign_flip_p_value(
                self.frame,
                "baseline",
                "final",
                samples=100,
                seed=7,
            ),
            1.0,
        )

    def test_paired_hd95_bootstrap_reports_each_model_and_delta(self) -> None:
        result = cluster_bootstrap_paired(
            self.frame,
            ["baseline"],
            ["final"],
            ["HD95"],
            samples=200,
            seed=23,
        )["HD95"]
        self.assertAlmostEqual(result["baseline"], 0.6)
        self.assertAlmostEqual(result["final"], 0.6)
        self.assertAlmostEqual(result["delta"], 0.0)
        self.assertLessEqual(result["baseline_ci95_lower"], result["baseline"])
        self.assertGreaterEqual(result["baseline_ci95_upper"], result["baseline"])
        self.assertEqual(result["delta_ci95_lower"], 0.0)
        self.assertEqual(result["delta_ci95_upper"], 0.0)

    def test_lesion_detection_metrics(self) -> None:
        result = detection_metrics_from_counts(tp=8, fp=2, fn=2)
        self.assertAlmostEqual(result["sensitivity"], 0.8)
        self.assertAlmostEqual(result["precision"], 0.8)
        self.assertAlmostEqual(result["f1"], 0.8)
        self.assertEqual(result["reference_lesions"], 10.0)

    def test_holm_adjustment_is_monotone(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.06)

    def test_brats_subject_identifier(self) -> None:
        self.assertEqual(subject_id("BraTS-GLI-01234-101"), "BraTS-GLI-01234")
        with self.assertRaises(ValueError):
            subject_id("unexpected-case")

    def test_claim_working_checklist_has_all_items_once(self) -> None:
        rows = checklist(completed_folds=3)
        self.assertEqual(len(rows), 44)
        self.assertEqual([row[0] for row in rows], list(range(1, 45)))

    def test_canonical_hash_is_mapping_order_independent(self) -> None:
        self.assertEqual(
            canonical_hash({"a": 1, "b": 2}),
            canonical_hash({"b": 2, "a": 1}),
        )

    def test_publication_bibliography_has_required_balanced_entries(self) -> None:
        root = Path(__file__).resolve().parents[2]
        text = (root / "paper_results_assets/publication_references.bib").read_text(
            encoding="utf-8"
        )
        self.assertEqual(text.count("{"), text.count("}"))
        for key in (
            "deVerdier2024BraTSPostTreatment",
            "Tejani2024CLAIM",
            "MaierHein2024MetricsReloaded",
            "Isensee2021nnUNet",
            "Isensee2019HDBET",
        ):
            self.assertIn(key, text)

    def test_figure_effect_formatting_preserves_near_zero_values(self) -> None:
        self.assertEqual(compact_effect(-2.703241196655437e-5), "-2.70e-5")
        self.assertEqual(compact_effect(0.0), "0 (unchanged)")
        self.assertEqual(
            compact_interval(-7.469171416805697e-5, 1.2260373824943268e-5),
            "[-7.47e-5, +1.23e-5]",
        )

    def test_interval_zero_is_numeric_not_unchanged_label(self) -> None:
        self.assertEqual(compact_interval(-0.2676, 0.0, digits=3), "[-0.268, 0.000]")


if __name__ == "__main__":
    unittest.main()
