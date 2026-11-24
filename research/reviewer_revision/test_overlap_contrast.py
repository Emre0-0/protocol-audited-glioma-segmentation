"""Analytic checks, using temporary synthetic scores only; no study estimates."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from score_overlap_contrast import REGIONS, contrast, read_scores


class OverlapContrastTest(unittest.TestCase):
    def test_scan_identity_and_metric_order(self):
        keys = [r + "_LesionWise_Score_Dice" for r in REGIONS]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.csv"
            def write(ids):
                with path.open("w", newline="") as out:
                    writer = csv.writer(out)
                    writer.writerow(["case_id", *keys])
                    for c in ids:
                        writer.writerow([c, *(np.arange(6) / 10 + (c == "B") / 4)])
            write(["B", "A"])
            scores = read_scores(path, ["A", "B"])
            np.testing.assert_allclose(scores[:, 0], [.25, .5])
            np.testing.assert_allclose(scores[0, 1:], np.arange(6) / 10)
            write(["A", "A"])
            with self.assertRaises(ValueError):
                read_scores(path, ["A", "B"])

    def test_repeated_patient_draws_and_sign(self):
        # Patient P contributes two scans with +0.25; Q contributes one with -0.125.
        # Drawing two patients therefore permits exactly three scan-weighted means.
        clean = np.full((3, 7), .5)
        exposed = clean + np.array([[.25], [.25], [-.125]])
        ids = ["P", "P", "Q"]
        def compare(a, b):
            return contrast(a, b, [0, 1, 2], np.random.default_rng(19), ids)
        d, draws = compare(clean, exposed)
        np.testing.assert_allclose(d, .125)
        np.testing.assert_array_equal(np.unique(draws), [-.125, .125, .25])
        rd, reversed_draws = compare(exposed, clean)
        np.testing.assert_array_equal(rd, -d)
        np.testing.assert_array_equal(reversed_draws, -draws)
        null, null_draws = compare(clean, clean)
        self.assertFalse(null.any() or null_draws.any())


if __name__ == "__main__":
    unittest.main()
