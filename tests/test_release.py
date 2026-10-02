from __future__ import annotations

import copy
from argparse import Namespace
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import nibabel as nib
from evaluate_brats2024_official_metrics import evaluate_case_subprocess, vendor_commit, REGIONS
from prepare_publication_data import index_development
from verify_release import read_csv, validate_protocol, verify_protocol, verify_statistics
from run_grouped import commands
from render_report import format_estimate
from research.grouped_delta_roi_v1.build_datasets import convert_npz_case
from research.grouped_et_topology_guard_v2.topology import metric_components, rc_topology_labels
from research.publication_integrity_v1.build_publication_statistics import (
    cluster_bootstrap_means, cluster_bootstrap_paired,
    cluster_case_weighted_sign_flip_p_value, holm_adjust, subject_weighted_mean,
)


class ReleaseTests(unittest.TestCase):
    def test_plot_does_not_round_nonzero_effect_to_zero(self):
        self.assertEqual(format_estimate(-2.703241e-5, effect=True), "-2.70e-05")
        self.assertEqual(format_estimate(0, effect=True), "0")

    def test_fixed_protocol_and_statistics(self):
        self.assertEqual(len(verify_protocol()), 5)
        verify_statistics()

    def test_subject_leak_is_rejected(self):
        folder = ROOT / "processed/brats_subject_grouped_protocol_v1"
        dev = read_csv(folder / "development_cases.csv")
        held = read_csv(folder / "locked_test_cases.csv")
        splits = json.loads((folder / "splits_final.json").read_text())
        modified = copy.deepcopy(dev)
        modified[0]["subject_id"] = held[0]["subject_id"]
        with self.assertRaisesRegex(ValueError, "subject overlap"):
            validate_protocol(splits, modified, held)

    def test_duplicate_scan_is_rejected(self):
        folder = ROOT / "processed/brats_subject_grouped_protocol_v1"
        dev = read_csv(folder / "development_cases.csv")
        held = read_csv(folder / "locked_test_cases.csv")
        splits = json.loads((folder / "splits_final.json").read_text())
        with self.assertRaisesRegex(ValueError, "Duplicate scan"):
            validate_protocol(splits, dev + dev[:1], held)

    def test_upstream_commit_is_not_export_repository_commit(self):
        self.assertEqual(vendor_commit(ROOT / "research/vendor/BraTS-2024-Metrics"), "f1131e340e155908553c226675644a6363b998b7")

    def test_upstream_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "vendor"
            shutil.copytree(ROOT / "research/vendor/BraTS-2024-Metrics", target)
            (target / "metrics_GLI.py").write_text("# changed\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                vendor_commit(target)

    def test_unversioned_directory_does_not_inherit_git(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(vendor_commit(Path(temp)))


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"subject_id": ["S1", "S1", "S2"], "score": [.2, .4, .9], "baseline": [.5, .6, .7], "final": [.5, .6, .7]})

    def test_subject_and_case_estimands_are_different(self):
        self.assertAlmostEqual(self.frame.score.mean(), .5)
        self.assertAlmostEqual(subject_weighted_mean(self.frame, ["score"])["score"], .6)

    def test_cluster_bootstrap_is_seeded(self):
        first = cluster_bootstrap_means(self.frame, ["score"], samples=100, seed=17)
        self.assertEqual(first, cluster_bootstrap_means(self.frame, ["score"], samples=100, seed=17))
        self.assertAlmostEqual(first["score"]["mean"], .5)

    def test_null_paired_change(self):
        result = cluster_bootstrap_paired(self.frame, ["baseline"], ["final"], ["HD95"], samples=100, seed=1)["HD95"]
        self.assertEqual(result["delta_ci95_lower"], 0)
        self.assertEqual(result["delta_ci95_upper"], 0)
        self.assertEqual(cluster_case_weighted_sign_flip_p_value(self.frame, "baseline", "final", samples=100, seed=1), 1)

    def test_holm_adjustment(self):
        values = holm_adjust({"a": .01, "b": .03, "c": .04})
        self.assertAlmostEqual(values["a"], .03)
        self.assertAlmostEqual(values["b"], .06)
        self.assertAlmostEqual(values["c"], .06)


class CommandTests(unittest.TestCase):
    def args(self, mode, fold):
        return Namespace(mode=mode, fold=fold, workers=1, device="cuda", resume=False, dry_run=True, output_dir=None)

    def test_fold_zero_uses_standard_trainer(self):
        cmd = commands(self.args("train", 0))[0]
        self.assertIn("nnUNetTrainer_250epochs", cmd)
        self.assertEqual(cmd[0], "nnUNetv2_train")

    def test_other_folds_use_historical_runner(self):
        for fold in range(1, 5):
            cmd = commands(self.args("train", fold))[0]
            self.assertIn("research/grouped_fivefold_cv_v11/run_fold.py", cmd)
            self.assertEqual(cmd[cmd.index("--fold") + 1], str(fold))

    def test_validation_selects_best(self):
        cmd = commands(self.args("validate", 0))[0]
        self.assertIn("--val_best", cmd)

    def test_preprocessing_does_not_replan(self):
        cmds = commands(self.args("preprocess", None))
        self.assertEqual([c[0] for c in cmds], ["nnUNetv2_extract_fingerprint", "nnUNetv2_preprocess"])
        self.assertIn("nnUNetResEncUNet10p5GBPlans", cmds[1])


class DataAndTopologyTests(unittest.TestCase):
    def test_conversion_preserves_channels_and_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = "BraTS-GLI-99999-100"
            image = np.arange(4 * 8**3, dtype=np.float32).reshape(4, 8, 8, 8)
            seg = np.zeros((8, 8, 8), dtype=np.uint8)
            seg[2:5, 2:5, 2:5] = 4
            np.savez_compressed(root / f"{case}.npz", image=image, seg=seg, case_id=case)
            convert_npz_case((case, str(root / f"{case}.npz"), str(root / "dataset")))
            for channel in range(4):
                result = nib.load(root / f"dataset/imagesTr/{case}_{channel:04d}.nii.gz")
                np.testing.assert_array_equal(np.asarray(result.dataobj), image[channel])
                np.testing.assert_array_equal(result.affine, np.eye(4))
            np.testing.assert_array_equal(np.asarray(nib.load(root / f"dataset/labelsTr/{case}.nii.gz").dataobj), seg)

    def test_missing_inputs_are_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                index_development(Path(temp), {"expected"})

    def test_index_excludes_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "development.npz").touch()
            (root / "holdout.npz").touch()
            self.assertEqual(set(index_development(root, {"development"})), {"development"})

    def test_topology_does_not_invent_rc(self):
        seg = np.zeros((20, 20, 20), dtype=np.uint8)
        seg[2:8, 2:8, 2:8] = 3
        self.assertEqual(int(rc_topology_labels(seg).max()), 0)

    def test_metric_component_minimum(self):
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[2:4, 2:4, 2:4] = True
        self.assertEqual(int(metric_components(mask, 3, 10).max()), 0)
        mask[10:14, 10:14, 10:14] = True
        result = metric_components(mask, 3, 10)
        self.assertTrue(np.any(result[10:14, 10:14, 10:14]))


class OfficialMetricSmokeTests(unittest.TestCase):
    def evaluate(self, miss_et=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gt = np.zeros((48, 48, 48), dtype=np.uint8)
            for label, corner in enumerate(((3, 3, 3), (28, 3, 3), (3, 28, 3), (28, 28, 28)), 1):
                x, y, z = corner
                gt[x:x+12, y:y+12, z:z+12] = label
            prediction = gt.copy()
            if miss_et:
                prediction[prediction == 3] = 0
            case = "BraTS-GLI-99999-100"
            for name, data in (("gt", gt), ("pred", prediction)):
                (root / name).mkdir()
                nib.save(nib.Nifti1Image(data, np.eye(4)), root / name / f"{case}.nii.gz")
            return evaluate_case_subprocess({"case_id": case, "prediction": str(root / "pred" / f"{case}.nii.gz"), "ground_truth": str(root / "gt" / f"{case}.nii.gz"), "vendor_dir": str(ROOT / "research/vendor/BraTS-2024-Metrics"), "cache_path": str(root / "cache" / f"{case}.json"), "work_root": str(root / "work")})

    def test_identical_masks_score_one_and_zero_hd95(self):
        row = self.evaluate()
        for region in REGIONS:
            self.assertAlmostEqual(row[f"{region}_LesionWise_Score_Dice"], 1, places=6)
            self.assertAlmostEqual(row[f"{region}_LesionWise_Score_HD95"], 0, places=6)

    def test_missed_et_has_official_penalty(self):
        row = self.evaluate(miss_et=True)
        self.assertEqual(row["ET_Num_FN"], 1)
        self.assertEqual(row["ET_LesionWise_Score_Dice"], 0)
        self.assertEqual(row["ET_LesionWise_Score_HD95"], 337)


if __name__ == "__main__":
    unittest.main()
