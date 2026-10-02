from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit archived HD-BET and NPZ preprocessing provenance."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/publication_integrity_v1/data_provenance"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def normalize_windows_path(value: Any) -> str:
    return str(PureWindowsPath(str(value))).lower()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def render_markdown(report: dict[str, Any]) -> str:
    normal = report["normal_t1_pretraining"]
    brats = report["brats_segmentation"]
    return "\n".join(
        [
            "# Data preprocessing provenance audit",
            "",
            f"- Status: **{report['status']}**",
            f"- Generated: {report['generated_at_utc']}",
            "",
            "## Normal T1 pretraining data",
            "",
            f"- Pretraining NPZ files: {normal['pretraining_npz_rows']} (all successful: {normal['pretraining_npz_all_success']}).",
            f"- ADNI HD-BET outputs: {normal['adni_hdbet_rows']} (all successful/return code 0: {normal['adni_hdbet_all_success']}).",
            f"- OASIS HD-BET outputs: {normal['oasis_hdbet_rows']} (all successful/return code 0: {normal['oasis_hdbet_all_success']}).",
            f"- Exact HD-BET-output to pretraining-input matches: {normal['exact_path_matches']}/{normal['pretraining_npz_rows']}.",
            "- ADNI runner provenance records CPU execution, disabled TTA, and saved brain masks.",
            "- The OASIS execution log is present, but its runner script/parameter list was not included in the archive.",
            "- NPZ conversion applied nonzero-voxel z-score normalization and did not retain NIfTI affine/orientation metadata.",
            "",
            "## BraTS segmentation data",
            "",
            f"- BraTS NPZ rows: {brats['npz_rows']} (all successful: {brats['npz_all_success']}).",
            "- The local NIfTI-to-NPZ script did not invoke HD-BET; it consumed challenge-preprocessed NIfTI volumes.",
            "- The script checked within-case modality/label shape consistency and applied per-channel nonzero z-score normalization.",
            "- The supplied NPZ format discarded source affine/orientation metadata. Later NIfTI reconstruction therefore uses an identity affine and must not be described as preserving scanner coordinates.",
            "- BraTS skull stripping, co-registration, and 1-mm resampling should be attributed to the source challenge preprocessing, not to this project code.",
            "",
            "## Claim boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eda = root / "_archive_inspect/eda/eda_outputs"
    scripts = root / "_archive_inspect/scripts/scripts"
    adni_path = eda / "hdbet_adni_log.csv"
    oasis_path = eda / "hdbet_selected_oasis_log.csv"
    pretrain_path = eda / "pretrain_t1_npz_log.csv"
    brats_path = eda / "brats2024_npz_full_log.csv"
    adni_runner = scripts / "run_hdbet_adni.py"
    pretrain_runner = scripts / "04_prepare_pretrain_t1_npz.py"
    brats_runner = scripts / "02_prepare_brats_npz.py"

    required = (
        adni_path,
        oasis_path,
        pretrain_path,
        brats_path,
        adni_runner,
        pretrain_runner,
        brats_runner,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing provenance evidence: {missing}")

    adni = pd.read_csv(adni_path)
    oasis = pd.read_csv(oasis_path)
    pretrain = pd.read_csv(pretrain_path)
    brats = pd.read_csv(brats_path)
    hdbet_outputs = {
        normalize_windows_path(value)
        for value in pd.concat(
            [adni["output_path"], oasis["output_path"]], ignore_index=True
        )
    }
    pretrain_inputs = {
        normalize_windows_path(value) for value in pretrain["input_path"]
    }
    exact_matches = len(hdbet_outputs.intersection(pretrain_inputs))
    adni_source = adni_runner.read_text(encoding="utf-8-sig")
    pretrain_source = pretrain_runner.read_text(encoding="utf-8-sig")
    brats_source = brats_runner.read_text(encoding="utf-8-sig")

    adni_success = bool(
        adni["status"].eq("success").all() and adni["returncode"].eq(0).all()
    )
    oasis_success = bool(
        oasis["status"].eq("success").all() and oasis["returncode"].eq(0).all()
    )
    pretrain_success = bool(pretrain["status"].eq("success").all())
    brats_success = bool(brats["status"].eq("success").all())
    source_checks = {
        "adni_runner_calls_hdbet": '"-i"' in adni_source and '"-o"' in adni_source,
        "adni_runner_cpu": '"-device", "cpu"' in adni_source,
        "adni_runner_disables_tta": '"--disable_tta"' in adni_source,
        "adni_runner_saves_mask": '"--save_bet_mask"' in adni_source,
        "pretrain_consumes_brain_outputs": "BRAIN_ROOT" in pretrain_source
        and '"*_brain.nii.gz"' in pretrain_source,
        "pretrain_nonzero_zscore": "normalize_nonzero_zscore" in pretrain_source,
        "brats_nonzero_zscore": "normalize_nonzero_zscore" in brats_source,
        "brats_no_local_hdbet_call": "hd-bet" not in brats_source.lower()
        and "hdbet" not in brats_source.lower(),
    }
    valid = (
        adni_success
        and oasis_success
        and pretrain_success
        and brats_success
        and len(pretrain_inputs) == 565
        and hdbet_outputs == pretrain_inputs
        and len(brats) == 1350
        and all(source_checks.values())
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_PROVENANCE_LIMITATIONS" if valid else "FAIL",
        "normal_t1_pretraining": {
            "pretraining_npz_rows": len(pretrain),
            "pretraining_npz_all_success": pretrain_success,
            "dataset_counts": pretrain["dataset"].value_counts().to_dict(),
            "adni_hdbet_rows": len(adni),
            "adni_hdbet_all_success": adni_success,
            "oasis_hdbet_rows": len(oasis),
            "oasis_hdbet_all_success": oasis_success,
            "exact_path_matches": exact_matches,
            "missing_pretraining_inputs_in_hdbet_logs": sorted(
                pretrain_inputs.difference(hdbet_outputs)
            ),
            "extra_hdbet_outputs": sorted(hdbet_outputs.difference(pretrain_inputs)),
            "oasis_runner_script_available": False,
        },
        "brats_segmentation": {
            "npz_rows": len(brats),
            "npz_all_success": brats_success,
            "legacy_split_counts": brats["split"].value_counts().to_dict(),
            "local_hdbet_applied_during_npz_conversion": False,
            "source_challenge_preprocessing_inherited": True,
            "affine_preserved_in_npz": False,
        },
        "source_checks": source_checks,
        "evidence_files": [file_record(path) for path in required],
        "claim_boundary": (
            "It is accurate to state that all 565 normal T1 NPZ inputs were traced "
            "to successful HD-BET outputs. It is not accurate to state that this "
            "project reran HD-BET on BraTS during NPZ conversion; BraTS skull stripping "
            "was inherited from source challenge preprocessing. Exact OASIS HD-BET "
            "command parameters and original spatial affines are not recoverable from "
            "the supplied archive."
        ),
    }
    (output_dir / "data_provenance.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "data_provenance.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "matches": exact_matches}))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
