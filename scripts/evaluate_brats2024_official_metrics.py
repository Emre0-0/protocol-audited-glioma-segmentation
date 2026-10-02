from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGIONS = ("WT", "TC", "NETC", "SNFH", "ET", "RC")


def vendor_commit(vendor_dir: Path) -> str | None:
    manifest_path = vendor_dir / "UPSTREAM.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = vendor_dir.resolve()
        for name, expected in manifest["files_sha256"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Unsafe upstream path: {name}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Upstream metric source differs: {name}")
        return str(manifest["commit"])
    if not (vendor_dir / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(vendor_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def evaluate_case(payload: dict[str, str]) -> dict[str, Any]:
    case_id = payload["case_id"]
    cache_path = Path(payload["cache_path"])
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    vendor_dir = Path(payload["vendor_dir"])
    work_root = Path(payload["work_root"])
    work_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # The official repository currently uses np.Inf, removed in NumPy 2.0.
    # This compatibility alias leaves the metric algorithm unchanged.
    np.Inf = np.inf  # type: ignore[attr-defined]
    sys.path.insert(0, str(vendor_dir))
    from metrics_GLI import get_LesionWiseResults

    old_cwd = Path.cwd()
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{case_id}_", dir=work_root))
    try:
        os.chdir(temp_dir)
        results_df, lesion_df = get_LesionWiseResults(
            payload["prediction"],
            payload["ground_truth"],
            "BraTS-GLI",
        )
        row: dict[str, Any] = {
            "case_id": case_id,
            "elapsed_seconds": time.perf_counter() - start,
        }
        for record in results_df.to_dict(orient="records"):
            region = str(record["Labels"])
            for key, value in record.items():
                if key == "Labels":
                    continue
                row[f"{region}_{key}"] = (
                    None if pd.isna(value) else float(value)
                )
        lesion_counts = lesion_df.groupby("Label", dropna=False).size().to_dict()
        for region in REGIONS:
            row[f"{region}_lesion_detail_rows"] = int(lesion_counts.get(region, 0))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        return row
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)


def evaluate_case_subprocess(payload: dict[str, str]) -> dict[str, Any]:
    cache_path = Path(payload["cache_path"])
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-case",
        payload["case_id"],
        "--prediction-dir",
        str(Path(payload["prediction"]).parent),
        "--ground-truth-dir",
        str(Path(payload["ground_truth"]).parent),
        "--vendor-dir",
        payload["vendor_dir"],
        "--cache-path",
        payload["cache_path"],
        "--work-root",
        payload["work_root"],
    ]
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=20 * 60,
        env=environment,
    )
    if result.returncode != 0:
        detail = (result.stdout + "\n" + result.stderr)[-6000:]
        raise RuntimeError(
            f"Official metric subprocess failed for {payload['case_id']} "
            f"with exit code {result.returncode}:\n{detail}"
        )
    if not cache_path.exists():
        raise RuntimeError(
            f"Official metric subprocess did not create cache for {payload['case_id']}"
        )
    return json.loads(cache_path.read_text(encoding="utf-8"))


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for region in REGIONS:
        item: dict[str, Any] = {
            "region": region,
            "case_count": int(len(rows)),
        }
        for metric in (
            "LesionWise_Score_Dice",
            "LesionWise_Score_HD95",
            "Legacy_Dice",
            "Legacy_HD95",
            "Num_TP",
            "Num_FP",
            "Num_FN",
            "Sensitivity",
            "Specificity",
        ):
            column = f"{region}_{metric}"
            if column not in rows:
                continue
            values = pd.to_numeric(rows[column], errors="coerce")
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_median"] = float(values.median())
            item[f"{metric}_sum"] = float(values.sum())
            item[f"{metric}_valid_cases"] = int(values.notna().sum())
        summary_rows.append(item)
    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate NIfTI predictions with the official BraTS 2024 GLI metric implementation."
    )
    parser.add_argument(
        "--prediction-dir",
        default="experiments/nnunet_predictions/fixed_val_5fold_best",
    )
    parser.add_argument(
        "--ground-truth-dir",
        default="processed/nnunet_raw/Dataset501_BraTS2024GliomaNPZ/labelsTr",
    )
    parser.add_argument(
        "--vendor-dir",
        default="research/vendor/BraTS-2024-Metrics",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/nnunet_ensemble_5seed_official_brats2024_metrics",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--single-case", default="")
    parser.add_argument("--cache-path", default="")
    parser.add_argument("--work-root", default="")
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir).resolve()
    ground_truth_dir = Path(args.ground_truth_dir).resolve()
    vendor_dir = Path(args.vendor_dir).resolve()
    metric_commit = vendor_commit(vendor_dir)
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "case_cache"
    work_root = output_dir / "tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    if args.single_case:
        case_id = args.single_case
        prediction = prediction_dir / f"{case_id}.nii.gz"
        ground_truth = ground_truth_dir / f"{case_id}.nii.gz"
        if not prediction.exists() or not ground_truth.exists():
            raise FileNotFoundError(
                f"Missing single-case input: prediction={prediction.exists()} "
                f"ground_truth={ground_truth.exists()} case={case_id}"
            )
        payload = {
            "case_id": case_id,
            "prediction": prediction.resolve().as_posix(),
            "ground_truth": ground_truth.resolve().as_posix(),
            "vendor_dir": str(vendor_dir),
            "cache_path": args.cache_path or str(cache_dir / f"{case_id}.json"),
            "work_root": args.work_root or str(work_root),
        }
        row = evaluate_case(payload)
        print(json.dumps({"case_id": case_id, "status": "complete", "metrics": row}))
        return

    predictions = sorted(prediction_dir.glob("*.nii.gz"))
    if args.limit > 0:
        predictions = predictions[: args.limit]
    if not predictions:
        raise FileNotFoundError(f"No NIfTI predictions found in {prediction_dir}")

    payloads: list[dict[str, str]] = []
    missing_ground_truth: list[str] = []
    for prediction in predictions:
        case_id = prediction.name.removesuffix(".nii.gz")
        ground_truth = ground_truth_dir / prediction.name
        if not ground_truth.exists():
            missing_ground_truth.append(case_id)
            continue
        payloads.append(
            {
                "case_id": case_id,
                "prediction": prediction.resolve().as_posix(),
                "ground_truth": ground_truth.resolve().as_posix(),
                "vendor_dir": str(vendor_dir),
                "cache_path": str(cache_dir / f"{case_id}.json"),
                "work_root": str(work_root),
            }
        )
    if missing_ground_truth:
        raise FileNotFoundError(
            f"Missing ground truth for {len(missing_ground_truth)} cases: {missing_ground_truth[:10]}"
        )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(evaluate_case_subprocess, payload): payload
            for payload in payloads
        }
        completed = 0
        for future in as_completed(futures):
            payload = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({"case_id": payload["case_id"], "error": repr(exc)})
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(payloads):
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 1e-8)
                eta = (len(payloads) - completed) / max(rate, 1e-8)
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(payloads),
                            "failures": len(failures),
                            "elapsed_minutes": round(elapsed / 60.0, 2),
                            "eta_minutes": round(eta / 60.0, 2),
                        }
                    ),
                    flush=True,
                )

    shutil.rmtree(work_root, ignore_errors=True)
    rows_df = pd.DataFrame(rows).sort_values("case_id") if rows else pd.DataFrame()
    rows_df.to_csv(output_dir / "case_metrics_official.csv", index=False)
    summary_df = summarize(rows_df) if not rows_df.empty else pd.DataFrame()
    summary_df.to_csv(output_dir / "summary_official.csv", index=False)
    (output_dir / "failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    metadata = {
        "metric_repository": "https://github.com/rachitsaluja/BraTS-2024-Metrics",
        "metric_repository_commit": metric_commit,
        "metric_script": "metrics_GLI.py",
        "prediction_dir": str(prediction_dir),
        "ground_truth_dir": str(ground_truth_dir),
        "requested_cases": len(payloads),
        "successful_cases": len(rows_df),
        "failed_cases": len(failures),
        "workers": args.workers,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        "numpy_2_compatibility_alias": "np.Inf = np.inf",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
