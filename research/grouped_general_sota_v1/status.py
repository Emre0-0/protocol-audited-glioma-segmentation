from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ROOT / "research" / "grouped_general_sota_v1" / "config.json").read_text(
        encoding="utf-8"
    )
)
EXPERIMENT = ROOT / CONFIG["paths"]["experiment_dir"]


def gpu_status() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def training_snapshot() -> dict[str, object]:
    trainer = CONFIG["nnunet"]["candidate_trainer"]
    plans = CONFIG["nnunet"]["plans"]
    configuration = CONFIG["nnunet"]["configuration"]
    fold = CONFIG["protocol"]["fold"]
    dataset = f"Dataset{CONFIG['dataset']['id']:03d}_{CONFIG['dataset']['name']}"
    folder = (
        ROOT
        / CONFIG["paths"]["results_root"]
        / dataset
        / f"{trainer}__{plans}__{configuration}"
        / f"fold_{fold}"
    )
    logs = sorted(folder.glob("training_log_*.txt"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return {"folder": str(folder), "log": None}
    text = logs[-1].read_text(encoding="utf-8", errors="replace")
    epochs = re.findall(r"Epoch (\d+)", text)
    times = [float(value) for value in re.findall(r"Epoch time: ([0-9.]+) s", text)]
    return {
        "folder": str(folder),
        "log": str(logs[-1]),
        "latest_epoch_zero_based": int(epochs[-1]) if epochs else None,
        "completed_epoch_count": len(times),
        "latest_epoch_seconds": times[-1] if times else None,
        "mean_epoch_seconds_last_5": (
            sum(times[-5:]) / len(times[-5:]) if times else None
        ),
        "best_checkpoint": (folder / "checkpoint_best.pth").exists(),
        "final_checkpoint": (folder / "checkpoint_final.pth").exists(),
    }


def main() -> None:
    status_path = EXPERIMENT / "status.json"
    status = (
        json.loads(status_path.read_text(encoding="utf-8-sig"))
        if status_path.exists()
        else {}
    )
    comparison_path = EXPERIMENT / "comparison" / "comparison.json"
    comparison = (
        json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison_path.exists()
        else None
    )
    print(
        json.dumps(
            {
                "pipeline": status,
                "training": training_snapshot(),
                "gpu": gpu_status(),
                "comparison": comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
