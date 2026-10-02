from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "grouped_delta_roi_v1"
CONFIG = json.loads((ROOT / "research" / "grouped_delta_roi_v1" / "config.json").read_text())


def latest_training_line() -> str | None:
    nnunet = CONFIG["nnunet"]
    fold = CONFIG["protocol"]["fold"]
    logs = sorted(
        (ROOT / "experiments" / "nnunet_results").glob(
            f"Dataset50[34]_*/{nnunet['trainer']}__{nnunet['plans']}__"
            f"{nnunet['configuration']}/fold_{fold}/training_log_*.txt"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not logs:
        return None
    lines = [line.strip() for line in logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()]
    useful = [line for line in lines if line]
    return useful[-1] if useful else None


def gpu_status() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> None:
    status_path = EXPERIMENT / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8-sig")) if status_path.exists() else {}
    payload = {
        "pipeline": status,
        "latest_training_line": latest_training_line(),
        "gpu": gpu_status(),
        "comparison_ready": (EXPERIMENT / "official_comparison.json").exists(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
