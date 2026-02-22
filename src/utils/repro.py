import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42):
    """Fix random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_run_metadata(run_dir: str, cfg: dict):
    """Save config YAML and git hash into the run folder."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Save git commit hash (best-effort)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"
    with open(run_dir / "git_hash.txt", "w") as f:
        f.write(commit + "\n")


def get_next_run_dir(base_dir: str) -> str:
    """Return the next run_NNN directory under base_dir."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("run_*"))
    if existing:
        last_num = int(existing[-1].name.split("_")[1])
        return str(base / f"run_{last_num + 1:03d}")
    return str(base / "run_001")
