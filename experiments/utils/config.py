"""Config loading, seeding, logging and reproducibility metadata."""

import glob
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.logger import logger as g_logger


REQUIRED_SECTIONS = ["experiment", "dataset", "model", "training", "checkpointing", "evaluation"]


def load_config(path: str) -> Dict[str, Any]:
    """Load and lightly validate the experiment config."""
    with open(path) as f:
        cfg = json.load(f)

    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"Config {path} is missing sections: {missing}")

    # The experiment ID is derived from the config file name (never from the
    # file contents), so artifacts stay addressable across config edits.
    cfg["experiment"]["id"] = os.path.splitext(os.path.basename(path))[0]

    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    """Compact content hash of the config (insensitive to JSON formatting)."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def read_stored_config_hash(output_dir: str, exp_id: str) -> Optional[str]:
    """Config hash recorded by the previous run of this experiment, if any."""
    path = os.path.join(output_dir, f"{exp_id}_metadata.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get("config_hash")
    except Exception:
        return None


def invalidate_artifacts(output_dir: str, models_dir: str, exp_id: str) -> None:
    """Delete every artifact produced by previous runs of this experiment ID."""
    for path in glob.glob(os.path.join(models_dir, f"{exp_id}_epoch_*.pt")):
        os.remove(path)
    final = os.path.join(models_dir, f"{exp_id}_final.pt")
    if os.path.exists(final):
        os.remove(final)

    if not os.path.isdir(output_dir):
        return
    # Only remove the whole directory when it is clearly dedicated to this
    # experiment ID; otherwise delete just the files carrying the ID.
    if os.path.basename(os.path.normpath(output_dir)) == exp_id:
        shutil.rmtree(output_dir)
    else:
        for path in glob.glob(os.path.join(output_dir, f"*{exp_id}*")):
            os.remove(path)


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply ``section.key=value`` command-line overrides in place."""
    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        # Best-effort JSON type coercion (numbers, booleans, null, strings).
        try:
            value = json.loads(value)
        except Exception:
            pass
        d[parts[-1]] = value
    return cfg


def set_seed(seed: int) -> None:
    """Make NumPy, PyTorch and Python random deterministic."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config_device: Optional[str]) -> torch.device:
    """Resolve device from config, defaulting to CUDA when available."""
    if config_device:
        return torch.device(config_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(output_dir: str, exp_id: str) -> logging.Logger:
    """Attach a per-experiment file handler to the global project logger.

    The global ``mcc`` logger (``src.logger``) already prints to the console;
    all experiment modules log through child loggers that propagate to it, so
    this file handler captures everything.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{exp_id}.log")

    # Replace any file handler left over from a previous run in this process.
    for h in list(g_logger.handlers):
        if isinstance(h, logging.FileHandler):
            g_logger.removeHandler(h)
            h.close()

    fh = logging.FileHandler(log_path)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    g_logger.addHandler(fh)

    return g_logger


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def write_metadata(cfg: Dict[str, Any], output_dir: str, start_time: str, cfg_hash: str) -> None:
    """Write a small JSON with everything needed to reproduce the run."""
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_repo_root(), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    metadata = {
        "experiment_id": cfg["experiment"]["id"],
        "config": cfg,
        "config_hash": cfg_hash,
        "start_time": start_time,
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "git_commit": git_commit,
        "command": " ".join(sys.argv),
    }
    out_path = os.path.join(output_dir, f"{cfg['experiment']['id']}_metadata.json")
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_config_copy(cfg: Dict[str, Any], output_dir: str) -> None:
    """Copy the resolved config into the output directory."""
    out_path = os.path.join(output_dir, f"{cfg['experiment']['id']}_config.json")
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
