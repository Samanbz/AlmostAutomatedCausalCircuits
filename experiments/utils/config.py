"""Config loading, seeding, artifact paths, logging and reproducibility metadata.

Artifact layout (per experiment ID and seed):

    results/<exp_id>/seed_<S>/{config.json,metadata.json,metrics.json,grids.npz,<exp_id>_seed<S>.log}
    models/<exp_id>/seed_<S>/<exp_id>_epoch_XXXX.pt
    models/<exp_id>/seed_<S>/<exp_id>_final.pt

Plotting / evaluation scripts save their outputs into the seed's results
directory (``results/<exp_id>/seed_<S>/``) by default.

The experiment ID is the config file name (never its contents), so artifacts
stay addressable across config edits; a content hash of the
checkpoint-relevant config sections is stored in the metadata and a config
change triggers invalidation of the stale artifacts (interactively, or
non-interactively via ``--yes-invalidate``).
"""

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

from experiments.utils.data import resolve_dataset_paths
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


# Model-section key defaults, mirroring the ``.get(...)`` fallbacks in
# ``build_circuit`` / ``create_md_circuit``. Keys left at these values produce
# identical circuits, so they are stripped from the checkpoint hash.
_MODEL_DEFAULTS = {
    "fairness_temperature": 1.0,
}

# Sections that influence the trained model or the training run. Edits confined
# to the other sections (evaluation, checkpointing, device/output paths) must
# not invalidate existing checkpoints.
_CHECKPOINT_SECTIONS = ("experiment", "dataset", "model", "training")


def checkpoint_hash(cfg: Dict[str, Any]) -> str:
    """Hash of only the checkpoint-relevant parts of the config.

    Covers ``experiment.seed``, ``dataset``, ``model`` (with default-valued
    keys stripped) and ``training``; evaluation settings and behavior-neutral
    model keys are ignored.
    """
    relevant = {}
    for section in _CHECKPOINT_SECTIONS:
        if section not in cfg:
            continue
        if section == "experiment":
            relevant[section] = {"seed": cfg[section].get("seed")}
        elif section == "model":
            relevant[section] = {
                k: v for k, v in cfg[section].items() if _MODEL_DEFAULTS.get(k) != v
            }
        else:
            relevant[section] = cfg[section]
    canonical = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------


def run_prefix(exp_id: str, seed: int) -> str:
    """Checkpoint file prefix for one seed run of an experiment."""
    return f"{exp_id}_seed{seed}"


def seed_dir(output_dir: str, seed: int) -> str:
    """Per-seed results directory."""
    return os.path.join(output_dir, f"seed_{seed}")


def models_seed_dir(models_dir: str, exp_id: str, seed: int) -> str:
    """Per-seed models directory: ``models/<exp_id>/seed_<S>/``.

    Checkpoint file names inside it carry the experiment id only (the seed is
    already in the path); the pre-seed-dir layout
    (``models/<exp_id>/<exp_id>_seed<S>_*.pt``) is handled as a legacy
    fallback by the loaders, not produced anymore.
    """
    return os.path.join(models_dir, exp_id, f"seed_{seed}")


def read_stored_checkpoint_hash(results_seed_dir: str) -> Optional[str]:
    """Checkpoint-relevant hash recorded by the previous run of this seed, if any.

    ``None`` when no metadata exists yet (first run) or it predates
    checkpoint-aware hashing.
    """
    path = os.path.join(results_seed_dir, "metadata.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get("checkpoint_hash")
    except Exception:
        return None


def invalidate_artifacts(results_seed_dir: str, models_dir: str, prefix: str) -> None:
    """Delete every artifact produced by previous runs of this seed run.

    ``models_dir`` is the per-seed models directory and ``prefix`` the
    checkpoint file prefix inside it.
    """
    for path in glob.glob(os.path.join(models_dir, f"{prefix}_epoch_*.pt")):
        os.remove(path)
    final = os.path.join(models_dir, f"{prefix}_final.pt")
    if os.path.exists(final):
        os.remove(final)
    if os.path.isdir(results_seed_dir):
        shutil.rmtree(results_seed_dir)
    # Remove the (now possibly empty) per-seed models directory and its
    # experiment-level parent.
    for d in (models_dir, os.path.dirname(models_dir)):
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)


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


def setup_logging(results_seed_dir: str, name: str) -> logging.Logger:
    """Attach a per-seed file handler to the global project logger.

    The global ``mcc`` logger (``src.logger``) already prints to the console;
    all experiment modules log through child loggers that propagate to it, so
    this file handler captures everything.
    """
    os.makedirs(results_seed_dir, exist_ok=True)
    log_path = os.path.join(results_seed_dir, f"{name}.log")

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


def data_file_hashes(cfg: Dict[str, Any]) -> Dict[str, str]:
    """SHA-256 of the dataset CSVs the run reads (for reproducibility)."""
    ds = cfg["dataset"]
    paths = resolve_dataset_paths(ds["data_dir"], ds["dataset"])
    hashes = {}
    for kind in ("observational", "interventional"):
        path = paths[kind]
        if not os.path.exists(path):
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        hashes[os.path.basename(path)] = h.hexdigest()[:16]
    return hashes


def write_metadata(
    cfg: Dict[str, Any],
    results_seed_dir: str,
    seed: int,
    start_time: str,
    end_time: str,
    cfg_hash: str,
    device: str,
) -> None:
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
        "seed": seed,
        "config": cfg,
        "config_hash": cfg_hash,
        "checkpoint_hash": checkpoint_hash(cfg),
        "start_time": start_time,
        "end_time": end_time,
        "hostname": platform.node(),
        "device": device,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "git_commit": git_commit,
        "command": " ".join(sys.argv),
        "data_file_hashes": data_file_hashes(cfg),
    }
    out_path = os.path.join(results_seed_dir, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_config_copy(cfg: Dict[str, Any], results_seed_dir: str) -> None:
    """Copy the resolved config into the seed results directory."""
    out_path = os.path.join(results_seed_dir, "config.json")
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
