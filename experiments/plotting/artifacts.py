"""Load experiment artifacts needed by the plotting scripts."""

import json
import os
import re
from typing import Any, Dict, Tuple

import torch

from experiments.utils.config import get_device
from experiments.utils.data import load_data
from experiments.utils.identification import identify_estimands
from experiments.utils.queries import compile_queries
from experiments.utils.training import find_latest_checkpoint, load_circuit


def load_results_config(results_seed_dir: str) -> Dict[str, Any]:
    """Load the effective config copied into the results seed directory."""
    path = os.path.join(results_seed_dir, "config.json")
    with open(path) as f:
        return json.load(f)


def _run_prefix(cfg: Dict[str, Any], seed: int) -> str:
    """Return the artifact prefix used by ``run_experiment.py``."""
    return f"{cfg['experiment']['id']}_seed{seed}"


def _seed_from_dir(results_seed_dir: str) -> int:
    """Infer the seed from a results directory named ``.../seed_<S>``."""
    match = re.search(r"seed_(\d+)", os.path.basename(results_seed_dir))
    if match is None:
        raise ValueError(f"Could not infer seed from results dir: {results_seed_dir}")
    return int(match.group(1))


def load_trained_artifacts(
    results_seed_dir: str, device: torch.device = None
) -> Tuple[Dict[str, Any], Any, Dict[str, Any], Dict[str, Any], torch.device]:
    """Load config, trained circuit, data_info, compiled query circuits, and device.

    Returns:
        (cfg, ac, data_info, query_acs, device)
    """
    cfg = load_results_config(results_seed_dir)
    if device is None:
        device = get_device(cfg["experiment"].get("device"))

    data_info = load_data(cfg)

    seed = _seed_from_dir(results_seed_dir)
    models_dir = cfg["experiment"]["models_dir"]
    prefix = _run_prefix(cfg, seed)
    final_path = os.path.join(models_dir, f"{prefix}_final.pt")
    if os.path.exists(final_path):
        model_path = final_path
    else:
        latest = find_latest_checkpoint(models_dir, prefix)
        if latest is None:
            raise FileNotFoundError(
                f"No final model or checkpoint found for {prefix} in {models_dir}\n"
                "Train the experiment first with experiments.run_experiment."
            )
        _, model_path = latest

    ac = load_circuit(model_path, device)
    estimands = identify_estimands(cfg, data_info)
    query_acs = compile_queries(ac, estimands, data_info)
    return cfg, ac, data_info, query_acs, device
