"""Evaluate every checkpoint of a trained seed run on held-out data.

For each model checkpoint of a results seed directory (the ``_final.pt`` model
plus every ``_epoch_XXXX.pt``), compute:

  * ``obs_test_nll`` — mean NLL of the *observational* held-out test rows
    under the base circuit;
  * ``do_nll`` — mean NLL of the *entire interventional* dataset under the
    compiled interventional circuit P(Y|do(X)).

Results are printed as a table and stored as ``checkpoint_nll.json`` /
``checkpoint_nll.csv`` in the seed results directory, next to ``metrics.json``.

Example:
    python -m experiments.evaluate_checkpoints \
        --results-dir experiments/results/config_backdoor_cont_Z1_100K/seed_27
"""

import argparse
import csv
import glob
import json
import os
import re

import torch

from experiments.plotting.artifacts import _run_prefix, _seed_from_dir, load_results_config
from experiments.utils.config import get_device, models_seed_dir
from experiments.utils.data import load_data
from experiments.utils.identification import identify_estimands
from experiments.utils.queries import compile_queries
from experiments.utils.training import compute_nll, load_circuit, resolve_eval_chunk_rows


def list_checkpoints(models_dir: str, prefix: str):
    """All ``<prefix>_epoch_XXXX.pt`` checkpoints plus ``<prefix>_final.pt``.

    Returns ``[(epoch_or_label, path)]`` sorted by epoch, with the final model
    (if present) last.
    """
    entries = []
    for path in glob.glob(os.path.join(models_dir, f"{prefix}_epoch_*.pt")):
        m = re.search(r"_epoch_(\d+)\.pt$", os.path.basename(path))
        if m is not None:
            entries.append((int(m.group(1)), path))
    entries.sort()
    final_path = os.path.join(models_dir, f"{prefix}_final.pt")
    if os.path.exists(final_path):
        entries.append((entries[-1][0] + 1 if entries else 0, final_path))
    if not entries:
        raise FileNotFoundError(f"No checkpoints for {prefix} in {models_dir}")
    return entries


def evaluate_checkpoints(results_dir: str) -> str:
    """Evaluate all checkpoints and write checkpoint_nll.{json,csv}."""
    cfg = load_results_config(results_dir)
    device = get_device(cfg["experiment"].get("device"))
    data_info = load_data(cfg)
    estimands = identify_estimands(cfg, data_info)
    chunk_rows = resolve_eval_chunk_rows(cfg, device)

    seed = _seed_from_dir(results_dir)
    exp_id = cfg["experiment"]["id"]
    models_dir = models_seed_dir(cfg["experiment"]["models_dir"], exp_id, seed)
    prefix = exp_id
    if not os.path.isdir(models_dir):
        # Legacy flat layout: models/<exp_id>/<exp_id>_seed<S>_*.pt
        models_dir = cfg["experiment"]["models_dir"]
        prefix = _run_prefix(cfg, seed)
    entries = list_checkpoints(models_dir, prefix)

    obs_test = data_info["df_test"].values
    do_full = data_info["df_do_full"].values

    rows = []
    for epoch, path in entries:
        ac = load_circuit(path, device)
        query_acs = compile_queries(ac, estimands, data_info)
        obs_test_nll = compute_nll(
            ac, torch.tensor(obs_test, dtype=torch.float32, device=device), chunk_rows
        )
        do_nll = compute_nll(
            query_acs["q_do_ac"],
            torch.tensor(do_full, dtype=torch.float32, device=device),
            chunk_rows,
        )
        rows.append(
            {
                "epoch": epoch,
                "obs_test_nll": round(obs_test_nll, 6),
                "do_nll": round(do_nll, 6),
            }
        )
        print(f"epoch {epoch:4d} | obs test NLL {obs_test_nll:.4f} | do NLL {do_nll:.4f}")

    json_path = os.path.join(results_dir, "checkpoint_nll.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "experiment_id": cfg["experiment"]["id"],
                "seed": seed,
                "n_obs_test": len(obs_test),
                "n_interventional": len(do_full),
                "checkpoints": rows,
            },
            f,
            indent=2,
        )
    csv_path = os.path.join(results_dir, "checkpoint_nll.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "obs_test_nll", "do_nll"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {json_path} and {csv_path}")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Observational-test and interventional NLL for every checkpoint."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to the experiment results seed directory (e.g. experiments/results/<id>/seed_27).",
    )
    args = parser.parse_args()
    evaluate_checkpoints(args.results_dir)


if __name__ == "__main__":
    main()
