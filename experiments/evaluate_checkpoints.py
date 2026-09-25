"""Evaluate every checkpoint of a trained seed run on held-out data.

For each model checkpoint of a results seed directory (the ``_final.pt`` model
plus every ``*_epoch_*.pt``), compute:

  * ``obs_test_nll`` — mean NLL of the *observational* held-out test rows
    under the base circuit;
  * ``do_nll`` — mean NLL of the *entire interventional* dataset under the
    compiled interventional circuit P(Y|do(X)).

Results are printed as a table and stored as ``checkpoint_nll.json`` /
``checkpoint_nll.csv`` in the seed results directory, next to ``metrics.json``.

Examples:
    python -m experiments.evaluate_checkpoints \
        --results-dir experiments/results/<exp_id>/seed_27
    python -m experiments.evaluate_checkpoints \
        --results-dir experiments/results/<exp_id> --seeds 0 1 2 3 4 --gpus 0,1,2
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

import torch

from experiments.plotting.artifacts import _run_prefix, _seed_from_dir, load_results_config
from experiments.run_experiment import parse_gpus
from experiments.utils.config import get_device, models_seed_dir
from experiments.utils.data import load_data
from experiments.utils.identification import identify_estimands
from experiments.utils.queries import compile_queries
from experiments.utils.training import compute_nll, load_circuit, resolve_eval_chunk_rows
from src.symbolic.identification import TractabilityError


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
    """Evaluate all checkpoints and write checkpoint_nll.{json,csv}.

    Runs without sufficient marginal determinism (e.g. an unconstrained
    circuit with empty ``md_sets``) cannot compile the interventional query —
    identification raises :class:`TractabilityError`.  In that case only the
    observational test NLL is computed and ``do_nll`` is None.
    """
    cfg = load_results_config(results_dir)
    device = get_device(cfg["experiment"].get("device"))
    data_info = load_data(cfg)
    try:
        estimands = identify_estimands(cfg, data_info)
    except TractabilityError as exc:
        print(f"Interventional query not tractable ({exc}); obs-test NLL only.")
        estimands = None
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
        obs_test_nll = compute_nll(
            ac,
            torch.tensor(obs_test, dtype=torch.float32, device=device),
            resolve_eval_chunk_rows(cfg, device, dense=True),
        )
        if estimands is not None:
            query_acs = compile_queries(ac, estimands, data_info)
            do_nll = round(
                compute_nll(
                    query_acs["q_do_ac"],
                    torch.tensor(do_full, dtype=torch.float32, device=device),
                    chunk_rows,
                ),
                6,
            )
        else:
            do_nll = None
        rows.append(
            {
                "epoch": epoch,
                "obs_test_nll": round(obs_test_nll, 6),
                "do_nll": do_nll,
            }
        )
        print(f"epoch {epoch:4d} | obs test NLL {obs_test_nll:.4f} | do NLL {do_nll}")

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


def run_seed_workers(results_dir: str, seeds: list[int], gpus: list[int]) -> None:
    """Evaluate each seed in its own subprocess, pinned round-robin to ``gpus``.

    ``results_dir`` may be the experiment results directory (``.../<exp_id>``)
    or one of its ``seed_<S>`` children; seed directories are reconstructed as
    ``<base>/seed_<S>`` and missing ones are skipped with a warning.
    """
    base = results_dir
    try:
        _seed_from_dir(results_dir)
    except ValueError:
        pass
    else:
        base = os.path.dirname(os.path.normpath(results_dir))

    procs = []
    for i, seed in enumerate(seeds):
        seed_dir = os.path.join(base, f"seed_{seed}")
        if not os.path.isdir(seed_dir):
            print(f"Skipping seed {seed}: {seed_dir} does not exist")
            continue
        env = os.environ.copy()
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)])
        cmd = [
            sys.executable,
            "-m",
            "experiments.evaluate_checkpoints",
            "--results-dir",
            seed_dir,
        ]
        print(f"Launching seed {seed} eval (GPU {env.get('CUDA_VISIBLE_DEVICES', 'cpu')})")
        procs.append((seed, subprocess.Popen(cmd, env=env)))

    failures = []
    for seed, proc in procs:
        code = proc.wait()
        if code != 0:
            failures.append((seed, code))
    if failures:
        raise SystemExit(f"Seed evals failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Observational-test and interventional NLL for every checkpoint."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to the experiment results directory (experiments/results/<id>) or a seed "
        "directory (.../seed_27).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Multiple seeds: evaluate each in its own subprocess (cluster mode).",
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpus,
        default=None,
        help="GPUs for a multi-seed run, e.g. --gpus 0,1,2 (default: all visible).",
    )
    args = parser.parse_args()
    if args.seeds:
        gpus = args.gpus if args.gpus is not None else list(range(torch.cuda.device_count()))
        run_seed_workers(args.results_dir, args.seeds, gpus)
        return
    evaluate_checkpoints(args.results_dir)


if __name__ == "__main__":
    main()
