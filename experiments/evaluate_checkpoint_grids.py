"""X-averaged KL of P(Y|do(X)) and P(Y|X) vs. analytical GT for every checkpoint.

Where ``evaluate_grids`` scores only the final model, this scores every
checkpoint (``*_epoch_*.pt`` + ``_final.pt``) so the accuracy *improvement
over training* can be plotted.  Per checkpoint and seed, writes
``checkpoint_kl.csv`` into the seed results directory:

    epoch, kl_do_gt_learned, kl_do_learned_gt, kl_obs_gt_learned,
    kl_obs_learned_gt, hellinger_do

The X-average uses the GT marginal of X (same quadrature weights as
``evaluate_grids``).  Seeds whose circuit does not support the interventional
query (e.g. unconstrained circuits) are skipped.

Examples:
    python -m experiments.evaluate_checkpoint_grids \
        --results-dir experiments/results/<exp_id>/seed_27
    python -m experiments.evaluate_checkpoint_grids \
        --results-dir experiments/results/<exp_id> --seeds 0 1 2 3 4 --gpus 0,1
"""

import argparse
import os

import numpy as np
import torch

from experiments.evaluate_checkpoints import list_checkpoints
from experiments.evaluate_grids import _hellinger, _kl, _x_weights
from experiments.plot_heatmaps import analytical_surfaces, load_ground_truth
from experiments.plotting.artifacts import _run_prefix, _seed_from_dir, load_results_config
from experiments.run_experiment import parse_gpus
from experiments.utils.config import get_device, models_seed_dir
from experiments.utils.data import discrete_categories, load_data
from experiments.utils.evaluation import _eval_on_grid, _grids_from_data
from experiments.utils.identification import identify_estimands
from experiments.utils.queries import compile_queries
from experiments.utils.training import load_circuit
from src.symbolic.identification import TractabilityError


def evaluate_checkpoint_grids(results_seed_dir: str) -> str:
    """Score every checkpoint of one seed run on the shared grids; idempotent."""
    seed = _seed_from_dir(results_seed_dir)
    out_path = os.path.join(results_seed_dir, "checkpoint_kl.csv")
    if os.path.exists(out_path):
        print(f"{out_path} already exists — skipping")
        return out_path

    cfg = load_results_config(results_seed_dir)
    device = get_device(cfg["experiment"].get("device"))
    data_info = load_data(cfg)
    estimands = identify_estimands(cfg, data_info)

    gt = load_ground_truth(data_info)
    if gt is None:
        raise FileNotFoundError("No SCM pickle found next to the dataset; KL needs it.")

    id_to_var = {v: k for k, v in data_info["var_to_id"].items()}
    x_name, y_name = id_to_var[data_info["x_id"]], id_to_var[data_info["y_id"]]
    x_cats = discrete_categories(cfg, data_info, x_name)
    y_cats = discrete_categories(cfg, data_info, y_name)
    x_discrete = x_cats is not None
    y_discrete = y_cats is not None

    eval_cfg = cfg.get("evaluation", {})
    x_grid, y_grid, _ = _grids_from_data(data_info, eval_cfg)
    if x_discrete:
        x_grid = np.asarray(x_cats, dtype=np.float64)
    if y_discrete:
        y_grid = np.asarray(y_cats, dtype=np.float64)
    dy = 1.0 if y_discrete else float(np.diff(y_grid).mean())

    gt_obs, gt_do = analytical_surfaces(gt, x_grid, y_grid, x_name, y_name, y_discrete=y_discrete)
    x_weights, _ = _x_weights(gt, x_grid, x_name, x_discrete)

    exp_id = cfg["experiment"]["id"]
    models_dir = models_seed_dir(cfg["experiment"]["models_dir"], exp_id, seed)
    prefix = exp_id
    if not os.path.isdir(models_dir):
        models_dir = cfg["experiment"]["models_dir"]
        prefix = _run_prefix(cfg, seed)

    rows = []
    for epoch, path in list_checkpoints(models_dir, prefix):
        ac = load_circuit(path, device)
        query_acs = compile_queries(ac, estimands, data_info)
        log_p_do, log_p_obs = _eval_on_grid(query_acs, data_info, x_grid, y_grid, device, cfg)
        learned_obs = np.exp(np.clip(log_p_obs.astype(np.float64), -745, 700))
        learned_do = np.exp(np.clip(log_p_do.astype(np.float64), -745, 700))

        def _avg(per_x):
            return float(np.nansum(per_x * x_weights))

        rows.append(
            {
                "epoch": epoch,
                "kl_do_gt_learned": _avg(_kl(gt_do, learned_do, dy, y_discrete)),
                "kl_do_learned_gt": _avg(_kl(learned_do, gt_do, dy, y_discrete)),
                "kl_obs_gt_learned": _avg(_kl(gt_obs, learned_obs, dy, y_discrete)),
                "kl_obs_learned_gt": _avg(_kl(learned_obs, gt_obs, dy, y_discrete)),
                "hellinger_do": float(
                    np.nansum(_hellinger(gt_do, learned_do, dy, y_discrete) * x_weights)
                ),
            }
        )
        r = rows[-1]
        print(
            f"epoch {epoch:4d} | do KL(g||l) {r['kl_do_gt_learned']:.4f} | "
            f"obs KL(g||l) {r['kl_obs_gt_learned']:.4f}"
        )

    import csv

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path}")
    return out_path


def run_seed_workers(results_dir: str, seeds: list[int], gpus: list[int]) -> None:
    """Evaluate each seed in its own subprocess, pinned round-robin to ``gpus``."""
    import subprocess
    import sys

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
            "experiments.evaluate_checkpoint_grids",
            "--results-dir",
            seed_dir,
        ]
        print(
            f"Launching seed {seed} checkpoint-KL eval (GPU {env.get('CUDA_VISIBLE_DEVICES', 'cpu')})"
        )
        procs.append((seed, subprocess.Popen(cmd, env=env)))

    failures = []
    for seed, proc in procs:
        code = proc.wait()
        if code != 0:
            failures.append((seed, code))
    if failures:
        raise SystemExit(f"Seed checkpoint-KL evals failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-checkpoint X-averaged KL of the query circuits vs analytical GT."
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
    try:
        evaluate_checkpoint_grids(args.results_dir)
    except TractabilityError as exc:
        raise SystemExit(f"Skipping (interventional query not tractable): {exc}") from None


if __name__ == "__main__":
    main()
