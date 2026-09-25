"""Grid-based accuracy metrics of the learned query circuits against the SCM ground truth.

For a trained seed run, evaluate the learned ``P(Y|X)`` and ``P(Y|do(X))`` on
the shared (X, Y) grid (same grids and discrete-variable handling as
``plot_heatmaps``), evaluate the exact analytical ground truth from the pickled
SCM (``scm.ground_truth()``), and compute per-X averaged:

  * MAE  — mean absolute error between the learned and GT surfaces;
  * KL   — both directions, ``KL(learned||GT)`` and ``KL(GT||learned)``, summed
    over Y (density x dy for continuous Y, plain sums for discrete Y);
  * Hellinger distance — in [0, 1], per X row and averaged over the GT
    marginal of X (``gt.marginal_density``/``marginal_prob``); the per-X
    arrays and X weights are stored for degradation-vs-density plots.

Stored per seed run (``results/<exp_id>/seed_<S>``):

  * ``grid_metrics.json`` — the metrics above plus grid/discrete metadata;
  * ``grids.npz`` — ``x_grid``, ``y_grid``, ``slice_contexts``,
    ``log_p_obs``, ``log_p_do`` (learned) and ``p_obs_gt``, ``p_do_gt``
    (analytical), so plotting can be redone offline from the artifacts.

Examples:
    python -m experiments.evaluate_grids \
        --results-dir experiments/results/<exp_id>/seed_27
    python -m experiments.evaluate_grids \
        --results-dir experiments/results/<exp_id> --seeds 0 1 2 3 4 --gpus 0,1 --plots
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

from experiments.plot_heatmaps import analytical_surfaces, load_ground_truth
from experiments.plotting.artifacts import _seed_from_dir, load_trained_artifacts
from experiments.run_experiment import parse_gpus
from experiments.utils.data import discrete_categories
from experiments.utils.evaluation import _eval_on_grid, _grids_from_data


def _kl(p, q, dy, discrete):
    """KL(p||q) summed over the Y axis (last axis), one value per X row.

    Terms with p == 0 contribute 0; rows where q == 0 while p > 0 are
    undefined (returned as NaN so they are excluded from the X-average).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p / np.where(q > 0, q, 1.0)), 0.0)
    rows = terms.sum(axis=-1) * (1.0 if discrete else dy)
    bad = (p > 0) & (q <= 0)
    return np.where(bad.any(axis=-1), np.nan, rows)


def _hellinger(p, q, dy, discrete):
    """Hellinger distance H(p, q) per X row, in [0, 1].

    H^2 = 1/2 * sum_y (sqrt(p) - sqrt(q))^2 dy  (plain sum for discrete Y).
    """
    sq = (np.sqrt(np.clip(p, 0, None)) - np.sqrt(np.clip(q, 0, None))) ** 2
    return np.sqrt(0.5 * sq.sum(axis=-1) * (1.0 if discrete else dy))


def _x_weights(gt, x_grid, x_name, x_discrete):
    """Ground-truth marginal density/mass of X on the grid, normalized.

    Returns (weights, description): quadrature weights that sum to 1 over the
    grid, so that ``sum(per_x * weights)`` approximates the integral of the
    per-X quantity against the GT marginal of X (exact sum for discrete X).
    """
    if x_discrete:
        w = np.array([gt.marginal_prob({x_name: float(v)}) for v in x_grid])
        return w / w.sum(), "gt_marginal_prob"
    pdf = np.asarray(gt.marginal_density([x_name]).pdf(x_grid), dtype=np.float64)
    w = pdf * np.gradient(x_grid)
    return w / w.sum(), "gt_marginal_density"


def compute_grid_metrics(results_seed_dir: str, make_plots: bool = False) -> str:
    """Compute learned-vs-GT grid metrics for one seed run; idempotent.

    Metrics and figures are independent artifacts: a rerun only recomputes
    whichever of ``grid_metrics.json`` / heatmap / slices is missing, so a
    job that crashed mid-way (e.g. CUDA OOM in the plot stage) can simply be
    retried.
    """
    seed = _seed_from_dir(results_seed_dir)
    metrics_path = os.path.join(results_seed_dir, "grid_metrics.json")

    if not os.path.exists(metrics_path):
        cfg, ac, data_info, query_acs, device = load_trained_artifacts(results_seed_dir)
        gt = load_ground_truth(data_info)
        if gt is None:
            raise FileNotFoundError(
                "No SCM pickle found next to the dataset; grid metrics need it."
            )

        id_to_var = {v: k for k, v in data_info["var_to_id"].items()}
        x_name, y_name = id_to_var[data_info["x_id"]], id_to_var[data_info["y_id"]]
        x_cats = discrete_categories(cfg, data_info, x_name)
        y_cats = discrete_categories(cfg, data_info, y_name)
        x_discrete = x_cats is not None
        y_discrete = y_cats is not None

        eval_cfg = cfg.get("evaluation", {})
        x_grid, y_grid, slice_contexts = _grids_from_data(data_info, eval_cfg)
        if x_discrete:
            x_grid = np.asarray(x_cats, dtype=np.float64)
        if y_discrete:
            y_grid = np.asarray(y_cats, dtype=np.float64)
        log_p_do, log_p_obs = _eval_on_grid(query_acs, data_info, x_grid, y_grid, device, cfg)
        # float64 before exp: float32 underflows to 0 below log ~ -87, which
        # both wrecks KL(GT||learned) (zero-density rows) and corrupts MAE at
        # sharp densities (N=128 reaches log-density ~ -120, i.e. ~1e-52).
        learned_obs = np.exp(np.clip(log_p_obs.astype(np.float64), -745, 700))
        learned_do = np.exp(np.clip(log_p_do.astype(np.float64), -745, 700))

        gt_obs, gt_do = analytical_surfaces(
            gt, x_grid, y_grid, x_name, y_name, y_discrete=y_discrete
        )
        dy = 1.0 if y_discrete else float(np.diff(y_grid).mean())

        hell_obs = _hellinger(gt_obs, learned_obs, dy, y_discrete)
        hell_do = _hellinger(gt_do, learned_do, dy, y_discrete)
        x_weights, x_weight_source = _x_weights(gt, x_grid, x_name, x_discrete)

        kl_obs_lg = _kl(learned_obs, gt_obs, dy, y_discrete)
        kl_obs_gl = _kl(gt_obs, learned_obs, dy, y_discrete)
        kl_do_lg = _kl(learned_do, gt_do, dy, y_discrete)
        kl_do_gl = _kl(gt_do, learned_do, dy, y_discrete)

        # All per-X-averaged scalars use the GT marginal of X (quadrature
        # weights below); the per-X arrays stay available unweighted.
        def _wavg(per_x):
            return float(np.nansum(per_x * x_weights))

        metrics = {
            "experiment_id": cfg["experiment"]["id"],
            "seed": seed,
            "n_x": len(x_grid),
            "n_y": len(y_grid),
            "x_discrete": x_discrete,
            "y_discrete": y_discrete,
            "x_grid": [float(v) for v in x_grid],
            "mae_obs": _wavg(np.abs(learned_obs - gt_obs).mean(axis=-1)),
            "mae_do": _wavg(np.abs(learned_do - gt_do).mean(axis=-1)),
            "kl_obs_learned_gt": _wavg(kl_obs_lg),
            "kl_obs_gt_learned": _wavg(kl_obs_gl),
            "kl_do_learned_gt": _wavg(kl_do_lg),
            "kl_do_gt_learned": _wavg(kl_do_gl),
            # per-X arrays (for degradation profiles over X)
            "kl_obs_learned_gt_per_x": kl_obs_lg.tolist(),
            "kl_obs_gt_learned_per_x": kl_obs_gl.tolist(),
            "kl_do_learned_gt_per_x": kl_do_lg.tolist(),
            "kl_do_gt_learned_per_x": kl_do_gl.tolist(),
            # Hellinger distance, per X row and averaged over the GT marginal
            # of X (weights below; per-X arrays kept for degradation plots).
            "hellinger_obs": float(np.sum(hell_obs * x_weights)),
            "hellinger_do": float(np.sum(hell_do * x_weights)),
            "hellinger_obs_per_x": hell_obs.tolist(),
            "hellinger_do_per_x": hell_do.tolist(),
            "x_weights": x_weights.tolist(),
            "x_weight_source": x_weight_source,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        np.savez_compressed(
            os.path.join(results_seed_dir, "grids.npz"),
            x_grid=x_grid,
            y_grid=y_grid,
            slice_contexts=slice_contexts,
            log_p_obs=log_p_obs,
            log_p_do=log_p_do,
            p_obs_gt=gt_obs,
            p_do_gt=gt_do,
        )
        print(f"Saved grid metrics to {metrics_path}")

    if make_plots:
        from experiments.plotting.artifacts import load_results_config

        exp_id = load_results_config(results_seed_dir)["experiment"]["id"]
        figures = [
            ("experiments.plot_heatmaps", f"{exp_id}_seed{seed}_heatmaps.png"),
            ("experiments.plot_x_support_slices", f"{exp_id}_seed{seed}_x_support_slices.png"),
        ]
        for module, filename in figures:
            if os.path.exists(os.path.join(results_seed_dir, filename)):
                continue
            subprocess.run(
                [sys.executable, "-m", module, "--results-dir", results_seed_dir],
                check=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
    return metrics_path


def run_seed_workers(results_dir: str, seeds: list[int], gpus: list[int], plots: bool) -> None:
    """Evaluate each seed in its own subprocess, pinned round-robin to ``gpus``."""
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
            "experiments.evaluate_grids",
            "--results-dir",
            seed_dir,
        ]
        if plots:
            cmd.append("--plots")
        print(f"Launching seed {seed} grid eval (GPU {env.get('CUDA_VISIBLE_DEVICES', 'cpu')})")
        procs.append((seed, subprocess.Popen(cmd, env=env)))

    failures = []
    for seed, proc in procs:
        code = proc.wait()
        if code != 0:
            failures.append((seed, code))
    if failures:
        raise SystemExit(f"Seed grid evals failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learned-vs-GT MAE/KL of P(Y|X) and P(Y|do(X)) on shared grids."
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
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Also render the heatmap and X-support slice figures into the seed directory.",
    )
    args = parser.parse_args()
    if args.seeds:
        gpus = args.gpus if args.gpus is not None else list(range(torch.cuda.device_count()))
        run_seed_workers(args.results_dir, args.seeds, gpus, args.plots)
        return
    compute_grid_metrics(args.results_dir, make_plots=args.plots)


if __name__ == "__main__":
    main()
