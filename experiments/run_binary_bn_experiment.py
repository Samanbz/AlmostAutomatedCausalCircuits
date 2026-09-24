"""Run a reproducible MD-circuit experiment on the real-world binary Bayesian-network
datasets of the MDNet paper (Wang & Kwiatkowska, AISTATS 2023; asia, child, win95pts,
andes; 1000-row binarized BN samples in data/*.pkl).

A comparison counterpart to MDNet on the same data: per seed run it

    1. loads the dataset pickle,
    2. builds the MD vtree with the repo's construction (``construct_optimal_md_vtree``,
       MD set = X∪Z) and the circuit with ``create_md_circuit`` using categorical
       input distributions,
    3. trains with batch EM (``experiments.utils.training.train_model``),
    4. stores config/metadata/metrics (observational NLL, backdoor-query MAE and
       Bernoulli log-loss) and the final model.

Artifacts (same layout as ``run_experiment``)::

    results/<exp_id>/seed_<S>/{config.json,metadata.json,metrics.json,<exp_id>_seed<S>.log}
    models/<exp_id>/<exp_id>_seed<S>_final.pt

Multi-seed runs for the cluster: pass ``--seeds 27 28 29`` to run each seed in its own
subprocess, pinned round-robin to ``--gpus`` via ``CUDA_VISIBLE_DEVICES``.

Example
-------
    python -m experiments.run_binary_bn_experiment --config configs/binary_bn/asia.json
    python -m experiments.run_binary_bn_experiment --config configs/binary_bn/asia.json \
        --seeds 27 28 29 --gpus 0 1 2
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from experiments.run_experiment import _handle_stale_config
from experiments.utils.config import (
    apply_overrides,
    checkpoint_hash,
    config_hash,
    get_device,
    load_config,
    models_seed_dir,
    run_prefix,
    save_config_copy,
    seed_dir,
    set_seed,
    setup_logging,
    write_metadata,
)
from experiments.utils.evaluation import write_metrics
from experiments.utils.training import (
    compute_nll,
    find_latest_checkpoint,
    load_circuit,
    resolve_eval_chunk_rows,
    save_final_model,
    train_model,
)
from src.construction.circuit_builder import create_md_circuit
from src.construction.learned_vtree import construct_optimal_md_vtree
from src.logger import logger as g_logger
from src.symbolic.arithmetic.circuit import eval_circuit
from src.symbolic.arithmetic.nodes.leaf_layer import CategoricalDistribution
from src.symbolic.arithmetic.query import compile_query
from src.symbolic.id_ast import make_cond, make_marg, make_p, make_prod


logger = g_logger.getChild("binary_bn_experiment")


def bernoulli_log_loss(p: float, target: float, eps: float = 1e-10) -> float:
    """Proper score of estimate p for a Bernoulli(target) outcome: scale-aware (errors
    at small p cost proportionally more than the same absolute error near p=1/2)."""
    p = min(max(p, eps), 1.0 - eps)
    return float(-(target * np.log(p) + (1.0 - target) * np.log(1.0 - p)))


def naive_adjustment(df, ds, var_to_idx):
    """Counting backdoor adjustment on the data (MDNet's run_naive_benchmark formula)."""
    from collections import defaultdict

    x_id = var_to_idx[ds["x"][0]]
    y_id = var_to_idx[ds["y"][0]]
    z_ids = [var_to_idx[v] for v in ds["z"]]
    x_val, y_val = ds["x_val"], ds["y_val"]
    target_y, other_y, z_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    arr = df.to_numpy()
    for row in arr:
        z_set = frozenset((z, row[z]) for z in z_ids)
        z_counts[z_set] += 1
        if row[x_id] == x_val:
            other_y[z_set] += 1
            if row[y_id] == y_val:
                target_y[z_set] += 1
    naive_do = sum(
        target_y[zs] / other_y[zs] * z_counts[zs] / len(arr) if other_y[zs] > 0 else 0
        for zs in z_counts
    )
    x_rows = arr[arr[:, x_id] == x_val]
    naive_obs = float((x_rows[:, y_id] == y_val).mean()) if len(x_rows) > 0 else 0.0
    return float(naive_do), naive_obs


def evaluate_backdoor(ac, df, ds, var_to_idx, device) -> dict:
    """Compile the backdoor estimand MARG_z[PROD[COND(y|x,z)[P(V)], MARG_x,y[P(V)]]] on
    the trained circuit and evaluate the two-point ratio at (x_val, y in {0,1}), plus
    the observational P(Y|X). Returns est_do/est_obs/naive_* and MAE / Bernoulli
    log-loss against the exact BN target."""
    import time as _time

    x_ids = [var_to_idx[v] for v in ds["x"]]
    y_ids = [var_to_idx[v] for v in ds["y"]]
    z_ids = [var_to_idx[v] for v in ds["z"]]
    x_val, y_val, target = ds["x_val"], ds["y_val"], ds["target"]
    n_vars = len(var_to_idx)

    # MDNet's protocol (their backdoor_experiment.py): marginalize the "other" vars W
    # out of the joint FIRST, then condition and take P(Z) — never let W couple through
    # the product. Estimand: MARG_z[PROD[COND(y|x,z)[MARG_w P(V)], MARG_x,y,w[P(V)]]].
    w_ids = [i for i in range(n_vars) if i not in set(x_ids) | set(y_ids) | set(z_ids)]

    ast_joint = make_p(set(x_ids) | set(y_ids) | set(z_ids) | set(w_ids))
    ast_xyz = make_marg(set(w_ids), ast_joint)  # P(X,Y,Z)
    ast_pz = make_marg(set(x_ids) | set(y_ids), ast_xyz)  # P(Z)
    ast_cond = make_cond(set(y_ids), set(x_ids) | set(z_ids), ast_xyz)  # P(Y|X,Z)
    ast_backdoor = make_marg(set(z_ids), make_prod([ast_cond, ast_pz]))  # P(Y|do(X))
    ast_obs = make_marg(set(z_ids), ast_joint)  # P(Y,X)

    t0 = _time.perf_counter()
    q_do = compile_query(ast_backdoor, ac, var_to_id={})
    q_obs = compile_query(ast_obs, ac, var_to_id={})

    rows = []
    for y in (0, 1):
        row = torch.full((n_vars,), float("nan"))  # all non-{x,y} vars marginalized
        for xi in x_ids:
            row[xi] = float(x_val)
        row[y_ids[0]] = float(y)
        rows.append(row)
    data = torch.stack(rows).to(device)
    with torch.no_grad():
        out_do = eval_circuit(q_do, data, keep_intermediates=False).reshape(2).float().cpu()
        out_obs = eval_circuit(q_obs, data, keep_intermediates=False).reshape(2).float().cpu()
    p_do, p_obs = torch.exp(out_do), torch.exp(out_obs)
    est_do = float(p_do[y_val] / p_do.sum())
    est_obs = float(p_obs[y_val] / p_obs.sum())
    naive_do, naive_obs = naive_adjustment(df, ds, var_to_idx)

    return {
        "est_do": est_do,
        "est_obs": est_obs,
        "naive_do": naive_do,
        "naive_obs": naive_obs,
        "target": target,
        "abs_error": abs(est_do - target),
        "log_loss": bernoulli_log_loss(est_do, target),
        "query_time": _time.perf_counter() - t0,
    }


def run_single(cfg: dict, config_path: str, cfg_hash: str, seed: int, yes_invalidate: bool) -> None:
    """Execute the pipeline for one seed: build, train, store artifacts."""
    g_logger.setLevel(logging.INFO)

    exp_id = cfg["experiment"]["id"]
    output_dir = cfg["experiment"]["output_dir"]
    # Per-seed models directory: models/<exp_id>/seed_<S>/ with <exp_id>_*.pt
    # files inside.  The log file keeps the seed-qualified name.
    models_dir = models_seed_dir(cfg["experiment"]["models_dir"], exp_id, seed)
    prefix = exp_id
    results_seed_dir = seed_dir(output_dir, seed)
    ckpt_hash = checkpoint_hash(cfg)

    _handle_stale_config(results_seed_dir, models_dir, prefix, ckpt_hash, yes_invalidate)

    os.makedirs(results_seed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    setup_logging(results_seed_dir, run_prefix(exp_id, seed))
    logger.info("Starting experiment %s (seed %d)", exp_id, seed)
    logger.info("Config path: %s (hash %s)", os.path.abspath(config_path), cfg_hash)
    save_config_copy(cfg, results_seed_dir)

    set_seed(seed)
    device = get_device(cfg["experiment"].get("device"))
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    ds = cfg["dataset"]
    df = pd.read_pickle(ds["path"])
    var_names = list(df)
    var_to_idx = {v: i for i, v in enumerate(var_names)}
    n_vars = len(var_names)
    data = torch.tensor(df.to_numpy(dtype=np.float32))
    logger.info(
        "Dataset %s: %d rows, %d vars; X=%s Y=%s |Z|=%d",
        ds["name"],
        df.shape[0],
        n_vars,
        ds["x"],
        ds["y"],
        len(ds["z"]),
    )

    # ------------------------------------------------------------------
    # Model: repo MD vtree + circuit, categorical leaves
    # ------------------------------------------------------------------
    x_ids = [var_to_idx[v] for v in ds["x"]]
    y_ids = [var_to_idx[v] for v in ds["y"]]
    z_ids = [var_to_idx[v] for v in ds["z"]]
    md_sets = [set(x_ids) | set(z_ids)]

    model_cfg = cfg["model"]
    md_vtree = construct_optimal_md_vtree(data, md_sets, keep_together=[(x_ids[0], y_ids[0])])

    categories = sorted(int(v) for v in pd.unique(df.values.ravel()))
    dists = {
        i: CategoricalDistribution(
            var=i, categories=categories, probabilities=[1.0 / len(categories)] * len(categories)
        )
        for i in range(n_vars)
    }

    ac = create_md_circuit(
        dists,
        md_vtree,
        num_nodes=model_cfg["num_nodes"],
        initialize_weights=True,
        fairness_temperature=model_cfg.get("fairness_temperature", 1.0),
        weight_softmax_temperature=model_cfg.get("weight_softmax_temperature", 0.5),
        max_leaf_num_nodes=model_cfg.get("max_leaf_num_nodes"),
        max_leaf_num_groups=model_cfg.get("max_leaf_num_groups"),
        max_sum_num_groups=model_cfg.get("max_sum_num_groups"),
        leaf_mixture_num_nodes=model_cfg.get("leaf_mixture_num_nodes"),
        leaf_mixture_num_groups=model_cfg.get("leaf_mixture_num_groups"),
    )
    ac.to(device)
    logger.info(
        "Built MD circuit (num_nodes=%d, params=%d)", model_cfg["num_nodes"], ac.num_parameters()
    )

    # ------------------------------------------------------------------
    # Train (resume from final model or latest checkpoint when present)
    # ------------------------------------------------------------------
    total_epochs = cfg["training"]["total_iters"] + 1
    start_epoch = 0
    final_path = os.path.join(models_dir, f"{prefix}_final.pt")
    if os.path.exists(final_path):
        ac = load_circuit(final_path, device)
        start_epoch = total_epochs
        logger.info("Found final model %s; skipping training.", final_path)
    else:
        latest = find_latest_checkpoint(models_dir, prefix)
        if latest is not None:
            ckpt_epoch, ckpt_path = latest
            ac = load_circuit(ckpt_path, device)
            start_epoch = min(ckpt_epoch, total_epochs)
            logger.info(
                "Resuming from checkpoint %s (epoch %d of %d).",
                ckpt_path,
                ckpt_epoch,
                total_epochs,
            )

    data_tensor = data.to(device)
    start_time = datetime.now(timezone.utc).isoformat()

    if start_epoch < total_epochs:
        train_model(
            ac, dists, data_tensor, cfg, device, models_dir, prefix, start_epoch=start_epoch
        )

    # ------------------------------------------------------------------
    # Post-training artifacts
    # ------------------------------------------------------------------
    eval_batch = resolve_eval_chunk_rows(cfg, device)
    metrics = {
        "seed": seed,
        "epochs_total": total_epochs,
        "obs_nll": compute_nll(ac, data_tensor, batch_size=eval_batch),
        "n_rows": int(df.shape[0]),
        "n_vars": n_vars,
    }
    metrics.update(evaluate_backdoor(ac, df, ds, var_to_idx, device))
    write_metrics(metrics, results_seed_dir)

    if cfg["checkpointing"].get("save_final", True):
        save_final_model(ac, models_dir, prefix)

    end_time = datetime.now(timezone.utc).isoformat()
    write_metadata(cfg, results_seed_dir, seed, start_time, end_time, cfg_hash, str(device))
    logger.info("Experiment %s (seed %d) finished.", exp_id, seed)


def run_seed_workers(
    config_path: str,
    seeds: list[int],
    gpus: list[int],
    overrides: list[str],
    yes_invalidate: bool,
) -> None:
    """Run each seed in its own subprocess, pinned round-robin to ``gpus``."""
    procs = []
    for i, seed in enumerate(seeds):
        env = os.environ.copy()
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)])
        cmd = [
            sys.executable,
            "-m",
            "experiments.run_binary_bn_experiment",
            "--config",
            config_path,
            "--seed",
            str(seed),
        ]
        for override in overrides:
            cmd += ["--override", override]
        if yes_invalidate:
            cmd.append("--yes-invalidate")
        logger.info("Launching seed %d (GPU %s)", seed, env.get("CUDA_VISIBLE_DEVICES", "cpu"))
        procs.append(subprocess.Popen(cmd, env=env))

    failures = []
    for seed, proc in zip(seeds, procs):
        code = proc.wait()
        if code != 0:
            failures.append((seed, code))
    if failures:
        logger.error("Seed runs failed: %s", failures)
        sys.exit(1)


def parse_gpus(arg: str) -> list[int]:
    """Parse a comma- or space-separated GPU list, e.g. ``"0,1,2"``."""
    return [int(tok) for tok in arg.replace(",", " ").split()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the repo's MD circuit on a real binary BN dataset (MDNet comparison)."
    )
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument(
        "--seed", type=int, default=None, help="Single seed (default: config experiment.seed)."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Multiple seeds: run each in its own subprocess (cluster mode).",
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpus,
        default=None,
        help="GPUs for a multi-seed run, e.g. --gpus 0,1,2 (default: all visible).",
    )
    parser.add_argument(
        "--yes-invalidate",
        action="store_true",
        help="Non-interactively delete stale artifacts when the config changed.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value, e.g. training.total_iters=10",
    )
    args = parser.parse_args()

    cfg_file = load_config(args.config)
    cfg_hash = config_hash(cfg_file)

    if args.seeds:
        gpus = args.gpus if args.gpus is not None else list(range(torch.cuda.device_count()))
        if args.seed is not None:
            logger.warning("Both --seed and --seeds given; --seeds takes precedence.")
        run_seed_workers(args.config, args.seeds, gpus, args.override, args.yes_invalidate)
        return

    seed = args.seed if args.seed is not None else cfg_file["experiment"].get("seed", 0)
    cfg = apply_overrides(cfg_file, args.override)
    run_single(cfg, args.config, cfg_hash, seed, args.yes_invalidate)


if __name__ == "__main__":
    main()
