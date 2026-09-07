"""Journal-grade backdoor adjustment experiment for Monarch Causal Circuits.

Mirrors the protocol in ``tests/test_joint_fit.py``:
train an MD circuit on observational data, then visualise the learned
P(Y|do(X)) and P(Y|X) against ground truth as 1-D slices and 2-D heatmaps,
with optional leaf plots.

All tunable settings live in an external JSON config file.  The experiment ID
is the config file name, and every artifact (model, checkpoints, plots, logs,
metadata) is named with that ID.  Runs are resumable: an existing final model
or checkpoint is reused instead of retraining.  A content hash of the config
is stored in the metadata; when the config changes, the user is asked whether
the stale artifacts should be invalidated before starting fresh.

All heavy lifting lives in ``experiments/utils/``; this file is only
orchestration.

Example
-------
    python -m experiments.run_backdoor_experiment --config configs/config_0.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import torch

from experiments.utils.config import (
    apply_overrides,
    config_hash,
    get_device,
    invalidate_artifacts,
    load_config,
    read_stored_config_hash,
    save_config_copy,
    set_seed,
    setup_logging,
    write_metadata,
)
from experiments.utils.data import build_circuit, load_data
from experiments.utils.plotting import (
    plot_2d_heatmaps,
    plot_conditional_densities,
    plot_leaves,
)
from experiments.utils.queries import compile_queries
from experiments.utils.training import (
    compute_nll,
    find_latest_checkpoint,
    load_circuit,
    resolve_eval_chunk_rows,
    save_checkpoint,
    save_final_model,
    train_model,
)
from experiments.utils.wandb_logging import (
    finish_wandb,
    init_wandb,
    log_plots,
    log_summary,
)
from src.logger import logger as g_logger


logger = g_logger.getChild("experiment")


def _handle_stale_config(output_dir: str, models_dir: str, exp_id: str, cfg_hash: str) -> None:
    """Ask to invalidate artifacts when the config changed since the last run."""
    metadata_path = os.path.join(output_dir, f"{exp_id}_metadata.json")
    if not os.path.exists(metadata_path):
        return  # first run for this experiment ID
    stored = read_stored_config_hash(output_dir, exp_id)
    if stored is not None and stored == cfg_hash:
        return
    if stored is None:
        logger.warning(
            "Experiment '%s' has results from a run that predates config hashing; "
            "they may not match the current config.",
            exp_id,
        )
    else:
        logger.warning(
            "The config for experiment '%s' changed since the last run "
            "(hash %s -> %s). Existing checkpoints and results were produced "
            "with the old config.",
            exp_id,
            stored,
            cfg_hash,
        )
    try:
        answer = input("Delete all model checkpoints and results for this config? [y/N] ")
    except EOFError:  # non-interactive shell: default to aborting
        answer = ""
    if answer.strip().lower() in ("y", "yes"):
        invalidate_artifacts(output_dir, models_dir, exp_id)
        logger.warning("Artifacts invalidated; starting fresh.")
    else:
        logger.error(
            "Aborted. Revert the config file, or re-run and answer 'y' to "
            "invalidate the stale artifacts."
        )
        sys.exit(1)


def run_experiment(cfg: dict, config_path: str, cfg_hash: str) -> None:
    """Execute the full experiment pipeline described by ``cfg``."""
    g_logger.setLevel(logging.INFO)

    exp_id = cfg["experiment"]["id"]
    output_dir = cfg["experiment"]["output_dir"]
    models_dir = cfg["experiment"]["models_dir"]

    _handle_stale_config(output_dir, models_dir, exp_id, cfg_hash)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    setup_logging(output_dir, exp_id)
    logger.info("Starting experiment %s", exp_id)
    logger.info("Config path: %s (hash %s)", os.path.abspath(config_path), cfg_hash)

    write_metadata(cfg, output_dir, datetime.now(timezone.utc).isoformat(), cfg_hash)
    save_config_copy(cfg, output_dir)
    wandb_run = init_wandb(cfg)

    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"].get("device"))
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Data + model
    # ------------------------------------------------------------------
    data_info = load_data(cfg)
    ac, dists = build_circuit(cfg, data_info["data"], data_info, device, output_dir)

    # Resume from existing artifacts when possible: prefer the final model
    # (training already completed for this config), otherwise resume from the
    # newest checkpoint.  ``start_epoch == total_epochs`` skips training.
    total_epochs = cfg["training"]["total_iters"] + 1
    start_epoch = 0
    final_path = os.path.join(models_dir, f"{exp_id}_final.pt")
    if os.path.exists(final_path):
        ac = load_circuit(final_path, device)
        start_epoch = total_epochs
        logger.info("Found final model %s; skipping training.", final_path)
    else:
        latest = find_latest_checkpoint(models_dir, exp_id)
        if latest is not None:
            ckpt_epoch, ckpt_path = latest
            ac = load_circuit(ckpt_path, device)
            start_epoch = min(ckpt_epoch, total_epochs)
            if ckpt_epoch >= total_epochs:
                logger.info(
                    "Checkpoint %s already covers the requested %d epochs; skipping training.",
                    ckpt_path,
                    total_epochs,
                )
            else:
                logger.info(
                    "Resuming from checkpoint %s (epoch %d of %d).",
                    ckpt_path,
                    ckpt_epoch,
                    total_epochs,
                )

    test_data = torch.tensor(data_info["df_test"].values, dtype=torch.float32, device=device)
    data_tensor = data_info["data"].to(device)
    do_train_data = torch.tensor(data_info["df_do"].values, dtype=torch.float32, device=device)
    do_test_data = torch.tensor(data_info["df_do_test"].values, dtype=torch.float32, device=device)

    eval_chunk_rows = resolve_eval_chunk_rows(cfg, device)

    # Optionally cap the size of the evaluation sets. The do-NLL passes cost
    # ~N^6 per row, so on large circuits a subset keeps before/after training
    # evaluation cheap with negligible impact on the metric.  The interventional
    # train set is only subsampled when per-epoch do-evaluation is off, since
    # that path requires it to match the observational train set row for row.
    eval_limit = cfg["evaluation"].get("eval_max_rows")
    if eval_limit:
        test_data = test_data[:eval_limit]
        do_test_data = do_test_data[:eval_limit]
        if cfg["training"].get("do_eval_every", 1) == 0:
            do_train_data = do_train_data[:eval_limit]
        logger.info("Evaluation sets capped at %d rows (evaluation.eval_max_rows).", eval_limit)

    # Compile the query circuits BEFORE training: they share parameter tensors
    # with the base circuit (shallow copies / lazy ProductWeights), so the
    # do-circuit tracks EM updates live and can be evaluated every epoch.
    query_acs = compile_queries(ac, data_info)
    q_do_ac = query_acs["q_do_ac"]

    # ------------------------------------------------------------------
    # Train (with optional initial checkpoint and periodic checkpoints)
    # ------------------------------------------------------------------
    test_nll_before = compute_nll(ac, test_data, batch_size=eval_chunk_rows)
    train_nll_before = compute_nll(ac, data_tensor, batch_size=eval_chunk_rows)
    # do_test_nll_before = compute_nll(q_do_ac, do_test_data, batch_size=eval_chunk_rows)
    # do_train_nll_before = compute_nll(q_do_ac, do_train_data, batch_size=eval_chunk_rows)
    logger.info("Test NLL (before training): %.4f", test_nll_before)
    logger.info("Train NLL (before training): %.4f", train_nll_before)
    # logger.info("Do-test NLL (before training): %.4f", do_test_nll_before)
    # logger.info("Do-train NLL (before training): %.4f", do_train_nll_before)

    if cfg["checkpointing"].get("save_initial", False) and start_epoch == 0:
        save_checkpoint(ac, 0, models_dir, exp_id)

    train_metrics = train_model(
        ac,
        dists,
        data_tensor,
        cfg,
        device,
        models_dir,
        exp_id,
        do_data=do_train_data,
        q_do_ac=q_do_ac,
        start_epoch=start_epoch,
        wandb_run=wandb_run,
    )

    # ------------------------------------------------------------------
    # Evaluate, persist metrics
    # ------------------------------------------------------------------
    test_nll_after = compute_nll(ac, test_data, batch_size=eval_chunk_rows)
    train_nll_after = train_metrics["train_nll"]
    # do_test_nll_after = compute_nll(q_do_ac, do_test_data, batch_size=eval_chunk_rows)
    # do_train_nll_after = compute_nll(q_do_ac, do_train_data, batch_size=eval_chunk_rows)
    logger.info("Test NLL (after training): %.4f", test_nll_after)
    logger.info("Train NLL (after training): %.4f", train_nll_after)
    # logger.info("Do-test NLL (after training): %.4f", do_test_nll_after)
    # logger.info("Do-train NLL (after training): %.4f", do_train_nll_after)

    metrics = {
        "test_nll_before": test_nll_before,
        "train_nll_before": train_nll_before,
        # "do_test_nll_before": do_test_nll_before,
        # "do_train_nll_before": do_train_nll_before,
        "train_nll_after": train_nll_after,
        "test_nll_after": test_nll_after,
        # "do_train_nll_after": do_train_nll_after,
        # "do_test_nll_after": do_test_nll_after,
    }
    metrics_path = os.path.join(output_dir, f"{exp_id}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log_summary(wandb_run, metrics)

    if cfg["checkpointing"].get("save_final", True):
        save_final_model(ac, models_dir, exp_id)

    # ------------------------------------------------------------------
    # Plots: 1-D slices, 2-D heatmaps, and optionally the leaves
    # ------------------------------------------------------------------
    eval_cfg = cfg["evaluation"]
    if eval_cfg.get("plot", True):
        var_to_name = {v: k for k, v in data_info["var_to_id"].items()}
        if eval_cfg.get("plot_leaves", True):
            plot_leaves(ac, var_to_name, output_dir, exp_id, df_obs=data_info["df_obs"])
        plot_conditional_densities(ac, query_acs, data_info, cfg, device, output_dir, exp_id)
        plot_2d_heatmaps(ac, query_acs, data_info, cfg, device, output_dir, exp_id)
        if cfg.get("wandb", {}).get("log_plots", True):
            log_plots(wandb_run, output_dir, exp_id)

    finish_wandb(wandb_run)
    logger.info("Experiment %s finished.", exp_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a reproducible backdoor-adjustment experiment."
    )
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value, e.g. training.total_iters=10",
    )
    args = parser.parse_args()

    cfg_file = load_config(args.config)
    cfg_hash = config_hash(cfg_file)  # hash the file contents, not CLI overrides
    cfg = apply_overrides(cfg_file, args.override)
    run_experiment(cfg, args.config, cfg_hash)


if __name__ == "__main__":
    main()
