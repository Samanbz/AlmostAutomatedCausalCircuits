"""Run a reproducible causal-circuit experiment: training only.

Follows the general experiment recipe, per seed run:

    1. Construct the MD circuit from the dataset and config.
    2. Train on the observational data; every ``checkpoint_every_epochs``
       epochs save a checkpoint and decay the step size.  No dataset-wide
       NLL evaluations happen here — they are run post-hoc from the saved
       checkpoints.
    3. Save the final model.

Identification, query compilation, evaluation and plotting are deliberately
NOT part of this script: a later step loads each checkpoint and evaluates it
on the test data (e.g. comparing how N=8/16/32/64/128 converge with
training), then plotting is a separate post-processing step.

Multi-seed runs for the cluster: pass ``--seeds 27 28 29`` to run each seed
in its own subprocess, pinned round-robin to ``--gpus`` (default: all visible
CUDA devices) via ``CUDA_VISIBLE_DEVICES``.  A single seed (the default)
runs in-process.  Averaging across seeds happens in post-processing.

Artifacts::

    results/<exp_id>/seed_<S>/{config.json,metadata.json,metrics.json,<exp_id>_seed<S>.log}
    models/<exp_id>/seed_<S>/<exp_id>_epoch_XXXX.pt
    models/<exp_id>/seed_<S>/<exp_id>_final.pt

Runs are resumable (an existing final model or checkpoint is reused instead
of retrained).  A content hash of the checkpoint-relevant config sections is
stored in the metadata; when the config changes, stale artifacts are
invalidated — interactively, or non-interactively with ``--yes-invalidate``.

All heavy lifting lives in ``experiments/utils/``; this file is only
orchestration.

Example
-------
    python -m experiments.run_experiment --config configs/config_0.json
    python -m experiments.run_experiment --config configs/config_0.json --seeds 27 28 29 --gpus 0 1 2
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

import torch

from experiments.utils.config import (
    apply_overrides,
    checkpoint_hash,
    config_hash,
    get_device,
    invalidate_artifacts,
    load_config,
    models_seed_dir,
    read_stored_checkpoint_hash,
    run_prefix,
    save_config_copy,
    seed_dir,
    set_seed,
    setup_logging,
    write_metadata,
)
from experiments.utils.data import build_circuit, load_data
from experiments.utils.evaluation import write_metrics
from experiments.utils.training import (
    find_latest_checkpoint,
    load_circuit,
    save_checkpoint,
    save_final_model,
    train_model,
)
from src.logger import logger as g_logger


logger = g_logger.getChild("experiment")


def _handle_stale_config(
    results_seed_dir: str, models_dir: str, prefix: str, ckpt_hash: str, yes_invalidate: bool
) -> None:
    """Ask to invalidate artifacts when the checkpoint-relevant config changed.

    Only ``dataset`` / ``model`` / ``training`` / ``experiment.seed`` changes
    (see ``checkpoint_hash``) trigger invalidation; evaluation edits never do.
    Metadata written before checkpoint-aware hashing has no
    ``checkpoint_hash`` and only gets a soft warning, not a prompt.
    """
    stored = read_stored_checkpoint_hash(results_seed_dir)
    if stored is None:
        return  # first run for this seed, or metadata predates hashing
    if stored == ckpt_hash:
        return
    logger.warning(
        "The config for seed run '%s' changed since the last run "
        "(checkpoint hash %s -> %s). Existing checkpoints and results were produced "
        "with the old config.",
        prefix,
        stored,
        ckpt_hash,
    )
    if yes_invalidate:
        invalidate_artifacts(results_seed_dir, models_dir, prefix)
        logger.warning("Artifacts invalidated; starting fresh.")
        return
    try:
        answer = input("Delete all model checkpoints and results for this seed run? [y/N] ")
    except EOFError:  # non-interactive shell: default to aborting
        answer = ""
    if answer.strip().lower() in ("y", "yes"):
        invalidate_artifacts(results_seed_dir, models_dir, prefix)
        logger.warning("Artifacts invalidated; starting fresh.")
    else:
        logger.error(
            "Aborted. Revert the config file, re-run with --yes-invalidate, or answer 'y'."
        )
        sys.exit(1)


def run_single(cfg: dict, config_path: str, cfg_hash: str, seed: int, yes_invalidate: bool) -> None:
    """Execute the full experiment pipeline for one seed."""
    g_logger.setLevel(logging.INFO)

    exp_id = cfg["experiment"]["id"]
    output_dir = cfg["experiment"]["output_dir"]
    # Per-seed models directory: models/<exp_id>/seed_<S>/ with <exp_id>_*.pt
    # files inside.  The log file keeps the seed-qualified name.
    models_dir = models_seed_dir(cfg["experiment"]["models_dir"], exp_id, seed)
    prefix = exp_id
    results_seed_dir = seed_dir(output_dir, seed)
    # Hash the *effective* config (after CLI overrides) — this is what the
    # stored metadata is compared against on the next run.
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
    # Data + model (recipe step 1)
    # ------------------------------------------------------------------
    data_info = load_data(cfg)

    ac, dists = build_circuit(cfg, data_info["data"], data_info, device)

    # Resume from existing artifacts when possible: prefer the final model
    # (training already completed for this config), otherwise resume from the
    # newest checkpoint.  ``start_epoch == total_epochs`` skips training.
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

    data_tensor = data_info["data"].to(device)

    # ------------------------------------------------------------------
    # Train (recipe step 4, with initial + periodic checkpoints, no NLL evals)
    # ------------------------------------------------------------------
    start_time = datetime.now(timezone.utc).isoformat()

    if cfg["checkpointing"].get("save_initial", False) and start_epoch == 0:
        save_checkpoint(ac, 0, models_dir, prefix)

    train_model(
        ac,
        dists,
        data_tensor,
        cfg,
        device,
        models_dir,
        prefix,
        start_epoch=start_epoch,
    )

    # ------------------------------------------------------------------
    # Post-training artifacts (recipe step 3): final model only.
    # Identification / query compilation / evaluation are post-hoc steps.
    # ------------------------------------------------------------------
    metrics = {
        "seed": seed,
        "epochs_total": total_epochs,
        "history": [],
        "note": "NLLs and query evaluations are run post-hoc from the saved checkpoints.",
    }
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
            "experiments.run_experiment",
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
        description="Run a reproducible causal-circuit experiment (training + evaluation)."
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
    cfg_hash = config_hash(cfg_file)  # hash the file contents, not CLI overrides

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
