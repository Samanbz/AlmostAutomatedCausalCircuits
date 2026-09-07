"""Batch-EM training loop with periodic checkpointing."""

import copy
import glob
import math
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

import torch

from src.logger import logger as g_logger
from src.symbolic.arithmetic.circuit import eval_circuit
from src.symbolic.arithmetic.nodes.leaf_layer import LogLinearSplineDistribution
from src.symbolic.arithmetic.train import SymbolicEMTrainer

from .wandb_logging import log_epoch


logger = g_logger.getChild("training")

# Rows per evaluation chunk on CPU. The compiled do-circuit holds ~N^4 floats
# per row in its layer outputs, so this must stay small for large node counts
# (at 32 units, 512 rows ≈ 64 MB per node tensor).
EVAL_CHUNK_ROWS = 512


def resolve_eval_chunk_rows(cfg: Dict[str, Any], device: torch.device) -> int:
    """Rows per evaluation chunk for NLL passes.

    ``evaluation.eval_batch_size`` overrides everything.  On CPU the
    conservative :data:`EVAL_CHUNK_ROWS` applies.  On CUDA the budget is ~8 GB
    of do-circuit stage temporaries, which grow as ~N^4 per row, so the chunk
    size scales as N^-4 (≈1300 rows at N=32, ≈80 at N=64).
    """
    override = cfg.get("evaluation", {}).get("eval_batch_size")
    if override:
        return int(override)
    if device.type != "cuda":
        return EVAL_CHUNK_ROWS
    n = cfg.get("model", {}).get("num_nodes", 32)
    per_row_bytes = 12e6 * (n / 32.0) ** 4
    return int(max(16, min(8192, 8e9 // per_row_bytes)))


def compute_nll(circuit, data: torch.Tensor, batch_size: int = EVAL_CHUNK_ROWS) -> float:
    """Mean negative log-likelihood on a tensor.

    Evaluated in batches with freed intermediates: the compiled query circuits
    hold ``rows x units^3`` floats per node when intermediates are kept, which
    exhausts memory on large node counts.
    """
    n = data.size(0)
    total = torch.zeros((), device=data.device, dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            log_probs = eval_circuit(
                circuit, data[i : i + batch_size], verbose=False, keep_intermediates=False
            )
            # Per-chunk float32 sum accumulated in float64 — matches the old
            # host-side accumulation bit-for-bit, without per-chunk host syncs.
            total += -log_probs.sum()
    return (total / n).item()


def find_latest_checkpoint(models_dir: str, exp_id: str) -> Optional[Tuple[int, str]]:
    """Return ``(epoch, path)`` of the newest ``<exp_id>_epoch_XXXX.pt``, if any."""
    best: Optional[Tuple[int, str]] = None
    for path in glob.glob(os.path.join(models_dir, f"{exp_id}_epoch_*.pt")):
        m = re.search(r"_epoch_(\d+)\.pt$", os.path.basename(path))
        if m is None:
            continue
        epoch = int(m.group(1))
        if best is None or epoch > best[0]:
            best = (epoch, path)
    return best


def load_circuit(path: str, device: torch.device):
    """Load a pickled circuit checkpoint onto ``device``."""
    try:
        ac = torch.load(path, weights_only=False)
    except TypeError:  # older torch without the weights_only argument
        ac = torch.load(path)
    ac.to(device)
    return ac


def train_model(
    ac,
    dists: Dict[int, Any],
    data: torch.Tensor,
    cfg: Dict[str, Any],
    device: torch.device,
    models_dir: str,
    exp_id: str,
    do_data: Optional[torch.Tensor] = None,
    q_do_ac=None,
    start_epoch: int = 0,
    wandb_run=None,
) -> Dict[str, float]:
    """Batch EM training loop with periodic checkpoints.

    This intentionally mirrors ``SymbolicEMTrainer.train`` exactly (including
    its ``n_iter + 1`` epoch convention and decay schedule) so results match
    the original ``test_joint_fit.py`` protocol.

    ``start_epoch`` resumes training as if ``start_epoch`` epochs have already
    been completed (step-size decay schedule included).  Note that the Adam
    moments of the leaf-parameter optimizer are not checkpointed, so a resumed
    run is not bit-identical to an uninterrupted one.

    If ``do_data`` and ``q_do_ac`` are given, every batch is additionally
    scored on the do-circuit using the interventional rows at the same batch
    indices, and the epoch-mean interventional NLL is logged alongside the
    observational one.  This evaluation is read-only and does not affect
    training.  Note that the compiled do-circuit is ~30x more expensive per
    evaluation than the base circuit, so ``do_eval_every`` controls how often
    this pass runs (0 = only the final evaluation after training).
    """
    train_cfg = cfg["training"]
    batch_size = train_cfg["batch_size"]
    # Match SymbolicEMTrainer.train: n_iter=total_iters runs total_iters + 1 epochs.
    total_epochs = train_cfg["total_iters"] + 1
    step_size = train_cfg["step_size"]
    decay_rate = train_cfg["decay_rate"]
    decay_every = train_cfg["decay_every"]
    log_interval = train_cfg["log_interval"]
    checkpoint_every = cfg["checkpointing"]["checkpoint_every_epochs"]

    has_spline = any(isinstance(d, LogLinearSplineDistribution) for d in dists.values())
    leaf_lr = train_cfg.get("leaf_lr")
    if leaf_lr is None:
        leaf_lr = 0.01 if has_spline else 0.05

    trainer = SymbolicEMTrainer(ac, leaf_lr=leaf_lr)
    data = data.to(device)

    do_eval_every = train_cfg.get("do_eval_every", 1)
    has_do_eval = do_data is not None and q_do_ac is not None and do_eval_every > 0
    eval_chunk_rows = resolve_eval_chunk_rows(cfg, data.device)
    if has_do_eval:
        do_data = do_data.to(device)
        if do_data.shape[0] != data.shape[0]:
            raise ValueError(
                f"Interventional train set has {do_data.shape[0]} rows, "
                f"expected {data.shape[0]} to match the observational train set."
            )

    N = data.shape[0]
    # Reconstruct the step size as it would stand after ``start_epoch`` epochs,
    # mirroring the decay rule in the loop below.
    decays_done = (
        sum(1 for m in range(start_epoch) if m % decay_every == 0) if decay_rate < 1.0 else 0
    )
    current_step_size = step_size * (decay_rate**decays_done)
    step_count = start_epoch * math.ceil(N / batch_size)

    train_start = time.perf_counter()
    ema_epoch_duration: Optional[float] = None
    ema_alpha = 0.3

    final_metrics: Dict[str, float] = {}

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.perf_counter()
        indices = torch.randperm(N, device=device)
        run_do_eval = has_do_eval and epoch % do_eval_every == 0

        # Accumulate as tensors to avoid a host sync on every step.
        obs_nll_sum = torch.zeros((), device=device)
        do_nll_sum = torch.zeros((), device=device)
        n_samples = 0

        for start in range(0, N, batch_size):
            idx = indices[start : start + batch_size]
            nll = trainer.em_step(data[idx], current_step_size)
            step_count += 1
            obs_nll_sum += nll * idx.shape[0]

            if run_do_eval:
                with torch.no_grad():
                    batch_do = do_data[idx]
                    for j in range(0, batch_do.shape[0], eval_chunk_rows):
                        do_log_probs = eval_circuit(
                            q_do_ac,
                            batch_do[j : j + eval_chunk_rows],
                            verbose=False,
                            keep_intermediates=False,
                        ).squeeze()
                        do_valid = torch.isfinite(do_log_probs)
                        do_nll_sum += -do_log_probs[do_valid].sum()

            n_samples += idx.shape[0]

        epoch_obs_nll = float(obs_nll_sum / n_samples)
        epoch_do_nll = float(do_nll_sum / n_samples) if run_do_eval else float("nan")

        epoch_duration = time.perf_counter() - epoch_start
        ema_epoch_duration = (
            epoch_duration
            if ema_epoch_duration is None
            else ema_alpha * epoch_duration + (1.0 - ema_alpha) * ema_epoch_duration
        )

        if log_interval > 0 and (epoch % log_interval == 0 or epoch == total_epochs - 1):
            elapsed = time.perf_counter() - train_start
            remaining = total_epochs - (epoch + 1)
            eta = ema_epoch_duration * remaining if ema_epoch_duration is not None else 0.0
            logger.info(
                "Epoch %3d | Step %5d | obs NLL %.4f | do NLL %.4f | step_size %.4f | "
                "epoch %s | elapsed %s | ETA %s",
                epoch,
                step_count,
                epoch_obs_nll,
                epoch_do_nll,
                current_step_size,
                _format_duration(epoch_duration),
                _format_duration(elapsed),
                _format_duration(eta),
            )
            log_epoch(
                wandb_run, epoch, epoch_obs_nll, epoch_do_nll, current_step_size, epoch_duration
            )

        if decay_rate < 1.0 and epoch % decay_every == 0:
            current_step_size *= decay_rate

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(ac, epoch + 1, models_dir, exp_id)

    final_metrics["train_nll"] = compute_nll(ac, data)
    logger.info("Training finished. Train NLL: %.4f", final_metrics["train_nll"])
    return final_metrics


def save_checkpoint(
    ac,
    epoch: int,
    models_dir: str,
    exp_id: str,
) -> str:
    """Save a CPU-resident copy of the circuit without mutating the trained model."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{exp_id}_epoch_{epoch:04d}.pt")
    checkpoint_ac = copy.deepcopy(ac)
    checkpoint_ac.to("cpu")
    torch.save(checkpoint_ac, path)
    logger.info("Saved checkpoint: %s", path)
    return path


def save_final_model(ac, models_dir: str, exp_id: str) -> str:
    """Save the final trained model as ``<exp_id>_final.pt``."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{exp_id}_final.pt")
    final_ac = copy.deepcopy(ac)
    final_ac.to("cpu")
    torch.save(final_ac, path)
    logger.info("Saved final model: %s", path)
    return path


def _format_duration(seconds: float) -> str:
    """Compact human-readable duration."""
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        minutes = int(seconds // 60)
        secs = int(round(seconds % 60))
        return f"{minutes}m {secs:02d}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(round(seconds % 60))
    return f"{hours}h {minutes:02d}m {secs:02d}s"
