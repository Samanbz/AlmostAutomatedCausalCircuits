"""Batch-EM training loop with periodic checkpointing.

Follows step 4 of the experiment recipe: every ``checkpoint_every_epochs``
epochs the loop saves a base-model checkpoint, and every ``decay_every``
epochs it decays the step size.  No dataset-wide NLL evaluations happen
during training — those are run post-hoc from the saved checkpoints, which
is much cheaper than re-evaluating the compiled do-circuit at every
checkpoint epoch.
"""

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


logger = g_logger.getChild("training")

# Rows per evaluation chunk on CPU. The compiled do-circuit holds ~N^4 floats
# per row in its layer outputs, so this must stay small for large node counts
# (at 32 units, 512 rows ≈ 64 MB per node tensor).
EVAL_CHUNK_ROWS = 512


def resolve_eval_chunk_rows(cfg: Dict[str, Any], device: torch.device, dense: bool = False) -> int:
    """Rows per evaluation chunk for NLL/grid passes.

    ``evaluation.eval_batch_size`` overrides everything.  On CPU the
    conservative :data:`EVAL_CHUNK_ROWS` applies.  On CUDA the budget is ~24 GB
    for do-circuit temporaries; empirically (V100, keep_intermediates=False)
    the do-circuit peak stays ~2.1 GiB at N=64 and ~2.4 GiB at N=128 for a
    4096-row chunk and grows only ~0.1 MB/row beyond that.

    The base circuit and the observational query circuits (numerator /
    denominator of P(Y|X)) are a different story: their layer outputs scale
    ~linearly with rows at ~1.1 MB/row * (N/64)^2 (measured 18.2 GiB at
    N=128 for a 4096-row chunk vs. 2.4 GiB for the do-circuit), so ``dense=True``
    uses a 12 GB budget and a 4096-row cap for them.
    """
    override = cfg.get("evaluation", {}).get("eval_batch_size")
    if override:
        return int(override)
    if device.type != "cuda":
        return EVAL_CHUNK_ROWS
    n = cfg.get("model", {}).get("num_nodes", 32)
    if dense:
        per_row_bytes = 1.1e6 * (n / 64.0) ** 2
        return int(max(256, min(4096, 12e9 // per_row_bytes)))
    per_row_bytes = 0.5e6 * (n / 64.0) ** 2
    return int(max(256, min(8192, 24e9 // per_row_bytes)))


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
            ).reshape(-1)
            # Per-chunk float32 sum accumulated in float64 — matches the old
            # host-side accumulation bit-for-bit, without per-chunk host syncs.
            total += -log_probs.sum()
    return (total / n).item()


def find_latest_checkpoint(models_dir: str, run_prefix: str) -> Optional[Tuple[int, str]]:
    """Return ``(epoch, path)`` of the newest ``<run_prefix>_epoch_XXXX.pt``, if any."""
    best: Optional[Tuple[int, str]] = None
    for path in glob.glob(os.path.join(models_dir, f"{run_prefix}_epoch_*.pt")):
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
    run_prefix: str,
    start_epoch: int = 0,
) -> None:
    """Batch EM training loop with periodic checkpoints.

    Mirrors ``SymbolicEMTrainer.train`` (including its ``n_iter + 1`` epoch
    convention and decay schedule) so results match the original
    ``test_joint_fit.py`` protocol.

    ``start_epoch`` resumes training as if ``start_epoch`` epochs have already
    been completed (step-size decay schedule included).  Note that the Adam
    moments of the leaf-parameter optimizer are not checkpointed, so a resumed
    run is not bit-identical to an uninterrupted one.

    No dataset-wide NLL evaluations happen during training — they are run
    post-hoc from the saved checkpoints, which avoids re-evaluating the
    compiled do-circuit (orders of magnitude more expensive than one training
    epoch) at every checkpoint epoch.
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

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.perf_counter()
        indices = torch.randperm(N, device=device)

        # Accumulate as tensors to avoid a host sync on every step.
        obs_nll_sum = torch.zeros((), device=device)
        n_samples = 0

        for start in range(0, N, batch_size):
            idx = indices[start : start + batch_size]
            nll = trainer.em_step(data[idx], current_step_size)
            step_count += 1
            obs_nll_sum += nll * idx.shape[0]
            n_samples += idx.shape[0]

        epoch_obs_nll = float(obs_nll_sum / n_samples)
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
                "Epoch %3d | Step %5d | obs NLL %.4f | step_size %.4f | "
                "epoch %s | elapsed %s | ETA %s",
                epoch,
                step_count,
                epoch_obs_nll,
                current_step_size,
                _format_duration(epoch_duration),
                _format_duration(elapsed),
                _format_duration(eta),
            )

        if decay_rate < 1.0 and epoch % decay_every == 0:
            current_step_size *= decay_rate

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(ac, epoch + 1, models_dir, run_prefix)

    logger.info("Training finished (no eval NLLs computed; use checkpoints for post-hoc eval).")


def save_checkpoint(ac, epoch: int, models_dir: str, run_prefix: str) -> str:
    """Save a CPU-resident copy of the circuit without mutating the trained model."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{run_prefix}_epoch_{epoch:04d}.pt")
    checkpoint_ac = copy.deepcopy(ac)
    checkpoint_ac.to("cpu")
    torch.save(checkpoint_ac, path)
    logger.info("Saved checkpoint: %s", path)
    return path


def save_final_model(ac, models_dir: str, run_prefix: str) -> str:
    """Save the final trained model as ``<run_prefix>_final.pt``."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{run_prefix}_final.pt")
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
