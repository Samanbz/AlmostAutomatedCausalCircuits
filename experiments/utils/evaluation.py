"""Grid evaluation and artifact storage for the plotting step.

Step 5 of the experiment recipe: after training, evaluate the learned
conditional densities on shared grids and store them (plus the NLL
histories from training) so that plotting — done in a separate step, possibly
on another machine — needs nothing but the artifacts in the seed directory.

Stored per seed run (``results/<exp_id>/seed_<S>/``):

  * ``grids.npz``
      - ``x_grid``         (n_x,)   X grid: evenly spaced quantiles of the
                         observational X (``grid_x_quantile_range``)
      - ``y_grid``         (n_y,)   Y grid: pooled obs + interventional Y range
      - ``slice_contexts`` (n_c,)   X values for 1-D slices — every curve in
                         this file is evaluated at these X values; the 1-D
                         slices are simply the matching rows of the 2-D arrays
      - ``log_p_do``       (n_x, n_y)  log P(Y|do(X)) from the compiled
                         interventional circuit
      - ``log_p_obs``      (n_x, n_y)  log P(Y|X) = log P(Y,X) - log P(X)
  * ``metrics.json`` — NLL histories (before / per-checkpoint / after).
"""

import json
import os
from typing import Any, Dict

import numpy as np
import torch

from src.logger import logger as g_logger
from src.symbolic.arithmetic.circuit import eval_circuit

from .training import resolve_eval_chunk_rows


logger = g_logger.getChild("evaluation")


def _grids_from_data(data_info: Dict[str, Any], eval_cfg: Dict[str, Any]):
    """Build the shared X / Y grids deterministically from the data."""
    df_obs = data_info["df_obs"]
    q_lo, q_hi = eval_cfg.get("grid_x_quantile_range", [0.005, 0.995])
    n_x = int(eval_cfg.get("grid_x_points", 100))
    n_y = int(eval_cfg.get("grid_y_points", 200))

    x_grid = np.quantile(df_obs["X"].values, np.linspace(q_lo, q_hi, n_x))
    pooled_y = np.concatenate([df_obs["Y"].values, data_info["df_do_full"]["Y"].values])
    y_grid = np.linspace(float(pooled_y.min()), float(pooled_y.max()), n_y)

    n_c = int(eval_cfg.get("slice_contexts", 16))
    slice_idx = np.unique(np.linspace(0, n_x - 1, n_c).round().astype(int))
    slice_contexts = x_grid[slice_idx]
    return x_grid, y_grid, slice_contexts


def _eval_on_grid(query_acs, data_info, x_grid, y_grid, device, cfg):
    """Evaluate log P(Y|do(X)) and log P(Y|X) on the (x, y) grid.

    Points are laid out x-major (all Y for one X consecutively) so each X
    value forms contiguous chunks; rows are evaluated with freed intermediates
    since the compiled circuits are memory-hungry per row.  The do-circuit and
    the observational (base-derived) query circuits get separate chunk sizes —
    see :func:`resolve_eval_chunk_rows`.
    """
    n_vars = data_info["n_vars"]
    x_id, y_id = data_info["x_id"], data_info["y_id"]
    z_ids = data_info["z_ids"]
    data_mean = data_info["data_mean"]

    n_x, n_y = len(x_grid), len(y_grid)
    pts = np.zeros((n_x * n_y, n_vars), dtype=np.float32)
    pts[:, x_id] = np.repeat(x_grid, n_y)
    pts[:, y_id] = np.tile(y_grid, n_x)
    for zid in z_ids:
        pts[:, zid] = data_mean[zid]
    pts_t = torch.tensor(pts, device=device)

    chunk_rows = resolve_eval_chunk_rows(cfg, device)
    chunk_dense = resolve_eval_chunk_rows(cfg, device, dense=True)

    log_p_do = np.empty(n_x * n_y, dtype=np.float32)
    log_p_num = np.empty(n_x * n_y, dtype=np.float32)

    q_do_ac = query_acs["q_do_ac"]
    obs_num_ac = query_acs["obs_num_ac"]
    obs_den_ac = query_acs["obs_den_ac"]
    with torch.no_grad():
        for i in range(0, pts_t.size(0), chunk_dense):
            batch = pts_t[i : i + chunk_dense]
            log_p_num[i : i + chunk_dense] = (
                eval_circuit(obs_num_ac, batch, verbose=False, keep_intermediates=False)
                .reshape(-1)
                .cpu()
                .numpy()
            )
        for i in range(0, pts_t.size(0), chunk_rows):
            batch = pts_t[i : i + chunk_rows]
            log_p_do[i : i + chunk_rows] = (
                eval_circuit(q_do_ac, batch, verbose=False, keep_intermediates=False)
                .reshape(-1)
                .cpu()
                .numpy()
            )
        # P(X) is constant over the Y grid: one row per X value suffices.
        den_rows = torch.tensor(pts[::n_y], dtype=torch.float32, device=device)
        log_p_den = (
            eval_circuit(obs_den_ac, den_rows, verbose=False, keep_intermediates=False)
            .reshape(-1)
            .cpu()
            .numpy()
        )

    log_p_do = log_p_do.reshape(n_x, n_y)
    log_p_obs = (log_p_num.reshape(n_x, n_y) - log_p_den[:, None]).astype(np.float32)
    return log_p_do, log_p_obs


def evaluate_and_store(
    query_acs: Dict[str, Any],
    data_info: Dict[str, Any],
    cfg: Dict[str, Any],
    device: torch.device,
    seed_dir: str,
) -> str:
    """Evaluate the learned densities on the shared grids and store them."""
    eval_cfg = cfg["evaluation"]
    x_grid, y_grid, slice_contexts = _grids_from_data(data_info, eval_cfg)

    log_p_do, log_p_obs = _eval_on_grid(query_acs, data_info, x_grid, y_grid, device, cfg)

    path = os.path.join(seed_dir, "grids.npz")
    np.savez_compressed(
        path,
        x_grid=x_grid,
        y_grid=y_grid,
        slice_contexts=slice_contexts,
        log_p_do=log_p_do,
        log_p_obs=log_p_obs,
    )
    logger.info(
        "Saved grid evaluations (%d x %d, %d slice contexts) to %s",
        len(x_grid),
        len(y_grid),
        len(slice_contexts),
        path,
    )
    return path


def write_metrics(metrics: Dict[str, Any], seed_dir: str) -> str:
    """Persist the run's metrics (NLL histories and final values)."""
    path = os.path.join(seed_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", path)
    return path
