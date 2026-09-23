"""Visualisation for the backdoor experiment: leaves, 1-D slices, 2-D heatmaps."""

import os
from typing import Any, Dict, Optional, Tuple

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import Normalize, PowerNorm

from src.logger import logger as g_logger
from src.symbolic.arithmetic.circuit import eval_circuit
from src.symbolic.arithmetic.nodes.leaf_layer import (
    GaussianLeafLayer,
    MixtureLeafLayer,
    SplineLeafLayer,
)
from src.utils.visualization import plot_gaussian_leaf, plot_mixture_leaf, plot_spline_leaf

from .circuit_inspect import resolve_context_xs
from .empirical import py_given_x_slice, py_given_x_surface_hist
from .queries import eval_conditionals_at_x
from .training import resolve_eval_chunk_rows


logger = g_logger.getChild("plotting")

# Fixed visualization constants (kept in code, not config, since they are not
# meant to be tuned per experiment).
HEATMAP_X_RES = 100
HEATMAP_DENSITY_GAMMA = 0.4
HEATMAP_DIFF_PERCENTILE = 98


# ---------------------------------------------------------------------------
# Shared axis derivation
# ---------------------------------------------------------------------------


def _derive_plot_axes(
    data_info: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    x_res: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Derive the shared (x, y) axes of the density plots from the data.

    Returns ``(grid_x, x_edges, x_range, y_range)`` where:

    * ``x_edges`` are ``x_res + 1`` bin edges spanning the full X range of the
      observational and interventional data combined; ``grid_x`` are the bin
      centers. These are the bins of the ground-truth histogram surfaces.
    * ``x_range`` is the widest span of grid columns where *both* datasets
      have at least ``heatmap_min_x_count`` samples in the X bin (the same
      criterion used to mask the ground-truth surfaces).
    * ``y_range`` covers the central 0.1-99.9 percentiles of Y pooled over
      the displayed x-span (both datasets), padded by 10%, so the tails of
      P(Y|X) stay visible for every X shown in the diagram.
    """
    df_obs = data_info["df_obs"]
    df_do = data_info["df_do"]
    x_obs = df_obs["X"].values
    x_do = df_do["X"].values

    x_all = np.concatenate([x_obs, x_do])
    x_edges = np.linspace(float(x_all.min()), float(x_all.max()), x_res + 1)
    grid_x = (x_edges[:-1] + x_edges[1:]) / 2.0

    min_x_count = eval_cfg["heatmap_min_x_count"]
    counts_obs, _ = np.histogram(x_obs, bins=x_edges)
    counts_do, _ = np.histogram(x_do, bins=x_edges)
    valid = (counts_obs >= min_x_count) & (counts_do >= min_x_count)
    if valid.any():
        idx = np.where(valid)[0]
        x_range = (float(grid_x[idx[0]]), float(grid_x[idx[-1]]))
        x_span = (float(x_edges[idx[0]]), float(x_edges[idx[-1] + 1]))
    else:
        x_range = (float(grid_x[0]), float(grid_x[-1]))
        x_span = (float(x_edges[0]), float(x_edges[-1]))

    y_pool = np.concatenate(
        [
            df_obs["Y"].values[(x_obs >= x_span[0]) & (x_obs <= x_span[1])],
            df_do["Y"].values[(x_do >= x_span[0]) & (x_do <= x_span[1])],
        ]
    )
    y_lo, y_hi = np.percentile(y_pool, [0.1, 99.9])
    pad = 0.1 * max(float(y_hi - y_lo), 1e-6)
    y_range = (float(y_lo - pad), float(y_hi + pad))

    return grid_x, x_edges, x_range, y_range


# ---------------------------------------------------------------------------
# Leaf plots
# ---------------------------------------------------------------------------


def _num_leaf_nodes(leaf) -> int:
    """Number of output nodes (plotted curves) for a leaf."""
    if isinstance(leaf, GaussianLeafLayer):
        return leaf.num_nodes
    if isinstance(leaf, SplineLeafLayer):
        return leaf.num_nodes
    if isinstance(leaf, MixtureLeafLayer):
        w = torch.exp(leaf.log_weights.log_weights).detach().cpu().numpy()
        if w.ndim == 6:
            w = w.squeeze(axis=(4, 5))
        return w.shape[1]
    return 1


def plot_leaves(
    ac,
    var_to_name: Dict[int, str],
    output_dir: str,
    exp_id: str,
    df_obs: Optional[pd.DataFrame] = None,
) -> None:
    """Plot every leaf distribution and overlay the empirical marginal."""
    logger.info("\n--- Plotting Leaves ---")
    for leaf_id in ac.get_leaves():
        leaf = ac.get_node_data(leaf_id)
        var = getattr(leaf, "var", None)
        var_name = var_to_name.get(var, f"var{var}")
        if isinstance(leaf, GaussianLeafLayer):
            axes = plot_gaussian_leaf(leaf)
        elif isinstance(leaf, MixtureLeafLayer):
            axes = plot_mixture_leaf(leaf)
        elif isinstance(leaf, SplineLeafLayer):
            axes = plot_spline_leaf(leaf)
        else:
            continue

        if df_obs is not None and var is not None and var_name in df_obs.columns:
            gt_values = df_obs[var_name].values
            num_nodes = _num_leaf_nodes(leaf)
            counts, bin_edges = np.histogram(gt_values, bins=100, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            for ax in np.asarray(axes).reshape(-1):
                ax.plot(
                    bin_centers,
                    counts,
                    drawstyle="steps-mid",
                    color="black",
                    linewidth=2,
                    label="GT marginal",
                )
                ax.plot(
                    bin_centers,
                    counts * num_nodes,
                    drawstyle="steps-mid",
                    color="red",
                    linewidth=2,
                    linestyle="--",
                    label=f"GT marginal × {num_nodes}",
                )
                ax.legend(loc="best", fontsize="small")

        out_path = os.path.join(output_dir, f"leaf_{exp_id}_{var_name}_distribution.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.debug(f"Saved leaf plot to {out_path}")


# ---------------------------------------------------------------------------
# 1-D conditional density slices
# ---------------------------------------------------------------------------


def plot_conditional_densities(
    ac,
    query_acs: Dict[str, Any],
    data_info: Dict[str, Any],
    cfg: Dict[str, Any],
    device: torch.device,
    output_dir: str,
    exp_id: str,
) -> None:
    """1-D slices of P(Y|do(X)) and P(Y|X), learned vs ground truth."""
    eval_cfg = cfg["evaluation"]
    df_obs = data_info["df_obs"]
    df_do = data_info["df_do"]

    contexts, context_labels = resolve_context_xs(ac, data_info, eval_cfg)
    logger.info(f"  Plotting at {len(contexts)} X value(s)")

    n_x = len(contexts)
    hist_bins = eval_cfg.get("hist_bins", 100)
    _, _, _, y_range = _derive_plot_axes(data_info, eval_cfg, HEATMAP_X_RES)
    y_edges = np.linspace(y_range[0], y_range[1], hist_bins + 1)
    grid_y = (y_edges[:-1] + y_edges[1:]) / 2.0
    dy = grid_y[1] - grid_y[0]

    n_cols = 2 if n_x >= 4 else 1
    n_rows = (n_x + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for row, (x_val, label) in enumerate(zip(contexts, context_labels)):
        ax = axes[row]
        p_do, p_obs = eval_conditionals_at_x(
            query_acs["q_do_ac"],
            query_acs["obs_num_ac"],
            query_acs["obs_den_ac"],
            x_val,
            grid_y,
            data_info,
            device,
        )
        # Ground truth: the same finite datasets the circuit saw, estimated
        # from the SLICE_WINDOW_COUNT samples nearest in X (adaptive window).
        gt_obs, lo_obs, hi_obs, hw_obs, n_obs = py_given_x_slice(df_obs, "X", "Y", x_val, grid_y)
        gt_do, lo_do, hi_do, hw_do, n_do = py_given_x_slice(df_do, "X", "Y", x_val, grid_y)

        ax.fill_between(grid_y, lo_do, hi_do, color="red", alpha=0.15, linewidth=0)
        ax.fill_between(grid_y, lo_obs, hi_obs, color="blue", alpha=0.15, linewidth=0)
        ax.plot(grid_y, p_do, color="red", alpha=0.5, label="Learned P(Y|do(X))", linewidth=2)
        ax.plot(grid_y, p_obs, color="blue", alpha=0.5, label="Learned P(Y|X)", linewidth=2)
        ax.plot(grid_y, gt_do, color="red", alpha=0.3, label="GT P(Y|do(X))", linewidth=2)
        ax.plot(grid_y, gt_obs, color="blue", alpha=0.3, label="GT P(Y|X)", linewidth=2)

        # Report the adaptive window half-widths: they control how much the
        # apparent GT conditional variance is inflated (slope * half_width).
        ax.text(
            0.02,
            0.97,
            f"GT window obs ±{hw_obs:.2f} (n={n_obs})\nGT window do ±{hw_do:.2f} (n={n_do})",
            transform=ax.transAxes,
            fontsize=6,
            va="top",
        )

        for arr, color, style in [
            (p_do, "red", "-"),
            (p_obs, "blue", "-"),
            (gt_do, "red", "--"),
            (gt_obs, "blue", "--"),
        ]:
            mean = np.sum(grid_y * arr) * dy
            ax.axvline(
                mean, color=color, alpha=0.5 if style == "-" else 0.2, linestyle=style, linewidth=1
            )

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Y")
        ax.set_ylabel("Density")
        if row == 0:
            ax.legend(loc="upper right", fontsize=6)
        ax.grid(True, alpha=0.3)

    out_file = os.path.join(
        output_dir,
        f"{exp_id}_tree_backdoor_{cfg['dataset']['dataset']}_"
        f"N{cfg['model']['num_nodes']}_{cfg['model']['md_sets']}.png",
    )
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    logger.info(f"Saved conditional density plot to {out_file}")


# ---------------------------------------------------------------------------
# 2-D heatmaps
# ---------------------------------------------------------------------------


def _eval_log_probs(circuit, inputs: torch.Tensor, batch_size: int) -> np.ndarray:
    """Batched circuit evaluation returning numpy log-probs."""
    log_probs = []
    for i in range(0, inputs.size(0), batch_size):
        batch = inputs[i : i + batch_size]
        with torch.no_grad():
            log_p = eval_circuit(circuit, batch, verbose=False, keep_intermediates=False).squeeze()
        log_probs.append(log_p.detach().cpu())
    return torch.cat(log_probs).numpy()


def plot_2d_heatmaps(
    ac,
    query_acs: Dict[str, Any],
    data_info: Dict[str, Any],
    cfg: Dict[str, Any],
    device: torch.device,
    output_dir: str,
    exp_id: str,
) -> None:
    """2x3 grid of learned / GT / difference for P(Y|do(X)) and P(Y|X)."""
    eval_cfg = cfg["evaluation"]
    hist_bins = eval_cfg.get("hist_bins", 100)
    x_res = HEATMAP_X_RES
    grid_x, x_edges, x_range, y_range = _derive_plot_axes(data_info, eval_cfg, x_res)
    y_edges = np.linspace(y_range[0], y_range[1], hist_bins + 1)
    grid_y = (y_edges[:-1] + y_edges[1:]) / 2.0
    y_res = len(grid_y)

    X, Y = np.meshgrid(grid_x, grid_y)

    pts = np.zeros((x_res * y_res, data_info["n_vars"]))
    pts[:, data_info["x_id"]] = X.flatten()
    pts[:, data_info["y_id"]] = Y.flatten()
    for zid in data_info["z_ids"]:
        pts[:, zid] = data_info["data_mean"][zid]
    pts_t = torch.tensor(pts, dtype=torch.float32, device=device)

    # The compiled do-circuit materializes ~N^4 floats per row in its bottom
    # ProductWeights broadcast, so the configured batch (sized for the config's
    # original node count) must be capped by the same device-aware budget the
    # NLL passes use — at N=64, 1024 rows would ask for ~64 GiB.
    eval_batch = resolve_eval_chunk_rows(cfg, device)
    circ_do = np.exp(_eval_log_probs(query_acs["q_do_ac"], pts_t, eval_batch)).reshape(y_res, x_res)
    log_num = _eval_log_probs(query_acs["obs_num_ac"], pts_t, eval_batch)
    log_den = _eval_log_probs(query_acs["obs_den_ac"], pts_t, eval_batch)
    circ_obs = np.exp(log_num - log_den).reshape(y_res, x_res)

    df_obs = data_info["df_obs"]
    df_do = data_info["df_do"]
    min_x_count = eval_cfg["heatmap_min_x_count"]
    gt_obs, _ = py_given_x_surface_hist(df_obs, "X", "Y", x_edges, y_edges, min_x_count=min_x_count)
    gt_do, _ = py_given_x_surface_hist(df_do, "X", "Y", x_edges, y_edges, min_x_count=min_x_count)

    x_mask_obs = ~np.isnan(gt_obs[0, :])
    x_mask_do = ~np.isnan(gt_do[0, :])
    circ_obs_masked = circ_obs.copy()
    circ_do_masked = circ_do.copy()
    circ_obs_masked[:, ~x_mask_obs] = np.nan
    circ_do_masked[:, ~x_mask_do] = np.nan

    diff_do = circ_do_masked - gt_do
    diff_obs = circ_obs_masked - gt_obs
    p99_diff = np.nanpercentile(
        np.concatenate([np.abs(diff_do).ravel(), np.abs(diff_obs).ravel()]),
        HEATMAP_DIFF_PERCENTILE,
    )
    diff_norm = Normalize(vmin=-p99_diff, vmax=p99_diff)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    density_vmin = np.nanmin([circ_do_masked, gt_do, circ_obs_masked, gt_obs])
    pooled = np.concatenate(
        [a[np.isfinite(a)] for a in (circ_do_masked, gt_do, circ_obs_masked, gt_obs)]
    )
    # Histograms can have single-sample spikes in the tails; cap the colour
    # scale at the 99.9th percentile so those cells saturate instead of
    # compressing the bulk of the density to black.
    density_vmax = max(float(np.percentile(pooled, 99.9)), density_vmin + 1e-12)
    density_norm = PowerNorm(gamma=HEATMAP_DENSITY_GAMMA, vmin=density_vmin, vmax=density_vmax)

    extent = [x_edges[0], x_edges[-1], grid_y[0], grid_y[-1]]
    mask_color = "lightgray"

    for ax in axes.flat:
        ax.set_xlim(x_range)

    # for ax in axes.flat:
    #     for x_split in get_x_split_points(ac, data_info["x_id"]):
    #         ax.axvline(x_split, color="white", linestyle="--", linewidth=1, alpha=0.5)

    cmap_do = plt.cm.viridis.with_extremes(bad=mask_color)
    cmap_diff = plt.cm.coolwarm.with_extremes(bad=mask_color)

    panels = [
        (circ_do_masked, cmap_do, density_norm, "Learned P(Y|do(X))"),
        (gt_do, cmap_do, density_norm, "Ground Truth P(Y|do(X))"),
        (diff_do, cmap_diff, diff_norm, "Difference (Learned - GT)"),
        (circ_obs_masked, cmap_do, density_norm, "Learned P(Y|X)"),
        (gt_obs, cmap_do, density_norm, "Ground Truth P(Y|X)"),
        (diff_obs, cmap_diff, diff_norm, "Difference (Learned - GT)"),
    ]
    for ax, (data, cmap, norm, title) in zip(axes.flat, panels):
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    out_file = os.path.join(
        output_dir,
        f"{exp_id}_heatmap_do_{cfg['dataset']['dataset']}_"
        f"N{cfg['model']['num_nodes']}_{cfg['model']['md_sets']}.png",
    )
    plt.savefig(out_file, dpi=150)
    plt.close()
    logger.info(f"Saved 2D heatmap grid to {out_file}")
