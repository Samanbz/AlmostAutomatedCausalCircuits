"""2x2 heatmap comparison of learned vs. analytical ground-truth densities.

For a trained experiment, plot ``P(Y|X)`` and ``P(Y|do(X))`` on a shared
``(X, Y)`` grid: the left column is the learned (compiled query) circuits, the
right column is the exact analytical ground truth from the pickled SCM
(``scm.ground_truth()``).  No difference panels — the figure is the direct
visual comparison, at the paper's (A4) proportions.

Example:
    python -m experiments.plot_heatmaps \
        --results-dir experiments/results/backdoor_cont_Z2_100K/seed_27 \
        --output-dir scratch/plots
"""

import argparse
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm

from experiments.plotting.artifacts import _seed_from_dir, load_trained_artifacts
from experiments.plotting.styles import apply_paper_style
from experiments.utils.data import discrete_categories
from experiments.utils.evaluation import _eval_on_grid, _grids_from_data
from experiments.utils.training import resolve_eval_chunk_rows


def load_ground_truth(data_info):
    """Return the SCM's exact GroundTruth engine, or None if no SCM pickle exists."""
    scm_path = data_info.get("scm_path")
    if scm_path is None or not os.path.exists(scm_path):
        return None
    with open(scm_path, "rb") as f:
        scm = pickle.load(f)
    return scm.ground_truth()


def analytical_surfaces(gt, x_grid, y_grid, x_name, y_name, y_discrete=False):
    """Exact P(Y|X=x) and P(Y|do(X=x)) on the (x, y) grid, shapes (n_x, n_y).

    Continuous Y: Gaussian-mixture density evaluated on the grid.  Discrete Y:
    exact category probabilities from ``marginal_prob`` (the GT engine refuses
    discrete targets in ``marginal_density``).
    """
    obs = np.empty((len(x_grid), len(y_grid)))
    do = np.empty((len(x_grid), len(y_grid)))
    for i, x in enumerate(x_grid):
        if y_discrete:
            obs[i] = [gt.marginal_prob({y_name: v}, evidence={x_name: float(x)}) for v in y_grid]
            do[i] = [gt.marginal_prob({y_name: v}, do={x_name: float(x)}) for v in y_grid]
        else:
            obs[i] = np.asarray(
                gt.marginal_density([y_name], evidence={x_name: float(x)}).pdf(y_grid)
            )
            do[i] = np.asarray(gt.marginal_density([y_name], do={x_name: float(x)}).pdf(y_grid))
    return obs, do


def _centered_extent(grid):
    """Half-step-padded axis limits so imshow cells center on the grid values."""
    step = float(np.diff(grid).min()) if len(grid) > 1 else 1.0
    return [grid[0] - 0.5 * step, grid[-1] + 0.5 * step]


def plot_heatmap_2x2(
    x_grid,
    y_grid,
    learned_obs,
    learned_do,
    gt_obs,
    gt_do,
    output_base,
    gamma=0.5,
    x_discrete=False,
    y_discrete=False,
    figsize=None,
):
    """Render the 2x2 comparison figure and save PNG + PDF."""
    extent = _centered_extent(x_grid) + _centered_extent(y_grid)
    surfaces = [learned_obs, gt_obs, learned_do, gt_do]
    vmax = max(float(np.nanmax(s)) for s in surfaces)
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)
    cmap = plt.cm.viridis

    titles = [
        [r"Learned $P(Y\mid X)$", r"Analytical $P(Y\mid X)$"],
        [r"Learned $P(Y\mid do(X))$", r"Analytical $P(Y\mid do(X))$"],
    ]
    data = [[learned_obs, gt_obs], [learned_do, gt_do]]

    fig, axes = plt.subplots(2, 2, figsize=figsize or (6, 4.5), constrained_layout=True)
    ims = []
    for row in range(2):
        for col in range(2):
            ax = axes[row][col]
            # Surfaces are indexed (x, y); imshow wants rows = Y, so transpose.
            im = ax.imshow(
                data[row][col].T,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap=cmap,
                norm=norm,
            )
            ims.append(im)
            ax.set_title(titles[row][col])
            ax.set_xlabel("X")
            if col == 0:
                ax.set_ylabel("Y")
            # Discrete axes: cells center on the categories — show them as ticks
            # instead of implying a continuous range between them.
            if x_discrete:
                ax.set_xticks(x_grid)
                ax.set_xticklabels([f"{v:g}" for v in x_grid])
            if y_discrete:
                ax.set_yticks(y_grid)
                ax.set_yticklabels([f"{v:g}" for v in y_grid])

    cbar_label = "probability" if y_discrete else "density"
    fig.colorbar(ims[0], ax=axes, shrink=0.85, label=cbar_label)
    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    fig.savefig(output_base + ".png")
    fig.savefig(output_base + ".pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2x2 heatmaps: learned vs. analytical P(Y|X) and P(Y|do(X))."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to the experiment results seed directory (e.g. experiments/results/<id>/seed_27).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the figure will be saved (default: the results seed directory).",
    )
    parser.add_argument("--output-file", default=None, help="Output filename stem.")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Figure size in inches, width height (default: 6 4.5).",
    )
    args = parser.parse_args()

    apply_paper_style()

    cfg, ac, data_info, query_acs, device = load_trained_artifacts(args.results_dir)

    gt = load_ground_truth(data_info)
    if gt is None:
        raise FileNotFoundError(
            "No SCM pickle found next to the dataset; analytical heatmaps need it."
        )

    id_to_var = {v: k for k, v in data_info["var_to_id"].items()}
    x_name, y_name = id_to_var[data_info["x_id"]], id_to_var[data_info["y_id"]]

    x_cats = discrete_categories(cfg, data_info, x_name)
    y_cats = discrete_categories(cfg, data_info, y_name)

    eval_cfg = cfg.get("evaluation", {})
    x_grid, y_grid, _ = _grids_from_data(data_info, eval_cfg)
    if x_cats is not None:
        # Discrete X: quantile grids interpolate fractional non-categories —
        # evaluate only at the true category values.
        x_grid = np.asarray(x_cats, dtype=np.float64)
    if y_cats is not None:
        y_grid = np.asarray(y_cats, dtype=np.float64)
    chunk_rows = resolve_eval_chunk_rows(cfg, device)
    log_p_do, log_p_obs = _eval_on_grid(query_acs, data_info, x_grid, y_grid, device, chunk_rows)
    learned_obs = np.exp(log_p_obs)
    learned_do = np.exp(log_p_do)

    gt_obs, gt_do = analytical_surfaces(
        gt, x_grid, y_grid, x_name, y_name, y_discrete=y_cats is not None
    )

    exp_id = cfg["experiment"]["id"]
    seed = _seed_from_dir(args.results_dir)
    output_file = args.output_file or f"{exp_id}_seed{seed}_heatmaps"
    output_base = os.path.join(args.output_dir or args.results_dir, output_file)

    plot_heatmap_2x2(
        x_grid,
        y_grid,
        learned_obs,
        learned_do,
        gt_obs,
        gt_do,
        output_base,
        gamma=float(eval_cfg.get("heatmap_density_gamma", 0.5)),
        x_discrete=x_cats is not None,
        y_discrete=y_cats is not None,
        figsize=tuple(args.figsize) if args.figsize is not None else None,
    )
    print(f"Saved heatmaps to {output_base}.png / .pdf")


if __name__ == "__main__":
    main()
