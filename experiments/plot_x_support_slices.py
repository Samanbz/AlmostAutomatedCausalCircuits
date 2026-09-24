"""Plot 1-D Y densities per X-leaf support for a trained experiment.

For each X-leaf support interval of the trained circuit, the script produces a
subpanel showing:
  - learned (circuit) P(Y|X=x) and P(Y|do(X=x))
  - analytical ground truth from the pickled SCM
  - a thin vertical marker at the mean of each curve

Every panel carries its own legend, so any cell can be lifted into the paper
as-is.  Colors are colorblind-safe (Okabe-Ito) and grayscale-distinguishable:
saturated hues for the circuit, light hues for the analytical GT; blue family
for P(Y|X), red/orange family for P(Y|do(X)).

Example:
    python -m experiments.plot_x_support_slices \
        --results-dir experiments/results/backdoor_cont_Z2_100K/seed_27
"""

import argparse
import os
import pickle
from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt

from experiments.plotting.artifacts import _seed_from_dir, load_trained_artifacts
from experiments.plotting.styles import apply_paper_style
from experiments.plotting.x_support import get_x_support_contexts
from experiments.utils.data import discrete_categories
from src.symbolic.arithmetic.circuit import eval_circuit


def build_y_grid(
    data_info: Dict[str, Any],
    y_bins: int = 200,
    quantile_range: tuple = (0.0, 1.0),
    y_cats: list = None,
) -> np.ndarray:
    """Y grid: observed categories for discrete Y, else an even-spaced grid over
    the pooled observational/interventional Y range."""
    if y_cats is not None:
        return np.asarray(y_cats, dtype=np.float64)
    y_obs = data_info["df_obs_full"]["Y"].values
    y_do = data_info["df_do_full"]["Y"].values
    pooled = np.concatenate([y_obs, y_do])
    lo, hi = np.quantile(pooled, quantile_range)
    return np.linspace(float(lo), float(hi), y_bins)


def load_ground_truth(data_info: Dict[str, Any]):
    """Return the SCM's exact GroundTruth engine, or None if no SCM pickle exists."""
    scm_path = data_info.get("scm_path")
    if scm_path is None or not os.path.exists(scm_path):
        return None
    with open(scm_path, "rb") as f:
        scm = pickle.load(f)
    return scm.ground_truth()


def analytical_densities_at_x(
    gt,
    x_val: float,
    grid_y: np.ndarray,
    x_name: str,
    y_name: str,
    y_discrete: bool = False,
) -> Dict[str, np.ndarray]:
    """Exact P(Y|X=x) and P(Y|do(X=x)) from the SCM ground-truth engine."""
    if y_discrete:
        # marginal_density refuses discrete targets; marginal_prob gives exact
        # category probabilities (and accepts discrete evidence).
        obs = [gt.marginal_prob({y_name: v}, evidence={x_name: float(x_val)}) for v in grid_y]
        do = [gt.marginal_prob({y_name: v}, do={x_name: float(x_val)}) for v in grid_y]
        return {"obs": np.asarray(obs), "do": np.asarray(do)}
    obs = gt.marginal_density([y_name], evidence={x_name: float(x_val)})
    do = gt.marginal_density([y_name], do={x_name: float(x_val)})
    return {"obs": np.asarray(obs.pdf(grid_y)), "do": np.asarray(do.pdf(grid_y))}


def circuit_densities_at_x(
    x_val: float,
    grid_y: np.ndarray,
    data_info: Dict[str, Any],
    query_acs: Dict[str, Any],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Evaluate learned P(Y|X=x) and P(Y|do(X=x)) on ``grid_y``."""
    n_y = len(grid_y)
    pts = np.zeros((n_y, data_info["n_vars"]), dtype=np.float32)
    pts[:, data_info["x_id"]] = x_val
    pts[:, data_info["y_id"]] = grid_y
    for zid in data_info["z_ids"]:
        pts[:, zid] = data_info["data_mean"][zid]
    pts_t = torch.tensor(pts, dtype=torch.float32, device=device)

    obs_num_ac = query_acs["obs_num_ac"]
    obs_den_ac = query_acs["obs_den_ac"]
    q_do_ac = query_acs["q_do_ac"]

    with torch.no_grad():
        log_num = eval_circuit(obs_num_ac, pts_t, verbose=False, keep_intermediates=False).reshape(
            -1
        )
        log_do = eval_circuit(q_do_ac, pts_t, verbose=False, keep_intermediates=False).reshape(-1)
        # P(X) is independent of Y: a single row is enough.
        den_pts = torch.tensor(pts[0:1], dtype=torch.float32, device=device)
        log_den = eval_circuit(
            obs_den_ac, den_pts, verbose=False, keep_intermediates=False
        ).reshape(-1)

    p_obs = np.exp(log_num.cpu().numpy() - log_den.cpu().numpy())
    p_do = np.exp(log_do.cpu().numpy())
    return {"obs": p_obs, "do": p_do}


def _curve_mean(grid_y: np.ndarray, density: np.ndarray, discrete: bool = False) -> float:
    """Mean of a density/pmf curve over the grid.

    Continuous: integral y p(y) dy / integral p(y) dy (trapezoid).  Discrete:
    the grid points are the categories and the curve a pmf, so the mean is the
    plain weighted average sum(y p) / sum(p).
    """
    d = np.nan_to_num(density)
    if discrete:
        mass = float(d.sum())
        if mass <= 0:
            return float("nan")
        return float((grid_y * d).sum() / mass)
    mass = float(np.sum(0.5 * (d[1:] + d[:-1]) * np.diff(grid_y)))
    if mass <= 0:
        return float("nan")
    moment = float(np.sum(0.5 * (grid_y[1:] * d[1:] + grid_y[:-1] * d[:-1]) * np.diff(grid_y)))
    return moment / mass


def plot_x_support_slices(
    data_info: Dict[str, Any],
    query_acs: Dict[str, Any],
    x_values: np.ndarray,
    labels: list,
    grid_y: np.ndarray,
    gt,
    device: torch.device,
    output_path: str,
    sharey: bool = False,
    y_discrete: bool = False,
    figsize=None,
) -> None:
    """Create the multi-panel slice figure and save it."""
    n = len(x_values)
    n_cols = 2 if n > 1 else 1
    n_rows = int(np.ceil(n / n_cols))

    x_name = [k for k, v in data_info["var_to_id"].items() if v == data_info["x_id"]][0]
    y_name = [k for k, v in data_info["var_to_id"].items() if v == data_info["y_id"]][0]

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize or (4.0 * n_cols, 3 * n_rows),
        squeeze=False,
        # Per-panel x-limits are chosen so every curve is fully visible, so
        # axes cannot be shared.
        sharex=False,
        sharey=sharey,
    )
    axes = axes.flatten()

    # Colorblind-safe (Okabe-Ito) and grayscale-distinguishable: saturated hues
    # for the circuit, light hues for the analytical GT; solid = P(Y|X),
    # dashed = P(Y|do(X)).
    blue, vermillion = "#0072B2", "#D55E00"
    sky_blue, orange = "#56B4E9", "#E69F00"
    line_kwargs = {
        "obs_circuit": {"color": blue, "linestyle": "-", "linewidth": 2.0},
        "do_circuit": {"color": vermillion, "linestyle": "-", "linewidth": 2.0},
        "obs_gt": {"color": sky_blue, "linestyle": "--", "linewidth": 1.6},
        "do_gt": {"color": orange, "linestyle": "--", "linewidth": 1.6},
    }
    mean_kwargs = {
        "obs_circuit": {"color": blue, "linestyle": ":", "linewidth": 1.2},
        "do_circuit": {"color": vermillion, "linestyle": ":", "linewidth": 1.2},
        "obs_gt": {"color": sky_blue, "linestyle": ":", "linewidth": 1.2},
        "do_gt": {"color": orange, "linestyle": ":", "linewidth": 1.2},
    }
    legend_entries = [
        (r"$P(Y\mid X)$ circuit", line_kwargs["obs_circuit"]),
        (r"$P(Y\mid do(X))$ circuit", line_kwargs["do_circuit"]),
        (r"$P(Y\mid X)$ analytical GT", line_kwargs["obs_gt"]),
        (r"$P(Y\mid do(X))$ analytical GT", line_kwargs["do_gt"]),
        ("curve mean", {"color": "0.3", "linestyle": ":", "linewidth": 1.2}),
    ]

    panel_peaks = []
    for ax, x_val, label in zip(axes, x_values, labels):
        circ = circuit_densities_at_x(x_val, grid_y, data_info, query_acs, device)
        curves = [("obs_circuit", circ["obs"]), ("do_circuit", circ["do"])]
        if gt is not None:
            gt_dens = analytical_densities_at_x(
                gt, x_val, grid_y, x_name, y_name, y_discrete=y_discrete
            )
            curves += [("obs_gt", gt_dens["obs"]), ("do_gt", gt_dens["do"])]

        # Discrete Y: the curves are pmfs over the category grid — render dodged
        # bars per category instead of connecting lines.
        if y_discrete:
            step = float(np.diff(grid_y).min()) if len(grid_y) > 1 else 1.0
            n_curves = len(curves)
            width = 0.8 * step / n_curves
        peaks = []
        for j, (key, density) in enumerate(curves):
            if y_discrete:
                ax.bar(
                    grid_y + (j - (n_curves - 1) / 2) * width,
                    density,
                    width=width,
                    color=line_kwargs[key]["color"],
                    linestyle=line_kwargs[key]["linestyle"],
                    linewidth=0.8,
                    edgecolor=line_kwargs[key]["color"],
                    fill=True,
                    alpha=0.75,
                )
            else:
                ax.plot(grid_y, density, **line_kwargs[key])
            mean = _curve_mean(grid_y, density, discrete=y_discrete)
            if np.isfinite(mean):
                ax.axvline(mean, alpha=0.7, **mean_kwargs[key])
            peaks.append(float(np.nan_to_num(density).max()))

        # Self-contained cell: every panel carries the full legend.
        entries = legend_entries if gt is not None else legend_entries[:3]
        ax.legend(
            [plt.Line2D([0], [0], **kw) for _, kw in entries],
            [lab for lab, _ in entries],
            fontsize=7,
            loc="best",
            framealpha=0.9,
        )

        # Data-driven limits: show wherever any curve carries non-negligible
        # mass; y starts at 0 with 12% headroom over the highest peak.
        curve_sum = sum(np.nan_to_num(d) for _, d in curves)
        mask = curve_sum > max(curve_sum.max() * 1e-3, 1e-12)
        idx = np.nonzero(mask)[0]
        if len(idx) > 1:
            pad = 0.03 * (grid_y[idx[-1]] - grid_y[idx[0]])
            ax.set_xlim(grid_y[idx[0]] - pad, grid_y[idx[-1]] + pad)
        peak_y = max(peaks)
        ax.set_ylim(0.0, peak_y * 1.12)
        panel_peaks.append(peak_y)

        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    if sharey and panel_peaks:
        shared_top = 1.12 * max(panel_peaks)
        for ax in axes[:n]:
            ax.set_ylim(0.0, shared_top)
            ax.sharey(axes[0])

    # Hide unused panels.
    for ax in axes[n:]:
        ax.axis("off")

    for ax in axes[:n]:
        ax.set_xlabel("Y", fontsize=10)
        ax.set_ylabel("probability" if y_discrete else "density", fontsize=10)

    fig.tight_layout(pad=0.6)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 1-D Y densities at one X value per leaf support."
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
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output filename (default: <exp_id>_seed<S>_x_support_slices.png).",
    )
    parser.add_argument(
        "--y-bins",
        type=int,
        default=None,
        help="Number of Y grid points for the circuit and analytical curves "
        "(default: evaluation.slice_y_points from the run config, else 200).",
    )
    parser.add_argument(
        "--y-quantile-range",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="Quantile range of pooled Y used to set the Y axis (default: 0.0-1.0, "
        "i.e. the full pooled range — matching explore_synthetic_data.py).",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=None,
        help="Plot at most this many X supports (evenly subsampled).",
    )
    parser.add_argument(
        "--sharey",
        action="store_true",
        help="Use a shared Y-axis across all panels.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Figure size in inches, width height (default: 4x3 per panel).",
    )
    args = parser.parse_args()

    apply_paper_style()

    cfg, ac, data_info, query_acs, device = load_trained_artifacts(args.results_dir)

    eval_cfg = cfg.get("evaluation", {})
    y_bins = args.y_bins if args.y_bins is not None else int(eval_cfg.get("slice_y_points", 400))

    id_to_var = {v: k for k, v in data_info["var_to_id"].items()}
    x_name = id_to_var[data_info["x_id"]]
    y_name = id_to_var[data_info["y_id"]]
    x_cats = discrete_categories(cfg, data_info, x_name)
    y_cats = discrete_categories(cfg, data_info, y_name)

    if x_cats is not None:
        # Discrete X: one panel per observed category (leaf "support intervals"
        # are DiscreteIntervals, not usable as continuous ranges).
        x_values = np.asarray(x_cats, dtype=np.float64)
        labels = [f"X = {v:g}" for v in x_cats]
    else:
        x_values, labels, _ = get_x_support_contexts(ac, data_info, df_obs=data_info["df_obs_full"])
    if args.max_contexts is not None and len(x_values) > args.max_contexts:
        idx = np.unique(np.linspace(0, len(x_values) - 1, args.max_contexts).round().astype(int))
        x_values, labels = x_values[idx], [labels[i] for i in idx]

    grid_y = build_y_grid(
        data_info,
        y_bins=y_bins,
        quantile_range=tuple(args.y_quantile_range),
        y_cats=y_cats,
    )

    gt = load_ground_truth(data_info)
    if gt is None:
        print("WARNING: no SCM pickle found next to the dataset; analytical GT curves omitted.")

    exp_id = cfg["experiment"]["id"]
    seed = _seed_from_dir(args.results_dir)
    output_file = args.output_file or f"{exp_id}_seed{seed}_x_support_slices.png"
    output_path = os.path.join(args.output_dir or args.results_dir, output_file)

    plot_x_support_slices(
        data_info,
        query_acs,
        x_values,
        labels,
        grid_y,
        gt,
        device,
        output_path,
        sharey=args.sharey,
        y_discrete=y_cats is not None,
        figsize=tuple(args.figsize) if args.figsize is not None else None,
    )
    print(f"Saved slice plot to {output_path}")


if __name__ == "__main__":
    main()
