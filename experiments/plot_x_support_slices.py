"""Plot 1-D Y densities per X-leaf support for a trained experiment.

For each X-leaf support interval of the trained circuit, the script produces a
subpanel showing:
  - learned (circuit) P(Y|X=x) and P(Y|do(X=x)) — solid lines
  - empirical ground truth — plain density histograms of the samples whose X
    falls in the support interval (no KDE: unbiased, assumption-free)
  - analytical ground truth from the pickled SCM — dotted lines

Example:
    python -m experiments.plot_x_support_slices \
        --results-dir experiments/results/backdoor_cont_Z2_100K/seed_27 \
        --output-dir scratch/plots
"""

import argparse
import os
import pickle
from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt

from experiments.plotting.artifacts import load_trained_artifacts
from experiments.plotting.empirical import empirical_histograms_at_x
from experiments.plotting.styles import COLORS, apply_paper_style
from experiments.plotting.x_support import get_x_support_contexts
from src.symbolic.arithmetic.circuit import eval_circuit


def build_y_grid(
    data_info: Dict[str, Any],
    y_bins: int = 200,
    quantile_range: tuple = (0.0, 1.0),
) -> np.ndarray:
    """Build an even-spaced Y grid covering the pooled observational/interventional data."""
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
) -> Dict[str, np.ndarray]:
    """Exact P(Y|X=x) and P(Y|do(X=x)) from the SCM ground-truth engine.

    The GT engine is analytical, so these are exact pointwise densities at
    ``x_val`` — the exact counterpart of the circuit curves plotted at the
    same representative X value. (The empirical histograms pool samples over
    the whole support interval, so they estimate the interval-averaged
    density; small deviations from the pointwise curves are pure binning
    effects.)
    """
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


def plot_x_support_slices(
    data_info: Dict[str, Any],
    query_acs: Dict[str, Any],
    x_values: np.ndarray,
    labels: list,
    intervals: list,
    grid_y: np.ndarray,
    bin_edges: np.ndarray,
    gt,
    device: torch.device,
    output_path: str,
    show_legend: bool = True,
    sharey: bool = False,
) -> None:
    """Create the multi-panel slice figure and save it."""
    n = len(x_values)
    n_cols = 2 if n > 1 else 1
    n_rows = int(np.ceil(n / n_cols))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    x_name = [k for k, v in data_info["var_to_id"].items() if v == data_info["x_id"]][0]
    y_name = [k for k, v in data_info["var_to_id"].items() if v == data_info["y_id"]][0]

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 3.5 * n_rows),
        squeeze=False,
        # Per-panel x-limits are chosen so every curve is fully visible, so
        # axes cannot be shared.
        sharex=False,
        sharey=sharey,
    )
    axes = axes.flatten()

    line_kwargs = {
        "obs_circuit": {"color": COLORS["obs"], "linestyle": "-", "linewidth": 2.0},
        "do_circuit": {"color": COLORS["do"], "linestyle": "-", "linewidth": 2.0},
        "obs_emp": {"color": COLORS["obs"], "linestyle": "-", "linewidth": 1.3, "alpha": 0.65},
        "do_emp": {"color": COLORS["do"], "linestyle": "-", "linewidth": 1.3, "alpha": 0.65},
        "obs_gt": {"color": COLORS["obs"], "linestyle": ":", "linewidth": 1.8},
        "do_gt": {"color": COLORS["do"], "linestyle": ":", "linewidth": 1.8},
    }
    legend_entries = [
        (r"$P(Y\mid X)$ circuit", line_kwargs["obs_circuit"]),
        (r"$P(Y\mid do(X))$ circuit", line_kwargs["do_circuit"]),
        (r"$P(Y\mid X)$ empirical hist", line_kwargs["obs_emp"]),
        (r"$P(Y\mid do(X))$ empirical hist", line_kwargs["do_emp"]),
        (r"$P(Y\mid X)$ analytical GT", line_kwargs["obs_gt"]),
        (r"$P(Y\mid do(X))$ analytical GT", line_kwargs["do_gt"]),
    ]
    # The analytical GT curves are exact pointwise densities at the same
    # representative X value as the circuit curves (see
    # analytical_densities_at_x); the histograms pool over the support bin.

    panel_peaks = []
    for ax, x_val, label, interval in zip(axes, x_values, labels, intervals):
        circ = circuit_densities_at_x(x_val, grid_y, data_info, query_acs, device)
        emp = empirical_histograms_at_x(
            data_info["df_obs_full"],
            data_info["df_do_full"],
            interval,
            bin_edges,
        )
        gt_dens = (
            analytical_densities_at_x(gt, x_val, grid_y, x_name, y_name) if gt is not None else None
        )

        # Empirical ground truth: step histograms (drawstyle keeps bins honest).
        ax.plot(
            bin_centers,
            emp["obs"],
            drawstyle="steps-mid",
            **line_kwargs["obs_emp"],
        )
        ax.plot(
            bin_centers,
            emp["do"],
            drawstyle="steps-mid",
            **line_kwargs["do_emp"],
        )
        # Learned circuit densities (solid) and analytical GT (dotted).
        ax.plot(grid_y, circ["obs"], **line_kwargs["obs_circuit"])
        ax.plot(grid_y, circ["do"], **line_kwargs["do_circuit"])
        if gt_dens is not None:
            ax.plot(grid_y, gt_dens["obs"], **line_kwargs["obs_gt"])
            ax.plot(grid_y, gt_dens["do"], **line_kwargs["do_gt"])

        # Data-driven limits: show the entirety of every curve and histogram.
        # The x-range spans wherever any curve carries non-negligible mass or
        # any histogram bin is nonempty; the y-range starts at 0 and adds 12%
        # headroom over the highest peak.
        curve_sum = np.nan_to_num(circ["obs"]) + np.nan_to_num(circ["do"])
        if gt_dens is not None:
            curve_sum = curve_sum + np.nan_to_num(gt_dens["obs"]) + np.nan_to_num(gt_dens["do"])
        mask = curve_sum > max(curve_sum.max() * 1e-3, 1e-12)
        hist_nonzero = (emp["obs"] > 0) | (emp["do"] > 0)
        if hist_nonzero.any():
            lo_i, hi_i = np.nonzero(hist_nonzero)[0][[0, -1]]
            mask |= (grid_y >= bin_edges[lo_i]) & (grid_y <= bin_edges[hi_i + 1])
        idx = np.nonzero(mask)[0]

        peaks = [
            np.nan_to_num(circ["obs"]).max(),
            np.nan_to_num(circ["do"]).max(),
            np.nan_to_num(emp["obs"]).max(),
            np.nan_to_num(emp["do"]).max(),
        ]
        if gt_dens is not None:
            peaks += [np.nan_to_num(gt_dens["obs"]).max(), np.nan_to_num(gt_dens["do"]).max()]
        peak_y = max(peaks)

        if len(idx) > 1:
            pad = 0.03 * (grid_y[idx[-1]] - grid_y[idx[0]])
            ax.set_xlim(grid_y[idx[0]] - pad, grid_y[idx[-1]] + pad)
        ax.set_ylim(0.0, peak_y * 1.12)
        panel_peaks.append(peak_y)

        ax.set_title(
            f"{label}  ($n_{{obs}}$={emp['n_obs']}, $n_{{do}}$={emp['n_do']})",
            fontsize=10,
        )
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
        ax.set_ylabel("density", fontsize=10)

    if show_legend:
        legend_lines = [plt.Line2D([0], [0], **kw) for _, kw in legend_entries]
        legend_labels = [lab for lab, _ in legend_entries]
        fig.legend(
            legend_lines,
            legend_labels,
            loc="upper center",
            ncol=3,
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, 1.0),
        )

    # Reserve top margin for the figure legend so it never overlaps panel titles.
    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.94 if show_legend else 1))
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
        default="scratch/plots",
        help="Directory where the figure will be saved.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output filename (default: <exp_id>_seed<S>_x_support_slices.png).",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=None,
        help="Number of histogram bins for the empirical ground truth "
        "(default: evaluation.hist_bins from the run config, else 60).",
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
        "--no-legend",
        action="store_true",
        help="Omit the figure legend.",
    )
    args = parser.parse_args()

    apply_paper_style()

    cfg, ac, data_info, query_acs, device = load_trained_artifacts(args.results_dir)

    eval_cfg = cfg.get("evaluation", {})
    hist_bins = args.hist_bins if args.hist_bins is not None else int(eval_cfg.get("hist_bins", 60))
    y_bins = (
        args.y_bins if args.y_bins is not None else int(eval_cfg.get("slice_y_points", 400))
    )

    x_values, labels, intervals = get_x_support_contexts(
        ac, data_info, df_obs=data_info["df_obs_full"]
    )
    if args.max_contexts is not None and len(x_values) > args.max_contexts:
        idx = np.unique(np.linspace(0, len(x_values) - 1, args.max_contexts).round().astype(int))
        x_values, labels, intervals = (
            x_values[idx],
            [labels[i] for i in idx],
            [intervals[i] for i in idx],
        )

    grid_y = build_y_grid(data_info, y_bins=y_bins, quantile_range=tuple(args.y_quantile_range))
    # Shared histogram bin edges spanning the same Y range.
    bin_edges = np.linspace(grid_y[0], grid_y[-1], hist_bins + 1)

    gt = load_ground_truth(data_info)
    if gt is None:
        print("WARNING: no SCM pickle found next to the dataset; analytical GT curves omitted.")

    exp_id = cfg["experiment"]["id"]
    seed = int(os.path.basename(args.results_dir).split("_")[-1])
    output_file = args.output_file or f"{exp_id}_seed{seed}_x_support_slices.png"
    output_path = os.path.join(args.output_dir, output_file)

    plot_x_support_slices(
        data_info,
        query_acs,
        x_values,
        labels,
        intervals,
        grid_y,
        bin_edges,
        gt,
        device,
        output_path,
        show_legend=not args.no_legend,
        sharey=args.sharey,
    )
    print(f"Saved slice plot to {output_path}")


if __name__ == "__main__":
    main()
