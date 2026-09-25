"""Cross-model slice comparison: the same Y-density slice for every model.

Renders one panel per config (default: the 400K backdoor N-series N=8..128)
showing the learned and analytical P(Y|X=x) / P(Y|do(X=x)) at a fixed X value
(default 3.5) for a single seed (--seed, default 0) -- heatmaps and slices
are single-seed visuals, aggregates live in the line plots. The panels share
the slice script's Okabe-Ito color palette; the bottom-right cell of the grid
carries the shared legend instead of a panel.

Example:
    python -m experiments.plot_slice_comparison \
        --x-value 3.5 --seed 0 --output-dir experiments/results/figures
"""

import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from experiments.plot_results import valid_seed_dirs
from experiments.plot_x_support_slices import (
    analytical_densities_at_x,
    build_y_grid,
    circuit_densities_at_x,
    load_ground_truth,
)
from experiments.plotting.artifacts import load_trained_artifacts
from experiments.plotting.styles import apply_paper_style


N_SERIES = [
    ("backdoor_cont_400K_N8", "N=8"),
    ("backdoor_cont_400K_N16", "N=16"),
    ("backdoor_cont_400K_N32", "N=32"),
    ("backdoor_cont_400K_N64", "N=64"),
    ("backdoor_cont_400K_N128", "N=128"),
]
DATA_SERIES = [
    ("backdoor_cont_25K_N64", "25K"),
    ("backdoor_cont_100K_N64", "100K"),
    ("backdoor_cont_400K_N64", "400K"),
    ("backdoor_cont_1200K_N64", "1.2M"),
]

# Same Okabe-Ito palette as plot_x_support_slices.py: saturated hues for the
# circuit, light hues for the analytical GT; solid = P(Y|X), dashed = do.
LINE_KWARGS = {
    "obs_circuit": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.8},
    "do_circuit": {"color": "#D55E00", "linestyle": "-", "linewidth": 1.8},
    "obs_gt": {"color": "#56B4E9", "linestyle": "--", "linewidth": 1.5},
    "do_gt": {"color": "#E69F00", "linestyle": "--", "linewidth": 1.5},
}
LEGEND_ENTRIES = [
    (r"learned $P(Y\mid X)$", LINE_KWARGS["obs_circuit"]),
    (r"learned $P(Y\mid do(X))$", LINE_KWARGS["do_circuit"]),
    (r"analytical $P(Y\mid X)$", LINE_KWARGS["obs_gt"]),
    (r"analytical $P(Y\mid do(X))$", LINE_KWARGS["do_gt"]),
]


def mean_densities_at_x(results_root, config, x_val, grid_y, device, gt_context=None, seed=0):
    """Single-seed circuit densities at x_val; GT added unless given externally.

    ``gt_context`` is a (gt, x_name, y_name) triple; when None it is built from
    the config's own SCM (used when every panel has its own GT). Heatmaps and
    slices are single-seed visuals; aggregates live in the line plots.
    """
    obs, do = [], []
    seed_dirs = valid_seed_dirs(results_root, config)
    seed_dirs = [d for d in seed_dirs if os.path.basename(d) == f"seed_{seed}"]
    if not seed_dirs:
        raise SystemExit(f"{config}: no valid seed_{seed} dir under {results_root}")
    for seed_dir in seed_dirs:
        _cfg, _ac, data_info, query_acs, dev = load_trained_artifacts(seed_dir, device=device)
        circ = circuit_densities_at_x(x_val, grid_y, data_info, query_acs, dev)
        obs.append(circ["obs"])
        do.append(circ["do"])
        if gt_context is None:
            gt = load_ground_truth(data_info)
            id_to_var = {v: k for k, v in data_info["var_to_id"].items()}
            gt_context = (
                gt,
                id_to_var[data_info["x_id"]],
                id_to_var[data_info["y_id"]],
            )
    out = {"obs_circuit": np.mean(obs, axis=0), "do_circuit": np.mean(do, axis=0)}
    gt, x_name, y_name = gt_context
    if gt is not None:
        gt_dens = analytical_densities_at_x(gt, x_val, grid_y, x_name, y_name)
        out["obs_gt"] = gt_dens["obs"]
        out["do_gt"] = gt_dens["do"]
    return out, seed


def plot_slice_comparison(
    results_root,
    output_dir,
    configs,
    x_val,
    device,
    y_bins=400,
    gt_config=None,
    output_name=None,
    seed=0,
):
    # The Y grid (and the GT unless --gt-config says otherwise) comes from the
    # first config; with --gt-config every panel is compared against that
    # config's analytical GT (e.g. 400K for the data-scaling comparison).
    first_cfg, _ = configs[0]
    first_seed = valid_seed_dirs(results_root, first_cfg)[0]
    _cfg, _ac, data_info, _queries, _dev = load_trained_artifacts(first_seed, device=device)
    grid_y = build_y_grid(data_info, y_bins=y_bins)

    gt = None
    gt_context = None
    if gt_config is not None:
        gt_seed = valid_seed_dirs(results_root, gt_config)[0]
        _gcfg, _gac, gt_info, _gq, _gd = load_trained_artifacts(gt_seed, device=device)
        gt = load_ground_truth(gt_info)
        if gt is None:
            raise SystemExit(f"{gt_config}: no SCM pickle found; analytical GT unavailable")
        id_to_var = {v: k for k, v in gt_info["var_to_id"].items()}
        gt_context = (
            gt,
            id_to_var[gt_info["x_id"]],
            id_to_var[gt_info["y_id"]],
        )

    n = len(configs)
    # +1 cell for the shared legend; keep the grid compact for few configs.
    n_cols = 3 if n >= 3 else n + 1
    n_rows = int(np.ceil((n + 1) / n_cols))
    figsize = (6.6, 3.9) if n >= 3 else (2.2 * n_cols, 2.6)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat = axes.flatten()

    x_lo, x_hi = np.inf, -np.inf
    for ax, (config, label) in zip(flat, configs):
        curves, used_seed = mean_densities_at_x(
            results_root, config, x_val, grid_y, device, gt_context=gt_context, seed=seed
        )
        for key, density in curves.items():
            ax.plot(grid_y, density, **LINE_KWARGS[key])
        curve_sum = sum(np.nan_to_num(d) for d in curves.values())
        mask = curve_sum > max(curve_sum.max() * 1e-3, 1e-12)
        idx = np.nonzero(mask)[0]
        if len(idx) > 1:
            x_lo, x_hi = min(x_lo, grid_y[idx[0]]), max(x_hi, grid_y[idx[-1]])
        ax.set_title(f"{label}  (seed {used_seed})", fontsize=9)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if np.isfinite(x_lo) and x_hi > x_lo:
        pad = 0.03 * (x_hi - x_lo)
        for ax in flat[:n]:
            ax.set_xlim(x_lo - pad, x_hi + pad)

    # Bottom-right cell: shared legend, no data panel.
    legend_ax = flat[n]
    legend_ax.axis("off")
    legend_ax.legend(
        [plt.Line2D([0], [0], **kw) for _, kw in LEGEND_ENTRIES],
        [lab for lab, _ in LEGEND_ENTRIES],
        loc="center",
        fontsize=9,
        frameon=False,
    )
    for ax in flat[n + 1 :]:
        ax.axis("off")

    for ax in flat[:n_cols]:
        ax.set_xlabel("Y", fontsize=9)
    for row in range(n_rows):
        flat[row * n_cols].set_ylabel("density", fontsize=9)

    os.makedirs(output_dir, exist_ok=True)
    name = output_name or f"fig_slice_comparison_X{x_val:g}"
    path = os.path.join(output_dir, name + ".png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/results")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--x-value", type=float, default=3.5)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[c for c, _ in N_SERIES],
        help="Config IDs, plotted in argument order (default: the 400K N-series).",
    )
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument(
        "--gt-config",
        default=None,
        help="Config whose SCM provides the analytical GT for every panel "
        "(default: each panel's own dataset GT).",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Figure filename stem (default: fig_slice_comparison_X<x>).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed whose model is plotted (default: 0); slices are single-seed visuals.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for circuit evaluation (default: cuda if available).",
    )
    args = parser.parse_args()

    # N_SERIES and DATA_SERIES collide on backdoor_cont_400K_N64; pick the
    # label map by which family the requested configs mostly belong to.
    n_hits = sum(c in dict(N_SERIES) for c in args.configs)
    d_hits = sum(c in dict(DATA_SERIES) for c in args.configs)
    label_map = dict(N_SERIES) if n_hits >= d_hits else dict(DATA_SERIES)
    labels = args.labels or [label_map.get(c, c) for c in args.configs]
    output_dir = args.output_dir or os.path.join(args.results_root, "figures")

    apply_paper_style()
    plot_slice_comparison(
        args.results_root,
        output_dir,
        list(zip(args.configs, labels)),
        args.x_value,
        torch.device(args.device),
        gt_config=args.gt_config,
        output_name=args.output_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
