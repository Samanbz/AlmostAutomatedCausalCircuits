"""Cross-model heatmap comparison: every model on one shared (X, Y) grid.

Renders an N_rows x 2 figure: the top row is the analytical ground truth
(P(Y|X), P(Y|do(X))); each following row is one config's learned surfaces for
a single seed (--seed, default 0) -- heatmaps are single-seed visuals,
aggregates live in the line plots. All panels share one color normalization
so densities are comparable across rows.

Two canonical uses:
  * --tag N   --configs backdoor_cont_400K_N8 ... N128   (capacity sweep)
  * --tag data --gt-config backdoor_cont_400K_N64 \
        --configs backdoor_cont_25K_N64 ... 1200K_N64    (data scaling)

Example:
    python -m experiments.plot_heatmap_comparison --tag N \
        --configs backdoor_cont_400K_N8 backdoor_cont_400K_N16 \
                  backdoor_cont_400K_N32 backdoor_cont_400K_N64 \
                  backdoor_cont_400K_N128
"""

import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm

from experiments.plot_heatmaps import analytical_surfaces, load_ground_truth
from experiments.plot_results import valid_seed_dirs
from experiments.plotting.artifacts import load_trained_artifacts
from experiments.plotting.styles import apply_paper_style
from experiments.utils.evaluation import _eval_on_grid, _grids_from_data


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
COLUMN_TITLES = [r"$P(Y\mid X)$", r"$P(Y\mid do(X))$"]


def surfaces_at_grid(results_root, config, x_grid, y_grid, device, seed=0):
    """Single-seed learned (obs, do) densities on the shared grid."""
    seed_dirs = [
        d for d in valid_seed_dirs(results_root, config) if os.path.basename(d) == f"seed_{seed}"
    ]
    if not seed_dirs:
        raise SystemExit(f"{config}: no valid seed_{seed} dir under {results_root}")
    cfg, _ac, data_info, query_acs, dev = load_trained_artifacts(seed_dirs[0], device=device)
    log_p_do, log_p_obs = _eval_on_grid(query_acs, data_info, x_grid, y_grid, dev, cfg)
    return np.exp(log_p_obs), np.exp(log_p_do), seed


def plot_heatmap_comparison(
    results_root,
    output_dir,
    configs,
    gt_config,
    tag,
    device,
    gamma=0.5,
    seed=0,
):
    # Reference grid + GT from the gt-config's dataset (all panels must align).
    gt_seed = valid_seed_dirs(results_root, gt_config)[0]
    gt_cfg, _ac, gt_info, _q, gt_dev = load_trained_artifacts(gt_seed, device=device)
    x_grid, y_grid, _ = _grids_from_data(gt_info, gt_cfg.get("evaluation", {}))
    id_to_var = {v: k for k, v in gt_info["var_to_id"].items()}
    gt = load_ground_truth(gt_info)
    if gt is None:
        raise SystemExit(f"{gt_config}: no SCM pickle found; analytical GT unavailable")
    gt_obs, gt_do = analytical_surfaces(
        gt, x_grid, y_grid, id_to_var[gt_info["x_id"]], id_to_var[gt_info["y_id"]]
    )

    rows = [("analytical GT", gt_obs, gt_do, None)]
    for config, label in configs:
        obs, do, used_seed = surfaces_at_grid(
            results_root, config, x_grid, y_grid, device, seed=seed
        )
        rows.append((f"{label}  (seed {used_seed})", obs, do, used_seed))

    n_rows = len(rows)
    # One full page for the N-series (6 rows); proportionally shorter otherwise.
    fig_h = 9.0 if n_rows >= 6 else 1.5 * n_rows + 0.5
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(6.8, fig_h), squeeze=False, constrained_layout=True
    )

    vmax = max(float(np.nanmax(s)) for _l, o, d, _n in rows for s in (o, d))
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)
    cmap = plt.cm.viridis
    extent = [
        x_grid[0] - 0.5 * float(np.diff(x_grid).min()),
        x_grid[-1] + 0.5 * float(np.diff(x_grid).min()),
        y_grid[0] - 0.5 * float(np.diff(y_grid).min()),
        y_grid[-1] + 0.5 * float(np.diff(y_grid).min()),
    ]

    ims = []
    for r, (label, obs, do, _n) in enumerate(rows):
        for col, surface in enumerate((obs, do)):
            ax = axes[r][col]
            im = ax.imshow(
                surface.T,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            ims.append(im)
            if r == 0:
                ax.set_title(COLUMN_TITLES[col], fontsize=10)
                if col == 0:
                    ax.text(
                        0.02,
                        0.95,
                        "analytical GT",
                        transform=ax.transAxes,
                        color="white",
                        fontsize=9,
                        va="top",
                    )
            elif col == 0:
                ax.set_title(label, fontsize=9, loc="left")
            if r < n_rows - 1:
                ax.tick_params(labelbottom=False)
            if col > 0:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=8)

    fig.supxlabel("X", fontsize=10)
    fig.supylabel("Y", fontsize=10)
    fig.colorbar(ims[0], ax=axes, shrink=0.45, label="density")

    os.makedirs(output_dir, exist_ok=True)
    name = f"fig_heatmap_comparison_{tag}"
    path = os.path.join(output_dir, name + ".png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/results")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--tag",
        required=True,
        help="Figure suffix, e.g. N or data (fig_heatmap_comparison_<tag>.png).",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[c for c, _ in N_SERIES],
        help="Config IDs, one row each in argument order (default: the 400K N-series).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Row labels (default: the known N-series / data-size labels).",
    )
    parser.add_argument(
        "--gt-config",
        default=None,
        help="Config whose dataset/SCM provides the shared grid and the top-row "
        "GT (default: the first of --configs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed whose model is plotted (default: 0); heatmaps are single-seed visuals.",
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
    gt_config = args.gt_config or args.configs[0]
    output_dir = args.output_dir or os.path.join(args.results_root, "figures")

    apply_paper_style()
    plot_heatmap_comparison(
        args.results_root,
        output_dir,
        list(zip(args.configs, labels)),
        gt_config,
        args.tag,
        torch.device(args.device),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
