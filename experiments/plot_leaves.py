"""Plot every learned leaf distribution for a trained experiment.

Each leaf gets its own figure. The learned per-node / per-group densities are
drawn by the existing visualization helpers in ``src.utils.visualization``, and
the empirical marginal of the corresponding variable (from the full
observational dataset) is overlaid as a black step histogram for reference.

Example:
    python -m experiments.plot_leaves \
        --results-dir experiments/results/backdoor_cont_Z2_100K/seed_27 \
        --output-dir scratch/plots/leaves
"""

import argparse
import os
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from experiments.plotting.artifacts import load_trained_artifacts
from experiments.plotting.styles import apply_paper_style
from src.symbolic.arithmetic.nodes.leaf_layer import (
    GaussianLeafLayer,
    MixtureLeafLayer,
    SplineLeafLayer,
)
from src.utils.visualization import plot_gaussian_leaf, plot_mixture_leaf, plot_spline_leaf


def _leaf_type_name(leaf) -> str:
    if isinstance(leaf, GaussianLeafLayer):
        return "gaussian"
    if isinstance(leaf, MixtureLeafLayer):
        return "mixture"
    if isinstance(leaf, SplineLeafLayer):
        return "spline"
    return type(leaf).__name__.lower().replace("leaflayer", "")


def _plot_leaf(
    leaf,
    var_name: str,
    df_obs: Optional[object],
    bins: int = 80,
    plot_components: bool = False,
):
    """Dispatch to the right leaf plotter and overlay the empirical marginal."""
    if isinstance(leaf, GaussianLeafLayer):
        axes = plot_gaussian_leaf(leaf)
    elif isinstance(leaf, MixtureLeafLayer):
        axes = plot_mixture_leaf(leaf, plot_components=plot_components)
    elif isinstance(leaf, SplineLeafLayer):
        axes = plot_spline_leaf(leaf)
    else:
        raise TypeError(f"Unsupported leaf type: {type(leaf).__name__}")

    axes = np.atleast_1d(axes).reshape(-1)

    if df_obs is not None and var_name in df_obs.columns:
        values = df_obs[var_name].values
        counts, edges = np.histogram(values, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        for ax in axes:
            ax.plot(
                centers,
                counts,
                drawstyle="steps-mid",
                color="black",
                linewidth=1.5,
                alpha=0.7,
                label="empirical marginal",
            )
            # Refresh legend if one exists.
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="best", fontsize="small")

    return axes


def plot_all_leaves(
    results_seed_dir: str,
    output_dir: str = "scratch/plots/leaves",
    bins: int = 80,
    plot_components: bool = False,
    no_empirical: bool = False,
) -> list:
    """Plot every leaf in the trained circuit and save one figure per leaf."""
    apply_paper_style()

    cfg, ac, data_info, _, _ = load_trained_artifacts(results_seed_dir)
    id_to_var = {v: k for k, v in data_info["var_to_id"].items()}
    df_obs = None if no_empirical else data_info["df_obs_full"]

    exp_id = cfg["experiment"]["id"]
    seed = int(os.path.basename(results_seed_dir).split("_")[-1])
    os.makedirs(output_dir, exist_ok=True)

    saved = []
    for leaf_id in ac.get_leaves():
        leaf = ac.get_node_data(leaf_id)
        var = getattr(leaf, "var", None)
        var_name = id_to_var.get(var, f"var{var}")
        leaf_type = _leaf_type_name(leaf)

        try:
            _plot_leaf(leaf, var_name, df_obs, bins=bins, plot_components=plot_components)
        except Exception as exc:
            plt.close("all")
            print(f"Skipping leaf {leaf_id} ({var_name}, {leaf_type}): {exc}")
            continue

        fig = plt.gcf()
        out_path = os.path.join(
            output_dir, f"{exp_id}_seed{seed}_leaf{leaf_id}_{var_name}_{leaf_type}.png"
        )
        fig.savefig(out_path)
        plt.close(fig)
        saved.append(out_path)
        print(f"Saved leaf {leaf_id} ({var_name}) to {out_path}")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot learned leaf distributions with empirical marginal overlays."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to the experiment results seed directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="scratch/plots/leaves",
        help="Directory where leaf figures are saved.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Histogram bins for the empirical marginal overlay.",
    )
    parser.add_argument(
        "--plot-components",
        action="store_true",
        help="For mixture leaves, also draw the weighted base components.",
    )
    parser.add_argument(
        "--no-empirical",
        action="store_true",
        help="Do not overlay the empirical marginal histogram.",
    )
    args = parser.parse_args()

    plot_all_leaves(
        args.results_dir,
        output_dir=args.output_dir,
        bins=args.bins,
        plot_components=args.plot_components,
        no_empirical=args.no_empirical,
    )


if __name__ == "__main__":
    main()
