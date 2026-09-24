"""Quickly explore a generated dataset: paired CSVs on disk + density-inspection plots.

Generates a dataset exactly like ``generate_synthetic_data.py`` (same CLI: pick a
skeleton, set sizes, mechanism knobs) and additionally writes, per (treatment,
outcome) pair:

* ``heatmap_<T>_<Y>.png`` — P(Y|X) and P(Y|do(X)), each as an empirical panel
  (from the paired datasets) and an exact-ground-truth panel. Empirical columns
  built from fewer than ``MIN_SAMPLES_PER_CURVE`` samples are left blank (light
  gray) instead of showing spiky tail-bin artifacts.
* ``slices_<T>_<Y>.png`` — one subplot per fixed treatment context X=x (arranged
  in a grid), each showing empirical (KDE) and exact P(Y|X=x) alongside empirical
  and exact P(Y|do(X=x)); contexts with fewer than ``MIN_SAMPLES_PER_CURVE``
  samples get their empirical curve skipped with a note in the panel.

Works for discrete, continuous and mixed layouts alike. Also prints the exact
|P(y|do(x)) - P(y|x)| effect gap per pair — useful for tuning confounding.

Example:
    python explore_synthetic_data.py --skeleton backdoor --kind mixed \
        --min_categories 2 --max_categories 4 --n_confounders 2 --regions 5 \
        --n_samples 50000 --seed 27
"""

import argparse
import os

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde

from scripts.generate_synthetic_data import (
    CASE_GUIDE,
    add_generation_arguments,
    build_skeleton_from_args,
    default_treatments,
    generate_paired_datasets,
    mechanism_kwargs_from_args,
    save_datasets,
)
from src.construction import effect_gap


# Minimum number of samples an empirical conditional estimate must be based on before it
# is plotted. Tail bins of continuous variables hold very few samples, and their empirical
# densities are extreme spiky artifacts that dominate the plot's color/height scale.
MIN_SAMPLES_PER_CURVE = 100


def _axis(values, kind, cardinality, bins):
    """Bin a variable: categories for discrete, equal-width bins for continuous.

    Returns (per-row bin index, bin centers, bin edges).
    """
    values = np.asarray(values)
    if kind == "discrete":
        edges = np.arange(-0.5, cardinality + 0.5)
        return values.astype(int), np.arange(cardinality), edges
    edges = np.linspace(values.min(), values.max(), bins + 1)
    idx = np.clip(np.digitize(values, edges) - 1, 0, bins - 1)
    return idx, 0.5 * (edges[:-1] + edges[1:]), edges


def _conditional_matrix(df, t, y, spec_t, spec_y, bins):
    t_idx, t_centers, _ = _axis(df[t].values, spec_t.kind, spec_t.cardinality, bins)
    y_idx, y_centers, _ = _axis(df[y].values, spec_y.kind, spec_y.cardinality, bins)
    counts = np.zeros((len(y_centers), len(t_centers)))
    np.add.at(counts, (y_idx, t_idx), 1.0)
    col_sums = counts.sum(axis=0, keepdims=True)
    return np.divide(
        counts,
        col_sums,
        out=np.full_like(counts, np.nan),
        where=col_sums >= MIN_SAMPLES_PER_CURVE,
    )


def _gt_matrix(gt, t, y, spec_t, spec_y, t_centers, y_centers, interventional):
    """Exact P(Y | do(T=t)) or P(Y | T=t) on the same bin grid (per-column normalized)."""
    matrix = np.zeros((len(y_centers), len(t_centers)))
    for j, ctx in enumerate(t_centers):
        kw = {"do": {t: ctx}} if interventional else {"evidence": {t: ctx}}
        if spec_y.kind == "discrete":
            for i, val in enumerate(y_centers):
                matrix[i, j] = gt.marginal_prob({y: int(val)}, **kw)
        else:
            gm = gt.marginal_density([y], **kw)
            pdf = np.maximum(gm.pdf(np.asarray(y_centers, dtype=np.float64)), 0.0)
            width = np.median(np.diff(y_centers)) if len(y_centers) > 1 else 1.0
            matrix[:, j] = pdf * width
        s = matrix[:, j].sum()
        if s > 0:
            matrix[:, j] /= s
    return matrix


def plot_heatmaps(df_obs, df_do, gt, t, y, spec_t, spec_y, bins, title, path):
    _, t_centers, t_edges = _axis(df_obs[t].values, spec_t.kind, spec_t.cardinality, bins)
    _, y_centers, y_edges = _axis(df_obs[y].values, spec_y.kind, spec_y.cardinality, bins)
    panels = [
        (_conditional_matrix(df_obs, t, y, spec_t, spec_y, bins), f"empirical P({y}|{t})"),
        (_conditional_matrix(df_do, t, y, spec_t, spec_y, bins), f"empirical P({y}|do({t}))"),
        (
            _gt_matrix(gt, t, y, spec_t, spec_y, t_centers, y_centers, False),
            f"exact P({y}|{t})",
        ),
        (
            _gt_matrix(gt, t, y, spec_t, spec_y, t_centers, y_centers, True),
            f"exact P({y}|do({t}))",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    for ax, (matrix, panel_title) in zip(axes.flat, panels):
        finite = matrix[np.isfinite(matrix) & (matrix > 0)]
        vmin = max(float(finite.min()) if len(finite) else 1e-8, 1e-8)
        vmax = float(np.nanmax(matrix))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin * 100
        norm = LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            norm=norm,
            extent=[t_edges[0], t_edges[-1], y_edges[0], y_edges[-1]],
            cmap=cmap,
        )
        ax.set_title(panel_title)
        ax.set_xlabel(t)
        ax.set_ylabel(y)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_slices(df_obs, df_do, gt, t, y, spec_t, spec_y, n_contexts, bins, title, path):
    if spec_t.kind == "discrete":
        contexts = np.arange(min(spec_t.cardinality, n_contexts)).astype(float)
    else:
        contexts = np.quantile(df_do[t].values, np.linspace(0.02, 0.98, n_contexts))

    def rows_at(df):
        _, _, edges = _axis(df[t].values, spec_t.kind, spec_t.cardinality, bins)
        if spec_t.kind == "discrete":
            return [df[df[t].values == int(ctx)][y].values for ctx in contexts]
        bin_idx = np.clip(np.digitize(df[t].values, edges) - 1, 0, bins - 1)
        out = []
        for ctx in contexts:
            ctx_bin = int(np.clip(np.digitize(ctx, edges) - 1, 0, bins - 1))
            out.append(df[bin_idx == ctx_bin][y].values)
        return out

    rows_obs, rows_do = rows_at(df_obs), rows_at(df_do)
    grid = np.linspace(
        min(df_obs[y].min(), df_do[y].min()), max(df_obs[y].max(), df_do[y].max()), 400
    )

    n = len(contexts)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)
    for ax, ctx, obs, do in zip(axes.flat, contexts, rows_obs, rows_do):
        ctx_gt = int(ctx) if spec_t.kind == "discrete" else float(ctx)
        ax.set_title(f"{t}={int(ctx)}" if spec_t.kind == "discrete" else f"{t}={ctx:.2f}")
        if spec_y.kind == "discrete":
            values = np.arange(spec_y.cardinality)
            if len(obs) >= MIN_SAMPLES_PER_CURVE:
                emp = np.array([(obs == v).mean() for v in values])
                ax.plot(values, emp, "o--", color="C0", alpha=0.7, label="emp P(Y|X)")
            if len(do) >= MIN_SAMPLES_PER_CURVE:
                emp = np.array([(do == v).mean() for v in values])
                ax.plot(values, emp, "o--", color="C1", alpha=0.7, label="emp P(Y|do(X))")
            exact = np.array([gt.marginal_prob({y: int(v)}, evidence={t: ctx_gt}) for v in values])
            ax.plot(values, exact, "s-", color="C0", alpha=0.5, label="GT P(Y|X)")
            exact = np.array([gt.marginal_prob({y: int(v)}, do={t: ctx_gt}) for v in values])
            ax.plot(values, exact, "s-", color="C1", alpha=0.5, label="GT P(Y|do(X))")
        else:
            if len(obs) >= MIN_SAMPLES_PER_CURVE and obs.max() > obs.min():
                ax.plot(grid, gaussian_kde(obs)(grid), color="C0", alpha=0.7, label="emp P(Y|X)")
            if len(do) >= MIN_SAMPLES_PER_CURVE and do.max() > do.min():
                ax.plot(grid, gaussian_kde(do)(grid), color="C1", alpha=0.7, label="emp P(Y|do(X))")
            gm = gt.marginal_density([y], evidence={t: ctx_gt})
            ax.plot(grid, gm.pdf(grid), "--", color="C0", alpha=0.9, label="GT P(Y|X)")
            gm = gt.marginal_density([y], do={t: ctx_gt})
            ax.plot(grid, gm.pdf(grid), "--", color="C1", alpha=0.9, label="GT P(Y|do(X))")
        low = [name for name, r in (("obs", obs), ("do", do)) if len(r) < MIN_SAMPLES_PER_CURVE]
        if low:
            ax.text(
                0.02,
                0.96,
                f"n<{MIN_SAMPLES_PER_CURVE}: skip {'+'.join(low)}",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                color="red",
            )
        ax.set_xlabel(y)
        ax.set_ylabel("density" if spec_y.kind == "continuous" else "probability")
        ax.legend(fontsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{title}\nP({y}|{t}=x) vs P({y}|do({t}=x)) slices")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a dataset and inspect its P(Y|X) / P(Y|do(X)) densities.",
        epilog=CASE_GUIDE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_generation_arguments(parser)
    parser.add_argument("--n_contexts", type=int, default=5, help="X contexts for 1D slices")
    parser.add_argument("--bins", type=int, default=60, help="Bins per continuous axis")
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=None,
        help="Plot output dir (default: <output_dir>/plots_<dataset name>)",
    )
    args = parser.parse_args()

    skeleton = build_skeleton_from_args(args)
    treatments = (
        tuple(args.treatments.split(",")) if args.treatments else default_treatments(skeleton)
    )
    outcomes = tuple(v.name for v in skeleton.variables if v.name == "Y" or v.name.startswith("Y_"))
    if not outcomes:
        raise ValueError("Skeleton has no outcome variables named 'Y'/'Y_*'.")

    mechanism_kwargs = mechanism_kwargs_from_args(args)
    df_obs, df_do, scm = generate_paired_datasets(
        skeleton,
        treatments=treatments,
        n_samples=args.n_samples,
        seed=args.seed,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        **mechanism_kwargs,
    )
    paths = save_datasets(df_obs, df_do, scm, skeleton, args, mechanism_kwargs)

    plots_dir = args.plots_dir or os.path.join(paths["output_dir"], f"plots_{paths['base_name']}")
    os.makedirs(plots_dir, exist_ok=True)

    gt = scm.ground_truth()
    for t in treatments:
        for y in outcomes:
            spec_t, spec_y = skeleton.spec_of(t), skeleton.spec_of(y)
            title = f"{paths['base_name']}  ({t} -> {y})"
            heatmap_path = os.path.join(plots_dir, f"heatmap_{t}_{y}.png")
            slices_path = os.path.join(plots_dir, f"slices_{t}_{y}.png")
            plot_heatmaps(df_obs, df_do, gt, t, y, spec_t, spec_y, args.bins, title, heatmap_path)
            plot_slices(
                df_obs,
                df_do,
                gt,
                t,
                y,
                spec_t,
                spec_y,
                args.n_contexts,
                args.bins,
                title,
                slices_path,
            )
            gap = effect_gap(scm, t, y)
            print(
                f"[{t} -> {y}] effect gap: mean L1 = {gap['mean_l1']:.3f}, "
                f"max L1 = {gap['max_l1']:.3f}"
            )
            print(f"Saved heatmap to {heatmap_path}")
            print(f"Saved slices to {slices_path}")


if __name__ == "__main__":
    main()
