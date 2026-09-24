"""Fit a single log-linear spline leaf group to an interesting univariate density.

This is the minimal circuit behind the paper's disjoint continuous leaves
(Section 5 / Appendix A): one ``LogLinearSplineLeafLayer`` group with K
components under a single uniform sum.  Concretely the sum is a
``MixtureLeafLayer``: a ``DenseWeights`` sum over the K leaf components whose
(absent) right child acts as a constant log(1) node, so the root computes

    log p(x) = logsum_k ( log w_k + log p_k(x) ),

initialized at w_k = 1/K — the equal-mass mixture of the log-linear spline
definition, and the sum weights are fixed there (never trained).  The splits
are initialized at the empirical 1/K-quantiles of the data (with boundary
heights matching the quantile widths), and training is batch EM via
``SymbolicEMTrainer`` (Adam on the spline parameters only).

Outputs the two-resolution figure used in the paper: the top row shows the
modelled vs. true CDF with the pinned quantile points (b_k, k/K); the bottom
row shows the fitted marginal (plus the initialization) with the K disjoint,
individually normalized component densities and the learned split points.

Example:
    python -m experiments.fit_univariate_logspline \
        --density skewed_bimodal --num-components 8 24 \
        --output-dir /plots/leaf_fit
"""

import argparse
import json
import math
import os
from typing import Callable, Dict, List, NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.plotting.styles import apply_paper_style
from src.symbolic.arithmetic.circuit import SymbolicArithmeticCircuit, eval_circuit
from src.symbolic.arithmetic.nodes.leaf_layer import (
    LogLinearSplineDistribution,
    LogLinearSplineLeafLayer,
    MixtureLeafLayer,
)
from src.symbolic.arithmetic.train import SymbolicEMTrainer
from src.symbolic.arithmetic.weights import DenseWeights
from src.utils import ContinuousInterval


class NamedDensity(NamedTuple):
    name: str
    description: str
    pdf: Callable[[np.ndarray], np.ndarray]
    sample: Callable[[np.random.Generator, int], np.ndarray]


def _norm_pdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    z = (x - loc) / scale
    return np.exp(-0.5 * z * z) / (scale * math.sqrt(2.0 * math.pi))


def _mixture_pdf(x: np.ndarray, components: List[Tuple[float, float, float]]) -> np.ndarray:
    return sum(w * _norm_pdf(x, m, s) for w, m, s in components)


def _mixture_sample(
    rng: np.random.Generator, n: int, components: List[Tuple[float, float, float]]
) -> np.ndarray:
    w = np.array([c[0] for c in components], dtype=np.float64)
    w = w / w.sum()
    idx = rng.choice(len(components), size=n, p=w)
    out = np.empty(n, dtype=np.float64)
    for k, (_, m, s) in enumerate(components):
        mask = idx == k
        out[mask] = rng.normal(m, s, size=int(mask.sum()))
    return out


_CLAW = [(0.5, 0.0, 1.0)] + [(0.1, k / 2.0 - 1.0, 0.1) for k in range(5)]

DENSITIES: Dict[str, NamedDensity] = {
    "gaussian": NamedDensity(
        "gaussian",
        "N(0, 1)",
        lambda x: _norm_pdf(x, 0.0, 1.0),
        lambda rng, n: rng.normal(0.0, 1.0, size=n),
    ),
    "skewed_bimodal": NamedDensity(
        "skewed_bimodal",
        "Marron-Wand skewed bimodal: 3/4 N(0,1) + 1/4 N(3/2, 1/3^2)",
        lambda x: _mixture_pdf(x, [(0.75, 0.0, 1.0), (0.25, 1.5, 1.0 / 3.0)]),
        lambda rng, n: _mixture_sample(rng, n, [(0.75, 0.0, 1.0), (0.25, 1.5, 1.0 / 3.0)]),
    ),
    "claw": NamedDensity(
        "claw",
        "Marron-Wand claw: 1/2 N(0,1) + sum_{k=0}^4 1/10 N(k/2 - 1, 1/10^2)",
        lambda x: _mixture_pdf(x, _CLAW),
        lambda rng, n: _mixture_sample(rng, n, _CLAW),
    ),
}


def empirical_quantile_supports(
    samples: np.ndarray, num_components: int
) -> Tuple[List[ContinuousInterval], np.ndarray]:
    """Initialize a leaf group from the data: disjoint intervals at the empirical
    ``1/K``-quantiles, plus boundary heights consistent with those widths.

    The heights follow the average density at each boundary under the equal-mass
    partition (``boundary_heights_from_splits`` in ``leaf_layer``, applied by the
    leaf when ``init_split_points`` is given), so the width identity of the
    construction reproduces (approximately) the quantile spacing — i.e. the
    model's actual initial splits land on the empirical quantiles.
    """
    b = np.quantile(samples, np.linspace(0.0, 1.0, num_components + 1)[1:-1])
    return [
        ContinuousInterval(
            b[i - 1] if i > 0 else float("-inf"),
            b[i] if i < num_components - 1 else float("inf"),
            include_low=True,
            include_high=(i == num_components - 1),
        )
        for i in range(num_components)
    ]


def build_logspline_circuit(
    num_components: int,
    supports: List[ContinuousInterval],
    quantile_splits: np.ndarray,
) -> Tuple[SymbolicArithmeticCircuit, LogLinearSplineLeafLayer, MixtureLeafLayer]:
    """One logspline leaf group under a uniform mixture sum (the minimal circuit).

    The mixture weights are fixed at 1/K (never trained): the patched marginal
    is then boundary-continuous (R2) and the modelled CDF is pinned at k/K at
    every split, exactly the equal-mass construction of the paper.
    """
    # base_mean/base_stddev are unused: with init_split_points the leaf derives
    # both the splits and the boundary heights from the data.
    spec = LogLinearSplineDistribution(var=0, base_mean=0.0, base_stddev=1.0)
    leaf = LogLinearSplineLeafLayer(
        spec,
        num_nodes=num_components,
        num_groups=1,
        node_supports=supports,
        init_split_points=quantile_splits,
    )

    w = np.full((1, 1, 1, num_components, 1, 1), 1.0 / num_components, dtype=np.float32)
    log_w = torch.log(torch.from_numpy(w)).detach()

    mixture = MixtureLeafLayer(base_dist=leaf, log_weights=DenseWeights(log_w))
    ac = SymbolicArithmeticCircuit()
    ac._add_node(mixture)
    return ac, leaf, mixture


def mixture_logpdf(ac: SymbolicArithmeticCircuit, xs: np.ndarray) -> np.ndarray:
    data = torch.tensor(xs, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        out = eval_circuit(ac, data, keep_intermediates=False)
    return out.reshape(-1).cpu().numpy()


def split_points(leaf: LogLinearSplineLeafLayer) -> np.ndarray:
    with torch.no_grad():
        return leaf._split_points().reshape(-1).cpu().numpy()


def mixture_weights(mixture: MixtureLeafLayer) -> np.ndarray:
    with torch.no_grad():
        return torch.exp(mixture.log_weights.log_weights).reshape(-1).cpu().numpy()


def cdf_from_pdf(xs: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    mass = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(xs))])
    return mass / mass[-1]


def fit_one_resolution(
    density: NamedDensity,
    num_components: int,
    samples: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    """Train one leaf group and return everything needed for the figure panels."""
    supports = empirical_quantile_supports(samples, num_components)
    quantile_splits = np.array([iv.high for iv in supports[:-1]])

    ac, leaf, mixture = build_logspline_circuit(
        num_components,
        supports,
        quantile_splits,
    )
    weights_init = mixture_weights(mixture)
    # Sanity: with the heights initialized to the empirical boundary densities,
    # the width identity should place the model's initial splits on the
    # empirical 1/K-quantiles.
    init_dev = float(np.max(np.abs(split_points(leaf) - quantile_splits)))

    # Tight plotting window: cut most of the tails so the interesting mass is
    # readable at both resolutions.
    lo, hi = np.quantile(samples, [0.001, 0.999])
    pad = 0.05 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 4001)

    trainer = SymbolicEMTrainer(ac, leaf_lr=args.leaf_lr)
    data = torch.tensor(samples, dtype=torch.float32).unsqueeze(1)
    trainer.train(
        data,
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        step_size=args.step_size,
        decay_rate=args.decay_rate,
        decay_every=args.decay_every,
        log_interval=max(1, args.n_iter // 10),
    )

    weights_final = mixture_weights(mixture)
    pdf_fit = np.exp(mixture_logpdf(ac, xs))
    splits = split_points(leaf)
    cdf_fit = cdf_from_pdf(xs, pdf_fit)
    cdf_true = cdf_from_pdf(xs, density.pdf(xs))

    # Quantile pinning check: the modelled CDF at the learned splits should hit
    # the cumulative mixture weights (exactly k/K while the sum stays uniform).
    cdf_at_splits = np.interp(splits, xs, cdf_fit)
    pin_target = np.cumsum(weights_final)[: len(splits)]
    print(
        f"K={num_components:3d} | final train NLL {_train_nll(ac, data):.4f}"
        f" | init splits vs quantiles max|dev| = {init_dev:.2e}"
        f" | max |F(b_k) - cumsum(w)| = {np.max(np.abs(cdf_at_splits - pin_target)):.2e}"
        f" | weights final {np.round(weights_final, 3)}"
    )

    return {
        "xs": xs,
        "pdf_fit": pdf_fit,
        "pdf_true": density.pdf(xs),
        "splits": splits,
        "cdf_fit": cdf_fit,
        "cdf_true": cdf_true,
        "weights_init": weights_init,
        "weights_final": weights_final,
    }


def _train_nll(ac: SymbolicArithmeticCircuit, data: torch.Tensor) -> float:
    with torch.no_grad():
        log_probs = eval_circuit(ac, data, keep_intermediates=False)
    return float(-log_probs.mean())


def plot_panels(
    density: NamedDensity,
    fits: List[Dict[str, np.ndarray]],
    ks: List[int],
    output_dir: str,
) -> str:
    """2x2 figure: CDF interpolation (top) and density fit (bottom).

    The density row shows, per column, ``K * p(x)`` for the modelled and the
    target density.  Since the mixture weight of every component is ``1/K`` and
    exactly one component is active at any ``x``, ``K * p_modelled`` is the
    height of the active component — the honest per-component counterpart of
    ``K * p_target`` (and its continuity across splits demonstrates R2).
    """
    apply_paper_style()
    col_labels = ["(a) coarse", "(b) fine"]

    # All four panels share one x-axis; the two density panels get independent
    # y-scales so the xK amplification is invisible — the target curve reads
    # identically at both resolutions, each on its own scale.
    fig, axes = plt.subplots(2, 2, figsize=(7, 4), sharex=True)

    for col, (fit, k) in enumerate(zip(fits, ks)):
        xs = fit["xs"]

        ax_cdf = axes[0][col]
        ax_cdf.plot(xs, fit["cdf_fit"], color="#1f77b4", lw=1.6, label="modelled CDF")
        ax_cdf.plot(xs, fit["cdf_true"], color="black", ls="--", lw=1.2, label="target CDF")
        pins = np.concatenate([[0.0], np.cumsum(fit["weights_final"])])
        ax_cdf.plot(
            fit["splits"],
            pins[1 : len(fit["splits"]) + 1],
            "o",
            color="#d62728",
            ms=3.5,
            label=r"knots $(\xi_k, k/K)$",
        )
        ax_cdf.set_ylabel("CDF")
        ax_cdf.set_title(f"{col_labels[col]}, $K$={k}")
        # Y-axis lives on the pinning levels k/K: ticks (labeled i/K) and a
        # thin horizontal line per level, so the knots visibly sit on them.
        levels = np.arange(0, k + 1) / k
        for lv in levels[1:-1]:
            ax_cdf.axhline(lv, color="gray", ls=":", lw=0.6, alpha=0.7)
        ax_cdf.set_yticks(levels)
        ax_cdf.set_yticklabels([f"{i}/{k}" for i in range(k + 1)], fontsize=6)
        if col == 1:
            ax_cdf.legend(loc="upper left", fontsize=7)

        ax_pdf = axes[1][col]
        # Modelled density as one continuous line; the disjoint per-component
        # pieces show as translucent colored fills beneath it (one color per
        # interval (xi_{j-1}, xi_j]); dashed vertical lines mark the splits.
        y_model = k * fit["pdf_fit"]
        bounds = np.concatenate([[-np.inf], fit["splits"], [np.inf]])
        for j in range(len(bounds) - 1):
            mask = (xs > bounds[j]) & (xs <= bounds[j + 1])
            ax_pdf.fill_between(
                xs[mask], 0.0, y_model[mask], color=plt.cm.tab10(j % 10), alpha=0.3, lw=0
            )
        for b in fit["splits"]:
            ax_pdf.axvline(b, color="gray", ls="--", lw=0.7, alpha=0.8)
        ax_pdf.plot(xs, y_model, color="#1f77b4", lw=1.8, label=r"$K\times$ modelled")
        ax_pdf.plot(
            xs,
            k * fit["pdf_true"],
            color="black",
            ls="--",
            lw=1.3,
            label=r"$K\times$ target",
        )
        ax_pdf.set_ylabel(r"density $\times\, K$")
        if col == 1:
            ax_pdf.legend(loc="upper left", fontsize=7, framealpha=0.9)

    axes[1][0].set_xlabel("value")
    axes[1][1].set_xlabel("value")
    for ax_cdf in axes[0]:
        ax_cdf.set_ylim(-0.03, 1.03)
    # Density y-scales: the coarse (left) panel must fit its own curves — it is
    # the spiky one and overshoots — and the fine (right) panel then uses
    # K2/K1 times that limit, so the two panels read on a common scale with
    # only the xK amplification factored out.
    k_left, k_right = ks[0], ks[1]
    fit_left = fits[0]
    ymax_left = max(k_left * fit_left["pdf_true"].max(), k_left * fit_left["pdf_fit"].max())
    ylim_left = 1.08 * ymax_left
    axes[1][0].set_ylim(0.0, ylim_left)
    axes[1][1].set_ylim(0.0, ylim_left * k_right / k_left)
    # fig.suptitle(f"Log-linear spline leaves fit to the {density.name} density")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f"loglinear_leaves_{density.name}_K{ks[0]}_K{ks[1]}")
    fig.savefig(base + ".png")
    fig.savefig(base + ".pdf")
    plt.close(fig)
    return base + ".png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--density", default="skewed_bimodal", choices=sorted(DENSITIES))
    parser.add_argument(
        "--num-components",
        type=int,
        nargs=2,
        default=[8, 24],
        metavar=("COARSE", "FINE"),
        help="Component counts for the two figure columns.",
    )
    parser.add_argument("--n-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--n-iter", type=int, default=50, help="Training epochs (plus one).")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--step-size", type=float, default=0.5, help="EM step size.")
    parser.add_argument("--decay-rate", type=float, default=0.7)
    parser.add_argument("--decay-every", type=int, default=10)
    parser.add_argument("--leaf-lr", type=float, default=0.01, help="Adam lr for spline params.")
    parser.add_argument("--output-dir", default="scratch/plots/leaf_fit")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    density = DENSITIES[args.density]
    print(f"Target density: {density.description}")
    rng = np.random.default_rng(args.seed)
    samples = density.sample(rng, args.n_samples)
    oracle_nll = float(-np.log(np.clip(density.pdf(samples), 1e-300, None)).mean())
    print(f"Oracle NLL (true density): {oracle_nll:.4f}")

    fits = [fit_one_resolution(density, k, samples, args) for k in sorted(args.num_components)]
    ks = sorted(args.num_components)
    out_path = plot_panels(density, fits, ks, args.output_dir)
    print(f"Saved figure: {out_path}")

    summary = {
        "density": density.description,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "resolutions": {
            str(k): {
                "splits": fits[i]["splits"].tolist(),
                "weights_init": fits[i]["weights_init"].tolist(),
                "weights_final": fits[i]["weights_final"].tolist(),
            }
            for i, k in enumerate(ks)
        },
    }
    summary_path = os.path.join(args.output_dir, f"loglinear_leaves_{density.name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
