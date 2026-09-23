"""Post-processing for the binary-BN (MDNet comparison) experiments: aggregate per-seed
metrics from ``experiments/results/binary_bn/<dataset>/seed_*/metrics.json``, print the
MAE / Bernoulli log-loss table (mean ± std across seeds), and save a grouped bar chart
comparing, per dataset, the true interventional target, the circuit's P(Y|do(X))
(mean ± std over seeds), the naive counting adjustment, and the circuit's
observational P(Y|X).

Example
-------
    conda activate pcs
    python -m experiments.plot_binary_bn_comparison
    python -m experiments.plot_binary_bn_comparison --datasets asia child --out /tmp/cmp.png
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def collect(results_root: str, datasets: list[str] | None):
    """Per dataset: target + per-seed metric dicts, from the results tree."""
    found = {}
    exp_dirs = sorted(glob.glob(os.path.join(results_root, "binary_bn", "*")))
    for exp_dir in exp_dirs:
        name = os.path.basename(exp_dir)
        if datasets and name not in datasets:
            continue
        seeds = {}
        target = None
        for seed_dir in sorted(glob.glob(os.path.join(exp_dir, "seed_*"))):
            metrics_path = os.path.join(seed_dir, "metrics.json")
            if not os.path.exists(metrics_path):
                continue
            m = json.load(open(metrics_path))
            if "est_do" not in m:
                continue  # predates query evaluation
            seed = m.get("seed", int(os.path.basename(seed_dir).split("_")[1]))
            seeds[seed] = m
            if target is None:
                cfg = json.load(open(os.path.join(seed_dir, "config.json")))
                target = cfg["dataset"]["target"]
        if seeds:
            found[name] = {"target": target, "seeds": seeds}
    return found


def report_table(found: dict) -> None:
    print(
        f"\n{'dataset':<10} {'target':>7} {'circ_do (mean±std)':>22} "
        f"{'MAE (mean±std)':>20} {'log-loss (mean±std)':>22} {'naive MAE':>10}"
    )
    for name, ds in found.items():
        runs = list(ds["seeds"].values())
        est = [r["est_do"] for r in runs]
        mae = [r["abs_error"] for r in runs]
        ll = [r["log_loss"] for r in runs]
        naive_mae = float(np.mean([abs(r["naive_do"] - ds["target"]) for r in runs]))

        def ms(xs):
            return f"{np.mean(xs):.4f} ± {np.std(xs, ddof=1) if len(xs) > 1 else 0.0:.4f}"

        print(
            f"{name:<10} {ds['target']:>7.4f} {ms(est):>22} {ms(mae):>20} "
            f"{ms(ll):>22} {naive_mae:>10.4f}"
        )


def plot_grouped_bars(found: dict, out_path: str) -> None:
    names = list(found)
    targets = [found[n]["target"] for n in names]
    circ_do = [np.mean([r["est_do"] for r in found[n]["seeds"].values()]) for n in names]
    circ_do_std = [
        np.std([r["est_do"] for r in found[n]["seeds"].values()], ddof=1)
        if len(found[n]["seeds"]) > 1
        else 0.0
        for n in names
    ]
    naive_do = [np.mean([r["naive_do"] for r in found[n]["seeds"].values()]) for n in names]
    circ_obs = [np.mean([r["est_obs"] for r in found[n]["seeds"].values()]) for n in names]

    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(2.0 + 1.4 * len(names), 5))
    ax.bar(x - 1.5 * width, targets, width, label="True P(Y|do(X))", color="#d62728")
    ax.bar(
        x - 0.5 * width,
        circ_do,
        width,
        yerr=circ_do_std,
        capsize=3,
        label="Circuit P(Y|do(X))",
        color="#ff9896",
    )
    ax.bar(x + 0.5 * width, naive_do, width, label="Naive counting P(Y|do(X))", color="#1f77b4")
    ax.bar(x + 1.5 * width, circ_obs, width, label="Circuit P(Y|X)", color="#aec7e8")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{n}\n(Z={len(json.load(open(f'experiments/results/binary_bn/{n}/seed_0/config.json'))['dataset']['z'])})"
            for n in names
        ]
    )
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved bar chart to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", default="experiments/results")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--out", default="experiments/results/binary_bn_comparison.png")
    args = parser.parse_args()

    found = collect(args.results_root, args.datasets)
    if not found:
        raise SystemExit(f"No binary_bn results with query metrics under {args.results_root}")
    report_table(found)
    plot_grouped_bars(found, args.out)


if __name__ == "__main__":
    main()
