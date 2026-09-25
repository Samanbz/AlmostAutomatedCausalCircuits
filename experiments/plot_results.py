"""Paper figures from the evaluation artifacts.

Reads per-seed artifacts written by ``evaluate_grids``, ``evaluate_checkpoints``
and ``evaluate_checkpoint_grids`` and renders:

  1. ``fig_kl_profile_N.png``  — per-X KL(P_GT||P_learned) of P(Y|do(X)), mean +/-
     std over seeds, overlaying the 400K N-series (N=8..128) and N=64 G1 over
     X in [mean - 2 sigma_X, mean + 2 sigma_X].
  2. ``fig_kl_profile_data.png``  — same, overlaying N=64 trained on 25K / 100K
     / 400K / 1.2M samples.
  3. ``fig_do_nll_vs_N.png``  — final interventional NLL vs. num_nodes N
     (400K family), mean +/- std over seeds.
  4. ``fig_kl_vs_N.png``  — X-averaged (GT marginal of X) KL(P_GT||P_learned)
     vs. N; do solid, obs dashed.
  5. ``fig_kl_over_training.png``  — X-averaged do-KL vs. training epoch from
     the checkpoints, for the 400K N-series + N64_G1.
  6. ``fig_do_nll_over_training.png``  — do NLL vs. epoch (same configs).
  7. ``tab_obs_nll_over_training.csv``  — observational test NLL (mean +/- std
     over seeds) at each checkpoint epoch, same configs plus the unconstrained
     N64 PC. Rendered as a table instead of a figure: the curves all converge
     within the first ~50 epochs and are visually flat afterwards.

Usage: python -m experiments.plot_results [--results-root experiments/results] [--output-dir experiments/results/figures]
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
from matplotlib import pyplot as plt

from experiments.plotting.styles import apply_paper_style


N_SERIES = [
    "backdoor_cont_400K_N8",
    "backdoor_cont_400K_N16",
    "backdoor_cont_400K_N32",
    "backdoor_cont_400K_N64",
    "backdoor_cont_400K_N128",
]
TRAINING_CURVES = N_SERIES[:-1] + [
    "backdoor_cont_400K_N64_G1"
]  # N8..N64 + G1 (+N128 via N_SERIES where asked)
LABELS = {
    "backdoor_cont_400K_N8": "N=8",
    "backdoor_cont_400K_N16": "N=16",
    "backdoor_cont_400K_N32": "N=32",
    "backdoor_cont_400K_N64": "N=64",
    "backdoor_cont_400K_N128": "N=128",
    "backdoor_cont_400K_N64_G1": "N=64 G1",
    "backdoor_cont_400K_N64_unconstrained": "N=64 unconstr.",
}
# Okabe-Ito palette (same family as the slice/heatmap figures): cool hues for
# small N warming to orange/vermillion for large N; G1 is black and dashed so
# it stays legible against white.
OKABE_ITO = ["#56B4E9", "#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#F0E442"]
SERIES_COLORS = {
    "backdoor_cont_400K_N8": ("#56B4E9", "-"),
    "backdoor_cont_400K_N16": ("#0072B2", "-"),
    "backdoor_cont_400K_N32": ("#009E73", "-"),
    "backdoor_cont_400K_N64": ("#E69F00", "-"),
    "backdoor_cont_400K_N128": ("#D55E00", "-"),
    "backdoor_cont_400K_N64_G1": ("#000000", "--"),
}
DATA_COLORS = {
    "backdoor_cont_25K_N64": "#56B4E9",
    "backdoor_cont_100K_N64": "#0072B2",
    "backdoor_cont_400K_N64": "#E69F00",
    "backdoor_cont_1200K_N64": "#D55E00",
}
# Slice-figure semantics: blue = P(Y|X), orange/vermillion = P(Y|do(X)).
CURVE_COLORS = {"do": "#D55E00", "obs": "#0072B2"}
# Half-column figures: fonts stay legible when included at natural size in the
# A4 paper (a full-width figure scaled to column width shrinks its labels).
COMPACT_FIGSIZE = (3.8, 2.6)


def read_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def valid_seed_dirs(results_root, config):
    """Seed dirs of this config that have a locally trained final model.

    Results dirs may contain stale seeds synced from other machines (e.g. a
    laptop's seed_27) whose checkpoint schedules differ and would corrupt the
    aggregated statistics — only seeds with ``models/<config>/.../seed_<S>/
    <config>_final.pt`` under this repo are counted.
    """
    cfg_path = os.path.join(results_root, config, "seed_0", "config.json")
    if not os.path.exists(cfg_path):
        cfg_path = glob.glob(os.path.join(results_root, config, "seed_*", "config.json"))[0]
    with open(cfg_path) as f:
        models_root = json.load(f)["experiment"]["models_dir"]  # per-config root
    dirs = []
    for seed_dir in sorted(glob.glob(os.path.join(results_root, config, "seed_*"))):
        seed = os.path.basename(seed_dir)
        final = os.path.join(models_root, config, seed, f"{config}_final.pt")
        if os.path.exists(final):
            dirs.append(seed_dir)
    return dirs


def mean_std_over_seeds(results_root, config, filename, value_keys):
    """Return (epochs, {key: (mean, std)}) aligned across seed dirs."""
    per_seed = []
    for seed_dir in valid_seed_dirs(results_root, config):
        path = os.path.join(seed_dir, filename)
        if not os.path.exists(path):
            continue
        rows = read_csv_rows(path)
        per_seed.append(
            {int(r["epoch"]): {k: _to_float(r.get(k)) for k in value_keys} for r in rows}
        )
    epochs = sorted({e for rows in per_seed for e in rows})
    stats = {}
    for key in value_keys:
        means, stds = [], []
        for e in epochs:
            vals = [rows[e][key] for rows in per_seed if e in rows and rows[e][key] == rows[e][key]]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else np.nan)
        stats[key] = (np.array(means), np.array(stds))
    return np.array(epochs), stats


def plot_band(ax, x, mean, std, label, linestyle="-", color=None):
    ax.plot(x, mean, label=label, linestyle=linestyle, color=color, linewidth=1.8)
    ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color, linewidth=0)


def save(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, name + ".png"))
    plt.close(fig)
    print(f"Saved {name}.png")


def x_std_range(results_root, config):
    """(mean, std) of observational X for the config's dataset."""
    import pandas as pd

    cfg_path = glob.glob(os.path.join(results_root, config, "seed_*", "config.json"))[0]
    with open(cfg_path) as f:
        cfg = json.load(f)
    obs_csv = sorted(
        glob.glob(os.path.join("data", cfg["dataset"]["dataset"], "observational_*.csv"))
    )[0]
    x = pd.read_csv(obs_csv, usecols=["X"]).values[:, 0]
    return float(np.mean(x)), float(np.std(x))


def style_ax(ax):
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def collect_kl_curves(results_root, config):
    """Per-seed (x_grid, kl_do_gt_learned_per_x) pairs for valid seeds."""
    grids, curves = [], []
    for seed_dir in valid_seed_dirs(results_root, config):
        path = os.path.join(seed_dir, "grid_metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            m = json.load(f)
        grids.append(np.array(m["x_grid"], dtype=float))
        curves.append(np.array(m["kl_do_gt_learned_per_x"], dtype=float))
    return grids, curves


def fig_kl_profile_overlay(results_root, output_dir, config_labels, figname, figsize=(5.5, 3.2)):
    """Overlay per-X do-KL curves of several configs on a common X window.

    ``config_labels`` is a list of (config, label, color, linestyle). Curves
    are interpolated per seed onto a shared grid over the intersection of the
    configs' [mu - 2 sigma_X, mu + 2 sigma_X] windows; mean +/- 1 std band.
    """
    windows, series = [], []
    for config, _label, _color, _ls in config_labels:
        grids, curves = collect_kl_curves(results_root, config)
        if not grids:
            print(f"skip {config}: no grid_metrics.json")
            continue
        mu, sd = x_std_range(results_root, config)
        windows.append((mu - 2 * sd, mu + 2 * sd))
        series.append((config, grids, curves))

    lo = max(w[0] for w in windows)
    hi = min(w[1] for w in windows)
    xs = np.linspace(lo, hi, 300)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for (_config, label, color, ls), (_c, grids, curves) in zip(config_labels, series):
        interp = np.array([np.interp(xs, g, c) for g, c in zip(grids, curves)])
        mean, std = interp.mean(axis=0), interp.std(axis=0)
        ax.plot(xs, mean, label=label, color=color, linestyle=ls, linewidth=1.8)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("X")
    ax.set_ylabel(r"KL$\left(P_{\mathrm{GT}}\,\|\,P_{\mathrm{learned}}\right)$ (nats)")
    ax.legend(loc="upper right", framealpha=0.9)
    style_ax(ax)
    save(fig, output_dir, figname)


def fig_vs_N(results_root, output_dir):
    def collect_nll(key):
        means, stds, ns = [], [], []
        for n, config in zip([8, 16, 32, 64, 128], N_SERIES):
            _, stats = mean_std_over_seeds(results_root, config, "checkpoint_nll.csv", [key])
            mean, std = stats[key][0][-1], stats[key][1][-1]
            means.append(mean)
            stds.append(std)
            ns.append(n)
        return np.array(ns), np.array(means), np.array(stds)

    def collect_metric(key):
        means, stds, ns = [], [], []
        for n, config in zip([8, 16, 32, 64, 128], N_SERIES):
            vals = []
            for seed_dir in valid_seed_dirs(results_root, config):
                p = os.path.join(seed_dir, "grid_metrics.json")
                if os.path.exists(p):
                    vals.append(json.load(open(p))[key])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            ns.append(n)
        return np.array(ns), np.array(means), np.array(stds)

    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
    ns, mean, std = collect_nll("do_nll")
    ax.errorbar(ns, mean, yerr=std, marker="o", capsize=3, color=CURVE_COLORS["do"], label="do NLL")
    ax.set_xlabel("num_nodes N")
    ax.set_ylabel("interventional NLL (lower is better)")
    ax.set_xticks(ns)
    style_ax(ax)
    save(fig, output_dir, "fig_do_nll_vs_N")

    fig, ax = plt.subplots(figsize=COMPACT_FIGSIZE, constrained_layout=True)
    for curve, style in [("do", "-"), ("obs", "--")]:
        ns, mean, std = collect_metric(f"kl_{curve}_gt_learned")
        label = r"$P(Y\mid do(X))$" if curve == "do" else r"$P(Y\mid X)$"
        ax.errorbar(
            ns,
            mean,
            yerr=std,
            marker="o",
            capsize=3,
            linestyle=style,
            color=CURVE_COLORS[curve],
            label=label,
        )
    ax.set_xlabel("num_nodes N")
    ax.set_ylabel(r"X-averaged KL (nats)")
    ax.set_xticks(ns)
    ax.legend(framealpha=0.9)
    style_ax(ax)
    save(fig, output_dir, "fig_kl_vs_N")


def fig_over_training(
    results_root, output_dir, csv_name, key, ylabel, figname, configs, max_epoch=None
):
    fig, ax = plt.subplots(figsize=(5.5, 3.4), constrained_layout=True)
    for i, config in enumerate(configs):
        color, linestyle = SERIES_COLORS.get(config, (OKABE_ITO[i % len(OKABE_ITO)], "-"))
        epochs, stats = mean_std_over_seeds(results_root, config, csv_name, [key])
        if len(epochs) == 0:
            print(f"skip {config}: no {csv_name}")
            continue
        mean, std = stats[key]
        if max_epoch is not None:
            keep = epochs <= max_epoch
            epochs, mean, std = epochs[keep], mean[keep], std[keep]
        plot_band(
            ax, epochs, mean, std, LABELS.get(config, config), color=color, linestyle=linestyle
        )
    ax.set_xlabel("training epoch")
    ax.set_ylabel(ylabel)
    ax.legend(framealpha=0.9)
    style_ax(ax)
    save(fig, output_dir, figname)


def write_obs_nll_table(results_root, output_dir, configs):
    """Observational test NLL (mean +/- std over seeds) per checkpoint epoch.

    One row per config, one column per checkpoint epoch; written as CSV and
    echoed as a markdown table (epochs where a config has no checkpoint are
    blank). Replaces the visually flat obs-NLL-over-training figure.
    """
    per_config = {}
    epochs = set()
    for config in configs:
        e, stats = mean_std_over_seeds(results_root, config, "checkpoint_nll.csv", ["obs_test_nll"])
        if len(e) == 0:
            print(f"skip {config}: no checkpoint_nll.csv")
            continue
        mean, std = stats["obs_test_nll"]
        per_config[config] = {int(ep): (m, s) for ep, m, s in zip(e, mean, std) if m == m}
        epochs.update(per_config[config])
    if not per_config:
        return
    epochs = sorted(epochs)

    path = os.path.join(output_dir, "tab_obs_nll_over_training.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_epochs"] + [f"nll@{e}" for e in epochs])
        for config in configs:
            if config not in per_config:
                continue
            row = [LABELS.get(config, config), len(per_config[config])]
            row += [
                f"{per_config[config][e][0]:.4f}±{per_config[config][e][1]:.4f}"
                if e in per_config[config]
                else ""
                for e in epochs
            ]
            writer.writerow(row)
    print(f"Saved {os.path.basename(path)}")

    header = "| config | " + " | ".join(f"NLL@{e}" for e in epochs) + " |"
    print("\n" + header)
    print("|" + "---|" * (len(epochs) + 1))
    for config in configs:
        if config not in per_config:
            continue
        cells = [
            f"{per_config[config][e][0]:.3f}±{per_config[config][e][1]:.3f}"
            if e in per_config[config]
            else "-"
            for e in epochs
        ]
        print(f"| {LABELS.get(config, config)} | " + " | ".join(cells) + " |")


def write_final_nll_table(results_root, output_dir, configs):
    """Final-checkpoint interventional and observational test NLL per config.

    One row per config: mean +/- std over seeds of the last checkpoint's
    do NLL (empty for configs without a compiled do-query, i.e. the
    unconstrained PC) and observational test NLL.
    """
    rows = []
    for config in configs:
        _e, stats = mean_std_over_seeds(
            results_root, config, "checkpoint_nll.csv", ["do_nll", "obs_test_nll"]
        )
        if len(_e) == 0:
            print(f"skip {config}: no checkpoint_nll.csv")
            continue
        row = {"config": LABELS.get(config, config)}
        for key in ("do_nll", "obs_test_nll"):
            mean, std = stats[key][0][-1], stats[key][1][-1]
            row[key] = f"{mean:.4f}±{std:.4f}" if mean == mean else ""
        rows.append(row)

    path = os.path.join(output_dir, "tab_final_nll.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "do_nll", "obs_test_nll"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {os.path.basename(path)}")
    print("\n| config | do NLL | obs test NLL |")
    print("|---|---|---|")
    for row in rows:
        print(f"| {row['config']} | {row['do_nll'] or '—'} | {row['obs_test_nll']} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/results")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.join(args.results_root, "figures")
    apply_paper_style()

    fig_kl_profile_overlay(
        args.results_root,
        output_dir,
        [
            (config, LABELS[config], SERIES_COLORS[config][0], SERIES_COLORS[config][1])
            for config in N_SERIES
        ]
        + [
            (
                "backdoor_cont_400K_N64_G1",
                LABELS["backdoor_cont_400K_N64_G1"],
                SERIES_COLORS["backdoor_cont_400K_N64_G1"][0],
                SERIES_COLORS["backdoor_cont_400K_N64_G1"][1],
            )
        ],
        "fig_kl_profile_N",
    )
    data_series = [
        ("backdoor_cont_25K_N64", "25K"),
        ("backdoor_cont_100K_N64", "100K"),
        ("backdoor_cont_400K_N64", "400K"),
        ("backdoor_cont_1200K_N64", "1.2M"),
    ]
    fig_kl_profile_overlay(
        args.results_root,
        output_dir,
        [(config, f"N=64, {label}", DATA_COLORS[config], "-") for config, label in data_series],
        "fig_kl_profile_data",
        figsize=COMPACT_FIGSIZE,
    )

    fig_vs_N(args.results_root, output_dir)
    n_and_g1 = N_SERIES + ["backdoor_cont_400K_N64_G1"]
    fig_over_training(
        args.results_root,
        output_dir,
        "checkpoint_kl.csv",
        "kl_do_gt_learned",
        r"do-KL$\left(P_{\mathrm{GT}}\|P_{\mathrm{learned}}\right)$ (nats)",
        "fig_kl_over_training",
        n_and_g1,
    )
    fig_over_training(
        args.results_root,
        output_dir,
        "checkpoint_nll.csv",
        "do_nll",
        "interventional NLL (lower is better)",
        "fig_do_nll_over_training",
        n_and_g1,
        max_epoch=300,
    )
    obs_configs = n_and_g1 + ["backdoor_cont_400K_N64_unconstrained"]
    write_obs_nll_table(args.results_root, output_dir, obs_configs)
    write_final_nll_table(args.results_root, output_dir, obs_configs)


if __name__ == "__main__":
    main()
