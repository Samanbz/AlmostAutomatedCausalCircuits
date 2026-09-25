"""Aggregate per-seed evaluation artifacts into a results table.

Reads, for every seed directory under ``experiments/results/<config>/``:

  * ``grid_metrics.json``  — MAE / KL / Hellinger of P(Y|X) and P(Y|do(X))
    against the analytical SCM ground truth (from ``evaluate_grids``);
  * ``checkpoint_nll.csv`` — observational-test and interventional NLL of the
    final model (from ``evaluate_checkpoints``).

Writes ``experiments/results/results_table.csv`` (one row per seed) and
``experiments/results/results_table_by_config.csv`` (mean +/- std over seeds),
and prints a markdown summary.

Usage: python -m experiments.make_results_table [--results-root experiments/results]
"""

import argparse
import csv
import glob
import json
import os


GRID_KEYS = [
    "mae_obs",
    "mae_do",
    "kl_obs_learned_gt",
    "kl_obs_gt_learned",
    "kl_do_learned_gt",
    "kl_do_gt_learned",
    "hellinger_obs",
    "hellinger_do",
]

FINAL_NLL_KEYS = ["obs_test_nll", "do_nll"]


def collect(results_root):
    rows = []
    for metrics_path in sorted(
        glob.glob(os.path.join(results_root, "*", "seed_*", "grid_metrics.json"))
    ):
        seed_dir = os.path.dirname(metrics_path)
        config = os.path.basename(os.path.dirname(seed_dir))
        seed = os.path.basename(seed_dir).removeprefix("seed_")
        # Ignore stale seed dirs synced from other machines: require the
        # locally trained final model (models_dir in the config is the
        # per-config root, i.e. models/<config>).
        with open(os.path.join(seed_dir, "config.json")) as f:
            models_root = json.load(f)["experiment"]["models_dir"]
        final = os.path.join(models_root, config, f"seed_{seed}", f"{config}_final.pt")
        if not os.path.exists(final):
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        row = {"config": config, "seed": int(seed)}
        for key in GRID_KEYS:
            row[key] = metrics.get(key)
        csv_path = os.path.join(seed_dir, "checkpoint_nll.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                last = list(csv.DictReader(f))[-1]
            for key in FINAL_NLL_KEYS:
                value = last.get(key)
                row[key] = float(value) if value not in (None, "") else None
        rows.append(row)
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def summarize(rows):
    import numpy as np

    by_config = {}
    for row in rows:
        by_config.setdefault(row["config"], []).append(row)
    out = []
    for config, rs in sorted(by_config.items()):
        agg = {"config": config, "n_seeds": len(rs)}
        for key in GRID_KEYS + FINAL_NLL_KEYS:
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))
        out.append(agg)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/results")
    args = parser.parse_args()

    rows = collect(args.results_root)
    if not rows:
        raise SystemExit(f"No grid_metrics.json found under {args.results_root}")

    per_seed_path = os.path.join(args.results_root, "results_table.csv")
    write_csv(per_seed_path, rows, ["config", "seed"] + GRID_KEYS + FINAL_NLL_KEYS)

    summary = summarize(rows)
    summary_path = os.path.join(args.results_root, "results_table_by_config.csv")
    write_csv(summary_path, summary, list(summary[0].keys()))

    print(f"Wrote {per_seed_path} ({len(rows)} seeds) and {summary_path}")
    print()
    print(
        "| config | seeds | H(obs) | H(do) | MAE(obs) | MAE(do) | KL do(l||g) | obs NLL | do NLL |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for agg in summary:

        def fmt(key, agg=agg):
            m, s = agg.get(key + "_mean"), agg.get(key + "_std")
            return f"{m:.4f}±{s:.4f}" if m is not None else "-"

        print(
            f"| {agg['config']} | {agg['n_seeds']} | {fmt('hellinger_obs')} | "
            f"{fmt('hellinger_do')} | {fmt('mae_obs')} | {fmt('mae_do')} | "
            f"{fmt('kl_do_learned_gt')} | {fmt('obs_test_nll')} | {fmt('do_nll')} |"
        )


if __name__ == "__main__":
    main()
