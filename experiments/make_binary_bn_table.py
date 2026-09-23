"""Regenerate the paper comparison table (ours vs MDNet on the real BN datasets).

Reads the stored per-seed artifacts of ``run_binary_bn_experiment`` (this repo) and the
per-run artifacts of ``causal-pc/wang_baseline_driver.py`` (the MDNet pipeline, run in
the ``wang-pc`` conda env — see causal-pc/README.md), and prints the MAE / Bernoulli
log-loss table (mean ± std).

Everything the table needs is reproducible from scratch:

1. **Ours** (bit-identical on CPU, verified): ::

       for d in asia child win95pts andes; do
         python -m experiments.run_binary_bn_experiment \
           --config configs/binary_bn/${d}.json --seeds 0 1 2 3 4
       done

   Data (data/<d>.pkl), configs, and runner are git-tracked. ``set_seed`` covers
   numpy, python-random and torch; the learned vtree is deterministic given the
   data. Pin ``experiment.device`` to a CPU (``"device": "cpu"``) or run on the
   same hardware: ``get_device`` defaults to CUDA when available and GPU float
   non-associativity can flip low-order bits.

2. **MDNet**: ``conda activate wang-pc && python wang_baseline_driver.py
   --runs 5`` inside causal-pc/ (writes wang_baseline_results_5run.json). The
   released code seeds the weights (PRNGKey(0)) but shuffles vtree scopes with an
   unseeded global RNG: individual runs are not bit-reproducible, only the
   run-distribution (mean ± std) is stable across batches.

Usage: ``python -m experiments.make_binary_bn_table``
"""

import json
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results" / "binary_bn"
MDNET_JSON = REPO_ROOT / "causal-pc" / "wang_baseline_results_5run.json"

DATASETS = ("asia", "child", "win95pts", "andes")
SEEDS = (0, 1, 2, 3, 4)


def _ms(vals):
    if len(vals) > 1:
        return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"
    return f"{statistics.mean(vals):.4f}"


def main() -> None:
    mdnet = json.load(open(MDNET_JSON))["summary"]

    header = (
        f"{'dataset':<10} {'target':>8} | "
        + "ours (learned vtree) MAE / log-loss        "
        + " | MDNet MAE / log-loss"
    )
    print(header)
    print("-" * len(header))
    for d in DATASETS:
        ms = [json.load(open(RESULTS / d / f"seed_{s}" / "metrics.json")) for s in SEEDS]
        target = ms[0]["target"]
        mae = [m["abs_error"] for m in ms]
        ll = [m["log_loss"] for m in ms]
        cell = f"{_ms(mae):>18s} / {_ms(ll):<18s}"
        w = mdnet[d]
        wcell = f"{w['mae_mean']:.4f} ± {w['mae_std']:.4f} / {w['ll_mean']:.4f} ± {w['ll_std']:.4f}"
        print(f"{d:<10} {target:>8.4f} | {cell} | {wcell}")


if __name__ == "__main__":
    main()
