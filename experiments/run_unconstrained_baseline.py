"""Expressivity baseline: N=64 circuit on backdoor_400K with NO MD sets.

Trains exactly the same architecture/leaf/training recipe as
``configs/backdoor_cont_400K_N64.json`` (the MD backdoor run) but with
``model.md_sets = []``: every vtree node is labeled universal, so every layer
is built fully dense — no marginal-determinism constraints anywhere. Comparing
this run's (post-hoc) test NLLs against the MD run's shows how much
expressivity the determinism family costs at fixed node budget N.

The derived config is written to
``configs/backdoor_cont_400K_N64_unconstrained.json`` (the experiment ID is the
file name, so artifacts live under results/models ``..._unconstrained``) and
execution is handed to ``experiments.run_experiment``: identical CLI
(``--seed`` / ``--seeds`` / ``--gpus`` / ``--override`` / ``--yes-invalidate``),
identical checkpoint cadence, resume and invalidation behavior.

Example
-------
    python -m experiments.run_unconstrained_baseline
    python -m experiments.run_unconstrained_baseline --seeds 27 28 29 --gpus 0 1 2
"""

import json
import os
import sys

from experiments import run_experiment
from experiments.utils.config import _repo_root


BASE_CONFIG = "configs/backdoor_cont_400K_N64.json"
BASELINE_CONFIG = "configs/backdoor_cont_400K_N64_unconstrained.json"
SUFFIX = "unconstrained"


def build_baseline_config() -> dict:
    """Derive the unconstrained config from the MD backdoor_400K_N64 config.

    Only the determinism family and the artifact directories change; dataset,
    leaves, N and the whole training section are inherited so the comparison
    is apples-to-apples.
    """
    with open(os.path.join(_repo_root(), BASE_CONFIG)) as f:
        cfg = json.load(f)

    cfg["model"]["md_sets"] = []  # no marginal determinism: fully dense layers

    exp = cfg["experiment"]
    for key in ("output_dir", "models_dir"):
        exp[key] = f"{exp[key]}_{SUFFIX}"

    return cfg


def main() -> None:
    path = os.path.join(_repo_root(), BASELINE_CONFIG)
    cfg = build_baseline_config()
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote derived baseline config to {path}")

    # Hand off to the standard runner; argv[1:] are its flags (--seed, --seeds, ...).
    sys.argv = ["experiments.run_experiment", "--config", BASELINE_CONFIG, *sys.argv[1:]]
    run_experiment.main()


if __name__ == "__main__":
    main()
