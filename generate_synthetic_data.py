import argparse
import multiprocessing as mp
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from src.symbolic.scm import build_synthetic_continuous_scm


# Per-process SCM and base seed, initialized once per worker.
_worker_scm = None
_worker_seed = None


def _init_worker(num_confounders, confounding_strength, seed):
    """Rebuild the same SCM in every worker process (avoids pickling the SCM)."""
    global _worker_scm, _worker_seed
    _worker_scm = build_synthetic_continuous_scm(num_confounders, confounding_strength, seed=seed)
    _worker_seed = seed


def _generate_interventional_chunk(args):
    """Generate interventional samples for one chunk of observational X values.

    For each observed X_i we sample one point from the intervened SCM do(X = X_i)
    with FRESH exogenous noise.  This is what makes the interventional dataset
    represent P(Y | do(X)) rather than reproducing the observational Y values.

    A deterministic per-chunk seed keeps the result reproducible while ensuring
    different chunks do not draw identical random sequences.
    """
    chunk_idx, x_vals = args
    n = len(x_vals)
    if n == 0:
        return pd.DataFrame()

    # Each chunk gets its own reproducible random stream.
    np.random.seed(_worker_seed + chunk_idx)

    data = {}
    for node_id in _worker_scm.topological_sort():
        parent_ids = _worker_scm.get_parents(node_id)
        parent_data = {pid: data[pid] for pid in parent_ids}
        mechanism = _worker_scm.get_node_data(node_id)

        if node_id == "X":
            # Fix X to the value observed in the paired row.
            data[node_id] = x_vals.astype(np.float64)
        else:
            # Fresh noise -> the interventional distribution, not the counterfactual.
            data[node_id] = mechanism(n_samples=n, **parent_data)

    return pd.DataFrame(data)


def generate_paired_datasets(
    n_samples: int = 18432,
    num_confounders: int = 2,
    confounding_strength: float = 1.0,
    seed: int = 27,
    n_workers: int = None,
    chunk_size: int = 256,
):
    """Generate a paired observational / interventional dataset from a backdoor SCM.

    The interventional dataset represents P(Y | do(X)): each row fixes X to the
    corresponding observational X_i and draws the remaining variables (Z, Y) from
    the intervened SCM with fresh noise.  Because Z is sampled from its marginal
    P(Z) in the intervened model, confounding is removed and P(Y | do(X)) differs
    from the observational P(Y | X) whenever confounding_strength > 0.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    scm = build_synthetic_continuous_scm(num_confounders, confounding_strength, seed=seed)

    print(f"Sampling {n_samples} observational points...")
    df_obs = scm.sample(n_samples)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(
        f"Generating paired interventional dataset with {n_workers} workers "
        f"(chunk_size={chunk_size})..."
    )

    x_vals = df_obs["X"].values

    # Build chunks.  Each chunk is just a slice of the observed X values; the
    # corresponding Z/Y are sampled fresh in the worker.
    chunks = []
    for chunk_idx, start in enumerate(range(0, n_samples, chunk_size)):
        end = min(start + chunk_size, n_samples)
        chunks.append((chunk_idx, x_vals[start:end]))

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(num_confounders, confounding_strength, seed),
    ) as executor:
        chunk_results = list(executor.map(_generate_interventional_chunk, chunks))

    df_do = pd.concat(chunk_results, ignore_index=True)

    return df_obs, df_do, scm


def main():
    parser = argparse.ArgumentParser(
        description="Generate paired observational/interventional datasets from a backdoor SCM."
    )
    parser.add_argument(
        "--n_samples", type=int, default=18432, help="Number of samples (should cover train+test)"
    )
    parser.add_argument("--num_confounders", type=int, default=2, help="Number of confounders")
    parser.add_argument(
        "--confounding_strength", type=float, default=1.0, help="Backdoor confounding strength"
    )
    parser.add_argument("--seed", type=int, default=27, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--n_workers", type=int, default=None, help="Number of worker processes (default: CPUs-1)"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=256, help="Number of samples per parallel chunk"
    )
    args = parser.parse_args()

    df_obs, df_do, scm = generate_paired_datasets(
        args.n_samples,
        args.num_confounders,
        args.confounding_strength,
        args.seed,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    # Plain names keep the consumption side (e.g. test_joint_fit.py) simple.
    obs_path = os.path.join(
        args.output_dir,
        f"observational_N{args.n_samples}_Z{args.num_confounders}_CF{args.confounding_strength}_S{args.seed}.csv",
    )
    do_path = os.path.join(
        args.output_dir,
        f"interventional_N{args.n_samples}_Z{args.num_confounders}_CF{args.confounding_strength}_S{args.seed}.csv",
    )
    scm_path = os.path.join(
        args.output_dir,
        f"scm_Z{args.num_confounders}_CF{args.confounding_strength}_S{args.seed}.pkl",
    )

    df_obs.to_csv(obs_path, index=False)
    df_do.to_csv(do_path, index=False)
    with open(scm_path, "wb") as f:
        pickle.dump(scm, f)

    print(f"Saved observational data to {obs_path}")
    print(f"Saved interventional data to {do_path}")
    print(f"Saved SCM to {scm_path}")


if __name__ == "__main__":
    main()
