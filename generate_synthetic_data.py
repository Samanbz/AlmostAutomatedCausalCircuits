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


def _init_worker(scm_kwargs):
    """Rebuild the same SCM in every worker process (avoids pickling the SCM)."""
    global _worker_scm, _worker_seed
    _worker_scm = build_synthetic_continuous_scm(**scm_kwargs)
    _worker_seed = scm_kwargs.get("seed", 0)


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


def _fmt(value):
    """Short, filename-friendly float representation."""
    return f"{float(value):.3g}"


def _filename_suffix(scm_kwargs):
    """Build a filename suffix that captures any non-default tunable parameters."""
    # direct_effect and obs_slope are already encoded in the base name.
    defaults = {
        "do_std": None,
        "obs_std": None,
        "x_confound_strength": 1.0,
        "x_noise_std": 1.0,
        "z_noise_std": 1.0,
        "y_noise_std": None,
    }
    prefix = {
        "do_std": "DS",
        "obs_std": "OB",
        "x_confound_strength": "XS",
        "x_noise_std": "XN",
        "z_noise_std": "ZN",
        "y_noise_std": "YN",
    }
    parts = []
    for key in prefix:
        if key in scm_kwargs and scm_kwargs[key] != defaults[key]:
            parts.append(f"{prefix[key]}{_fmt(scm_kwargs[key])}")
    return "_".join(parts)


def generate_paired_datasets(
    n_samples: int = 18432,
    num_confounders: int = 4,
    seed: int = 27,
    n_workers: int = None,
    chunk_size: int = 256,
    **scm_kwargs,
):
    """Generate a paired observational / interventional dataset from a backdoor SCM.

    The interventional dataset represents P(Y | do(X)): each row fixes X to the
    corresponding observational X_i and draws the remaining variables (Z, Y) from
    the intervened SCM with fresh noise.

    Extra keyword arguments are forwarded to ``build_synthetic_continuous_scm``,
    so you can set ``direct_effect``, ``obs_slope``, ``do_std``, ``obs_std``,
    etc. to control the generated regime.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    scm_kwargs.update(
        {
            "num_confounders": num_confounders,
            "seed": seed,
        }
    )
    scm = build_synthetic_continuous_scm(**scm_kwargs)

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
        initargs=(scm_kwargs,),
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
    parser.add_argument("--num_confounders", type=int, default=4, help="Number of confounders")
    parser.add_argument("--seed", type=int, default=27, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--n_workers", type=int, default=None, help="Number of worker processes (default: CPUs-1)"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=256, help="Number of samples per parallel chunk"
    )

    # Tunable SCM parameters.
    parser.add_argument(
        "--direct_effect", type=float, default=1.0, help="Slope of P(Y|do(X)) (default: 1.0)"
    )
    parser.add_argument(
        "--obs_slope",
        type=float,
        default=None,
        help="Slope of P(Y|X).  Defaults to direct_effect + 1.0",
    )
    parser.add_argument("--do_std", type=float, default=None, help="Std of P(Y|do(X))")
    parser.add_argument(
        "--obs_std", type=float, default=None, help="Std of P(Y|X); must be <= do_std"
    )
    parser.add_argument(
        "--x_strength",
        type=float,
        default=1.0,
        help="Coefficient of each Z in the X equation (default: 1.0)",
    )
    parser.add_argument(
        "--x_noise", type=float, default=1.0, help="Std of X exogenous noise (default: 1.0)"
    )
    parser.add_argument(
        "--z_noise", type=float, default=1.0, help="Std of Z exogenous noise (default: 1.0)"
    )
    parser.add_argument(
        "--y_noise", type=float, default=None, help="Std of Y exogenous noise (default: 1.0)"
    )

    args = parser.parse_args()

    scm_kwargs = {
        "direct_effect": args.direct_effect,
        "x_confound_strength": args.x_strength,
        "x_noise_std": args.x_noise,
        "z_noise_std": args.z_noise,
    }
    if args.obs_slope is not None:
        scm_kwargs["obs_slope"] = args.obs_slope
    if args.do_std is not None:
        scm_kwargs["do_std"] = args.do_std
    if args.obs_std is not None:
        scm_kwargs["obs_std"] = args.obs_std
    if args.y_noise is not None:
        scm_kwargs["y_noise_std"] = args.y_noise

    df_obs, df_do, scm = generate_paired_datasets(
        args.n_samples,
        args.num_confounders,
        args.seed,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        **scm_kwargs,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    suffix = _filename_suffix(scm_kwargs)
    if suffix:
        suffix = "_" + suffix

    obs_slope = args.obs_slope if args.obs_slope is not None else args.direct_effect + 1.0
    base_name = (
        f"N{args.n_samples}_Z{args.num_confounders}_"
        f"DE{args.direct_effect}_OS{obs_slope}_S{args.seed}"
    )
    obs_path = os.path.join(args.output_dir, f"observational_{base_name}{suffix}.csv")
    do_path = os.path.join(args.output_dir, f"interventional_{base_name}{suffix}.csv")
    scm_path = os.path.join(args.output_dir, f"scm_{base_name}{suffix}.pkl")

    df_obs.to_csv(obs_path, index=False)
    df_do.to_csv(do_path, index=False)
    with open(scm_path, "wb") as f:
        pickle.dump(scm, f)

    print(f"Saved observational data to {obs_path}")
    print(f"Saved interventional data to {do_path}")
    print(f"Saved SCM to {scm_path}")


if __name__ == "__main__":
    main()
