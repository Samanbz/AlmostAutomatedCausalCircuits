"""Dataset loading and MD-circuit construction."""

import os
from glob import glob
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch

from src.construction.circuit_builder import create_md_circuit
from src.construction.learned_vtree import construct_optimal_md_vtree
from src.construction.vtree_spec import build_vtree_from_spec
from src.logger import logger as g_logger
from src.symbolic.arithmetic.nodes import CategoricalDistribution, GaussianDistribution
from src.symbolic.arithmetic.nodes.leaf_layer import LogLinearSplineDistribution


logger = g_logger.getChild("data")


def _prefix_class(letter: str, var_names: List[str]) -> Set[str]:
    """Resolve one shorthand letter to the set of variables it refers to.

    ``"Z"``/``"M"``/``"W"`` match every ``Z_*``/``M_*``/``W_*`` column; a
    letter with no prefixed columns matches the bare column (``"X"`` -> ``{"X"}``).
    """
    prefixed = {v for v in var_names if v.startswith(letter + "_")}
    if prefixed:
        return prefixed
    if letter in var_names:
        return {letter}
    return set()


def resolve_md_set_names(
    model_cfg: Dict[str, Any], var_names: List[str], z_names: List[str]
) -> List[Set[str]]:
    """Resolve ``model.md_sets`` to a list of MD-scope name sets.

    Accepted forms (each MD set is a scope of variables the circuit will be
    marginally deterministic over — also T-ID's determinism family):

    * key absent        -> ``[{"X", *z_names}]`` (legacy hard-coded behavior)
    * ``"XZ"`` (string) -> one set; each character is a prefix class
    * ``["XZ", "XM"]``  -> one set per string
    * ``[["X", "M_0"]]``-> explicit variable names per set
    """
    md_cfg = model_cfg.get("md_sets")
    if md_cfg is None:
        return [{"X", *z_names}]

    specs: List[Any]
    if isinstance(md_cfg, str):
        specs = [md_cfg]
    else:
        specs = list(md_cfg)

    md_set_names: List[Set[str]] = []
    for spec in specs:
        if isinstance(spec, str):
            scope: Set[str] = set()
            for letter in spec:
                cls = _prefix_class(letter, var_names)
                if not cls:
                    raise ValueError(
                        f'model.md_sets: letter "{letter}" in "{spec}" matches no dataset '
                        f"column (available: {var_names})"
                    )
                scope |= cls
        else:
            scope = set(spec)
            unknown = scope - set(var_names)
            if unknown:
                raise ValueError(
                    f"model.md_sets: unknown variables {sorted(unknown)} (available: {var_names})"
                )
        if not scope:
            raise ValueError(f"model.md_sets: empty MD set in spec {spec!r}")
        md_set_names.append(scope)

    return md_set_names


def resolve_dataset_paths(data_dir: str, dataset_name: str) -> Dict[str, str]:
    """Resolve dataset file paths, preferring the per-dataset directory layout.

    Accepted layouts (first match wins):

    New layout, bare names (hand-curated):
        {data_dir}/{dataset_name}/observational.csv
        {data_dir}/{dataset_name}/interventional.csv
        {data_dir}/{dataset_name}/scm.pkl
        {data_dir}/{dataset_name}/meta.json

    New layout, generated names (generate_synthetic_data.py --dataset_name):
        {data_dir}/{dataset_name}/observational_<base_name>.csv
        {data_dir}/{dataset_name}/interventional_<base_name>.csv
        {data_dir}/{dataset_name}/scm_<base_name>.pkl
        {data_dir}/{dataset_name}/meta_<base_name>.json

    Fallback flat layout (legacy):
        {data_dir}/observational_{dataset_name}.csv
        ...
    """
    new_dir = os.path.join(data_dir, dataset_name)

    def first_existing(*candidates: str) -> str:
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    def glob_first(directory: str, pattern: str) -> str:
        matches = sorted(glob(os.path.join(directory, pattern)))
        return matches[0] if matches else os.path.join(directory, pattern)

    return {
        "observational": first_existing(
            os.path.join(new_dir, "observational.csv"),
            glob_first(new_dir, "observational_*.csv"),
            os.path.join(data_dir, f"observational_{dataset_name}.csv"),
        ),
        "interventional": first_existing(
            os.path.join(new_dir, "interventional.csv"),
            glob_first(new_dir, "interventional_*.csv"),
            os.path.join(data_dir, f"interventional_{dataset_name}.csv"),
        ),
        "scm": first_existing(
            os.path.join(new_dir, "scm.pkl"),
            glob_first(new_dir, "scm_*.pkl"),
            os.path.join(data_dir, f"scm_{dataset_name}.pkl"),
        ),
        "meta": first_existing(
            os.path.join(new_dir, "meta.json"),
            glob_first(new_dir, "meta_*.json"),
            os.path.join(data_dir, f"meta_{dataset_name}.json"),
        ),
    }


def load_data(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load observational / interventional CSVs and split train/test."""
    ds = cfg["dataset"]
    data_dir = ds["data_dir"]
    dataset_name = ds["dataset"]
    train_size = ds["train_size"]

    paths = resolve_dataset_paths(data_dir, dataset_name)
    obs_path = paths["observational"]
    do_path = paths["interventional"]

    if not os.path.exists(obs_path) or not os.path.exists(do_path):
        raise FileNotFoundError(
            f"Dataset not found. Expected one of:\n"
            f"  {os.path.join(data_dir, dataset_name, 'observational.csv')}\n"
            f"  {os.path.join(data_dir, dataset_name, 'interventional.csv')}\n"
            "Generate it first, e.g. with generate_synthetic_data.py."
        )

    df_obs_full = pd.read_csv(obs_path)
    df_do_full = pd.read_csv(do_path)

    n_train = int(train_size * len(df_obs_full))
    n_test = len(df_obs_full) - n_train

    df_obs = df_obs_full.iloc[:n_train]
    df_test = df_obs_full.iloc[n_train : n_train + n_test]
    df_do = df_do_full.iloc[:n_train]
    df_do_test = df_do_full.iloc[n_train : n_train + n_test]

    data_np = df_obs.values
    data = torch.tensor(data_np, dtype=torch.float32)
    data_mean = data.mean(0).numpy()
    data_std = data.std(0).numpy()

    var_names = list(df_obs.columns)
    var_to_id = {v: i for i, v in enumerate(var_names)}
    n_vars = len(var_names)

    n_confounders = cfg["model"]["n_confounders"]
    z_candidates = sorted([v for v in var_names if v.startswith("Z")])
    if len(z_candidates) < n_confounders:
        raise ValueError(
            f"Config asks for {n_confounders} confounders but only found {z_candidates} "
            f"in columns {var_names}"
        )
    z_names = z_candidates[:n_confounders]
    z_ids = [var_to_id[z] for z in z_names]
    x_id = var_to_id["X"]
    y_id = var_to_id["Y"]
    V_names = set(var_names)

    # MD scopes the circuit is built with (model.md_sets); T-ID's determinism
    # family is derived from these so both speak about the same scopes.
    md_set_names = resolve_md_set_names(cfg["model"], var_names, z_names)
    md_sets = [{var_to_id[v] for v in s} for s in md_set_names]

    logger.info(
        "Loaded %d observational samples (%d train / %d test)",
        len(df_obs_full),
        n_train,
        n_test,
    )
    logger.info("Variables: %s", var_names)
    logger.info("Confounders: %s", z_names)
    logger.info("MD sets: %s", [sorted(s) for s in md_set_names])
    # logger.info("Data mean: %s", data_mean.round(4).tolist())
    # logger.info("Data std:  %s", data_std.round(4).tolist())

    return {
        "df_test": df_test,
        "df_obs": df_obs,
        "df_obs_full": df_obs_full,
        "df_do": df_do,
        "df_do_test": df_do_test,
        "df_do_full": df_do_full,
        "data": data,
        "data_mean": data_mean,
        "data_std": data_std,
        "var_to_id": var_to_id,
        "z_names": z_names,
        "z_ids": z_ids,
        "md_set_names": md_set_names,
        "md_sets": md_sets,
        "x_id": x_id,
        "y_id": y_id,
        "n_vars": n_vars,
        "V_names": V_names,
        "scm_path": paths["scm"] if os.path.exists(paths["scm"]) else None,
        "meta_path": paths["meta"] if os.path.exists(paths["meta"]) else None,
    }


def _make_leaf_distribution(
    var: int,
    dist_cfg: Dict[str, Any],
    data_mean: np.ndarray,
    data_std: np.ndarray,
    categories: List[Any] = None,
) -> Any:
    """Instantiate a leaf distribution from the config block."""
    dist_type = dist_cfg.get("distribution", "log_linear_spline").lower()

    if dist_type == "gaussian":
        return GaussianDistribution(
            var=var,
            base_mean=float(data_mean[var]),
            base_stddev=float(data_std[var]),
        )
    if dist_type == "categorical":
        if categories is None:
            raise ValueError(
                "Categorical leaf requires discrete observed values; "
                "got a continuous column (or no categories passed)."
            )
        return CategoricalDistribution(
            var=var,
            categories=list(categories),
            probabilities=[1.0 / len(categories)] * len(categories),
        )
    if dist_type in {"log_linear_spline", "spline"}:
        return LogLinearSplineDistribution(
            var=var,
            base_mean=float(data_mean[var]),
            base_stddev=float(data_std[var]),
        )
    raise ValueError(f"Unsupported leaf distribution: {dist_type}")


def discrete_categories(
    cfg: Dict[str, Any], data_info: Dict[str, Any], var_name: str
) -> Optional[List[float]]:
    """Sorted observed category values if ``var_name`` uses a categorical leaf.

    Returns None for continuous variables (spline/Gaussian leaves).  Plotting
    uses this to pick category grids / bar rendering instead of quantile
    grids / density curves.
    """
    leaf_cfg = cfg.get("model", {}).get("leaf", {})
    block = leaf_cfg.get(var_name, leaf_cfg.get("default", {}))
    if block.get("distribution", "log_linear_spline").lower() != "categorical":
        return None
    return sorted(float(v) for v in data_info["df_obs_full"][var_name].unique())


def build_circuit(
    cfg: Dict[str, Any],
    data: torch.Tensor,
    data_info: Dict[str, Any],
    device: torch.device,
):
    """Build the MD circuit and move it to the target device."""
    model_cfg = cfg["model"]
    num_nodes = model_cfg["num_nodes"]

    x_id = data_info["x_id"]
    y_id = data_info["y_id"]

    md_sets = data_info["md_sets"]
    vtree_spec = model_cfg.get("vtree")
    if vtree_spec:
        # Config-pinned shape (Newick-style, e.g. "((Z,X),(M,Y))"); the MD
        # labeling is still computed from md_sets. Use for structures whose
        # COAST kernels require a specific decomposition that learning does
        # not reliably find (e.g. colliderdoor).
        md_vtree = build_vtree_from_spec(vtree_spec, data_info["var_to_id"])
        md_vtree.compute_md_labeling(md_sets)
        logger.info("Using config-pinned vtree: %s", vtree_spec)
    else:
        md_vtree = construct_optimal_md_vtree(data, md_sets, keep_together=[(x_id, y_id)])

    leaf_cfg = model_cfg.get("leaf", {})
    dists = {}
    df_obs = data_info["df_obs"]
    for name, vid in data_info["var_to_id"].items():
        dists[vid] = _make_leaf_distribution(
            vid,
            leaf_cfg.get(name, leaf_cfg.get("default", {"distribution": "log_linear_spline"})),
            data_info["data_mean"],
            data_info["data_std"],
            categories=sorted(int(v) for v in df_obs[name].unique())
            if leaf_cfg.get(name, leaf_cfg.get("default", {}))
            .get("distribution", "log_linear_spline")
            .lower()
            == "categorical"
            else None,
        )

    # Empirical per-variable samples so spline leaves initialize their splits
    # at the empirical 1/H-quantiles (matching boundary heights) instead of the
    # spec's Gaussian quantiles.
    leaf_quantile_data = {
        vid: data[:, vid].detach().cpu().numpy() for vid in data_info["var_to_id"].values()
    }

    ac = create_md_circuit(
        dists,
        md_vtree,
        num_nodes=num_nodes,
        initialize_weights=True,
        fairness_temperature=model_cfg.get("fairness_temperature", 1.0),
        weight_softmax_temperature=model_cfg.get("weight_softmax_temperature", 0.5),
        max_leaf_num_nodes=model_cfg.get("max_leaf_num_nodes"),
        max_leaf_num_groups=model_cfg.get("max_leaf_num_groups"),
        max_sum_num_groups=model_cfg.get("max_sum_num_groups"),
        leaf_mixture_num_nodes=model_cfg.get("leaf_mixture_num_nodes"),
        leaf_mixture_num_groups=model_cfg.get("leaf_mixture_num_groups"),
        leaf_quantile_data=leaf_quantile_data,
    )
    ac.to(device)

    logger.info("Built MD circuit with %d nodes on %s", num_nodes, device)
    logger.info("Circuit parameters: %d", ac.num_parameters())
    return ac, dists
