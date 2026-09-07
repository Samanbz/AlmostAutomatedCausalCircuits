"""Dataset loading and MD-circuit construction."""

import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from src.construction.circuit_builder import create_md_circuit
from src.construction.learned_vtree import construct_optimal_md_vtree
from src.logger import logger as g_logger
from src.symbolic.arithmetic.nodes import GaussianDistribution
from src.symbolic.arithmetic.nodes.leaf_layer import LogLinearSplineDistribution
from src.symbolic.io_utils import plot_dag


logger = g_logger.getChild("data")


def load_data(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load observational / interventional CSVs and split train/test."""
    ds = cfg["dataset"]
    data_dir = ds["data_dir"]
    dataset_name = ds["dataset"]
    train_size = ds["train_size"]

    obs_path = os.path.join(data_dir, f"observational_{dataset_name}.csv")
    do_path = os.path.join(data_dir, f"interventional_{dataset_name}.csv")

    if not os.path.exists(obs_path) or not os.path.exists(do_path):
        raise FileNotFoundError(
            f"Dataset not found. Expected:\n  {obs_path}\n  {do_path}\n"
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
    df_do_test = df_do_full.iloc[n_train : n_train + n_test]

    data_np = df_obs.values
    data = torch.tensor(data_np, dtype=torch.float32)
    data_mean = data.mean(0).numpy()
    data_std = data.std(0).numpy()

    var_names = list(df_obs.columns)
    var_to_id = {v: i for i, v in enumerate(var_names)}
    n_vars = len(var_names)

    n_confounders = cfg["model"]["n_confounders"]
    z_names = [f"Z{i}" for i in range(n_confounders)]
    z_ids = [var_to_id[z] for z in z_names]
    x_id = var_to_id["X"]
    y_id = var_to_id["Y"]
    V_names = set(var_names)

    logger.info(
        "Loaded %d observational samples (%d train / %d test)",
        len(df_obs_full),
        n_train,
        n_test,
    )
    logger.info("Variables: %s", var_names)
    logger.info("Confounders: %s", z_names)
    # logger.info("Data mean: %s", data_mean.round(4).tolist())
    # logger.info("Data std:  %s", data_std.round(4).tolist())

    return {
        "df_test": df_test,
        "df_obs": df_obs,
        "df_do": df_do,
        "df_do_test": df_do_test,
        "data": data,
        "data_mean": data_mean,
        "data_std": data_std,
        "var_to_id": var_to_id,
        "z_names": z_names,
        "z_ids": z_ids,
        "x_id": x_id,
        "y_id": y_id,
        "n_vars": n_vars,
        "V_names": V_names,
    }


def _make_leaf_distribution(
    var: int,
    dist_cfg: Dict[str, Any],
    data_mean: np.ndarray,
    data_std: np.ndarray,
) -> Any:
    """Instantiate a leaf distribution from the config block."""
    dist_type = dist_cfg.get("distribution", "log_linear_spline").lower()
    scale = dist_cfg.get("base_stddev_scale", 2.0)

    if dist_type == "gaussian":
        return GaussianDistribution(
            var=var,
            base_mean=float(data_mean[var]),
            base_stddev=float(data_std[var]) * scale,
        )
    if dist_type in {"log_linear_spline", "spline"}:
        return LogLinearSplineDistribution(
            var=var,
            base_mean=float(data_mean[var]),
            base_stddev=float(data_std[var]) * scale,
        )
    raise ValueError(f"Unsupported leaf distribution: {dist_type}")


def build_circuit(
    cfg: Dict[str, Any],
    data: torch.Tensor,
    data_info: Dict[str, Any],
    device: torch.device,
    output_dir: str,
):
    """Build the MD circuit and move it to the target device."""
    model_cfg = cfg["model"]
    n_confounders = model_cfg["n_confounders"]
    num_nodes = model_cfg["num_nodes"]
    prioritize = "hardware" if model_cfg["prioritize"] == "H" else "expressivity"

    z_ids = data_info["z_ids"]
    x_id = data_info["x_id"]
    y_id = data_info["y_id"]

    md_sets = [set([x_id] + z_ids)]
    md_vtree = construct_optimal_md_vtree(
        data,
        md_sets,
        prioritize=prioritize,
        keep_together=[(x_id, y_id)],
    )

    vtree_plot_path = os.path.join(
        output_dir, f"md_vtree_{n_confounders}_{model_cfg['md_sets']}_{model_cfg['prioritize']}.svg"
    )
    plot_dag(md_vtree, output_path=vtree_plot_path)
    logger.info("Saved vTree plot to %s", vtree_plot_path)

    leaf_cfg = model_cfg.get("leaf", {})
    dists = {}
    for name, vid in data_info["var_to_id"].items():
        dists[vid] = _make_leaf_distribution(
            vid,
            leaf_cfg.get(name, leaf_cfg.get("default", {"distribution": "log_linear_spline"})),
            data_info["data_mean"],
            data_info["data_std"],
        )

    ac = create_md_circuit(
        dists,
        md_vtree,
        num_nodes=num_nodes,
        initialize_weights=True,
        fairness_temperature=model_cfg.get("fairness_temperature", 1.0),
    )
    ac.to(device)

    logger.info("Built MD circuit with %d nodes on %s", num_nodes, device)
    return ac, dists
