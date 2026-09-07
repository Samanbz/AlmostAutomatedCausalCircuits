"""Causal query compilation and learned-conditional evaluation."""

from typing import Any, Dict, Tuple

import numpy as np
import torch

from src.logger import logger as g_logger
from src.symbolic.arithmetic.circuit import eval_circuit
from src.symbolic.arithmetic.query import compile_query
from src.symbolic.id_ast import make_cond, make_marg, make_p, make_prod


logger = g_logger.getChild("queries")


def compile_queries(
    ac,
    data_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Compile the query circuits needed for the P(Y|X) / P(Y|do(X)) plots.

    Returns only:
      * ``obs_num_ac`` — P(Y, X)
      * ``obs_den_ac`` — P(X)
      * ``q_do_ac``    — P(Y | do(X)) = sum_z P(Y | X, Z) P(Z)
    """
    V_names = data_info["V_names"]
    z_names = data_info["z_names"]
    base_root_id = ac.get_roots()[0]
    var_to_id = data_info["var_to_id"]

    p_all = make_p(V_names)

    # P(Y, X, Z) with other vars marginalized.
    other_vars = V_names - {"X", "Y"} - set(z_names)
    joint_xyz_ast = make_marg(other_vars, p_all)

    # Observational P(Y | X) = P(Y, X) / P(X)
    obs_num_ast = make_marg(set(z_names), joint_xyz_ast)
    obs_den_ast = make_marg({"Y"}, obs_num_ast)

    # Backdoor adjustment: sum_z P(Y | X, Z) P(Z)
    cond_yxz_ast = make_cond({"Y"}, {"X"} | set(z_names), joint_xyz_ast)
    marg_z_ast = make_marg({"X", "Y"}, p_all)
    do_ast = make_marg(set(z_names), make_prod([cond_yxz_ast, marg_z_ast]))

    logger.info("Compiling query circuits...")
    obs_num_ac, _ = compile_query(obs_num_ast, ac, base_root_id, var_to_id)
    obs_den_ac, _ = compile_query(obs_den_ast, ac, base_root_id, var_to_id)
    q_do_ac, _ = compile_query(do_ast, ac, base_root_id, var_to_id)
    logger.info("Query circuits compiled.")

    return {
        "obs_num_ac": obs_num_ac,
        "obs_den_ac": obs_den_ac,
        "q_do_ac": q_do_ac,
    }


def eval_conditionals_at_x(
    q_do_ac,
    obs_num_ac,
    obs_den_ac,
    x_val: float,
    grid_y: np.ndarray,
    data_info: Dict[str, Any],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate learned P(Y|do(X)) and P(Y|X) at a single X value."""
    n_vars = data_info["n_vars"]
    x_id = data_info["x_id"]
    y_id = data_info["y_id"]
    z_ids = data_info["z_ids"]
    data_mean = data_info["data_mean"]

    pts = np.zeros((len(grid_y), n_vars))
    pts[:, x_id] = x_val
    pts[:, y_id] = grid_y
    for zid in z_ids:
        pts[:, zid] = data_mean[zid]
    pts_t = torch.tensor(pts, dtype=torch.float32, device=device)

    with torch.no_grad():
        log_do = eval_circuit(q_do_ac, pts_t, verbose=False).squeeze().detach().cpu().numpy()
        p_do = np.exp(log_do)
        log_num = eval_circuit(obs_num_ac, pts_t, verbose=False).squeeze().detach().cpu().numpy()
        log_den = eval_circuit(obs_den_ac, pts_t, verbose=False).squeeze().detach().cpu().numpy()
        p_obs = np.exp(log_num - log_den)

    return p_do, p_obs
