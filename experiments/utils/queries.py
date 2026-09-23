"""Causal query compilation: estimand ASTs -> executable circuits.

:func:`experiments.utils.identification.identify_estimands` (T-ID) produces
the estimand ASTs; this module compiles them into circuits that share
parameter tensors with the base circuit, so they track EM updates live and
can be evaluated at any point during training.
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch

from src.logger import logger as g_logger
from src.symbolic.arithmetic.circuit import eval_circuit
from src.symbolic.arithmetic.query import compile_query


logger = g_logger.getChild("queries")


def compile_queries(ac, estimands: Dict[str, Any], data_info: Dict[str, Any]) -> Dict[str, Any]:
    """Compile the query circuits the experiment evaluates.

    Args:
      * ``estimands`` — output of :func:`identify_estimands`
        (``obs_num`` = P(Y,X), ``obs_den`` = P(X), ``do`` = T-ID COAST for
        P(Y|do(X)), ``info`` = identification metadata).

    Returns:
      * ``obs_num_ac`` — P(Y, X)
      * ``obs_den_ac`` — P(X)
      * ``q_do_ac``    — P(Y | do(X)), identified by T-ID
      * ``identification_info`` — complexity K, determinism family, graph edges
    """
    var_to_id = data_info["var_to_id"]

    logger.info("Compiling query circuits...")
    obs_num_ac = compile_query(estimands["obs_num"], ac, var_to_id)
    obs_den_ac = compile_query(estimands["obs_den"], ac, var_to_id)
    q_do_ac = compile_query(estimands["do"], ac, var_to_id)
    logger.info("Query circuits compiled.")
    logger.info(
        "Query circuit parameters: P(Y,X)=%d, P(X)=%d, P(Y|do(X))=%d",
        obs_num_ac.num_parameters(),
        obs_den_ac.num_parameters(),
        q_do_ac.num_parameters(),
    )

    return {
        "obs_num_ac": obs_num_ac,
        "obs_den_ac": obs_den_ac,
        "q_do_ac": q_do_ac,
        "identification_info": estimands["info"],
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
        log_do = eval_circuit(q_do_ac, pts_t).squeeze().detach().cpu().numpy()
        p_do = np.exp(log_do)
        log_num = eval_circuit(obs_num_ac, pts_t).squeeze().detach().cpu().numpy()
        log_den = eval_circuit(obs_den_ac, pts_t).squeeze().detach().cpu().numpy()
        p_obs = np.exp(log_num - log_den)

    return p_do, p_obs
