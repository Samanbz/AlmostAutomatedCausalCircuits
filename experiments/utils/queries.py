"""Causal query compilation: estimand ASTs -> executable circuits.

:func:`experiments.utils.identification.identify_estimands` (T-ID) produces
the estimand ASTs; this module compiles them into circuits that share
parameter tensors with the base circuit, so they track EM updates live and
can be evaluated at any point during training.
"""

from typing import Any, Dict

from src.logger import logger as g_logger
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
    base_root_id = ac.get_roots()[0]
    var_to_id = data_info["var_to_id"]

    logger.info("Compiling query circuits...")
    obs_num_ac, _ = compile_query(estimands["obs_num"], ac, base_root_id, var_to_id)
    obs_den_ac, _ = compile_query(estimands["obs_den"], ac, base_root_id, var_to_id)
    q_do_ac, _ = compile_query(estimands["do"], ac, base_root_id, var_to_id)
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
