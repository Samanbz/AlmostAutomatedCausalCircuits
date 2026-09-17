"""Estimand identification via T-ID.

Step 2 of the experiment recipe: identify the interventional estimand
``P(outcome | do(treatment))`` from the dataset's SCM with the real T-ID
algorithm (:func:`src.symbolic.identification.ID`), running on the latent
projection of the pickled SCM (hidden confounders become bidirected edges).

The determinism family ``D`` — what the base circuit is marginally
deterministic over — is taken from the MD sets the circuit was actually
built with (``model.md_sets``, resolved in
:mod:`experiments.utils.data`), so the tractability check is always
relative to the circuit at hand, never to an assumed one.

Optional config section::

    "identification": {
        "treatment": "X",                       # intervened variable (default "X")
        "outcome": "Y",                         # target variable     (default "Y")
        "determinisms": [["X", "M"]]            # override D; default: the circuit's md_sets
    }

Raises :class:`IdentificationError` when the query is not identifiable from
the observational distribution, and :class:`TractabilityError` when
identification needs a conditioning scope the base circuit is not marginally
deterministic over — in that case choose a richer ``model.md_sets``.
"""

import pickle
from typing import Any, Dict, List, Set

from src.construction.latent_projection import latent_projection
from src.logger import logger as g_logger
from src.symbolic.causal_graph import CausalGraph
from src.symbolic.id_ast import EstimandAST, make_marg, make_p
from src.symbolic.identification import (
    ID,
    IdentificationError,
    TractabilityError,
    UncompiledCircuit,
)


logger = g_logger.getChild("identification")


def _load_observed_graph(data_info: Dict[str, Any]) -> CausalGraph:
    """Latent projection of the pickled SCM onto the observed variables."""
    scm_path = data_info.get("scm_path")
    if scm_path is None:
        raise FileNotFoundError(
            "T-ID needs the SCM pickle (scm.pkl) next to the dataset CSVs; "
            "regenerate the dataset with generate_synthetic_data.py."
        )
    with open(scm_path, "rb") as f:
        scm = pickle.load(f)
    graph = latent_projection(scm)
    missing = set(graph.nodes) - set(data_info["V_names"])
    if missing:
        raise ValueError(
            f"Observed graph variables {sorted(map(str, missing))} are missing from the "
            f"dataset columns {sorted(map(str, data_info['V_names']))}."
        )
    return graph


def _format_ast(ast: EstimandAST) -> str:
    """Compact indented rendering of the COAST for the log file."""
    lines: List[str] = []

    def rec(node_id: int, indent: int = 0) -> None:
        lines.append("  " * indent + str(ast.get_node_data(node_id)))
        for child_id, _ in ast.get_outgoing_edges(node_id):
            rec(child_id, indent + 1)

    rec(ast.get_root())
    return "\n".join(lines)


def identify_estimands(cfg: Dict[str, Any], data_info: Dict[str, Any]) -> Dict[str, Any]:
    """Identify ``P(outcome | do(treatment))`` and build observational ``P(outcome | treatment)``.

    Returns:
      * ``obs_num`` — P(outcome, treatment)
      * ``obs_den`` — P(treatment)
      * ``do``      — T-ID COAST (EstimandAST) for P(outcome | do(treatment))
      * ``info``    — treatment/outcome, complexity exponent K, determinism family,
                      and the observed graph edges (for metadata / debugging)
    """
    id_cfg = cfg.get("identification", {})
    treatment = id_cfg.get("treatment", "X")
    outcome = id_cfg.get("outcome", "Y")
    v_names: Set[str] = set(data_info["V_names"])
    for name in (treatment, outcome):
        if name not in v_names:
            raise ValueError(
                f"identification: '{name}' is not among the dataset columns "
                f"{sorted(map(str, v_names))}"
            )

    # Determinism family: what the base circuit is marginally deterministic over
    # (its MD sets), unless the config overrides it explicitly.
    det_cfg = id_cfg.get("determinisms")
    det_names = [set(s) for s in det_cfg] if det_cfg is not None else data_info["md_set_names"]
    determinisms = [frozenset(s) for s in det_names]
    unknown = set().union(*det_names) - v_names if det_names else set()
    if unknown:
        raise ValueError(
            f"identification.determinisms reference unknown variables "
            f"{sorted(map(str, unknown))}; dataset columns are {sorted(map(str, v_names))}"
        )

    graph = _load_observed_graph(data_info)
    logger.info(
        "Observed graph: directed=%s bidirected=%s",
        sorted(map(str, graph.directed)),
        sorted(map(str, graph.bidirected)),
    )
    logger.info(
        "Determinism family D = %s",
        [sorted(map(str, d)) for d in determinisms],
    )

    p = UncompiledCircuit(
        ast=make_p(v_names),
        determinisms=set(determinisms),
        complexity=1.0,
        is_base=True,
    )
    try:
        result = ID({outcome}, {treatment}, p, graph)
    except IdentificationError as e:
        raise IdentificationError(
            f"P({outcome}|do({treatment})) is not identifiable from the observational "
            f"distribution on this graph (directed={sorted(map(str, graph.directed))}, "
            f"bidirected={sorted(map(str, graph.bidirected))})."
        ) from e
    except TractabilityError as e:
        raise TractabilityError(
            f"{e} The base circuit only guarantees marginal determinism over "
            f"{[sorted(map(str, d)) for d in determinisms]}; choose a richer "
            f'"model.md_sets" (or set "identification.determinisms") so the '
            f"required conditioning scopes are covered."
        ) from e

    logger.info(
        "T-ID identified P(%s|do(%s)): complexity K = %.1f, determinisms %s",
        outcome,
        treatment,
        result.complexity,
        [sorted(map(str, d)) for d in result.determinisms],
    )
    logger.info("COAST:\n%s", _format_ast(result.ast))

    # Observational P(outcome|treatment) = P(outcome, treatment) / P(treatment).
    # No graph needed; both compiled from the base circuit.
    p_all = make_p(v_names)
    obs_num = make_marg(v_names - {treatment, outcome}, p_all)  # P(outcome, treatment)
    obs_den = make_marg({outcome}, obs_num)  # P(treatment)

    return {
        "obs_num": obs_num,
        "obs_den": obs_den,
        "do": result.ast,
        "info": {
            "treatment": treatment,
            "outcome": outcome,
            "complexity": result.complexity,
            "determinisms": [sorted(map(str, d)) for d in determinisms],
            "graph_directed": sorted(map(str, graph.directed)),
            "graph_bidirected": sorted(map(str, graph.bidirected)),
        },
    }
