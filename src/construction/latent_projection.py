"""Verma's latent projection: project an SCM with hidden variables onto a CausalGraph.

Given a :class:`~src.symbolic.scm.graph.StructuralCausalModel` whose endogenous nodes may
be flagged hidden (``StructuralCausalModel.hidden_variables``), the released/observed
graph is the *latent projection* onto the observed nodes only: an acyclic directed mixed
graph (ADMG) with directed edges (A -> B) and bidirected edges (A <-> B) encoding
dependencies routed through marginalized latents.

Reference: Panayiotou et al. 2026 (CausalProfiler), Appendix C, Algorithm 3
(Verma's latent projection).

The projection is purely structural: mechanisms are untouched, and the full SCM remains
the ground-truth object (latents are marginalized analytically by the ground-truth
engine).
"""

from typing import Any, Dict, Optional, Set, Tuple

from src.symbolic.causal_graph import CausalGraph, _canonical_pair
from src.symbolic.scm import StructuralCausalModel


def _observed_reachable(
    scm: StructuralCausalModel, source: Any, latent: Set[Any], observed: Set[Any]
) -> Set[Any]:
    """Observed nodes reachable from ``source`` via a directed path whose
    intermediate nodes are all latent."""
    reached = set()
    frontier = [source]
    visited = {source}
    while frontier:
        node = frontier.pop()
        for child in scm.get_children(node):
            if child == source:
                continue
            if child in observed:
                reached.add(child)
            elif child in latent and child not in visited:
                visited.add(child)
                frontier.append(child)
    return reached


def latent_projection(scm: StructuralCausalModel, hidden: Optional[Set[Any]] = None) -> CausalGraph:
    """Verma's latent projection of ``scm`` onto its observed nodes.

    The observed set V_O is every node that is neither hidden nor exogenous.
    For observed nodes A, B:

    - a directed edge A -> B is added iff a directed path A -> ... -> B exists whose
      intermediate nodes are all latent;
    - a bidirected edge A <-> B is added iff a collider-free path between A and B
      exists whose intermediate nodes are all latent. In a DAG this is equivalent to
      a latent trek: some latent node L with latent-intermediate directed paths
      L ->> A and L ->> B (L is the trek's topmost source).

    Exogenous variables (noise-style nodes) count as latent for path traversal but
    are never part of V_O, so observed nodes sharing an exogenous parent (a
    c-component) get a bidirected edge.

    ``hidden`` defaults to ``scm.hidden_variables``.
    """
    hidden = set(scm.hidden_variables) if hidden is None else set(hidden)
    exogenous = set(scm.exogenous_variables)
    latent = hidden | exogenous

    observed: Set[Any] = set()
    for node in scm.topological_sort():
        if node not in latent:
            observed.add(node)
    nodes = [n for n in scm.topological_sort() if n in observed]

    directed_edges: Set[Tuple[Any, Any]] = set()
    bidirected_edges: Set[Tuple[Any, Any]] = set()

    # Directed edges: BFS from each observed source through latent intermediates.
    reachability: Dict[Any, Set[Any]] = {}
    for a in nodes:
        reachability[a] = _observed_reachable(scm, a, latent, observed)
        for b in reachability[a]:
            if b != a:
                directed_edges.add((a, b))

    # Bidirected edges: for every latent trek source L, pair up its observed reach.
    all_nodes = set(scm.topological_sort())
    for latent_source in latent:
        if latent_source not in all_nodes:
            continue
        reached = _observed_reachable(scm, latent_source, latent, observed)
        reached_list = sorted(reached, key=repr)
        for i, a in enumerate(reached_list):
            for b in reached_list[i + 1 :]:
                bidirected_edges.add(_canonical_pair(a, b))

    return CausalGraph(nodes, directed_edges, bidirected_edges)
