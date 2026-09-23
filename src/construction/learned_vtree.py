"""VTree learning from data.

Primary algorithm: bottom-up induction of Liang, Bekker & Van den Broeck,
"Learning the Structure of Probabilistic Sentential Decision Diagrams" (UAI
2017), Section 3. Pairwise mutual information is estimated from data; vtrees
are built bottom-up by pairing, level by level, the components with maximum
average pairwise MI (pMI) via maximum-weight perfect matching. The paper shows
this bottom-up variant outperforms the top-down balanced min-cut variant,
because most interactions occur between small numbers of variables, making the
lower vtree levels more important. An optional MD labeling
(:meth:`VTree.compute_md_labeling`, Wang & Kwiatkowska) then turns the learned
vtree into a regular MD-vtree for causal tractability.

:func:`learn_liang_ps_md_vtree` additionally forces the (pseudo-)mixing layer
serving the finest required determinism to the root and keeps outcome
variables on the density side of every pseudo-mixing layer, warning when the
MD family forces additional pseudo-mixing layers deeper in the tree.

Both public MD constructors learn the structure with this bottom-up induction
and then label it with the Wang & Kwiatkowska MD labeling algorithm; there is no
top-down partitioning path.
"""

from typing import Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch

from src.construction.circuit_builder import LayerType
from src.logger import logger as g_logger
from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet


logger = g_logger.getChild("learned_vtree")

__all__ = [
    "learn_liang_vtree",
    "learn_liang_ps_md_vtree",
    "construct_optimal_vtree",
    "construct_optimal_md_vtree",
]


# ---------------------------------------------------------------------------
# Empirical pairwise mutual information
# ---------------------------------------------------------------------------


def _discretize_column(col: np.ndarray, max_bins: int) -> np.ndarray:
    """Integer-coded bins for one column; identity codes when already discrete."""
    _, codes = np.unique(col, return_inverse=True)
    n_unique = codes.max() + 1
    if n_unique <= max_bins:
        return codes.astype(np.int64)
    # Continuous / high-cardinality: quantile bins.
    ranks = np.argsort(np.argsort(col))
    bins = np.minimum((ranks * max_bins) // len(col), max_bins - 1)
    return bins.astype(np.int64)


def estimate_pairwise_mi_matrix(data: torch.Tensor, max_bins: int = 64) -> np.ndarray:
    """Exact empirical pairwise MI matrix.

    Each column is discretized to at most ``max_bins`` bins (quantile bins for
    continuous columns, identity codes for discrete ones), and MI is computed
    from the joint frequency table:

        MI(X, Y) = sum_xy p(x, y) log(p(x, y) / (p(x) p(y))).

    Returns a symmetric ``[n_vars, n_vars]`` array with zeros on the diagonal.
    """
    arr = data.detach().cpu().numpy().astype(np.float64)
    n_rows, n_vars = arr.shape
    cols = [_discretize_column(arr[:, i], max_bins) for i in range(n_vars)]
    bins = [int(c.max()) + 1 for c in cols]

    mi = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            joint = np.bincount(cols[i] * bins[j] + cols[j], minlength=bins[i] * bins[j])
            joint = joint.reshape(bins[i], bins[j]) / n_rows
            pi = joint.sum(axis=1, keepdims=True)
            pj = joint.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = joint * np.log(joint / (pi @ pj))
            mi[i, j] = mi[j, i] = float(np.nansum(terms))
    return mi


# ---------------------------------------------------------------------------
# Bottom-up vtree induction (Liang et al., UAI 2017, Sec. 3)
# ---------------------------------------------------------------------------


def _avg_pairwise_mi(mi: np.ndarray, a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    """pMI(A, B) = average pairwise MI over all cross pairs (paper, Eq. pMI)."""
    if len(a) == 1 and len(b) == 1:
        return float(mi[a[0], b[0]])
    sub = mi[np.ix_(list(a), list(b))]
    return float(sub.mean())


def _max_weight_perfect_matching(
    components: List[Tuple[int, ...]], mi: np.ndarray
) -> List[Tuple[int, int]]:
    """Maximum-weight perfect matching over the complete pMI-weighted graph.

    With an odd number of components, one is left unmatched (carried over to
    the next level), keeping the vtree near-balanced.
    """
    n = len(components)
    if n < 2:
        return []
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=_avg_pairwise_mi(mi, components[i], components[j]))
    matching = nx.max_weight_matching(G, maxcardinality=True)
    return sorted(tuple(sorted(e)) for e in matching)


def _liang_induce(
    vt: VTree,
    mi: np.ndarray,
    variables: List[int],
    keep_together: Optional[List[Tuple[int, int]]] = None,
) -> int:
    """Bottom-up pMI pairing over a subset of variables (Liang et al., Sec. 3).

    All nodes are added to ``vt``; returns the root node id of the induced
    subtree. ``mi`` is the full MI matrix — only the submatrix over
    ``variables`` is used, so several subtrees can be learned from one MI
    estimate.
    """
    var_set = set(variables)
    components: List[Tuple[Tuple[int, ...], int]] = []
    used: Set[int] = set()
    for u, v in keep_together or []:
        if u == v or u in used or v in used or u not in var_set or v not in var_set:
            continue
        # A keep_together pair becomes a two-leaf subtree, so the variables are
        # merged at the first level (split as deep/late in the vtree as
        # possible). Vtree leaves must be single variables.
        left_id = vt.add_node(VNode(scope=BitSet([u])))
        right_id = vt.add_node(VNode(scope=BitSet([v])))
        vid = vt.add_node(VNode(scope=BitSet([u, v])))
        vt.add_children(vid, left_id, right_id)
        components.append(((u, v), vid))
        used.update((u, v))
    for var in variables:
        if var not in used:
            vid = vt.add_node(VNode(scope=BitSet([var])))
            components.append(((var,), vid))

    sub_mi = mi[np.ix_(list(variables), list(variables))]
    pos = {v: i for i, v in enumerate(variables)}

    # Bottom-up levels: pair max-pMI components until one remains (the root).
    while len(components) > 1:
        local_tuples = [tuple(pos[v] for v in c[0]) for c in components]
        matching = _max_weight_perfect_matching(local_tuples, sub_mi)
        matched = {i for pair in matching for i in pair}
        next_level: List[Tuple[Tuple[int, ...], int]] = []
        for i, j in matching:
            (vars_i, vid_i), (vars_j, vid_j) = components[i], components[j]
            scope = BitSet(list(vars_i) + list(vars_j))
            vid = vt.add_node(VNode(scope=scope))
            vt.add_children(vid, vid_i, vid_j)
            next_level.append((vars_i + vars_j, vid))
        for i in range(len(components)):
            if i not in matched:
                next_level.append(components[i])  # odd carry-over
        components = next_level

    return components[0][1]


def learn_liang_vtree(
    data: torch.Tensor,
    keep_together: Optional[List[Tuple[int, int]]] = None,
    max_bins: int = 64,
) -> VTree:
    """Learn a vtree by bottom-up pMI induction (Liang et al., UAI 2017).

    Starts from singleton variables at the bottom and repeatedly pairs the
    components with maximum average pairwise mutual information, level by
    level, via maximum-weight perfect matching. Low-MI splits end up high in
    the vtree (near the root), where Proposition 1 context-specific
    independences justify few primes per decision node.

    ``keep_together`` pairs are pre-merged as first-level components, so they
    are split as deep (late) in the vtree as possible.
    """
    n_vars = data.shape[1]
    mi = estimate_pairwise_mi_matrix(data, max_bins=max_bins)
    vt = VTree()
    _liang_induce(vt, mi, list(range(n_vars)), keep_together)
    return vt


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------


def construct_optimal_vtree(
    data: torch.Tensor,
    keep_together: Optional[List[Tuple[int, int]]] = None,
    max_bins: int = 64,
) -> VTree:
    """Learn a data-driven vtree (Liang et al. bottom-up pMI induction)."""
    return learn_liang_vtree(data, keep_together=keep_together, max_bins=max_bins)


def construct_optimal_md_vtree(
    data: torch.Tensor,
    md_sets: List[Set[int]],
    keep_together: Optional[List[Tuple[int, int]]] = None,
    max_bins: int = 64,
) -> VTree:
    """Learn a vtree (Liang et al. bottom-up pMI induction) and label it for
    marginal determinism with the Wang & Kwiatkowska labeling algorithm
    (:meth:`VTree.compute_md_labeling`).

    The MD sets need not be kept contiguous during learning: the labeling
    algorithm produces a regular MD-vtree for any binary variable tree, which
    is both necessary and sufficient for the MD circuit construction and the
    query compiler.
    """
    _validate_md_family(md_sets)
    vt = learn_liang_vtree(data, keep_together=keep_together, max_bins=max_bins)
    vt.compute_md_labeling(md_sets)
    return vt


# ---------------------------------------------------------------------------
# PS-root MD vtree learner
# ---------------------------------------------------------------------------


def _validate_md_family(md_sets: List[Set[int]]) -> List[Set[int]]:
    """Intersection-closure validation shared by the MD vtree constructors."""
    md_sets = [set(s) for s in md_sets]
    for i in range(len(md_sets)):
        for j in range(i + 1, len(md_sets)):
            intersection = md_sets[i] & md_sets[j]
            if intersection not in md_sets:
                raise ValueError(
                    f"MD sets must be closed under intersection. "
                    f"{md_sets[i]} ∩ {md_sets[j]} = {intersection} is missing."
                )
    return md_sets


def learn_liang_ps_md_vtree(
    data: torch.Tensor,
    md_sets: List[Set[int]],
    y_vars: Iterable[int],
    keep_together: Optional[List[Tuple[int, int]]] = None,
    max_bins: int = 64,
    root_md_set: Optional[Set[int]] = None,
) -> VTree:
    """Learn a Liang bottom-up pMI vtree with the (pseudo-)mixing layer at the root.

    The root split is forced to separate ``root_md_set`` (default: the
    intersection of all required determinisms) from the remaining variables, so
    that

    * the root sum layer is the mixing layer serving conditioning on that
      determinism. Only the root may be a pseudo-mixing (support-overlapping)
      layer without consequences, since it has no parent that could rely on
      disjoint support of its children;
    * the outcome variables ``y_vars`` always sit on the universal (density)
      side of every pseudo-mixing layer — for the root a *left* pseudo-mixing
      layer, so Y ends up on the right. (A left pseudo-mixing layer only
      models the effect of its left (MD) side on its right side, never the
      reverse, so causes belong on the left and outcomes on the right.)

    Both subtrees are learned from data by Liang et al.'s bottom-up pMI
    induction. If the MD family forces *additional* pseudo-mixing layers below
    the root (an inclusion chain of N determinisms yields N-1 PS layers, e.g.
    {M,Z,X} ⊃ {M,X} ⊃ {X}), a warning is logged:
    conditioning on the md-set of a non-root PS layer alone is not normalized,
    because such layers share support across parent units by construction.

    Raises if any md-set contains an outcome variable (the Y-side rule would
    be unfulfillable) or if the forced root split leaves no variables on the
    density side.
    """

    n_vars = data.shape[1]
    y_set = set(y_vars)
    if not all(0 <= v < n_vars for v in y_set):
        raise ValueError(f"Outcome variables {y_set} out of range for {n_vars} columns.")

    md_sets = _validate_md_family(md_sets)
    for s in md_sets:
        if s & y_set:
            raise ValueError(
                f"MD set {sorted(s)} contains outcome variables {sorted(s & y_set)}; "
                "outcomes must stay on the density side of pseudo-mixing layers."
            )

    if root_md_set is None:
        root_md_set = set.intersection(*md_sets)
    root_md_set = set(root_md_set)
    if root_md_set not in md_sets:
        raise ValueError(f"root_md_set {sorted(root_md_set)} is not a member of the MD family.")

    selector = sorted(root_md_set)
    rest = [v for v in range(n_vars) if v not in root_md_set]
    if not rest:
        raise ValueError("The forced root split leaves no variables on the density side.")

    mi = estimate_pairwise_mi_matrix(data, max_bins=max_bins)
    kt = [tuple(p) for p in (keep_together or [])]
    kt_selector = [p for p in kt if p[0] in root_md_set and p[1] in root_md_set]
    kt_rest = [p for p in kt if p[0] not in root_md_set and p[1] not in root_md_set]
    dropped = [p for p in kt if p not in kt_selector and p not in kt_rest]
    if dropped:
        logger.warning(
            f"learn_liang_ps_md_vtree: dropping keep_together pairs crossing the root "
            f"split {root_md_set} | rest: {dropped}"
        )

    vt = VTree()
    left_root = _liang_induce(vt, mi, selector, kt_selector)
    right_root = _liang_induce(vt, mi, rest, kt_rest)
    root = vt.add_node(VNode(scope=BitSet(selector + rest)))
    vt.add_children(root, left_root, right_root)

    vt.compute_md_labeling(md_sets)

    # --- Verify the PS structure and report violations / extra PS layers. ---
    ps_layers: List[Tuple[int, BitSet, BitSet]] = []  # (vid, scope, selector scope)

    def _walk(vid: int) -> None:
        children = vt.get_children_pair(vid)
        if children is None:
            return
        l_vid, r_vid = children
        node = vt.get_node_data(vid)
        l_node = vt.get_node_data(l_vid)
        r_node = vt.get_node_data(r_vid)
        layer_type = LayerType.from_md_sets(node.md_set, l_node.md_set, r_node.md_set)
        if layer_type in (LayerType.PS_LEFT_MIXING, LayerType.PS_RIGHT_MIXING):
            sel_scope = l_node.scope if layer_type == LayerType.PS_LEFT_MIXING else r_node.scope
            ps_layers.append((vid, node.scope, sel_scope))
            if y_set & set(sel_scope):
                logger.warning(
                    f"Pseudo-mixing layer at scope {set(node.scope)} has outcome variables "
                    f"{sorted(y_set & set(sel_scope))} on its selector (md) side."
                )
        _walk(l_vid)
        _walk(r_vid)

    _walk(root)

    root_is_ps = ps_layers and ps_layers[0][0] == root
    if not ps_layers:
        logger.info(
            "learn_liang_ps_md_vtree: root is a classical disjoint mixing layer "
            "(no pseudo-mixing needed for this MD family)."
        )
    elif not root_is_ps:
        logger.warning("learn_liang_ps_md_vtree: root is not a pseudo-mixing layer.")
    if len(ps_layers) > 1:
        extras = [f"scope {set(scope)} (selector {set(sel)})" for vid, scope, sel in ps_layers[1:]]
        logger.warning(
            f"learn_liang_ps_md_vtree: {len(ps_layers)} pseudo-mixing layers — an inclusion "
            "chain of determinisms forces PS layers below the root. Conditioning on the "
            f"md-set of a non-root PS layer alone will not be normalized. Extra PS layers: {extras}"
        )

    return vt
