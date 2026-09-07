import itertools

import networkx as nx
import numpy as np
import pymetis
import torch

from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet, estimate_pairwise_mi


def build_skeleton(dag: nx.DiGraph | None, num_variables: int) -> nx.Graph:
    """
    Phase 1: Build the undirected structural skeleton.
    If a DAG is provided, parents are moralized and edges are made undirected.
    If no DAG is provided, return a complete graph over variables.
    """
    skeleton = nx.Graph()
    skeleton.add_nodes_from(range(num_variables))

    if dag is None:
        # Fully connected if no ground-truth skeleton given
        for i in range(num_variables):
            for j in range(i + 1, num_variables):
                skeleton.add_edge(i, j)
        return skeleton

    # Start with all undirected edges
    for u, v in dag.edges():
        skeleton.add_edge(u, v)

    # Moralize: marry parents of the same child
    for node in dag.nodes():
        parents = list(dag.predecessors(node))
        for p1, p2 in itertools.combinations(parents, 2):
            skeleton.add_edge(p1, p2)

    return skeleton


def apply_md_constraints(skeleton: nx.Graph, md_sets: list[set[int]], md_weight: float = 1e9):
    """
    Phase 1b: Constraint Mapping.
    Assign a positive infinity (or highly un-cuttable weight) to edges
    within the identical MD-set. This prevents heuristic min-cut algorithms
    from ever splitting variables that must be processed atomically.
    """
    for md_set in md_sets:
        nodes = list(md_set)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if not skeleton.has_edge(u, v):
                    skeleton.add_edge(u, v)
                skeleton[u][v]["weight"] = md_weight


def evaluate_data_mi(skeleton: nx.Graph, data: torch.Tensor):
    """
    Phase 2: Calculate empirical mutual information on skeleton edges.
    Leaves +infty constraints alone.
    """
    # Find edges that need to be evaluated (i.e. those without weight=1e9)
    edges_to_eval = [(u, v) for u, v, d in skeleton.edges(data=True) if d.get("weight", 0.0) < 1e8]

    if edges_to_eval:
        mi_scores = estimate_pairwise_mi(data, edges_to_eval)

        # Add a very small jitter to avoid 0 weight edge deletion issues with partitioners
        for u, v in edges_to_eval:
            skeleton[u][v]["weight"] = mi_scores[(u, v)] + 1e-6


class LearnedVTreeBuilder:
    def __init__(
        self,
        prioritize: str = "hardware",
        md_sets: list[set[int]] | None = None,
        keep_together: list[tuple[int, int]] | None = None,
    ):
        self.vt = VTree()
        self.prioritize = prioritize
        self.md_sets = md_sets or []
        self.keep_together = [tuple(sorted((u, v))) for u, v in (keep_together or [])]

    def _apply_pymetis(self, G: nx.Graph) -> tuple[list[int], list[int]]:
        nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        adjacency = []
        eweights = []

        for n in nodes:
            neighbors = list(G.neighbors(n))
            adj_n = [node_to_idx[nbr] for nbr in neighbors]
            adjacency.append(np.array(adj_n, dtype=np.int32))
            weights_n = [int(G[n][nbr]["weight"] * 1e5) for nbr in neighbors]
            eweights.extend(weights_n)

        _, parts = pymetis.part_graph(2, adjacency=adjacency, eweights=eweights)

        left_vars = [nodes[i] for i, p in enumerate(parts) if p == 0]
        right_vars = [nodes[i] for i, p in enumerate(parts) if p == 1]
        return left_vars, right_vars

    def _apply_greedy_modularity(self, G: nx.Graph) -> tuple[list[int], list[int]]:
        communities = nx.community.greedy_modularity_communities(G, weight="weight")
        left_vars = list(communities[0])
        right_vars = [node for c in communities[1:] for node in c]
        return left_vars, right_vars

    def _apply_kernighan_lin(self, G: nx.Graph) -> tuple[list[int], list[int]]:
        # KL bisection is sensitive to initial partition; retry with multiple
        # seeds and pick the partition with the lowest cut weight.
        best_cut = float("inf")
        best_left, best_right = None, None
        for seed in range(20):
            left, right = nx.community.kernighan_lin_bisection(
                G, weight="weight", max_iter=10, seed=seed
            )
            cut = sum(G[u][v]["weight"] for u in left for v in right if G.has_edge(u, v))
            if cut < best_cut:
                best_cut = cut
                best_left, best_right = list(left), list(right)
        return best_left, best_right

    def _bisect_graph(self, G: nx.Graph, scope_list: list[int]) -> tuple[list[int], list[int]]:
        scope_set = set(scope_list)

        # Contract any keep_together pairs that are fully inside this scope.
        # This forces the partitioner to put the paired variables on the same
        # side, so they are split as late/deep in the tree as possible.
        G_contracted = G.copy()
        merged = {n: {n} for n in G_contracted.nodes()}
        for u, v in self.keep_together:
            if (
                u in scope_set
                and v in scope_set
                and u in G_contracted
                and v in G_contracted
                and u != v
            ):
                keep, discard = (v, u) if v > u else (u, v)
                G_contracted = nx.contracted_nodes(G_contracted, keep, discard, self_loops=False)
                merged[keep].update(merged.pop(discard))

        components = list(nx.connected_components(G_contracted))
        if len(components) > 1:
            left_super = list(components[0])
            right_super = [node for c in components[1:] for node in c]
            left_vars = sorted([x for sn in left_super for x in merged[sn]])
            right_vars = sorted([x for sn in right_super for x in merged[sn]])
            return left_vars, right_vars

        # Expressivity: split on the largest md-set boundary first.
        # Find the largest md-set that is a proper subset of the current scope.
        if self.prioritize == "expressivity" and self.md_sets:
            best_md = None
            for md in sorted(self.md_sets, key=len, reverse=True):
                md_in_scope = md & scope_set
                if md_in_scope and md_in_scope != scope_set:
                    best_md = md_in_scope
                    break
            if best_md is not None:
                # If an explicit MD split would separate a keep_together pair,
                # fall back to the partitioner so the pair stays together.
                cuts_pair = any(
                    (u in best_md) != (v in best_md)
                    for u, v in self.keep_together
                    if u in scope_set and v in scope_set
                )
                if not cuts_pair:
                    left_vars = sorted(best_md)
                    right_vars = sorted(scope_set - best_md)
                    return left_vars, right_vars

        try:
            if self.prioritize == "expressivity":
                try:
                    left_super, right_super = self._apply_pymetis(G_contracted)
                except ImportError:
                    left_super, right_super = self._apply_greedy_modularity(G_contracted)
            else:
                left_super, right_super = self._apply_kernighan_lin(G_contracted)

            if not left_super or not right_super:
                raise ValueError("Bisection failed")

            left_vars = sorted([x for sn in left_super for x in merged[sn]])
            right_vars = sorted([x for sn in right_super for x in merged[sn]])
            return left_vars, right_vars
        except (nx.NetworkXError, ValueError):
            half = len(scope_list) // 2
            return scope_list[:half], scope_list[half:]

    def _partition_recursively(self, G: nx.Graph) -> int:
        scope_list = list(G.nodes)
        curr_scope = BitSet(scope_list)
        curr_vnode = VNode(scope=curr_scope)
        curr_id = self.vt.add_node(curr_vnode)

        if len(scope_list) == 1:
            return curr_id

        left_vars, right_vars = self._bisect_graph(G, scope_list)

        left_id = self._partition_recursively(G.subgraph(left_vars))
        right_id = self._partition_recursively(G.subgraph(right_vars))

        self.vt.add_children(curr_id, left_id, right_id)
        return curr_id

    def build(self, skeleton: nx.Graph) -> VTree:
        self._partition_recursively(skeleton)
        return self.vt


def construct_optimal_vtree(
    data: torch.Tensor,
    dag: nx.DiGraph | None = None,
    prioritize: str = "hardware",
    keep_together: list[tuple[int, int]] | None = None,
) -> VTree:
    """
    Builds a pure data-driven and skeleton-constrained VTree.
    """
    n_vars = data.shape[1]
    skeleton = build_skeleton(dag, n_vars)
    evaluate_data_mi(skeleton, data)

    builder = LearnedVTreeBuilder(prioritize=prioritize, keep_together=keep_together)
    return builder.build(skeleton)


def construct_optimal_md_vtree(
    data: torch.Tensor,
    md_sets: list[set[int]],
    dag: nx.DiGraph | None = None,
    prioritize: str = "hardware",  # or "expressivity"
    keep_together: list[tuple[int, int]] | None = None,
) -> VTree:
    """
    Builds a VTree that explicitly enforces MD-Set constraints.
    These sets will not be sliced apart during recursive min-cut partitioning,
    thus guaranteeing causal tractability.

    MD sets must be closed under intersection: if A and B are in md_sets, then
    A ∩ B must also be in md_sets.

    ``keep_together`` is a list of variable pairs that should stay on the same
    side of every bisection for as long as possible (i.e. they are split as
    deep in the tree as possible).
    """
    # Validate intersection-closure
    md_sets_bs = [BitSet(s) for s in md_sets]
    for i in range(len(md_sets_bs)):
        for j in range(i + 1, len(md_sets_bs)):
            intersection = md_sets_bs[i].intersection(md_sets_bs[j])
            if not any(intersection == s for s in md_sets_bs):
                raise ValueError(
                    f"MD sets must be closed under intersection. "
                    f"{set(md_sets_bs[i])} ∩ {set(md_sets_bs[j])} = {set(intersection)} is missing."
                )

    n_vars = data.shape[1]
    skeleton = build_skeleton(dag, n_vars)
    apply_md_constraints(skeleton, md_sets, md_weight=1e9)
    evaluate_data_mi(skeleton, data)

    builder = LearnedVTreeBuilder(
        prioritize=prioritize, md_sets=md_sets, keep_together=keep_together
    )
    vt = builder.build(skeleton)

    vt.compute_md_labeling(md_sets)
    return vt
