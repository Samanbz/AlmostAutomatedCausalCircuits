import itertools

import networkx as nx
import numpy as np
import torch

from src.symbolic.vtree import MDVTree, VNode, VTree
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


def partition_recursively(
    vt: VTree, parent_id: int, G: nx.Graph, node_counter: int, prioritize: str = "hardware"
) -> tuple[int, int]:
    """
    Phase 3: Recursive Partitioning.
    If prioritize="hardware", uses Kernighan-Lin to strictly bisect variables (balanced trees).
    If prioritize="expressivity", uses unconstrained Stoer-Wagner min-cut (unbalanced trees with fewer correlation cuts).
    """
    scope_list = list(G.nodes)
    if len(scope_list) == 1:
        # We are at a leaf
        leaf_scope = BitSet(scope_list)
        leaf_vnode = VNode(scope=leaf_scope)
        leaf_id = node_counter
        vt.add_node(leaf_id, leaf_vnode)
        return leaf_id, node_counter + 1

    # Attempt bisection
    # If the sub-graph is extremely disconnected, try to get connected components
    components = list(nx.connected_components(G))
    if len(components) > 1:
        # Found completely disconnected components, split along that!
        left_vars = list(components[0])
        right_vars = [node for c in components[1:] for node in c]
    else:
        try:
            if prioritize == "expressivity":
                try:
                    import pymetis

                    nodes = list(G.nodes())
                    node_to_idx = {n: i for i, n in enumerate(nodes)}

                    adjacency = []
                    eweights = []

                    for n in nodes:
                        neighbors = list(G.neighbors(n))
                        adj_n = [node_to_idx[nbr] for nbr in neighbors]
                        adjacency.append(np.array(adj_n, dtype=np.int32))

                        # PyMetis expects integer weights. Our weights are typically
                        # +1e9 for MD-Sets or 0.0 - 1.0 for correlations.
                        weights_n = [int(G[n][nbr]["weight"] * 1e5) for nbr in neighbors]
                        eweights.extend(weights_n)

                    # Unbalanced C++ min-cut solver
                    _, parts = pymetis.part_graph(2, adjacency=adjacency, eweights=eweights)

                    left_vars = [nodes[i] for i, p in enumerate(parts) if p == 0]
                    right_vars = [nodes[i] for i, p in enumerate(parts) if p == 1]

                except ImportError:
                    # Fallback to greedy python heuristic if pymetis is missing
                    communities = nx.community.greedy_modularity_communities(G, weight="weight")
                    left_vars = list(communities[0])
                    right_vars = [node for c in communities[1:] for node in c]
            else:
                # Use kernighan lin bisection. It tries to divide into two roughly equal halves
                # while minimizing edge cut weights.
                set_left, set_right = nx.community.kernighan_lin_bisection(
                    G, weight="weight", max_iter=10
                )
                left_vars, right_vars = list(set_left), list(set_right)

            # Failsafe if partition returns an empty set
            if not left_vars or not right_vars:
                raise ValueError("Bisection failed")

        except (nx.NetworkXError, ValueError):
            # Fallback: simple split if the graph structure causes bisection failure
            half = len(scope_list) // 2
            left_vars = scope_list[:half]
            right_vars = scope_list[half:]

    # Recursive steps
    # Internal node
    curr_scope = BitSet(scope_list)
    curr_vnode = VNode(scope=curr_scope)
    curr_id = node_counter
    vt.add_node(curr_id, curr_vnode)

    node_counter += 1

    left_id, node_counter = partition_recursively(
        vt, curr_id, G.subgraph(left_vars), node_counter, prioritize
    )
    right_id, node_counter = partition_recursively(
        vt, curr_id, G.subgraph(right_vars), node_counter, prioritize
    )

    vt.add_children(curr_id, left_id, right_id)

    return curr_id, node_counter


def construct_optimal_vtree(
    data: torch.Tensor, dag: nx.DiGraph | None = None, prioritize: str = "hardware"
) -> VTree:
    """
    Builds a pure data-driven and skeleton-constrained VTree.
    """
    n_vars = data.shape[1]
    skeleton = build_skeleton(dag, n_vars)
    evaluate_data_mi(skeleton, data)

    vt = VTree()
    root_id, _ = partition_recursively(vt, -1, skeleton, 0, prioritize)

    return vt


def construct_optimal_md_vtree(
    data: torch.Tensor,
    md_sets: list[set[int]],
    dag: nx.DiGraph | None = None,
    prioritize: str = "hardware",
) -> MDVTree:
    """
    Builds a VTree that explicitly enforces MD-Set constraints.
    These sets will not be sliced apart during recursive min-cut partitioning,
    thus guaranteeing causal tractability.
    """
    n_vars = data.shape[1]

    # 1. Structural Skeleton Extraction
    skeleton = build_skeleton(dag, n_vars)

    # 2. Causal constraints via supernodes (+infty weights)
    apply_md_constraints(skeleton, md_sets, md_weight=1e9)

    # 3. Massively Parallel Batch Correlation
    evaluate_data_mi(skeleton, data)

    # 4. Constrained VTree Instantiation (Min-Cut)
    vt = VTree()
    root_id, _ = partition_recursively(vt, -1, skeleton, 0, prioritize)

    return MDVTree.from_vtree(vt, md_sets)
