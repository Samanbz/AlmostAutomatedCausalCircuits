import itertools
import random
from typing import Dict, List, Tuple

import numpy as np

from src.logger import logger as g_logger
from src.symbolic import (
    DataPartitionNode,
    DataRegionGraph,
    DataRegionNode,
    Distribution,
    MDVTree,
    ProductNode,
    SumNode,
    SymbolicArithmeticCircuit,
    VTree,
)
from src.utils import BitSet, DataSlice, DiscreteInterval, Support


logger = g_logger.getChild("xpc")


def get_random_constraints(
    conj_dists: Dict[int, Distribution],
) -> Tuple[Support, Support]:
    """Generate random constraints by sampling from distributions and cutting at the sampled point."""
    constraints = Support()
    r_constraints = Support()

    for varg_node_idx, dist in conj_dists.items():
        rand_cut_point = dist.sample()
        left_dist, right_dist = dist.split_at(rand_cut_point)
        if random.random() < 0.5:
            left_dist, right_dist = right_dist, left_dist
        constraints.add(varg_node_idx, left_dist.var_support)
        r_constraints.add(varg_node_idx, right_dist.var_support)

    return constraints, r_constraints


def get_factorial_constraints(
    conj_dists: Dict[int, Distribution],
) -> List[Support]:
    """Generate mutually exclusive constraints forming a full factorial grid split."""
    var_splits = {}
    for var, dist in conj_dists.items():
        if isinstance(dist.var_support, DiscreteInterval) and len(dist.var_support.values) > 1:
            valid_cuts = sorted(dist.var_support.values)[:-1]
            rand_cut_point = random.choice(valid_cuts)
        else:
            rand_cut_point = dist.sample()

        logger.debug(
            f"Variable {var}: sampled cut point {rand_cut_point} from distribution {dist}."
        )
        left_support, right_support = dist.var_support.split_at(rand_cut_point)
        # Randomize left/right order to avoid bias
        opts = (
            [right_support, left_support]
            if random.random() < 0.5
            else [left_support, right_support]
        )
        var_splits[var] = opts

    vars_list = list(conj_dists.keys())
    supports = []

    # Generate all 2^k combinations
    for combo in itertools.product(*[var_splits[v] for v in vars_list]):
        s = Support()
        for i, v in enumerate(vars_list):
            s.add(v, combo[i])
        supports.append(s)

    return supports


def apply_constraint(
    data_slice: DataSlice,
    constraint: Support,
) -> BitSet:
    """
    Applies a single hyper-rectangle constraint to the data slice
    and returns the row IDs that fall into it.
    """
    if not data_slice.row_ids:
        return BitSet()

    assert all(i in data_slice.col_ids for i in list(constraint.intervals.keys())), (
        "Constraint variables must be in data slice columns."
    )

    mask = np.ones(len(data_slice.row_ids), dtype=bool)

    for varg_node_idx, interval in constraint:
        vals = data_slice.get_column(varg_node_idx)
        mask &= interval.contains(vals)

    # Convert active rows to indices
    current_row_mask = data_slice.row_ids.to_numpy(data_slice.data.shape[0])
    active_indices = np.flatnonzero(current_row_mask)

    # Filter indices based on mask
    subset_indices = active_indices[mask]

    subset_ids_mask = np.zeros_like(current_row_mask)
    subset_ids_mask[subset_indices] = True
    subset_ids = BitSet.from_bool_mask(subset_ids_mask)

    return subset_ids


def partition_randomly(
    data_slice: DataSlice,
    split_arity: int,  # Kept for signature compatibility, but overridden by 2^k geometry
    min_examples: int,
    conj_dists: Dict[int, Distribution],
    max_tries: int = 20,
    subsample_size: int = None,
) -> List[DataSlice]:
    if len(data_slice) < 2 * min_examples:
        logger.debug(
            f"Insufficient examples ({len(data_slice)}) to partition data slice on vars {list(conj_dists.keys())}."
        )
        return []

    logger.debug(
        f"Factorial partitioning data slice with {len(data_slice)} examples on {len(conj_dists)} vars."
    )

    test_slice = data_slice
    test_min_examples = min_examples
    if subsample_size is not None and len(data_slice) > subsample_size:
        active_indices = np.flatnonzero(data_slice.row_ids.to_numpy(data_slice.data.shape[0]))
        subsample_indices = np.random.choice(active_indices, size=subsample_size, replace=False)
        sub_mask = np.zeros(data_slice.data.shape[0], dtype=bool)
        sub_mask[subsample_indices] = True
        test_slice = DataSlice(
            data_slice.data,
            BitSet.from_bool_mask(sub_mask),
            data_slice.col_ids,
            data_slice.constraints,
        )
        test_min_examples = max(1, int(min_examples * (subsample_size / len(data_slice))))

    for _ in range(max_tries):
        supports = get_factorial_constraints(conj_dists)
        populated_branches = 0

        # First pass: Test efficiency on the (sub)sample
        for constraint in supports:
            subset_row_ids = apply_constraint(
                data_slice=test_slice,
                constraint=constraint,
            )
            if len(subset_row_ids) >= test_min_examples:
                populated_branches += 1

        # For a split to be useful, it should distribute data into at least 2 branches
        if populated_branches >= 2:
            logger.debug(
                f"\tAccepted factorial split with {populated_branches}/{len(supports)} heavily populated branches on test sample."
            )

            # Second pass: Apply accepted constraint to full data slice
            candidate_slices = []
            for constraint in supports:
                full_subset_row_ids = apply_constraint(
                    data_slice=data_slice,
                    constraint=constraint,
                )
                new_slice = DataSlice(
                    data_slice.data,
                    full_subset_row_ids,
                    data_slice.col_ids,
                    constraints=data_slice.constraints.intersect(constraint),
                )
                candidate_slices.append(new_slice)

            return candidate_slices

    logger.debug(f"Failed to find a viable factorial partition after {max_tries} tries.")
    return []


def construct_random_data_region_graph(
    data: np.ndarray,
    input_dists: Dict[int, Distribution],
    min_examples: int,
    split_arity: int,
    var_decomp: VTree,
    subsample_size: int = None,
) -> DataRegionGraph:
    assert split_arity >= 2, "Split arity must be at least 2."
    assert data.ndim == 2, "Data must be a 2D array."
    assert var_decomp is not None, "Variable decomposition (VTree) is required."

    node_counter = 0

    def next_id():
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        return nid

    rg = DataRegionGraph()
    num_rows, num_cols = data.shape

    root_region = DataRegionNode(scope=BitSet.full(num_cols), row_ids=BitSet.full(num_rows))
    root_id = next_id()
    rg.add_node(root_id, root_region)

    # Map DataRegionNode ID -> VTree Node ID
    rnode_to_vnode: Dict[int, int] = {root_id: var_decomp.get_root()}

    P: List[Tuple[int, DataRegionNode]] = [(root_id, root_region)]

    while P:
        # Random region from P
        idx = random.randint(0, len(P) - 1)
        rg_node_id, rg_node = P.pop(idx)

        v_id = rnode_to_vnode[rg_node_id]

        if var_decomp.is_leaf(v_id):
            continue

        # Get children (left is always conj_vars), v_id is not leaf, so children are not None
        conj_v_id, rest_v_id = var_decomp.get_children_pair(v_id)

        conj_vars = var_decomp.get_node_data(conj_v_id).scope

        current_data_slice = rg_node.get_data_slice(data)
        current_constraints = rg_node.constraints
        constrainted_input_dists = {
            var: dist.constrain_to(current_constraints[var]) if var in current_constraints else dist
            for var, dist in input_dists.items()
        }

        conj_dists = {var: constrainted_input_dists[var] for var in conj_vars}

        if not conj_dists:
            data_slices = [current_data_slice]
        else:
            data_slices = partition_randomly(
                data_slice=current_data_slice,
                conj_dists=conj_dists,
                split_arity=split_arity,
                min_examples=min_examples,
                subsample_size=subsample_size,
            )
            if not data_slices:
                logger.debug("Failed to find valid split. Falling back to single partition.")
                data_slices = [current_data_slice]

        new_partitions: List[Tuple[int, DataPartitionNode]] = []
        for data_slice in data_slices:
            partition = DataPartitionNode(
                scope=rg_node.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints,  # carry over constraints
            )
            partition_id = next_id()
            rg.add_node(partition_id, partition)
            rg.add_edge(rg_node_id, partition_id)
            new_partitions.append((partition_id, partition))

        for p_id, p_node in new_partitions:
            # Filter constraints for children based on their scope.
            l_constraints = p_node.constraints.filter_by_vars(conj_vars)
            l_region = DataRegionNode(
                scope=conj_vars,
                row_ids=p_node.row_ids,
                constraints=l_constraints,
            )

            r_scope = rg_node.scope.difference(conj_vars)
            r_constraints = p_node.constraints.filter_by_vars(r_scope)
            r_region = DataRegionNode(
                scope=r_scope,
                row_ids=p_node.row_ids,
                constraints=r_constraints,
            )
            l_region_id = next_id()
            r_region_id = next_id()

            rg.add_node(l_region_id, l_region)
            rg.add_node(r_region_id, r_region)
            rg.add_edge(p_id, l_region_id)
            rg.add_edge(p_id, r_region_id)

            # Map new regions to corresponding VNodes
            rnode_to_vnode[l_region_id] = conj_v_id
            if rest_v_id is not None:
                rnode_to_vnode[r_region_id] = rest_v_id

            # Add to Queue
            P.append((l_region_id, l_region))
            P.append((r_region_id, r_region))

    return rg


def construct_random_md_data_region_graph(
    data: np.ndarray,
    input_dists: Dict[int, Distribution],
    min_examples: int,
    split_arity: int,
    md_var_decomp: MDVTree,
    num_repetitions: int = 1,
    num_sums: int = 1,
    num_inputs: int = 1,
    subsample_size: int = None,
) -> DataRegionGraph:
    assert split_arity >= 2, "Split arity must be at least 2."
    assert data.ndim == 2, "Data must be a 2D array."
    assert md_var_decomp is not None, (
        "Marginal deterministic variable decomposition (MDVTree) is required."
    )

    node_counter = 0

    def next_id():
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        return nid

    rg = DataRegionGraph()
    num_rows, num_cols = data.shape

    root_md_vid = md_var_decomp.get_root()
    root_md_vnode = md_var_decomp.get_node_data(root_md_vid)
    root_is_universal = root_md_vnode.md_set.is_universal

    root_region = DataRegionNode(
        scope=BitSet.full(num_cols),
        row_ids=BitSet.full(num_rows),
        num_sums=num_sums if root_is_universal else 1,
        num_inputs=num_inputs if root_is_universal else 1,
    )
    root_id = next_id()
    rg.add_node(root_id, root_region)

    # Map DataRegionNode ID -> VTree Node ID
    rg_nid_to_md_vid: Dict[int, int] = {root_id: root_md_vid}

    def partition_recursive(rg_node_id: int):
        rg_node = rg.get_node_data(rg_node_id)
        md_vid = rg_nid_to_md_vid[rg_node_id]
        md_vnode = md_var_decomp.get_node_data(md_vid)
        if md_var_decomp.is_leaf(md_vid):
            return  # No further partitioning needed

        logger.debug(
            f"Partitioning RG node {rg_node_id} with MDVTree node {md_vid} on vars {md_vnode.md_set}."
        )

        current_data_slice = rg_node.get_data_slice(data)
        current_constraints = rg_node.constraints
        # logger.debug(f"Current constraints: {current_constraints}")

        is_universal = md_vnode.md_set.is_universal

        conj_dists = {
            var: dist.constrain_to(current_constraints[var]) if var in current_constraints else dist
            for var, dist in input_dists.items()
            if var in md_vnode.md_set and var in rg_node.scope
        }

        if is_universal:
            data_slices = [current_data_slice for _ in range(num_repetitions)]
        elif not conj_dists:
            data_slices = [current_data_slice]
        else:
            data_slices = partition_randomly(
                data_slice=current_data_slice,
                split_arity=split_arity,
                min_examples=min_examples,
                conj_dists=conj_dists,
                subsample_size=subsample_size,
            )
            # Fall back to a single, full partition if no valid split configurations are found
            if not data_slices:
                logger.debug("Failed to find valid split. Falling back to single partition.")
                data_slices = [current_data_slice]

        for data_slice in data_slices:
            l_md_child_id, r_md_child_id = md_var_decomp.get_children_pair(
                md_vid
            )  # Guaranteed to be truthy since md_vid is not a leaf
            l_md_child = md_var_decomp.get_node_data(l_md_child_id)
            r_md_child = md_var_decomp.get_node_data(r_md_child_id)

            p_node_id = next_id()
            p_node = DataPartitionNode(
                scope=md_vnode.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints,
            )
            rg.add_node(p_node_id, p_node)

            l_rg_node_id = next_id()
            print(
                f"Creating left region with scope {l_md_child.scope} and constraints {data_slice.constraints.filter_by_vars(l_md_child.scope)} from md_vnode {md_vid} with md_set {md_vnode.md_set}"
            )
            l_rg_node = DataRegionNode(
                scope=l_md_child.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints.filter_by_vars(l_md_child.scope),
                num_sums=num_sums if l_md_child.md_set.is_universal else 1,
                num_inputs=num_inputs if l_md_child.md_set.is_universal else 1,
            )
            rg.add_node(l_rg_node_id, l_rg_node)
            rg_nid_to_md_vid[l_rg_node_id] = l_md_child_id

            r_rg_node_id = next_id()
            r_rg_node = DataRegionNode(
                scope=r_md_child.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints.filter_by_vars(r_md_child.scope),
                num_sums=num_sums if r_md_child.md_set.is_universal else 1,
                num_inputs=num_inputs if r_md_child.md_set.is_universal else 1,
            )
            rg.add_node(r_rg_node_id, r_rg_node)
            rg_nid_to_md_vid[r_rg_node_id] = r_md_child_id

            rg.add_edge(p_node_id, l_rg_node_id)
            rg.add_edge(p_node_id, r_rg_node_id)

            rg.add_edge(rg_node_id, p_node_id)

            partition_recursive(l_rg_node_id)
            partition_recursive(r_rg_node_id)

    partition_recursive(root_id)
    return rg


def construct_spn_from_region_graph(
    rg: DataRegionGraph,
    input_dists: Dict[int, Distribution],
    alpha: float = 0.01,
) -> SymbolicArithmeticCircuit:
    node_counter = 1

    def next_id():
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        return nid

    circuit = SymbolicArithmeticCircuit()

    rg_root_id = rg.get_roots()[0]
    region_to_ac_nodes: Dict[int, List[int]] = {}

    import copy

    # Pass 1: Regions
    for rg_node_id in rg.topological_sort():
        rg_node = rg.get_node_data(rg_node_id)
        if not isinstance(rg_node, DataRegionNode):
            continue

        is_leaf_region = len(rg._adj[rg_node_id]) == 0
        ac_ids = []

        if is_leaf_region:
            assert len(rg_node.scope) == 1, f"Leaf region scope must be 1, got {rg_node.scope}"
            var_id = list(rg_node.scope)[0]
            base_dist = input_dists[var_id]

            for _ in range(rg_node.num_inputs):
                leaf_id = next_id()
                # Use deepcopy for independence
                dist_copy = copy.deepcopy(base_dist)
                leaf_node = dist_copy.constrain_to(rg_node.constraints.get(var_id))
                circuit.add_node(leaf_id, leaf_node)
                ac_ids.append(leaf_id)
        else:
            # Need some standard Support for the node
            full_support = Support({var: input_dists[var].var_support for var in rg_node.scope})
            node_support = full_support.intersect(rg_node.constraints)

            for _ in range(rg_node.num_sums):
                print(
                    f"Creating SumNode for region {rg_node_id} with scope {rg_node.scope} and support {node_support}, num_sums={rg_node.num_sums}"
                )
                sum_id = next_id()
                sum_node = SumNode(support=node_support)
                circuit.add_node(sum_id, sum_node)
                ac_ids.append(sum_id)

        region_to_ac_nodes[rg_node_id] = ac_ids

    # Pass 2: Partitions
    for p_id in rg.topological_sort():
        p_node = rg.get_node_data(p_id)
        if not isinstance(p_node, DataPartitionNode):
            continue

        parents = rg.get_parents(p_id)
        assert len(parents) == 1, "DataPartitionNode must have exactly one parent."
        parent_region_id = parents[0]
        parent_r_node = rg.get_node_data(parent_region_id)

        children = list(rg._adj[p_id].keys())
        assert len(children) == 2, "Partition node must have exactly two child regions."
        l_r_id, r_r_id = children

        parent_sums = region_to_ac_nodes.get(parent_region_id, [])
        l_children = region_to_ac_nodes.get(l_r_id, [])
        r_children = region_to_ac_nodes.get(r_r_id, [])

        full_support = Support({var: input_dists[var].var_support for var in parent_r_node.scope})
        node_support = full_support.intersect(p_node.constraints)

        # XPC proportions
        num_partitions = len(rg.get_children(parent_region_id))
        proportion_base = (len(p_node.row_ids) + alpha) / (
            len(parent_r_node.row_ids) + alpha * num_partitions
        )

        for l_ac in l_children:
            for r_ac in r_children:
                prod_id = next_id()
                prod_node = ProductNode(support=node_support)
                circuit.add_node(prod_id, prod_node)

                circuit.add_edge(prod_id, l_ac)
                circuit.add_edge(prod_id, r_ac)

                for parent_sum_id in parent_sums:
                    # Break symmetry if dense RAT-SPN style, else standard proportion
                    weight = proportion_base
                    if parent_r_node.num_sums > 1 or num_partitions > 1:
                        weight = proportion_base * (0.5 + random.random())
                    circuit.add_edge(parent_sum_id, prod_id, data=weight)

    # Normalize weights
    for node_id in circuit.topological_sort():
        if isinstance(circuit.get_node_data(node_id), SumNode):
            outgoing = circuit.get_outgoing_edges(node_id)
            if not outgoing:
                continue

            total_weight = sum(edge[1] for edge in outgoing)
            if total_weight > 0:
                for child_id, weight in outgoing:
                    circuit.set_edge_data(node_id, child_id, weight / total_weight)

    return circuit
