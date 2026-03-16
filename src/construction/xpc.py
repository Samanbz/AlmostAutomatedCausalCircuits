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
from src.utils import BitSet, DataSlice, Support


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
        rand_cut_point = dist.sample()
        logger.debug(
            f"Variable {var}: sampled cut point {rand_cut_point} from distribution {dist}."
        )
        left_dist, right_dist = dist.split_at(rand_cut_point)
        # Randomize left/right order to avoid bias, though combinatorial product covers all
        opts = [left_dist.var_support, right_dist.var_support]
        if random.random() < 0.5:
            opts = [opts[1], opts[0]]
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
) -> List[DataSlice]:
    if len(data_slice) < 2 * min_examples:
        logger.debug(
            f"Insufficient examples ({len(data_slice)}) to partition data slice on vars {list(conj_dists.keys())}."
        )
        return []

    logger.debug(
        f"Factorial partitioning data slice with {len(data_slice)} examples on {len(conj_dists)} vars."
    )

    for _ in range(max_tries):
        supports = get_factorial_constraints(conj_dists)
        candidate_slices = []
        populated_branches = 0

        for constraint in supports:
            subset_row_ids = apply_constraint(
                data_slice=data_slice,
                constraint=constraint,
            )

            # We track populated branches to ensure the split is actually separating data
            if len(subset_row_ids) >= min_examples:
                populated_branches += 1

            new_slice = DataSlice(
                data_slice.data,
                subset_row_ids,
                data_slice.col_ids,
                constraints=data_slice.constraints.intersect(constraint),
            )
            candidate_slices.append(new_slice)

        # For a split to be useful, it should distribute data into at least 2 branches
        if populated_branches >= 2:  # FIXME: So we just lose data if we can't find a good split!?
            logger.debug(
                f"\tAccepted factorial split with {populated_branches}/{len(supports)} heavily populated branches."
            )
            # logger. debug(f"Factorial constraints: {supports}")
            return candidate_slices

    logger.debug(f"Failed to find a viable factorial partition after {max_tries} tries.")
    return []


def construct_random_data_region_graph(
    data: np.ndarray,
    input_dists: Dict[int, Distribution],
    min_examples: int,
    split_arity: int,
    var_decomp: VTree,
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
            )
            if not data_slices:
                logger.debug(f"Failed to find valid split. Falling back to single partition.")
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

    root_region = DataRegionNode(scope=BitSet.full(num_cols), row_ids=BitSet.full(num_rows))
    root_id = next_id()
    rg.add_node(root_id, root_region)

    # Map DataRegionNode ID -> VTree Node ID
    rg_nid_to_md_vid: Dict[int, int] = {root_id: md_var_decomp.get_root()}

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

        conj_dists = {
            var: dist.constrain_to(current_constraints[var]) if var in current_constraints else dist
            for var, dist in input_dists.items()
            if var in md_vnode.md_set
        }

        if not conj_dists:
            data_slices = [current_data_slice]
        else:
            data_slices = partition_randomly(
                data_slice=current_data_slice,
                split_arity=split_arity,
                min_examples=min_examples,
                conj_dists=conj_dists,
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
            l_rg_node = DataRegionNode(
                scope=l_md_child.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints.filter_by_vars(l_md_child.scope),
            )
            rg.add_node(l_rg_node_id, l_rg_node)
            rg_nid_to_md_vid[l_rg_node_id] = l_md_child_id

            r_rg_node_id = next_id()
            r_rg_node = DataRegionNode(
                scope=r_md_child.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints.filter_by_vars(r_md_child.scope),
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
) -> SymbolicArithmeticCircuit:
    node_counter = 1

    def next_id():
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        return nid

    def handle_leaf(scope: BitSet, constraints: Support, p_ac_id: int):
        if len(scope) == 1:
            var = next(iter(scope))
            leaf_node = input_dists[var].constrain_to(constraints.get(var))
            leaf_id = next_id()
            circuit.add_node(leaf_id, leaf_node)
            circuit.add_edge(p_ac_id, leaf_id)
        else:
            dummy_sum_node_id = next_id()
            dummy_sum_node = SumNode(
                support=Support({var: input_dists[var].var_support for var in scope}).intersect(
                    constraints
                )
            )
            circuit.add_node(dummy_sum_node_id, dummy_sum_node)
            circuit.add_edge(p_ac_id, dummy_sum_node_id)
            prod_node_id = next_id()
            prod_node = ProductNode(
                support=Support(
                    {var: input_dists[var].var_support for var in rg_node.scope}
                ).intersect(rg_node.constraints)
            )
            circuit.add_node(prod_node_id, prod_node)
            circuit.add_edge(dummy_sum_node_id, prod_node_id, 1.0)  # Dummy weight

            rg_node_vars = list(scope)
            left_scope = rg_node_vars[: len(rg_node_vars) // 2]
            right_scope = rg_node_vars[len(rg_node_vars) // 2 :]

            handle_leaf(
                BitSet(left_scope), constraints.filter_by_vars(BitSet(left_scope)), prod_node_id
            )
            handle_leaf(
                BitSet(right_scope), constraints.filter_by_vars(BitSet(right_scope)), prod_node_id
            )

    circuit = SymbolicArithmeticCircuit()

    rg_root = rg.get_node_data(rg.get_roots()[0])
    ac_root = SumNode(
        support=Support({var: input_dists[var].var_support for var in rg_root.scope}).intersect(
            rg_root.constraints
        )
    )

    circuit.add_node(0, ac_root)

    rg_node_to_ac_node: Dict[int, int] = {}
    rg_node_to_ac_node[rg.get_roots()[0]] = 0

    for rg_node_id in rg.topological_sort():
        rg_node = rg.get_node_data(rg_node_id)
        if rg_node_id == rg.get_roots()[0]:
            continue

        if isinstance(rg_node, DataRegionNode):
            rg_node_parents = rg.get_parents(rg_node_id)  # TODO Use tree?
            assert len(rg_node_parents) == 1, (
                "DataRegionNode must have exactly one parent DataPartitionNode."
            )
            p_id = rg_node_parents[0]
            p_ac_id = rg_node_to_ac_node[p_id]
            p_ac_node = circuit.get_node_data(p_ac_id)
            assert isinstance(p_ac_node, ProductNode), (
                "Parent AC node must be a ProductNode corresponding to the partition."
            )

            if rg.is_leaf(rg_node_id):
                handle_leaf(rg_node.scope, rg_node.constraints, p_ac_id)
            else:
                # Create SumNode for inner RegionNode
                region_ac_node = SumNode(
                    support=Support(
                        {var: input_dists[var].var_support for var in rg_node.scope}
                    ).intersect(rg_node.constraints)
                )
                region_ac_id = next_id()
                circuit.add_node(region_ac_id, region_ac_node)
                circuit.add_edge(p_ac_id, region_ac_id)
                rg_node_to_ac_node[rg_node_id] = region_ac_id

        elif isinstance(rg_node, DataPartitionNode):
            partition_ac_node = ProductNode(
                support=Support(
                    {var: input_dists[var].var_support for var in rg_node.scope}
                ).intersect(rg_node.constraints)
            )
            partition_ac_id = next_id()
            circuit.add_node(partition_ac_id, partition_ac_node)

            rg_node_parents = rg.get_parents(rg_node_id)
            assert len(rg_node_parents) == 1, (
                "DataPartitionNode must have exactly one parent DataRegionNode."
            )
            rg_parent_id = rg_node_parents[0]
            rg_parent_node = rg.get_node_data(rg_parent_id)
            proportion = (
                (len(rg_node.row_ids) / len(rg_parent_node.row_ids))
                if len(rg_parent_node.row_ids) > 0
                else 0.0
            )
            ac_parent_id = rg_node_to_ac_node[rg_parent_id]
            ac_parent_node = circuit.get_node_data(ac_parent_id)
            assert isinstance(ac_parent_node, SumNode), (
                "Parent AC node must be a SumNode corresponding to the region."
            )

            circuit.add_edge(ac_parent_id, partition_ac_id, proportion)
            rg_node_to_ac_node[rg_node_id] = partition_ac_id
        else:
            raise ValueError(f"Unknown region graph node type: {type(rg_node)}")

    return circuit
