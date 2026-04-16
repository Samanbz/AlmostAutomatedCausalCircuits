import random
from typing import Dict, List, Tuple

import numpy as np

from src.construction.partitioning import partition_randomly
from src.logger import logger as g_logger
from src.symbolic import (
    DataPartitionNode,
    DataRegionGraph,
    DataRegionNode,
    Distribution,
    MDVTree,
    PartitionNode,
    RegionGraph,
    RegionNode,
    VTree,
)
from src.utils import BitSet


logger = g_logger.getChild("region_graph_builder")


class RegionGraphBuilder:
    def __init__(self, num_features: int, depth: int, num_repetitions: int):
        self.num_features = num_features
        self.depth = depth
        self.num_repetitions = num_repetitions
        self.rg = RegionGraph()
        self.node_counter = 0
        self.scope_to_id: Dict[BitSet, int] = {}

    def _next_id(self) -> int:
        nid = self.node_counter
        self.node_counter += 1
        return nid

    def _get_or_create_region_node(self, scope: BitSet) -> int:
        if scope in self.scope_to_id:
            return self.scope_to_id[scope]

        nid = self._next_id()
        r_node = RegionNode(scope=scope)
        self.rg.add_node(nid, r_node)
        self.scope_to_id[scope] = nid
        return nid

    def _split_recursive(self, scope: BitSet, depth: int):
        rid = self._get_or_create_region_node(scope)

        if len(scope) < 2:
            return

        partition_id = self._next_id()
        p_node = PartitionNode(scope=scope)
        self.rg.add_node(partition_id, p_node)
        self.rg.add_edge(rid, partition_id)

        shuffled = list(scope)
        random.shuffle(shuffled)
        mid = len(shuffled) // 2
        scope1 = BitSet(shuffled[:mid])
        scope2 = BitSet(shuffled[mid:])

        r1_id = self._get_or_create_region_node(scope1)
        r2_id = self._get_or_create_region_node(scope2)

        self.rg.add_edge(partition_id, r1_id)
        self.rg.add_edge(partition_id, r2_id)

        if depth > 1:
            if len(scope1) > 1:
                self._split_recursive(scope1, depth - 1)
            if len(scope2) > 1:
                self._split_recursive(scope2, depth - 1)

    def build(self) -> RegionGraph:
        root_scope = BitSet(range(self.num_features))
        for _ in range(self.num_repetitions):
            self._split_recursive(root_scope, self.depth)
        return self.rg


class DataRegionGraphBuilder:
    def __init__(
        self,
        data: np.ndarray,
        input_dists: Dict[int, Distribution],
        min_examples: int,
        split_arity: int,
        var_decomp: VTree,
        subsample_size: int = None,
    ):
        assert split_arity >= 2, "Split arity must be at least 2."
        assert data.ndim == 2, "Data must be a 2D array."
        assert var_decomp is not None, "Variable decomposition (VTree) is required."
        self.data = data
        self.input_dists = input_dists
        self.min_examples = min_examples
        self.split_arity = split_arity
        self.var_decomp = var_decomp
        self.subsample_size = subsample_size
        self.rg = DataRegionGraph()
        self.node_counter = 0

    def _next_id(self) -> int:
        nid = self.node_counter
        self.node_counter += 1
        return nid

    def build(self) -> DataRegionGraph:
        num_rows, num_cols = self.data.shape
        root_region = DataRegionNode(scope=BitSet.full(num_cols), row_ids=BitSet.full(num_rows))
        root_id = self._next_id()
        self.rg.add_node(root_id, root_region)
        rnode_to_vnode: Dict[int, int] = {root_id: self.var_decomp.get_root()}
        P: List[Tuple[int, DataRegionNode]] = [(root_id, root_region)]

        while P:
            idx = random.randint(0, len(P) - 1)
            rg_node_id, rg_node = P.pop(idx)
            v_id = rnode_to_vnode[rg_node_id]

            if self.var_decomp.is_leaf(v_id):
                continue

            conj_v_id, rest_v_id = self.var_decomp.get_children_pair(v_id)
            conj_vars = self.var_decomp.get_node_data(conj_v_id).scope

            current_data_slice = rg_node.get_data_slice(self.data)
            current_constraints = rg_node.constraints
            constrainted_input_dists = {
                var: dist.constrain_to(current_constraints[var])
                if var in current_constraints
                else dist
                for var, dist in self.input_dists.items()
            }
            conj_dists = {var: constrainted_input_dists[var] for var in conj_vars}

            if not conj_dists:
                data_slices = [current_data_slice]
            else:
                data_slices = partition_randomly(
                    data_slice=current_data_slice,
                    conj_dists=conj_dists,
                    split_arity=self.split_arity,
                    min_examples=self.min_examples,
                    subsample_size=self.subsample_size,
                )
                if not data_slices:
                    logger.debug("Failed to find valid split. Falling back to single partition.")
                    data_slices = [current_data_slice]

            new_partitions: List[Tuple[int, DataPartitionNode]] = []
            for data_slice in data_slices:
                partition = DataPartitionNode(
                    scope=rg_node.scope,
                    row_ids=data_slice.row_ids,
                    constraints=data_slice.constraints,
                )
                partition_id = self._next_id()
                self.rg.add_node(partition_id, partition)
                self.rg.add_edge(rg_node_id, partition_id)
                new_partitions.append((partition_id, partition))

            for p_id, p_node in new_partitions:
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
                l_region_id = self._next_id()
                r_region_id = self._next_id()

                self.rg.add_node(l_region_id, l_region)
                self.rg.add_node(r_region_id, r_region)
                self.rg.add_edge(p_id, l_region_id)
                self.rg.add_edge(p_id, r_region_id)

                rnode_to_vnode[l_region_id] = conj_v_id
                if rest_v_id is not None:
                    rnode_to_vnode[r_region_id] = rest_v_id

                P.append((l_region_id, l_region))
                P.append((r_region_id, r_region))

        return self.rg


class MDDataRegionGraphBuilder:
    def __init__(
        self,
        data: np.ndarray,
        input_dists: Dict[int, Distribution],
        min_examples: int,
        split_arity: int,
        md_var_decomp: MDVTree,
        num_repetitions: int = 1,
        num_sums: int = 1,
        num_inputs: int = 1,
        subsample_size: int = None,
    ):
        assert split_arity >= 2, "Split arity must be at least 2."
        assert data.ndim == 2, "Data must be a 2D array."
        assert md_var_decomp is not None, "Marginal deterministic variable decomposition required."
        self.data = data
        self.input_dists = input_dists
        self.min_examples = min_examples
        self.split_arity = split_arity
        self.md_var_decomp = md_var_decomp
        self.num_repetitions = num_repetitions
        self.num_sums = num_sums
        self.num_inputs = num_inputs
        self.subsample_size = subsample_size
        self.rg = DataRegionGraph()
        self.node_counter = 0
        self.rg_nid_to_md_vid: Dict[int, int] = {}

    def _next_id(self) -> int:
        nid = self.node_counter
        self.node_counter += 1
        return nid

    def _partition_recursive(self, rg_node_id: int):
        rg_node = self.rg.get_node_data(rg_node_id)
        md_vid = self.rg_nid_to_md_vid[rg_node_id]
        md_vnode = self.md_var_decomp.get_node_data(md_vid)
        if self.md_var_decomp.is_leaf(md_vid):
            return

        logger.debug(
            f"Partitioning RG node {rg_node_id} with MDVTree node {md_vid} on vars {md_vnode.md_set}."
        )

        current_data_slice = rg_node.get_data_slice(self.data)
        current_constraints = rg_node.constraints
        is_universal = md_vnode.md_set.is_universal

        conj_dists = {
            var: dist.constrain_to(current_constraints[var]) if var in current_constraints else dist
            for var, dist in self.input_dists.items()
            if var in md_vnode.md_set and var in rg_node.scope
        }

        if is_universal:
            data_slices = [current_data_slice for _ in range(self.num_repetitions)]
        elif not conj_dists:
            data_slices = [current_data_slice]
        else:
            data_slices = partition_randomly(
                data_slice=current_data_slice,
                split_arity=self.split_arity,
                min_examples=self.min_examples,
                conj_dists=conj_dists,
                subsample_size=self.subsample_size,
            )
            if not data_slices:
                logger.debug("Failed to find valid split. Falling back to single partition.")
                data_slices = [current_data_slice]

        for data_slice in data_slices:
            l_md_child_id, r_md_child_id = self.md_var_decomp.get_children_pair(md_vid)
            l_md_child = self.md_var_decomp.get_node_data(l_md_child_id)
            r_md_child = self.md_var_decomp.get_node_data(r_md_child_id)

            p_node_id = self._next_id()
            p_node = DataPartitionNode(
                scope=md_vnode.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints,
            )
            self.rg.add_node(p_node_id, p_node)

            l_rg_node_id = self._next_id()
            l_rg_node = DataRegionNode(
                scope=l_md_child.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints.filter_by_vars(l_md_child.scope),
                num_sums=self.num_sums if l_md_child.md_set.is_universal else 1,
                num_inputs=self.num_inputs if l_md_child.md_set.is_universal else 1,
            )
            self.rg.add_node(l_rg_node_id, l_rg_node)
            self.rg_nid_to_md_vid[l_rg_node_id] = l_md_child_id

            r_rg_node_id = self._next_id()
            r_rg_node = DataRegionNode(
                scope=r_md_child.scope,
                row_ids=data_slice.row_ids,
                constraints=data_slice.constraints.filter_by_vars(r_md_child.scope),
                num_sums=self.num_sums if r_md_child.md_set.is_universal else 1,
                num_inputs=self.num_inputs if r_md_child.md_set.is_universal else 1,
            )
            self.rg.add_node(r_rg_node_id, r_rg_node)
            self.rg_nid_to_md_vid[r_rg_node_id] = r_md_child_id

            self.rg.add_edge(p_node_id, l_rg_node_id)
            self.rg.add_edge(p_node_id, r_rg_node_id)
            self.rg.add_edge(rg_node_id, p_node_id)

            self._partition_recursive(l_rg_node_id)
            self._partition_recursive(r_rg_node_id)

    def build(self) -> DataRegionGraph:
        num_rows, num_cols = self.data.shape
        root_md_vid = self.md_var_decomp.get_root()
        root_md_vnode = self.md_var_decomp.get_node_data(root_md_vid)
        root_is_universal = root_md_vnode.md_set.is_universal

        root_region = DataRegionNode(
            scope=BitSet.full(num_cols),
            row_ids=BitSet.full(num_rows),
            num_sums=self.num_sums if root_is_universal else 1,
            num_inputs=self.num_inputs if root_is_universal else 1,
        )
        root_id = self._next_id()
        self.rg.add_node(root_id, root_region)
        self.rg_nid_to_md_vid[root_id] = root_md_vid

        self._partition_recursive(root_id)
        return self.rg
