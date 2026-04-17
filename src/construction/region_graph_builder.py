from typing import Dict, List

from src.logger import logger as g_logger
from src.symbolic import (
    Distribution,
    MDRegionGraph,
    MDVTree,
    PartitionNode,
    RegionGraph,
    RegionNode,
)
from src.utils.node_allocator import NodeAllocator


logger = g_logger.getChild("region_graph_builder")


class MDRegionGraphBuilder:
    def __init__(
        self,
        input_dists: Dict[int, Distribution],
        md_var_decomp: MDVTree,
        num_sums: int = 1,
        num_inputs: int = 1,
    ):
        self.input_dists = input_dists
        self.md_var_decomp = md_var_decomp
        self.num_sums = num_sums
        self.num_inputs = num_inputs
        self.rg = MDRegionGraph()
        self.node_allocator = NodeAllocator()
        self.cache: Dict[int, List[int]] = {}

    def _next_id(self) -> int:
        return self.node_allocator.next_id()

    def _get_group_key(self, support: Dict[int, Distribution], vars_subset: set) -> tuple:
        key_parts = []
        for v in sorted(vars_subset):
            key_parts.append((v, support[v]))
        return tuple(key_parts)

    def _build_recursive(self, md_vid: int, dists: Dict[int, Distribution]) -> List[int]:
        if md_vid in self.cache:
            return self.cache[md_vid]

        md_vnode = self.md_var_decomp.get_node_data(md_vid)

        if self.md_var_decomp.is_leaf(md_vid):
            res = self._mix_regions(md_vnode, dists)
            self.cache[md_vid] = res
            return res

        l_vid, r_vid = self.md_var_decomp.get_children_pair(md_vid)
        l_scope = self.md_var_decomp.get_node_data(l_vid).scope
        r_scope = self.md_var_decomp.get_node_data(r_vid).scope

        l_child_regions = self._build_recursive(l_vid, {v: dists[v] for v in l_scope})
        r_child_regions = self._build_recursive(r_vid, {v: dists[v] for v in r_scope})

        res = self._synthesize_regions(md_vnode, l_child_regions, r_child_regions, dists)
        self.cache[md_vid] = res
        return res

    def _mix_regions(self, md_vnode, dists: Dict[int, Distribution]) -> List[int]:
        """Base case: The Mixing rule for building Region Graph Leaves."""
        var = list(md_vnode.scope)[0]
        is_md = not md_vnode.md_set.is_universal and var in md_vnode.md_set
        res = []

        if is_md:
            # Split support into num_inputs intervals
            for sd in dists[var].split(self.num_inputs):
                new_dist = dists.copy()
                new_dist[var] = sd
                leaf_id = self._next_id()
                leaf_node = RegionNode(scope=md_vnode.scope, support=new_dist, num_inputs=1)
                self.rg.add_node(leaf_id, leaf_node)
                res.append(leaf_id)
        else:
            # Unconstrained leaf: num_inputs copies of full support
            for _ in range(self.num_inputs):
                leaf_id = self._next_id()
                leaf_node = RegionNode(scope=md_vnode.scope, support=dists, num_inputs=1)
                self.rg.add_node(leaf_id, leaf_node)
                res.append(leaf_id)
        return res

    def _synthesize_regions(
        self,
        md_vnode,
        l_child_regions: List[int],
        r_child_regions: List[int],
        dists: Dict[int, Distribution],
    ) -> List[int]:
        """Recursive case: The Synthesizing rule for building ancestor Region Nodes and Partition Nodes."""
        if md_vnode.md_set.is_universal:
            L_parent = set()
        else:
            L_parent = set(md_vnode.md_set).intersection(md_vnode.scope)

        # Group child pairs by their combined support on L_parent
        groups = {}
        for l_rid in l_child_regions:
            l_node = self.rg.get_node_data(l_rid)
            for r_rid in r_child_regions:
                r_node = self.rg.get_node_data(r_rid)

                # The support of the parent partition is the UNION of child supports
                # But to preserve determinism, we group them ONLY by the support
                # of the variables that are actually in the parent's MD-set.
                p_support = {**l_node.support, **r_node.support}
                group_key = self._get_group_key(p_support, L_parent)

                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append((l_rid, r_rid, p_support))

        parent_regions = []
        n_sums = self.num_sums if len(L_parent) == 0 else 1

        for group_key, pairs in groups.items():
            partition_ids = []
            for l_rid, r_rid, p_support in pairs:
                p_id = self._next_id()
                p_node = PartitionNode(scope=md_vnode.scope, support=p_support)
                self.rg.add_node(p_id, p_node)
                self.rg.add_edge(p_id, l_rid)
                self.rg.add_edge(p_id, r_rid)
                partition_ids.append(p_id)

            # The parent region node must have the support of ONLY the MD variables it dictates.
            parent_region_support = dists.copy()
            for v, s in group_key:
                parent_region_support[v] = s

            for _ in range(n_sums):
                r_id = self._next_id()
                r_node = RegionNode(scope=md_vnode.scope, support=parent_region_support, num_sums=1)
                self.rg.add_node(r_id, r_node)
                for p_id in partition_ids:
                    self.rg.add_edge(r_id, p_id)
                parent_regions.append(r_id)

        return parent_regions

    def build(self) -> MDRegionGraph:
        root_md_vid = self.md_var_decomp.get_root()
        root_regions = self._build_recursive(root_md_vid, self.input_dists)

        # Force a single root node for the SPN compilation
        if len(root_regions) > 1:
            root_id = self._next_id()
            root_scope = self.md_var_decomp.get_node_data(root_md_vid).scope
            root_node = RegionNode(scope=root_scope, support=self.input_dists, num_sums=1)
            self.rg.add_node(root_id, root_node)
            for r_id in root_regions:
                r_node = self.rg.get_node_data(r_id)
                p_id = self._next_id()
                p_node = PartitionNode(scope=root_scope, support=r_node.support)
                self.rg.add_node(p_id, p_node)
                self.rg.add_edge(p_id, r_id)
                self.rg.add_edge(root_id, p_id)

        return self.rg
