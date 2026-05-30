from typing import Dict

from src.logger import logger as g_logger
from src.symbolic.arithmetic import (
    Distribution,
)
from src.symbolic.region_graph import LayerType, PartitionNode, RegionGraph, RegionNode
from src.symbolic.vtree import VTree
from src.utils.bitset import BitSet
from src.utils.support import Support


logger = g_logger.getChild("region_graph_builder")


class RegionGraphBuilder:
    """
    Builds a Marginally Deterministic Region Graph (RegionGraph) from an VTree
    and base leaf distributions based on the Net architecture.
    """

    def __init__(
        self,
        input_dists: Dict[int, Distribution],
        md_vtree: VTree,
    ):
        self.input_dists = input_dists
        self.md_vtree = md_vtree

    def build(self) -> RegionGraph:
        rg = RegionGraph()
        md_root = self.md_vtree.get_root()
        self._build_recursive(rg, md_root)
        return rg

    def _build_recursive(
        self,
        rg: RegionGraph,
        md_vid: int,
        parent_md_set: BitSet | None = None,
    ) -> int:
        md_vnode = self.md_vtree._nodes[md_vid]
        md_set = md_vnode.md_set

        # A node is constrained (must provide disjoint support upward) iff
        # its md_set is a non-universal subset of the parent's md_set.
        # Otherwise it's unconstrained (leaves don't split, sums are dense).
        if parent_md_set is None:
            is_constrained = not md_set.is_universal  # root: constrained if it has any md-set
        elif md_set.is_universal:
            is_constrained = False
        else:
            is_constrained = md_set.issubset(parent_md_set) and not parent_md_set.is_universal

        if not self.md_vtree._adj[md_vid]:
            dist = self.input_dists[md_vnode.scope.min]
            reg_node = RegionNode(
                scope=md_vnode.scope,
                support=dist.support,
                md_set=md_set,
                layer_type=LayerType.UNIVERSAL,
                is_constrained=is_constrained,
            )
            reg_id = rg._add_node(reg_node)
            return reg_id

        l_child_id, r_child_id = self.md_vtree._children_pair[md_vid]
        vt_nodes = self.md_vtree._nodes
        l_child_node = vt_nodes[l_child_id]
        r_child_node = vt_nodes[r_child_id]
        l_md_set, r_md_set = l_child_node.md_set, r_child_node.md_set

        layer_type = LayerType.from_md_sets(md_set, l_md_set, r_md_set)

        l_region_id = self._build_recursive(rg, l_child_id, parent_md_set=md_set)
        r_region_id = self._build_recursive(rg, r_child_id, parent_md_set=md_set)

        rg_nodes = rg._nodes
        scope = md_vnode.scope

        l_support = rg_nodes[l_region_id].support
        r_support = rg_nodes[r_region_id].support
        part_support = Support.fast_disjoint_union(l_support, r_support)
        part_node = PartitionNode(scope=scope, support=part_support)
        part_id = rg._add_node(part_node)
        rg._add_edge(part_id, l_region_id)
        rg._add_edge(part_id, r_region_id)

        rg_id = rg._add_node(
            RegionNode(
                scope=md_vnode.scope,
                support=part_support,
                md_set=md_set,
                layer_type=layer_type,
                is_constrained=is_constrained,
            )
        )
        rg._add_edge(rg_id, part_id)

        return rg_id
