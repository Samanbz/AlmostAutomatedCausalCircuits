from typing import Any, Dict

from src.utils import BitSet

from .arithmetic.distributions import Distribution
from .base import DirectedAcyclicGraph, Node


class RegionGraphNode(Node):
    def __init__(
        self,
        scope: BitSet,
        support: Dict[int, Distribution] = None,
        num_sums: int = 1,
        num_inputs: int = 1,
    ):
        self.scope = scope
        self.support = support or {}
        self.num_sums = num_sums
        self.num_inputs = num_inputs

    def __repr__(self):
        return f"{self.__class__.__name__}(scope={self.scope}, support={self.support})"


class RegionNode(RegionGraphNode):
    pass


class PartitionNode(RegionGraphNode):
    pass


class RegionGraph(DirectedAcyclicGraph[int, RegionGraphNode, Any]):
    def _get_base_node_config(self) -> Dict[type, Dict[str, Any]]:
        config = super()._get_base_node_config()

        def support_label(n: RegionGraphNode) -> str:
            s_parts = []
            for v, d in sorted(n.support.items()):
                s_parts.append(f"{v}: {d.var_support}")
            return "\n".join(s_parts)

        config.update(
            {
                RegionNode: {
                    "color": "#ccffcc",
                    "label": lambda n: f"Region\nScope: {list(n.scope)}\n{support_label(n)}",
                },
                PartitionNode: {
                    "color": "#ccccff",
                    "label": lambda n: f"Partition\nScope: {list(n.scope)}\n{support_label(n)}",
                },
            }
        )
        return config


class MDRegionGraph(RegionGraph):
    """
    Marginally Deterministic Region Graph.
    A specific type of RegionGraph produced by MDRegionGraphBuilder
    and consumed by MDCircuitBuilder to ensure correct graph structures.
    """

    pass
