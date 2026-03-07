from typing import Any, Dict

from src.utils import BitSet

from .base import DirectedAcyclicGraph, Node


class RegionGraphNode(Node):
    """Base class for nodes in a region graph."""

    def __init__(self, scope: BitSet):
        self.scope = scope

    def __repr__(self):
        return f"{self.__class__.__name__}(scope={self.scope})"


class RegionNode(RegionGraphNode):
    """Represents a region in the region graph."""

    pass


class PartitionNode(RegionGraphNode):
    """Represents a partition in the region graph."""

    pass


class RegionGraph(DirectedAcyclicGraph[int, RegionGraphNode, Any]):
    """
    A DAG representing a region graph.
    Nodes are identified by integers and contain Node objects (RegionNode, PartitionNode, etc.).
    """

    def _get_base_node_config(self) -> Dict[type, Dict[str, Any]]:
        config = super()._get_base_node_config()

        config.update(
            {
                RegionNode: {
                    "color": "#ffcc99",
                    "label": lambda n: f"Region\nScope: {list(n.scope)}",
                    "shape": "box",
                },
                PartitionNode: {
                    "color": "#99ccff",
                    "label": lambda n: f"Partition\nScope: {list(n.scope)}",
                    "shape": "ellipse",
                },
            }
        )
        return config
