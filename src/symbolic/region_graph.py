from typing import Any

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

    pass
