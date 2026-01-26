from collections import deque
from typing import Any, Dict, Generator, Generic, List, Optional, Tuple, TypeVar

import numpy as np


# Generic types for Node ID (K), Node Payload (N) and Edge Payload (E)
K = TypeVar("K")
N = TypeVar("N")
E = TypeVar("E")


class DirectedAcyclicGraph(Generic[K, N, E]):
    """
    A generic DAG implementation.
    K: Type of node identifier.
    N: Type of data stored in nodes.
    E: Type of data stored in edges.
    """

    def __init__(self):
        self._nodes: Dict[K, N] = {}
        # Adjacency: parent_id -> {child_id: edge_data}
        self._adj: Dict[K, Dict[K, E]] = {}
        # Reverse Adjacency: child_id -> {parent_id: edge_data}
        self._rev_adj: Dict[K, Dict[K, E]] = {}

    def add_node(self, node_id: K, data: N) -> None:
        """Adds a node to the graph."""
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists.")
        self._nodes[node_id] = data
        self._adj[node_id] = {}
        self._rev_adj[node_id] = {}

    def add_edge(self, source: K, target: K, data: E = None) -> None:
        """Adds a directed edge from source to target."""
        if source not in self._nodes or target not in self._nodes:
            raise KeyError(f"Source '{source}' or Target '{target}' not found.")

        # TODO: Check for cycles before adding (Simple DFS check could be added here for strictness,
        # but topological sort will catch it later).

        self._adj[source][target] = data
        self._rev_adj[target][source] = data

    def get_parents(self, node_id: K) -> List[K]:
        """Returns a list of parent node IDs."""
        return list(self._rev_adj[node_id].keys())

    def get_node_data(self, node_id: K) -> N:
        return self._nodes[node_id]

    def get_edge_data(self, source: K, target: K) -> E:
        return self._adj[source][target]

    def topological_sort(self) -> Generator[K, None, None]:
        """
        Yields node_ids in topological order using Kahn's Algorithm.
        """
        # Calculate in-degrees based on existing edges
        in_degree = {u: len(self._rev_adj[u]) for u in self._nodes}
        queue = deque([u for u, deg in in_degree.items() if deg == 0])

        visited_count = 0
        while queue:
            u = queue.popleft()
            yield u
            visited_count += 1

            # Decrement in-degree for neighbors
            for v in self._adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if visited_count != len(self._nodes):
            raise RuntimeError(
                f"Cycle detected in Graph! Visited {visited_count}/{len(self._nodes)} nodes."
            )


class Node:
    """Base class for nodes used in graphs."""

    pass


class ArithmeticNode(Node):
    """Represents an arithmetic operation node."""

    def __init__(self, scope: Optional[Tuple[int, ...]] = None):
        self.scope = scope

    def __repr__(self):
        return f"{self.__class__.__name__}(scope={self.scope})"


class SumNode(ArithmeticNode):
    """Represents a sum operation."""

    pass


class ProductNode(ArithmeticNode):
    """Represents a product operation."""

    pass


class RegionGraphNode(Node):
    """Base class for nodes in a region graph."""

    def __init__(self, scope: Tuple[int, ...]):
        self.scope = scope

    def __repr__(self):
        return f"{self.__class__.__name__}(scope={self.scope})"


class LeafNode(ArithmeticNode):
    """Represents a leaf distribution (e.g., Gaussian) in the SPN."""

    def __init__(self, scope: Tuple[int, ...]):
        self.scope = scope


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


class SymbolicArithmeticCircuit(DirectedAcyclicGraph[int, ArithmeticNode, Any]):
    """
    A DAG representing a symbolic arithmetic circuit.
    Nodes are identified by integers and contain Node objects (SumNode, ProductNode, etc.).
    """

    pass
