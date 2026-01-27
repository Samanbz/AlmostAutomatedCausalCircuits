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


class Interval:
    """Represents a mathematical interval [low, high), (low, high], etc."""

    def __init__(
        self,
        low: float,
        high: float,
        include_low: bool = True,
        include_high: bool = False,
    ):
        self.low = low
        self.high = high
        self.include_low = include_low
        self.include_high = include_high

    def contains(self, value: float) -> bool:
        lower_check = (value >= self.low) if self.include_low else (value > self.low)
        upper_check = (value <= self.high) if self.include_high else (value < self.high)
        return lower_check and upper_check

    def intersect(self, other: "Interval") -> "Interval":
        new_low = max(self.low, other.low)
        new_high = min(self.high, other.high)

        # Determine strictness of bounds
        if self.low == other.low:
            new_include_low = self.include_low and other.include_low
        else:
            new_include_low = self.include_low if self.low > other.low else other.include_low

        if self.high == other.high:
            new_include_high = self.include_high and other.include_high
        else:
            new_include_high = self.include_high if self.high < other.high else other.include_high

        return Interval(new_low, new_high, new_include_low, new_include_high)

    def __repr__(self):
        left = "[" if self.include_low else "("
        right = "]" if self.include_high else ")"
        return f"{left}{self.low}, {self.high}{right}"


class Distribution(LeafNode):
    """Base class for probability distributions used as leaves in SPNs."""

    pass


class GaussianDistribution(Distribution):
    """Represents a Gaussian distribution leaf node."""

    def __init__(self, scope: Tuple[int, ...], mean: float, stddev: float):
        super().__init__(scope)
        self.mean = mean
        self.stddev = stddev

    def __repr__(self):
        return f"GaussianDistribution(scope={self.scope}, mean={self.mean}, stddev={self.stddev})"


class CategoricalDistribution(Distribution):
    """Represents a Categorical distribution leaf node."""

    def __init__(self, scope: Tuple[int, ...], categories: List[Any], probabilities: List[float]):
        super().__init__(scope)
        self.categories = categories
        self.probabilities = probabilities

    def __repr__(self):
        return f"CategoricalDistribution(scope={self.scope}, categories={self.categories}, probabilities={self.probabilities})"


class UniformDistribution(Distribution):
    """Represents a Uniform distribution leaf node."""

    def __init__(self, scope: Tuple[int, ...], low: float, high: float):
        super().__init__(scope)
        self.low = low
        self.high = high

    def __repr__(self):
        return f"UniformDistribution(scope={self.scope}, low={self.low}, high={self.high})"


class TruncatedDistribution(Distribution):
    """
    Represents a distribution restricted to a specific interval.
    """

    def __init__(
        self,
        scope: Tuple[int, ...],
        base_distribution: Distribution,
        interval: Interval,
    ):
        super().__init__(scope)
        self.base_distribution = base_distribution
        self.interval = interval

    def __repr__(self):
        return f"TruncatedDistribution(scope={self.scope}, base={self.base_distribution}, interval={self.interval})"


class DataRegionGraphNode(RegionGraphNode):
    """RegionGraphNode that holds data slice information and constraints."""

    def __init__(
        self,
        scope: Tuple[int, ...],
        row_ids: List[int],
        constraints: Dict[int, Interval] = None,
    ):
        super().__init__(scope)
        self.row_ids = row_ids
        self.constraints = constraints if constraints is not None else {}

    def get_data_slice(self, data: np.ndarray) -> np.ndarray:
        return data[np.ix_(self.row_ids, self.scope)]


class DataRegionNode(DataRegionGraphNode, RegionNode):
    pass


class DataPartitionNode(DataRegionGraphNode, PartitionNode):
    pass


class DataRegionGraph(DirectedAcyclicGraph[int, DataRegionGraphNode, Any]):
    """RegionGraph that holds data slices at each node."""

    pass
