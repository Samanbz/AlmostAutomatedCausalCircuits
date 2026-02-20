from collections import deque
from typing import Any, Dict, Generator, Generic, List, Optional, Tuple, TypeVar


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

    def get_roots(self) -> List[K]:
        """Returns a list of root node IDs (nodes with in-degree 0)."""
        return [n for n in self._nodes if not self._rev_adj[n]]

    def get_children(self, node_id: K) -> List[K]:
        """Returns a list of children node IDs."""
        if node_id not in self._adj:
            return []
        return list(self._adj[node_id].keys())

    def get_leaves(self) -> List[K]:
        """Returns a list of leaf node IDs (nodes with out-degree 0)."""
        return [n for n in self._nodes if not self._adj[n]]

    def is_leaf(self, node_id: K) -> bool:
        """Returns True if the node has no children."""
        return node_id not in self._adj or len(self._adj[node_id]) == 0

    @property
    def node_config(self) -> Dict[N, Dict[str, Any]]:
        """
        Returns the default node configuration for plotting.
        Should return a dictionary mapping Node types to style attributes.
        """
        return {}

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


class Tree(DirectedAcyclicGraph[K, N, E]):
    """
    A specific type of DAG that enforces tree properties (single root, unique parents).
    """

    def add_edge(self, source: K, target: K, data: E = None) -> None:
        """Adds a directed edge from source to target, enforcing tree properties."""

        # Enforce unique parents (in-degree <= 1)
        if len(self.get_parents(target)) > 0:
            raise ValueError(
                f"Node '{target}' already has a parent. Trees do not allow multi-parent nodes."
            )

        super().add_edge(source, target, data)

    def get_root(self) -> K:
        """Returns the single root node ID."""
        roots = self.get_roots()

        if not roots:
            if not self._nodes:
                raise ValueError("Tree is empty.")
            raise ValueError("Graph has cycle, no root found.")

        if len(roots) > 1:
            raise ValueError(
                f"Tree has multiple roots: {roots}. It might be a disconnected forest."
            )

        return roots[0]


class BinaryTree(Tree[K, N, E]):
    """
    A binary tree with explicit left/right child semantics.
    Enforces that each node is either a leaf (no children) or has exactly two children.
    """

    def __init__(self):
        super().__init__()
        # Maps parent_id -> (left_child_id, right_child_id)
        self._children_pair: Dict[K, Tuple[K, K]] = {}

    def add_children(self, parent: K, left_child: K, right_child: K, data: E = None) -> None:
        """Adds both children to the parent node atomically."""
        if parent in self._children_pair:
            raise ValueError(f"Node '{parent}' already has children.")
        # Add edges via parent class (enforces tree properties)
        super().add_edge(parent, left_child, data)
        super().add_edge(parent, right_child, data)
        self._children_pair[parent] = (left_child, right_child)

    def add_edge(self, source: K, target: K, data: E = None) -> None:
        """Disabled: Use add_children() to add both children at once."""
        raise NotImplementedError(
            "BinaryTree requires adding both children at once via add_children()."
        )

    def get_children_pair(self, node_id: K) -> Optional[Tuple[K, K]]:
        """Returns (left_child, right_child) tuple, or None if leaf."""
        return self._children_pair.get(node_id)


class Node:
    """Base class for nodes used in graphs."""

    pass
