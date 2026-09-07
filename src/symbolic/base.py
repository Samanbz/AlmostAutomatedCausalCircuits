from collections import deque
from typing import Any, Dict, Generator, Generic, List, Optional, Tuple, TypeVar

from src.utils import IncrementalNodeAllocator, NodeAllocator


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

    def __init__(self, node_allocator: Optional[NodeAllocator[K]] = None):
        self._nodes: Dict[K, N] = {}
        # Adjacency: parent_id -> {child_id: edge_data}
        self._adj: Dict[K, Dict[K, E]] = {}
        # Reverse Adjacency: child_id -> {parent_id: edge_data}
        self._rev_adj: Dict[K, Dict[K, E]] = {}
        # Memoized topological orders; cleared on any structural mutation.
        self._topo_cache: Dict[bool, List[K]] = {}
        self._in_degree_cache: Dict[K, int] = {}

        if node_allocator is None:
            # Default to IncrementalNodeAllocator if K is int
            self.node_allocator = IncrementalNodeAllocator()  # type: ignore
        else:
            self.node_allocator = node_allocator
        # For generating unique node IDs

    def _clear_structure_caches(self) -> None:
        """Invalidate memoized graph walks after a structural mutation."""
        self._topo_cache.clear()
        self._in_degree_cache.clear()

    def in_degrees(self) -> Dict[K, int]:
        """In-degree of every node, memoized; cleared on structural mutation."""
        if not self._in_degree_cache and self._nodes:
            self._in_degree_cache = {u: len(self._rev_adj[u]) for u in self._nodes}
        return self._in_degree_cache

    def add_node(self, data: N) -> K:
        """Adds a node to the graph."""
        node_id = self.node_allocator.next_id()
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists.")
        self._nodes[node_id] = data
        self._adj[node_id] = {}
        self._rev_adj[node_id] = {}
        self._clear_structure_caches()
        return node_id

    def _add_node(self, data: N) -> K:
        """Adds a node without duplicate check — caller must guarantee uniqueness."""
        node_id = self.node_allocator.next_id()
        self._nodes[node_id] = data
        self._adj[node_id] = {}
        self._rev_adj[node_id] = {}
        self._clear_structure_caches()
        return node_id

    def _add_node_explicit(self, node_id: K, data: N) -> None:
        """Adds a node with a specific ID."""
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists.")
        self._nodes[node_id] = data
        self._adj[node_id] = {}
        self._rev_adj[node_id] = {}
        self._clear_structure_caches()

    def add_edge(self, source: K, target: K, data: E = None) -> None:
        """Adds a directed edge from source to target."""
        if source not in self._nodes or target not in self._nodes:
            raise KeyError(f"Source '{source}' or Target '{target}' not found.")
        self._adj[source][target] = data
        self._rev_adj[target][source] = data
        self._clear_structure_caches()

    def _add_edge(self, source: K, target: K, data: E = None) -> None:
        """Adds an edge without existence checks — caller must guarantee both nodes exist."""
        self._adj[source][target] = data
        self._rev_adj[target][source] = data
        self._clear_structure_caches()

    def remove_edge(self, source: K, target: K) -> None:
        """Removes a directed edge from source to target."""
        if source in self._adj and target in self._adj[source]:
            del self._adj[source][target]
            del self._rev_adj[target][source]
            self._clear_structure_caches()

    def remove_node(self, node_id: K) -> None:
        """Removes a node and all its incident edges."""
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found.")

        # Remove outgoing edges
        for target in list(self._adj[node_id].keys()):
            self.remove_edge(node_id, target)

        # Remove incoming edges
        for source in list(self._rev_adj[node_id].keys()):
            self.remove_edge(source, node_id)

        del self._adj[node_id]
        del self._rev_adj[node_id]
        del self._nodes[node_id]
        self._clear_structure_caches()

    def get_parents(self, node_id: K) -> List[K]:
        """Returns a list of parent node IDs."""
        return list(self._rev_adj[node_id].keys())

    def get_grandparents(self, node_id: K) -> List[K]:
        """Returns a list of grandparent node IDs."""
        grandparents = set()
        for parent in self.get_parents(node_id):
            grandparents.update(self.get_parents(parent))
        return list(grandparents)

    def get_incoming_edges(self, node_id: K) -> List[Tuple[K, E]]:
        """Returns a list of (parent_id, edge_data) tuples for incoming edges."""
        return list(self._rev_adj[node_id].items())

    def get_outgoing_edges(self, node_id: K) -> List[Tuple[K, E]]:
        """Returns a list of (child_id, edge_data) tuples for outgoing edges."""
        return list(self._adj[node_id].items())

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

    def get_grandchildren(self, node_id: K) -> List[K]:
        """Returns a list of grandchildren node IDs."""
        grandchildren = set()
        for child in self.get_children(node_id):
            grandchildren.update(self.get_children(child))
        return list(grandchildren)

    def get_leaves(self) -> List[K]:
        """Returns a list of leaf node IDs (nodes with out-degree 0)."""
        return [n for n in self._nodes if not self._adj[n]]

    def is_leaf(self, node_id: K) -> bool:
        """Returns True if the node has no children."""
        return node_id not in self._adj or len(self._adj[node_id]) == 0

    def _get_base_node_config(self) -> Dict[type, Dict[str, Any]]:
        """
        Returns the base node configuration for plotting.
        Subclasses should override this method to provide specific styles.
        """
        return {}

    def get_node_config(self, show_node_id: bool = False) -> Dict[type, Dict[str, Any]]:
        """
        Returns the node configuration for plotting.
        If show_node_id is True, dynamically wraps labels to safely include node IDs.
        """
        config = self._get_base_node_config()

        if show_node_id:
            # Match nodes by object identity in case of unhashable nodes or overlapping equality
            node_ids = {id(n): k for k, n in self._nodes.items()}

            wrapped_config = {}
            for cls, style in config.items():
                new_style = style.copy()
                orig_label = style.get("label", "")

                def make_wrapper(orig_lbl):
                    def wrapper(node):
                        nid = node_ids.get(id(node), "?")

                        if callable(orig_lbl):
                            base_str = str(orig_lbl(node))
                        else:
                            base_str = str(orig_lbl)

                        if base_str:
                            return f"ID: {nid}\n{base_str}"
                        return f"ID: {nid}"

                    return wrapper

                new_style["label"] = make_wrapper(orig_label)
                wrapped_config[cls] = new_style
            return wrapped_config

        return config

    def topological_sort(self, reverse: bool = False) -> Generator[K, None, None]:
        """
        Yields node_ids in topological order using Kahn's Algorithm.

        The order is computed once and memoized on the instance; any structural
        mutation (adding/removing nodes or edges) invalidates the cache.
        """
        cached = self._topo_cache.get(reverse)
        if cached is None:
            cached = list(self._kahn_topological_sort(reverse))
            self._topo_cache[reverse] = cached
        return iter(cached)

    def _kahn_topological_sort(self, reverse: bool = False) -> Generator[K, None, None]:
        """Uncached Kahn's algorithm — see :meth:`topological_sort`."""
        # Calculate in-degrees based on existing edges
        if reverse:
            # Out-degree for reverse topological sort
            degree = {u: len(self._adj[u]) for u in self._nodes}
            queue = deque([u for u, deg in degree.items() if deg == 0])
            adjacency = self._rev_adj
        else:
            # In-degree for normal topological sort
            degree = {u: len(self._rev_adj[u]) for u in self._nodes}
            queue = deque([u for u, deg in degree.items() if deg == 0])
            adjacency = self._adj

        visited_count = 0
        while queue:
            u = queue.popleft()
            yield u
            visited_count += 1

            # Decrement degree for neighbors
            for v in adjacency[u]:
                degree[v] -= 1
                if degree[v] == 0:
                    queue.append(v)

        if visited_count != len(self._nodes):
            raise RuntimeError(
                f"Cycle detected in Graph! Visited {visited_count}/{len(self._nodes)} nodes."
            )

    def layered_topological_sort(self, reverse: bool = False) -> Generator[List[K], None, None]:
        """
        Yields lists of node_ids in topological order, grouped by layers.
        Nodes in the same layer have the same distance from the root(s).
        """

        if reverse:
            # Out-degree for reverse layering
            degree = {u: len(self._adj[u]) for u in self._nodes}
            queue = deque([u for u, deg in degree.items() if deg == 0])
        else:
            # In-degree for normal layering
            degree = {u: len(self._rev_adj[u]) for u in self._nodes}
            queue = deque([u for u, deg in degree.items() if deg == 0])

        visited_count = 0
        while queue:
            layer_size = len(queue)
            current_layer = []
            for _ in range(layer_size):
                u = queue.popleft()
                current_layer.append(u)
                visited_count += 1

                neighbors = self._rev_adj[u] if reverse else self._adj[u]
                for v in neighbors:
                    degree[v] -= 1
                    if degree[v] == 0:
                        queue.append(v)

            yield current_layer

        if visited_count != len(self._nodes):
            raise RuntimeError(
                f"Cycle detected in Graph! Visited {visited_count}/{len(self._nodes)} nodes."
            )

    def print_in_order(self) -> None:
        """Prints the node labels in alphabetical topological order."""
        for layer in self.layered_topological_sort():
            for node_id in layer:
                node = self.get_node_data(node_id)
                config = self.get_node_config()

                # Fetch matching style configuration handling subclass inheritances properly
                style = {}
                for cls, c in config.items():
                    if isinstance(node, cls):
                        style = c
                        break

                label_val = style.get("label", str(node))
                if callable(label_val):
                    label_str = str(label_val(node))
                else:
                    label_str = str(label_val)

                print(f"ID: {node_id} | {label_str.replace(chr(10), ' ')}")


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

    def get_parent(self, node_id: K) -> Optional[K]:
        """Returns the parent node ID, or None if root."""
        parents = self.get_parents(node_id)
        if not parents:
            return None
        return parents[0]


class BinaryTree(Tree[K, N, E]):
    """
    A binary tree with explicit left/right child semantics.
    Enforces that each node is either a leaf (no children) or has exactly two children.
    """

    def __init__(self, node_allocator: Optional[NodeAllocator[K]] = None):
        super().__init__(node_allocator)
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
