from typing import TypeVar, Generic, Dict, List, Generator
from collections import deque

# Generic types for Node Payload (N) and Edge Payload (E)
N = TypeVar('N') 
E = TypeVar('E') 

class DirectedAcyclicGraph(Generic[N, E]):
    """
    A generic DAG implementation.
    N: Type of data stored in nodes.
    E: Type of data stored in edges.
    """
    def __init__(self):
        self._nodes: Dict[str, N] = {}
        # Adjacency: parent_id -> {child_id: edge_data}
        self._adj: Dict[str, Dict[str, E]] = {}
        # Reverse Adjacency: child_id -> {parent_id: edge_data}
        self._rev_adj: Dict[str, Dict[str, E]] = {}

    def add_node(self, node_id: str, data: N) -> None:
        """Adds a node to the graph."""
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists.")
        self._nodes[node_id] = data
        self._adj[node_id] = {}
        self._rev_adj[node_id] = {}

    def add_edge(self, source: str, target: str, data: E = None) -> None:
        """Adds a directed edge from source to target."""
        if source not in self._nodes or target not in self._nodes:
            raise KeyError(f"Source '{source}' or Target '{target}' not found.")
        
        # TODO: Check for cycles before adding (Simple DFS check could be added here for strictness, 
        # but topological sort will catch it later).
        
        self._adj[source][target] = data
        self._rev_adj[target][source] = data

    def get_parents(self, node_id: str) -> List[str]:
        """Returns a list of parent node IDs."""
        return list(self._rev_adj[node_id].keys())

    def get_node_data(self, node_id: str) -> N:
        return self._nodes[node_id]

    def get_edge_data(self, source: str, target: str) -> E:
        return self._adj[source][target]

    def topological_sort(self) -> Generator[str, None, None]:
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
            raise RuntimeError(f"Cycle detected in Graph! Visited {visited_count}/{len(self._nodes)} nodes.")