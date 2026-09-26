"""Causal graph topology: the single mixed-graph class of the framework.

A :class:`CausalGraph` is an acyclic directed mixed graph (ADMG) over an ordered
node list with directed edges (``a -> b``) and bidirected edges (``a <-> b``,
canonically ordered). It is the *pure topology* object: it carries no mechanisms
or distributions. Two roles consume it:

* :func:`src.construction.latent_projection.latent_projection` projects an
  :class:`~src.symbolic.scm.graph.StructuralCausalModel` onto its observed nodes
  (hidden confounders become bidirected edges).
* :func:`src.symbolic.identification.t_id` runs causal identification on it
  (it also accepts an SCM directly and projects it internally).

Graph algorithms live here and nowhere else: ancestors, districts
(C-components), d-separation via ancestral moralization, minimal separators,
subgraph surgery, and topological order.
"""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _canonical_pair(a: Any, b: Any) -> Tuple[Any, Any]:
    """Order an unordered pair canonically (robust to mixed str/int node ids)."""
    return tuple(sorted((a, b), key=repr))  # type: ignore[return-type]


class CausalGraph:
    """Immutable acyclic directed mixed graph over an ordered node list.

    Parameters follow the ADMG convention: ``nodes`` in topological
    (declaration) order, ``directed_edges`` as ``(a, b)`` pairs meaning
    ``a -> b``, ``bidirected_edges`` as unordered pairs meaning ``a <-> b``
    (stored canonically ordered). Edges referencing unknown nodes are dropped.
    """

    def __init__(
        self,
        nodes: Iterable[Any],
        directed_edges: Iterable[Tuple[Any, Any]] = (),
        bidirected_edges: Iterable[Tuple[Any, Any]] = (),
    ):
        node_tuple = tuple(nodes)
        node_set = set(node_tuple)
        self.nodes: Tuple[Any, ...] = node_tuple
        self.directed: frozenset = frozenset(
            (a, b) for a, b in directed_edges if a in node_set and b in node_set
        )
        self.bidirected: frozenset = frozenset(
            _canonical_pair(a, b) for a, b in bidirected_edges if a in node_set and b in node_set
        )

    @classmethod
    def from_sets(
        cls,
        nodes: Iterable[Any],
        directed_edges: Iterable[Tuple[Any, Any]] = (),
        bidirected_edges: Iterable[Tuple[Any, Any]] = (),
    ) -> "CausalGraph":
        return cls(nodes, directed_edges, bidirected_edges)

    def has_edge(self, a: Any, b: Any) -> bool:
        """True iff the directed edge ``a -> b`` is present."""
        return (a, b) in self.directed

    def has_bidirected(self, a: Any, b: Any) -> bool:
        """True iff the bidirected edge ``a <-> b`` is present (order-insensitive)."""
        return _canonical_pair(a, b) in self.bidirected

    def parents(self, v: Any) -> Set[Any]:
        return {a for a, b in self.directed if b == v}

    def children(self, v: Any) -> Set[Any]:
        return {b for a, b in self.directed if a == v}

    def spouses(self, v: Any) -> Set[Any]:
        out = set()
        for a, b in self.bidirected:
            if a == v:
                out.add(b)
            elif b == v:
                out.add(a)
        return out

    def ancestors(self, s: Iterable[Any]) -> Set[Any]:
        """Directed ancestors of ``s`` (including ``s`` itself)."""
        anc = set(s)
        queue = list(s)
        while queue:
            node = queue.pop()
            for parent in self.parents(node):
                if parent not in anc:
                    anc.add(parent)
                    queue.append(parent)
        return anc

    def districts(self, subset: Optional[Set[Any]] = None) -> List[Set[Any]]:
        """C-components (districts): connectivity via bidirected edges."""
        nodes = set(self.nodes) if subset is None else set(subset)
        comps: List[Set[Any]] = []
        unvisited = set(nodes)
        while unvisited:
            start = unvisited.pop()
            comp = {start}
            queue = [start]
            while queue:
                cur = queue.pop()
                for nb in self.spouses(cur):
                    if nb in unvisited:
                        unvisited.remove(nb)
                        comp.add(nb)
                        queue.append(nb)
            comps.append(comp)
        return comps

    def remove_nodes(self, x: Set[Any]) -> "CausalGraph":
        """``G \\ x``: delete the nodes and all incident edges."""
        keep = [n for n in self.nodes if n not in x]
        return CausalGraph(
            keep,
            ((a, b) for a, b in self.directed if a not in x and b not in x),
            ((a, b) for a, b in self.bidirected if a not in x and b not in x),
        )

    def remove_outgoing(self, x: Set[Any]) -> "CausalGraph":
        r"""``G_{\bar{x}}``: delete outgoing edges of ``x`` (x keeps its parents)."""
        return CausalGraph(
            self.nodes,
            ((a, b) for a, b in self.directed if a not in x),
            self.bidirected,
        )

    def remove_incoming(self, x: Set[Any]) -> "CausalGraph":
        r"""``G_{\underline{x}}``: delete incoming edges of ``x`` (x keeps its children)."""
        return CausalGraph(
            self.nodes,
            ((a, b) for a, b in self.directed if b not in x),
            self.bidirected,
        )

    def induced_subgraph(self, s: Set[Any]) -> "CausalGraph":
        """``G_S``: keep only nodes in ``s`` and edges among them."""
        s = set(s)
        return CausalGraph(
            (n for n in self.nodes if n in s),
            ((a, b) for a, b in self.directed if a in s and b in s),
            ((a, b) for a, b in self.bidirected if a in s and b in s),
        )

    def d_separated(self, x: Any, y: Any, given: Set[Any]) -> bool:
        """d-separation on the mixed graph via ancestral moralization.

        ``x ⊥d y | given`` iff every node of ``x`` is disconnected from every
        node of ``y`` in the moral graph of ``An(x ∪ y ∪ given)`` after
        removing ``given``.  ``x`` and ``y`` may be single nodes or sets.
        """
        xs = set(x) if isinstance(x, (set, frozenset)) else {x}
        ys = set(y) if isinstance(y, (set, frozenset)) else {y}
        blocked = set(given)
        if (xs & ys) - blocked:
            return False
        relevant = self.ancestors(xs | ys | blocked)
        sub = self.induced_subgraph(relevant)
        # Undirected skeleton: parent-child + spouse links.
        adj: Dict[Any, Set[Any]] = {n: set() for n in sub.nodes}

        def link(a, b):
            adj[a].add(b)
            adj[b].add(a)

        for a, b in sub.directed:
            link(a, b)
        for a, b in sub.bidirected:
            link(a, b)
        child_parents: Dict[Any, Set[Any]] = {n: set() for n in sub.nodes}
        for a, b in sub.directed:
            child_parents[b].add(a)
        for v in sub.nodes:
            fam = child_parents[v] | self.spouses(v)
            fam = [n for n in fam if n in adj]
            for i in range(len(fam)):
                for j in range(i + 1, len(fam)):
                    link(fam[i], fam[j])
        # Remove conditioning set.
        seen = set()
        stack = [n for n in xs if n not in blocked]
        while stack:
            cur = stack.pop()
            if cur in seen or cur in blocked:
                continue
            seen.add(cur)
            if cur in ys:
                return False
            stack.extend(adj[cur])
        return True

    def minimal_separator(self, v: Any, candidates: Set[Any]) -> Set[Any]:
        """Minimal ``S ⊆ candidates`` with ``v ⊥d candidates \\ S | S``.

        Greedy deletion from ``candidates``; ties broken by node order.
        """
        remaining = [c for c in self.nodes if c in candidates]
        sep = list(remaining)
        for cand in remaining:
            trial = [s for s in sep if s != cand]
            if self.d_separated(v, cand, set(trial)):
                sep = trial
        return set(sep)

    def topological_order(self, subset: Optional[Set[Any]] = None) -> List[Any]:
        nodes = [n for n in self.nodes if subset is None or n in subset]
        node_set = set(nodes)
        indeg = {n: len(self.parents(n) & node_set) for n in nodes}
        queue = [n for n in nodes if indeg[n] == 0]
        order = []
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for child in self.children(cur) & node_set:
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)
        return order

    def __str__(self) -> str:
        directed = sorted(self.directed, key=repr)
        bidirected = sorted(self.bidirected, key=repr)
        lines = [f"CausalGraph(nodes={list(self.nodes)})"]
        lines.append("  directed:   " + (", ".join(f"{a} -> {b}" for a, b in directed) or "none"))
        lines.append(
            "  bidirected: " + (", ".join(f"{a} <-> {b}" for a, b in bidirected) or "none")
        )
        return "\n".join(lines)
