from typing import Any, Dict

from src.graph import BinaryTree, Node
from src.utils import BitSet


class VNode(Node):
    """A node in a variable tree (vtree) representing a subset of variables."""

    def __init__(self, scope: BitSet):
        super().__init__()
        self.scope = scope

    def __repr__(self):
        return f"VNode(scope={self.scope})"


class VTree(BinaryTree[int, VNode, None]):
    """A variable tree (vtree) represented as a directed acyclic graph."""

    @property
    def node_config(self) -> Dict[type, Dict[str, Any]]:
        return {
            VNode: {
                "shape": "box",
                "label": lambda n: f"Scope: {list(n.scope)}",
                "color": "lightblue",
            }
        }


class MDVNode(VNode):
    """A node in an md-vtree (marginal deterministic vtree) representing a subset of variables and their marginal determinism specification (md-set)."""

    def __init__(self, scope: BitSet, md_set: BitSet):
        super().__init__(scope)
        self.md_set = md_set

    def __repr__(self):
        return f"MDVNode(scope={self.scope}, md_set={self.md_set})"


class MDVTree(BinaryTree[int, MDVNode, None]):
    """An md-vtree (marginal deterministic vtree) represented as a directed acyclic graph."""

    @classmethod
    def from_vtree(cls, vtree: VTree, md_sets: Dict[int, BitSet]) -> "MDVTree":
        """
        Compute optimal labeling of a VTree given a list of marginal determinism sets following the
        algorithm described in Wang & Kwiatkowska (202?).

        :param vtree: The input VTree to label
        :type vtree: VTree
        :param md_sets: List of required marginal determinism sets to use for labeling
        :type md_sets: List[BitSet]
        :return: An MDVTree with the same structure as the input VTree, but with optimal labels assigned to each node.
        :rtype: MDVTree
        """
        md_sets = [BitSet(s) for s in md_sets]

        def get_label_for_scope(scope: BitSet) -> BitSet:
            label = BitSet.universal()
            for md_set in md_sets:
                if not md_set.intersection(scope).is_empty:
                    label = label.intersection(md_set)
            if not label.is_universal:
                label = label.intersection(scope)
            return label

        md_vtree = MDVTree()

        def label_recursive(vid: int):
            vtree_node = vtree.get_node_data(vid)
            label = get_label_for_scope(vtree_node.scope)

            md_vnode = MDVNode(scope=vtree_node.scope, md_set=label)
            md_vtree.add_node(vid, md_vnode)

            children = vtree.get_children_pair(vid)
            if children is None:
                return

            l_vid, r_vid = children
            label_recursive(l_vid)
            label_recursive(r_vid)

            md_vtree.add_children(vid, l_vid, r_vid)

        root_id = vtree.get_root()
        label_recursive(root_id)
        return md_vtree

    @property
    def node_config(self) -> Dict[type, Dict[str, Any]]:
        return {
            MDVNode: {
                "shape": "box",
                "label": lambda n: f"Scope: {list(n.scope)}\nMD-Set: {list(n.md_set) if not n.md_set.is_universal else 'Universal'}",
                "color": "lightgreen",
            }
        }
