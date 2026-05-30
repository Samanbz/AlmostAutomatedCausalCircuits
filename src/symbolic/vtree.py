from typing import Any, Dict, List, Optional, Set

from src.utils import BitSet

from .base import BinaryTree, Node


class VNode(Node):
    """A node in a variable tree (vtree) representing a subset of variables."""

    def __init__(self, scope: BitSet, md_set: Optional[BitSet] = None):
        super().__init__()
        self.scope = scope
        self.md_set = md_set if md_set is not None else BitSet.universal()

    def __repr__(self):
        return f"VNode(scope={self.scope}, md_set={self.md_set})"


class VTree(BinaryTree[int, VNode, None]):
    """A variable tree (vtree) represented as a directed acyclic graph."""

    def compute_md_labeling(self, md_sets: List[Set[int]]) -> None:
        """
        Compute optimal labeling of a VTree given a list of marginal determinism sets following the
        algorithm described in Wang & Kwiatkowska (202?).

        :param md_sets: List of required marginal determinism sets to use for labeling
        :type md_sets: List[Set[int]]
        """

        md_sets: List[BitSet] = [BitSet(s) for s in md_sets]

        def get_label_for_scope(scope: BitSet) -> BitSet:
            label = BitSet.universal()
            for md_set in md_sets:
                if not md_set.intersection(scope).is_empty:
                    label = label.intersection(md_set)
            if not label.is_universal:
                label = label.intersection(scope)
            return label

        def label_recursive(vid: int):
            vtree_node = self.get_node_data(vid)
            label = get_label_for_scope(vtree_node.scope)
            vtree_node.md_set = label

            children = self.get_children_pair(vid)
            if children is None:
                return

            l_vid, r_vid = children
            label_recursive(l_vid)
            label_recursive(r_vid)

        root_id = self.get_root()
        label_recursive(root_id)

    def _get_base_node_config(self) -> Dict[type, Dict[str, Any]]:
        config = super()._get_base_node_config()

        config.update(
            {
                VNode: {
                    "shape": "box",
                    "label": lambda n: f"Scope: {list(n.scope)}\nMD-Set: {list(n.md_set) if not n.md_set.is_universal else 'Univ.'}",
                    "color": "lightblue",
                }
            }
        )
        return config
