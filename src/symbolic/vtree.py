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

    # TODO: Does it make sense to have a Tree class that VTree can inherit from? Does assuming tree properties (e.g. a node has at most one parent) help?

    @property
    def node_config(self) -> Dict[type, Dict[str, Any]]:
        return {
            VNode: {
                "shape": "box",
                "label": lambda n: f"Scope: {list(n.scope)}",
                "color": "lightblue",
            }
        }
