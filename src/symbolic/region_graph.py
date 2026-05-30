from enum import Enum
from typing import Any, Dict

from src.utils import BitSet
from src.utils.support import Support

from .base import DirectedAcyclicGraph, Node


class LayerType(Enum):
    LEFT_MIXING = "left-mixing"
    RIGHT_MIXING = "right-mixing"
    SYNTHESIZING = "synthesizing"
    UNIVERSAL = "universal"

    @classmethod
    def from_md_sets(
        cls, parent_mdset: BitSet, l_child_mdset: BitSet, r_child_mdset: BitSet
    ) -> "LayerType":
        if parent_mdset.is_universal or parent_mdset.is_empty:
            return cls.UNIVERSAL
        if parent_mdset == l_child_mdset:
            return cls.LEFT_MIXING
        if parent_mdset == r_child_mdset:
            return cls.RIGHT_MIXING
        if parent_mdset == l_child_mdset.union(r_child_mdset):
            return cls.SYNTHESIZING
        raise ValueError("Invalid md-sets relationship for LayerType")


class RegionGraphNode(Node):
    def __init__(
        self,
        scope: BitSet,
        support: Support = None,
    ):
        self.scope = scope
        self.support = support or {}

    def __repr__(self):
        return f"{self.__class__.__name__}(scope={self.scope}, support={self.support})"


class RegionNode(RegionGraphNode):
    def __init__(
        self,
        scope: BitSet,
        support: Support,
        md_set: BitSet,
        layer_type: LayerType,
        is_constrained: bool = True,
    ):
        super().__init__(scope, support)
        self.md_set = md_set
        self.layer_type = layer_type
        self.is_constrained = is_constrained

    def __repr__(self):
        c = "C" if self.is_constrained else "U"
        return (
            super().__repr__()[:-1] + f" md_set={self.md_set}, layer_type={self.layer_type}, {c})"
        )


class PartitionNode(RegionGraphNode):
    pass


class RegionGraph(DirectedAcyclicGraph[int, RegionGraphNode, Any]):
    def _get_base_node_config(self) -> Dict[type, Dict[str, Any]]:
        config = super()._get_base_node_config()

        def support_label(n: RegionGraphNode) -> str:
            s_parts = []
            for v, d in sorted(n.support.intervals.items()):
                s_parts.append(f"{v}: {d}")
            return "\n".join(s_parts)

        config.update(
            {
                RegionNode: {
                    "color": "#ccffcc",
                    "label": lambda n: f"Region [{n.layer_type.value}]\nScope: {list(n.scope)}\nMD: {list(n.md_set) if not n.md_set.is_universal else 'Univ.'}\n{support_label(n)}",
                },
                PartitionNode: {
                    "color": "#ccccff",
                    "label": lambda n: f"Partition\nScope: {list(n.scope)}\n{support_label(n)}",
                },
            }
        )
        return config
