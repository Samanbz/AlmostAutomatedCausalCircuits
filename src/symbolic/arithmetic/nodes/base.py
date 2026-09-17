from typing import List, Optional

import torch

from src.symbolic.base import Node
from src.utils import BitSet, Support


class ArithmeticNode(Node):
    """Represents an arithmetic operation node."""

    def __init__(
        self,
        support: Support = None,
        num_nodes: int = 1,
        num_groups: int = 1,
        node_supports: Optional[List[Support]] = None,
        md_set: Optional[BitSet] = None,
    ):
        self.support = support
        self.num_groups = num_groups
        self.num_nodes = num_nodes
        self.node_supports = (
            node_supports
            if node_supports is not None
            else ([support] * num_nodes if support is not None else None)
        )
        self.md_set = md_set
        self.marg_scope = BitSet()

    @property
    def scope(self) -> BitSet:
        if self.support is None:
            return BitSet()
        return self.support.scope

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def to(self, device: torch.device):
        """Moves any tensor attributes of this node to the given device."""
        return self

    def __repr__(self):
        scope_str = self.support.scope if self.support is not None else None
        return f"{self.__class__.__name__}(scope={scope_str}, #nodes={self.num_nodes}, #groups={self.num_groups})"
