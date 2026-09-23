from typing import Optional

import torch

from src.symbolic.base import Node
from src.utils import BitSet


class ArithmeticNode(Node):
    """Represents an arithmetic operation node."""

    def __init__(
        self,
        scope: Optional[BitSet] = None,
        md_set: Optional[BitSet] = None,
        num_groups: int = 1,
        num_nodes: int = 1,
    ):
        self.scope = scope
        self.md_set = md_set
        self.num_groups = num_groups
        self.num_nodes = num_nodes

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def to(self, device: torch.device):
        """Moves any tensor attributes of this node to the given device."""
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(scope={self.scope}, #nodes={self.num_nodes}, #groups={self.num_groups})"
