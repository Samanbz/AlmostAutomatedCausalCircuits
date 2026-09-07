from typing import List, Optional

import torch

from src.logger import logger as g_logger
from src.symbolic.arithmetic.weights import Weights
from src.utils import BitSet, Support

from .base import ArithmeticNode


logger = g_logger.getChild("SumLayer")


class SumLayer(ArithmeticNode):
    """Represents a sum operation.

    Attributes:
        sparse: If True, weights are block-diagonal (partitioned).
                If False, weights are dense.
    """

    def __init__(
        self,
        num_nodes: int,
        num_groups: int,
        md_set: BitSet,
        support: Support = None,
        node_supports: Optional[List[Support]] = None,
    ):
        super().__init__(
            support=support,
            num_nodes=num_nodes,
            num_groups=num_groups,
            node_supports=node_supports,
            md_set=md_set,
        )
        self.log_weights: Weights = None

    def to(self, device: torch.device):
        if getattr(self, "log_weights", None) is not None:
            self.log_weights.to(device)
        return super().to(device)

    def __repr__(self):
        shape_str = ""
        if getattr(self, "log_weights", None) is not None:
            shape = list(self.log_weights.shape)
            shape_str = f" shape={shape}"
        scope_str = self.support.scope if self.support is not None else None
        return f"{self.__class__.__name__}(scope={scope_str}, num_nodes={self.num_nodes}, num_groups={self.num_groups}){shape_str}"

    def forward(
        self, data: torch.Tensor, children_outputs: tuple[torch.Tensor] = None
    ) -> torch.Tensor:
        assert children_outputs is not None and len(children_outputs) == 2, (
            "SumLayer must have exactly 2 children (left and right)."
        )
        assert self.log_weights is not None, (
            "SumLayer log_weights must be initialized before forward pass."
        )

        left_out, right_out = children_outputs

        return self.log_weights.forward(left_out, right_out)

    def update_params(
        self,
        data: torch.Tensor,
        step_size: float = 1.0,
        smoothing: float = 1e-4,
        valid_mask: torch.Tensor = None,
    ):
        """
        Updates the weights using EMA on the computed gradient counts.
        """
        if self.log_weights is None:
            return

        self.log_weights.update_params(step_size=step_size, smoothing=smoothing)
