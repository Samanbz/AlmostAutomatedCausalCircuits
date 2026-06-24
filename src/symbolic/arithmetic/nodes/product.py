import torch

from .base import ArithmeticNode


class ProductNode(ArithmeticNode):
    """Represents a product operation."""

    pass


class KroneckerProductNode(ProductNode):
    """Represents a product operation (Cartesian/Kronecker)."""

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        assert children_outputs is not None and len(children_outputs) >= 1
        if len(children_outputs) == 1:
            return children_outputs[0]
        L = children_outputs[0]
        R = children_outputs[1]
        B, h_L = L.shape
        h_R = R.shape[1]
        outer = L.unsqueeze(2) + R.unsqueeze(1)
        return outer.reshape(B, h_L * h_R)
