import torch

from src.logger import logger as g_logger
from src.utils import BitSet, Support

from .base import ArithmeticNode


logger = g_logger.getChild("SumNode")


class SumNode(ArithmeticNode):
    """Represents a sum operation.

    Attributes:
        sparse: If True, weights are block-diagonal (partitioned).
                If False, weights are dense.
    """

    def __init__(
        self,
        unit_count: int = 1,
        md_set: BitSet = None,
        sparse: bool = True,
        support: Support = None,
        unit_supports: list[Support] = None,
    ):
        super().__init__(
            support=support, unit_count=unit_count, unit_supports=unit_supports, md_set=md_set
        )
        self.sparse = sparse
        self.log_weights = None

    def __repr__(self):
        shape_str = ""
        if getattr(self, "log_weights", None) is not None:
            shape = list(self.log_weights.shape)
            if len(shape) == 3:
                shape_str = f" shape=[{shape[0]}, {shape[1] * shape[2]}]"
            else:
                shape_str = f" shape={shape}"
        scope_str = self.support.scope if self.support is not None else None
        return f"{self.__class__.__name__}(scope={scope_str}, unit_count={self.unit_count}){shape_str}"

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        assert children_outputs is not None and len(children_outputs) == 1
        assert self.log_weights is not None, (
            "SumNode log_weights must be initialized before forward pass."
        )

        child_out = children_outputs[0]
        child_out = child_out.clamp(min=-1e10)
        logger.debug(
            "SumNode forward: sparse=%s unit_count=%d log_weights=%s child_out=%s",
            self.sparse,
            self.unit_count,
            list(self.log_weights.shape),
            list(child_out.shape),
        )
        inputs = self.log_weights.unsqueeze(0) + child_out.unsqueeze(1)
        return torch.logsumexp(inputs, dim=2)

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
        if self.log_weights is None or self.log_weights.grad is None:
            return

        # Gradients of LL w.r.t log_weights are exactly the expected counts
        counts = self.log_weights.grad.nan_to_num(0.0).clamp(min=0.0)

        if self.sparse:
            # Infer mask from current weights (the ones that are close to clamp min)
            mask = (self.log_weights.data > -20.0).float()
            counts = counts * mask
            probs = (counts + smoothing * mask) / (
                counts.sum(dim=-1, keepdim=True) + smoothing * mask.sum(dim=-1, keepdim=True)
            )
            probs = probs.nan_to_num(0.0)
        else:
            # Normalize to get the batch M-step estimate
            probs = (counts + smoothing) / (
                counts.sum(dim=-1, keepdim=True) + smoothing * counts.shape[-1]
            )

        # EMA update in probability space
        old_probs = torch.exp(self.log_weights.data)
        new_probs = (1.0 - step_size) * old_probs + step_size * probs

        # Apply mask again in probability space to ensure exact zeros stay zeros (clamped to 1e-15)
        if self.sparse:
            new_probs = torch.where(
                mask > 0,
                new_probs,
                torch.tensor(1e-15, dtype=new_probs.dtype, device=new_probs.device),
            )

        self.log_weights.data.copy_(torch.log(new_probs.clamp(min=1e-20)))
        self.log_weights.grad.zero_()
