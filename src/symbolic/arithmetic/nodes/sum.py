import torch

from .base import ArithmeticNode


class SumNode(ArithmeticNode):
    """Represents a sum operation."""

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        assert children_outputs is not None and len(children_outputs) == 1
        child_out = children_outputs[0]
        B = child_out.shape[0]
        h_in = child_out.shape[1]
        
        prods_per_sum = h_in // self.unit_count
        weights = getattr(self, "weights", None)
        if weights is None:
            weights = torch.ones(self.unit_count, prods_per_sum, device=data.device) / prods_per_sum
            
        log_w = torch.log(weights.clamp(min=1e-12))
        child_blocks = child_out.reshape(B, self.unit_count, prods_per_sum)
        
        # log_w: [h_out, prods_per_sum]
        # child_blocks: [B, h_out, prods_per_sum]
        # Broadcast sum over prods_per_sum
        return torch.logsumexp(log_w.unsqueeze(0) + child_blocks, dim=2)


class UniversalSumNode(SumNode):
    """Represents a dense, unpartitioned sum operation over all children products."""

    def forward(
        self, data: torch.Tensor, children_outputs: list[torch.Tensor] = None
    ) -> torch.Tensor:
        assert children_outputs is not None and len(children_outputs) == 1
        child_out = children_outputs[0]
        h_in = child_out.shape[1]
        weights = getattr(self, "weights", None)
        if weights is None:
            weights = torch.ones(self.unit_count, h_in, device=data.device) / h_in
        log_w = torch.log(weights.clamp(min=1e-12))
        return torch.logsumexp(log_w.unsqueeze(0) + child_out.unsqueeze(1), dim=2)
