import math
from typing import List

import torch
from torch import nn


class TensorizedLayer(nn.Module):
    def __init__(self, out_idx: List[int] = None):
        super().__init__()
        self.out_idx = out_idx

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class ProductLayer(TensorizedLayer):
    def __init__(self, out_idx: List[int], left_idx: List[int], right_idx: List[int]):
        super().__init__(out_idx)
        self.left_idx = left_idx
        self.right_idx = right_idx

    def forward(self, global_buffer: torch.Tensor) -> torch.Tensor:
        res = global_buffer[:, self.left_idx] + global_buffer[:, self.right_idx]
        global_buffer[:, self.out_idx] = res
        return res


class SafeLogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        output = torch.logsumexp(input, dim=dim)
        ctx.save_for_backward(input, output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        dim = ctx.dim

        out_unsqueezed = output.unsqueeze(dim)
        mask = torch.isneginf(out_unsqueezed)

        safe_output = torch.where(mask, torch.zeros_like(out_unsqueezed), out_unsqueezed)
        safe_input = torch.where(mask, torch.zeros_like(input), input)

        softmax = torch.exp(safe_input - safe_output)
        softmax = torch.where(mask, torch.zeros_like(softmax), softmax)

        grad_input = grad_output.unsqueeze(dim) * softmax
        return grad_input, None


class GaussianInputLayer(TensorizedLayer):
    def __init__(
        self,
        out_idx: List[int],
        means: torch.Tensor,
        stds: torch.Tensor,
        lows: torch.Tensor,
        highs: torch.Tensor,
        scopes: List[int],
    ):
        super().__init__(out_idx)
        self.means = nn.Parameter(means, requires_grad=False)
        self.stds = nn.Parameter(stds, requires_grad=False)
        self.register_buffer("lows", lows)
        self.register_buffer("highs", highs)
        self.scopes = scopes

    def forward(self, global_buffer: torch.Tensor, input_values: torch.Tensor) -> torch.Tensor:
        x = input_values[:, self.scopes]
        self.saved_x = x.detach()

        z = (x - self.means) / self.stds
        log_phi_z = -0.5 * math.log(2 * math.pi) - 0.5 * (z**2)
        log_pdf_x = log_phi_z - torch.log(self.stds)

        cdf_high = 0.5 * (1 + torch.erf((self.highs - self.means) / (self.stds * math.sqrt(2))))
        cdf_low = 0.5 * (1 + torch.erf((self.lows - self.means) / (self.stds * math.sqrt(2))))
        Z = cdf_high - cdf_low

        trunc_log_pdf = log_pdf_x - torch.log(Z.clamp(min=1e-10))

        out_of_bounds = (x < self.lows) | (x > self.highs)
        trunc_log_pdf = torch.where(
            out_of_bounds, torch.full_like(trunc_log_pdf, float("-inf")), trunc_log_pdf
        )

        self.saved_log_pdf = trunc_log_pdf.requires_grad_(True)
        self.saved_log_pdf.retain_grad()

        global_buffer[:, self.out_idx] = self.saved_log_pdf

        return trunc_log_pdf

    def update_params(self):
        with torch.no_grad():
            valid_grads = torch.clamp(self.saved_log_pdf.grad, min=0.0)

            responsibilities = valid_grads + 1e-15
            total_resp = responsibilities.sum(dim=0)
            valid_mask = total_resp > 1e-5

            sufficient_stats = torch.stack([self.saved_x, self.saved_x**2], dim=0)

            exp_params = (responsibilities.unsqueeze(0) * sufficient_stats).sum(
                dim=1
            ) / total_resp.clamp(min=1e-15)

            new_means = exp_params[0]
            new_vars = exp_params[1] - (new_means**2)
            new_stds = torch.sqrt(new_vars.clamp(min=1e-5))

            self.means.copy_(torch.where(valid_mask, new_means, self.means))
            self.stds.copy_(torch.where(valid_mask, new_stds, self.stds))
