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
        is_nan = torch.isnan(x)
        x_safe = torch.where(is_nan, torch.zeros_like(x), x)
        self.saved_x = x_safe.detach()
        self.nan_mask = is_nan.detach()

        z = (x_safe - self.means) / self.stds
        log_phi_z = -0.5 * math.log(2 * math.pi) - 0.5 * (z**2)
        log_pdf_x = log_phi_z - torch.log(self.stds)

        cdf_high = 0.5 * (1 + torch.erf((self.highs - self.means) / (self.stds * math.sqrt(2))))
        cdf_low = 0.5 * (1 + torch.erf((self.lows - self.means) / (self.stds * math.sqrt(2))))
        Z = cdf_high - cdf_low

        trunc_log_pdf = log_pdf_x - torch.log(Z.clamp(min=1e-10))

        out_of_bounds = (x_safe < self.lows) | (x_safe > self.highs)
        trunc_log_pdf = torch.where(
            out_of_bounds, torch.full_like(trunc_log_pdf, float("-inf")), trunc_log_pdf
        )

        trunc_log_pdf = torch.where(is_nan, torch.zeros_like(trunc_log_pdf), trunc_log_pdf)

        self.saved_log_pdf = trunc_log_pdf.requires_grad_(True)
        self.saved_log_pdf.retain_grad()

        global_buffer[:, self.out_idx] = self.saved_log_pdf

        return trunc_log_pdf

    def update_params(self):
        with torch.no_grad():
            valid_grads = torch.clamp(self.saved_log_pdf.grad, min=0.0)
            valid_grads = valid_grads.masked_fill(self.nan_mask, 0.0)

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


class UniformInputLayer(TensorizedLayer):
    def __init__(
        self,
        out_idx: List[int],
        lows: torch.Tensor,
        highs: torch.Tensor,
        scopes: List[int],
    ):
        super().__init__(out_idx)
        self.register_buffer("lows", lows)
        self.register_buffer("highs", highs)
        self.scopes = scopes
        self.register_buffer("log_pdf", -torch.log((highs - lows).clamp(min=1e-10)))

    def forward(self, global_buffer: torch.Tensor, input_values: torch.Tensor) -> torch.Tensor:
        x = input_values[:, self.scopes]
        is_nan = torch.isnan(x)
        x_safe = torch.where(is_nan, torch.zeros_like(x), x)

        out_of_bounds = (x_safe < self.lows) | (x_safe > self.highs)

        batch_size = x.size(0)
        log_probs = self.log_pdf.unsqueeze(0).expand(batch_size, -1)

        log_probs = torch.where(out_of_bounds, torch.full_like(log_probs, float("-inf")), log_probs)
        log_probs = torch.where(is_nan, torch.zeros_like(log_probs), log_probs)

        global_buffer[:, self.out_idx] = log_probs
        return log_probs

    def update_params(self):
        pass


class CategoricalInputLayer(TensorizedLayer):
    def __init__(
        self,
        out_idx: List[int],
        categories: torch.Tensor,
        log_probabilities: torch.Tensor,
        scopes: List[int],
    ):
        super().__init__(out_idx)
        self.register_buffer("categories", categories)
        self.log_probabilities = nn.Parameter(log_probabilities)
        self.scopes = scopes

    def forward(self, global_buffer: torch.Tensor, input_values: torch.Tensor) -> torch.Tensor:
        x = input_values[:, self.scopes]
        is_nan = torch.isnan(x)
        x_safe = torch.where(is_nan, torch.zeros_like(x), x)

        self.saved_x = x_safe.detach()
        self.nan_mask = is_nan.detach()

        x_expanded = x_safe.unsqueeze(-1)
        matches = x_expanded == self.categories.unsqueeze(0)

        log_probs = self.log_probabilities.unsqueeze(0).expand(x.size(0), -1, -1)
        matched_log_probs = log_probs.masked_fill(~matches, float("-inf"))

        node_log_probs = torch.logsumexp(matched_log_probs, dim=-1)
        node_log_probs = torch.where(is_nan, torch.zeros_like(node_log_probs), node_log_probs)

        self.saved_log_pdf = node_log_probs.requires_grad_(True)
        self.saved_log_pdf.retain_grad()

        global_buffer[:, self.out_idx] = self.saved_log_pdf
        return node_log_probs

    def update_params(self):
        with torch.no_grad():
            if getattr(self, "saved_log_pdf", None) is None or self.saved_log_pdf.grad is None:
                return

            valid_grads = torch.clamp(self.saved_log_pdf.grad, min=0.0)
            valid_grads = valid_grads.masked_fill(self.nan_mask, 0.0)

            x_expanded = self.saved_x.unsqueeze(-1)
            matches = x_expanded == self.categories.unsqueeze(0)

            responsibilities = valid_grads + 1e-15
            resp_expanded = responsibilities.unsqueeze(-1).masked_fill(~matches, 0.0)

            total_resp_per_cat = resp_expanded.sum(dim=0)
            total_resp = total_resp_per_cat.sum(dim=-1, keepdim=True)
            valid_mask = (total_resp > 1e-5).squeeze(-1)

            new_probs = total_resp_per_cat / total_resp.clamp(min=1e-15)
            new_log_probs = torch.log(new_probs.clamp(min=1e-10))

            self.log_probabilities.copy_(
                torch.where(valid_mask.unsqueeze(-1), new_log_probs, self.log_probabilities)
            )
