from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


LOG_ZERO = -46.0517  # log(1e-20)


class _SafeLogSumExp(torch.autograd.Function):
    """Like ``torch.logsumexp`` but returns zero gradients for all-``-inf`` slices.

    PyTorch's native ``logsumexp`` returns ``-inf`` when all inputs are ``-inf``,
    but its backward pass produces NaN gradients (because it computes
    ``exp(x - out)`` with ``x == out == -inf``). This wrapper masks those
    slices so gradients are zero instead of NaN.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, dim, keepdim: bool):
        output = torch.logsumexp(input, dim=dim, keepdim=keepdim)
        ctx.save_for_backward(input, output)
        ctx.dim = dim
        ctx.keepdim = keepdim
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, output = ctx.saved_tensors
        dim = ctx.dim
        keepdim = ctx.keepdim

        if not keepdim:
            dims = dim if isinstance(dim, (tuple, list)) else (dim,)
            dims_set = set(dims)
            target_shape = []
            out_ptr = 0
            for i in range(input.ndim):
                if i in dims_set:
                    target_shape.append(1)
                else:
                    target_shape.append(output.shape[out_ptr])
                    out_ptr += 1
            output = output.view(target_shape)
            grad_output = grad_output.view(target_shape)

        # Clamp output so that ``input - output`` is never ``-inf - (-inf)``.
        # Where the true output is -inf, all inputs are -inf, so the clamped
        # difference is still -inf and exp(...) is 0.
        safe_output = output.clamp(min=-1e10)
        mask = ~torch.isneginf(output)
        # Zero out grad_output on all-``-inf`` slices before multiplying, so that
        # any NaN/inf upstream gradients in those slices do not leak into the
        # returned gradient.
        safe_grad_output = torch.where(mask, grad_output, torch.zeros_like(grad_output))
        grad_input = torch.where(
            mask,
            safe_grad_output * torch.exp(input - safe_output),
            torch.zeros_like(input),
        )
        return grad_input, None, None


def safe_logsumexp(input: torch.Tensor, dim, keepdim: bool = False) -> torch.Tensor:
    return _SafeLogSumExp.apply(input, dim, keepdim)


class Weights(ABC):
    """Abstract base class for lazy weights. Duck-types as a torch.Tensor for basic ops."""

    log_weights: torch.Tensor

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Returns the logical shape of the weights."""
        pass

    @abstractmethod
    def forward(self, left_out: torch.Tensor, right_out: torch.Tensor) -> torch.Tensor:
        """
        Applies the weights to the combined product of the left and right children.
        """
        pass

    @abstractmethod
    def uniformize(self) -> "Weights":
        """Returns a new Weights object with log-weights set to 0.0 (weights = 1.0)"""
        pass

    @abstractmethod
    def to(self, device: torch.device) -> "Weights":
        """Moves the underlying tensors to the given device."""
        pass

    @abstractmethod
    def detach(self) -> "Weights":
        """Returns a new Weights object with detached tensors."""
        pass

    @abstractmethod
    def requires_grad_(self, requires_grad: bool) -> "Weights":
        """Sets requires_grad on underlying tensors."""
        pass

    @abstractmethod
    def update_params(self, step_size: float = 1.0, smoothing: float = 1e-4) -> None:
        """EM update."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(shape={list(self.shape)})"


class DenseWeights(Weights):
    """Standard explicit dense weights for a SumLayer."""

    def __init__(self, log_weights: torch.Tensor):
        self.log_weights = log_weights

    def to(self, device: torch.device) -> "Weights":
        if isinstance(self.log_weights, torch.Tensor):
            self.log_weights = self.log_weights.to(device)
            if not self.log_weights.is_leaf:
                self.log_weights = self.log_weights.detach().requires_grad_(
                    self.log_weights.requires_grad
                )
        return self

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.log_weights.shape)

    def uniformize(
        self,
        other_child_axis: Optional[str] = None,
    ) -> "Weights":
        if other_child_axis is None:
            return DenseWeights(torch.zeros_like(self.log_weights))

        assert other_child_axis is not None and other_child_axis in ["left", "right"]

        # weights have shape [G, U, G_L, L, G_R, R]
        total = safe_logsumexp(
            self.log_weights,
            dim=(4, 5) if other_child_axis == "right" else (2, 3),
            keepdim=True,
        )
        return DenseWeights(self.log_weights - total)

    def forward(
        self, left_out: torch.Tensor, right_out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        w = self.log_weights
        G, U, G_L, L, G_R, R = w.shape
        B = left_out.shape[0]

        if right_out is None:
            # right_out is absent, effectively 0 in log space since we don't multiply it
            prod = left_out.view(B, G_L, L, 1, 1)
        else:
            prod = left_out.view(B, G_L, L, 1, 1) + right_out.view(B, 1, 1, G_R, R)

        out = safe_logsumexp(prod.unsqueeze(1).unsqueeze(2) + w.unsqueeze(0), dim=(3, 4, 5, 6))
        return out.view(B, G, U)

    def detach(self) -> "DenseWeights":
        return DenseWeights(self.log_weights.detach())

    def requires_grad_(self, requires_grad: bool) -> "DenseWeights":
        self.log_weights.requires_grad_(requires_grad)
        return self

    def update_params(self, step_size: float = 1.0, smoothing: float = 1e-4) -> None:
        if self.log_weights is None or self.log_weights.grad is None:
            return

        counts = self.log_weights.grad.nan_to_num(0.0).clamp(min=0.0)

        sum_dims = (2, 3, 4, 5)
        denom = counts.sum(dim=sum_dims, keepdim=True) + smoothing * (
            counts.shape[2] * counts.shape[3] * counts.shape[4] * counts.shape[5]
        )
        probs = (counts + smoothing) / denom

        old_probs = torch.exp(self.log_weights.data)
        new_probs = (1.0 - step_size) * old_probs + step_size * probs

        self.log_weights.data.copy_(torch.log(new_probs.clamp(min=1e-20)))
        self.log_weights.grad.zero_()

    def __str__(self) -> str:
        w = torch.exp(self.log_weights)
        lines = []
        if w.ndim == 3 and w.numel() <= 64:
            lines.append(f"weights: shape={list(w.shape)}")
            h_l = w.shape[1]
            for i in range(h_l):
                row_strs = []
                for u in range(w.shape[0]):
                    row_str = "[" + ", ".join(f"{x:.5f}" for x in w[u, i]) + "]"
                    row_strs.append(row_str)
                lines.append("  " + " | ".join(row_strs))
        elif w.numel() <= 64:
            lines.append("weights:")
            if w.ndim == 1:
                w_str = "[" + ", ".join(f"{x:.5f}" for x in w) + "]"
                lines.append(f"  {w_str}")
            elif w.ndim == 2:
                for row in w:
                    w_str = "[" + ", ".join(f"{x:.5f}" for x in row) + "]"
                    lines.append(f"  {w_str}")
        else:
            lines.append(f"weights: shape={list(w.shape)}")
        return "\n".join(lines)


class SparseWeights(Weights):
    """Masked weights that enforce structural zeros."""

    def __init__(self, log_weights: torch.Tensor, mask: torch.Tensor = None):
        self.log_weights = log_weights
        if mask is None:
            mask = (log_weights.data > -20.0).float()
        self.mask = mask

    def to(self, device: torch.device) -> "Weights":
        if isinstance(self.log_weights, torch.Tensor):
            self.log_weights = self.log_weights.to(device)
            if not self.log_weights.is_leaf:
                self.log_weights = self.log_weights.detach().requires_grad_(
                    self.log_weights.requires_grad
                )
        if isinstance(self.mask, torch.Tensor):
            self.mask = self.mask.to(device)
        return self

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.log_weights.shape)

    def uniformize(
        self,
        other_child_axis: Optional[str] = None,
    ) -> "Weights":
        if other_child_axis is None:
            new_weights = torch.zeros_like(self.log_weights)
            new_weights = torch.where(self.mask == 0, float("-inf"), new_weights)
            return SparseWeights(new_weights, self.mask)

        assert other_child_axis is not None and other_child_axis in ["left", "right"]

        total = safe_logsumexp(
            self.log_weights,
            dim=(4, 5) if other_child_axis == "right" else (2, 3),
            keepdim=True,
        )
        new_weights = self.log_weights - total
        new_weights = torch.where(self.mask == 0, float("-inf"), new_weights)
        return SparseWeights(new_weights, self.mask)

    def forward(
        self, left_out: torch.Tensor, right_out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        w = self.log_weights
        G, U, G_L, L, G_R, R = w.shape
        B = left_out.shape[0]

        if right_out is None:
            child = left_out.view(B, G_L, L, 1, 1)
        else:
            child = left_out.view(B, G_L, L, 1, 1) + right_out.view(B, 1, 1, G_R, R)

        child = child.unsqueeze(1).unsqueeze(2)  # shape: [B, 1, 1, G_L, L, G_R, R]
        w = w.unsqueeze(0)
        mask = self.mask.unsqueeze(0)

        val = torch.where(mask > 0, child, torch.tensor(float("-inf"), device=child.device))

        out = safe_logsumexp(val + w, dim=(3, 4, 5, 6))
        return out.view(B, G, U)

    def detach(self) -> "SparseWeights":
        return SparseWeights(
            self.log_weights.detach(), self.mask.detach() if self.mask is not None else None
        )

    def requires_grad_(self, requires_grad: bool) -> "SparseWeights":
        self.log_weights.requires_grad_(requires_grad)
        return self

    def update_params(self, step_size: float = 1.0, smoothing: float = 1e-4) -> None:
        if self.log_weights is None or self.log_weights.grad is None:
            return

        counts = self.log_weights.grad.nan_to_num(0.0).clamp(min=0.0)
        counts = counts * self.mask

        sum_dims = (2, 3, 4, 5)
        mask_sum = self.mask.sum(dim=sum_dims, keepdim=True)
        denom = counts.sum(dim=sum_dims, keepdim=True) + smoothing * mask_sum
        probs = (counts + smoothing * self.mask) / denom
        probs = probs.nan_to_num(0.0)

        old_probs = torch.exp(self.log_weights.data)
        new_probs = (1.0 - step_size) * old_probs + step_size * probs

        new_probs = torch.where(
            self.mask > 0,
            new_probs,
            torch.tensor(1e-15, dtype=new_probs.dtype, device=new_probs.device),
        )

        self.log_weights.data.copy_(torch.log(new_probs.clamp(min=1e-20)))
        self.log_weights.grad.zero_()

    def __str__(self) -> str:
        w = torch.exp(self.log_weights) * self.mask
        lines = []
        if w.ndim == 3 and w.numel() <= 64:
            lines.append(f"weights: shape={list(w.shape)}")
            h_l = w.shape[1]
            for i in range(h_l):
                row_strs = []
                for u in range(w.shape[0]):
                    row_str = "[" + ", ".join(f"{x:.5f}" for x in w[u, i]) + "]"
                    row_strs.append(row_str)
                lines.append("  " + " | ".join(row_strs))
        elif w.numel() <= 64:
            lines.append("weights:")
            if w.ndim == 1:
                w_str = "[" + ", ".join(f"{x:.5f}" for x in w) + "]"
                lines.append(f"  {w_str}")
            elif w.ndim == 2:
                for row in w:
                    w_str = "[" + ", ".join(f"{x:.5f}" for x in row) + "]"
                    lines.append(f"  {w_str}")
        else:
            lines.append(f"weights: shape={list(w.shape)}")
        return "\n".join(lines)


class ProductWeights(Weights):
    """Lazily computes the structural product of two distributions with mixed expansion."""

    def __init__(
        self,
        w1: Weights,
        w2: Weights,
        expand_U: bool,
        expand_L: bool,
        expand_R: bool,
    ):
        self.w1: Weights = w1
        self.w2: Weights = w2

        self.expand_U = expand_U
        self.expand_L = expand_L
        self.expand_R = expand_R

    def to(self, device: torch.device) -> "Weights":
        if hasattr(self.w1, "to"):
            self.w1.to(device)
        if hasattr(self.w2, "to"):
            self.w2.to(device)
        return self

    @property
    def shape(self):
        return self.log_weights.shape

    @property
    def log_weights(self) -> torch.Tensor:
        lw1: torch.Tensor = self.w1.log_weights
        lw2: torch.Tensor = self.w2.log_weights

        G1, U1, G_L1, L1, G_R1, R1 = lw1.shape
        G2, U2, G_L2, L2, G_R2, R2 = lw2.shape

        new_G = G1 * G2
        new_G_L = G_L1 * G_L2
        new_G_R = G_R1 * G_R2
        new_U = U1 * U2 if self.expand_U else U1
        new_L = (L1 * L2) if self.expand_L else L1
        new_R = (R1 * R2) if self.expand_R else R1

        if self.expand_L and self.expand_R:  # Kronecker
            # [G, U, G_L, L, G_R, R] -> [G1, G2, U1, U2, G_L1, G_L2, L1, L2, G_R1, G_R2, R1, R2]
            w1_shape = (G1, 1, U1, 1, G_L1, 1, L1, 1, G_R1, 1, R1, 1)
            w2_shape = (1, G2, 1, U2, 1, G_L2, 1, L2, 1, G_R2, 1, R2)

        elif not self.expand_L and self.expand_R:
            # [G, U, G_L, L, G_R, R] -> [G1, G2, U1, U2, G_L1, G_L2, L, G_R1, G_R2, R1, R2]
            w1_shape = (G1, 1, U1, 1, G_L1, 1, L1, G_R1, 1, R1, 1)
            w2_shape = (1, G2, 1, U2, 1, G_L2, L2, 1, G_R2, 1, R2)

        elif self.expand_L and not self.expand_R:
            # [G, U, G_L, L, G_R, R] -> [G1, G2, U1, U2, G_L1, G_L2, L1, L2, G_R1, G_R2, R]
            w1_shape = (G1, 1, U1, 1, G_L1, 1, L1, 1, G_R1, 1, R1)
            w2_shape = (1, G2, 1, U2, 1, G_L2, 1, L2, 1, G_R2, R2)

        else:  # Hadamard
            # [G, U, G_L, L, G_R, R] -> [G1, G2, U1, U2, G_L1, G_L2, L, G_R1, G_R2, R]
            w1_shape = (G1, 1, U1, 1, G_L1, 1, L1, G_R1, 1, R1)
            w2_shape = (1, G2, 1, U2, 1, G_L2, L2, 1, G_R2, R2)

        w_out = lw1.view(w1_shape) + lw2.view(w2_shape)

        if not self.expand_U:
            if U1 == 1 and U2 == 1:
                w_out = w_out.squeeze(2)
                new_U = 1
            elif U1 == 1:
                w_out = w_out.squeeze(2)
                new_U = U2
            elif U2 == 1:
                w_out = w_out.squeeze(3)
                new_U = U1
            else:
                w_out = w_out.diagonal(dim1=2, dim2=3).movedim(-1, 2)
                new_U = min(U1, U2)
        else:
            new_U = U1 * U2

        w_out = w_out.reshape(new_G, new_U, new_G_L, new_L, new_G_R, new_R)
        return w_out

    def uniformize(self) -> "Weights":
        return ProductWeights(
            self.w1.uniformize(),
            self.w2.uniformize(),
            expand_L=self.expand_L,
            expand_R=self.expand_R,
            expand_U=self.expand_U,
        )

    def forward(
        self, left_in: torch.Tensor, right_in: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Instead of realizing the full Kronecker product of weights, we contract sequentially.
        lw1: torch.Tensor = getattr(self.w1, "log_weights", self.w1)
        lw2: torch.Tensor = getattr(self.w2, "log_weights", self.w2)

        G1, U1, G_L1, L1, G_R1, R1 = lw1.shape
        G2, U2, G_L2, L2, G_R2, R2 = lw2.shape
        B = left_in.shape[0]

        if self.expand_L:
            left_r = left_in.view(B, G_L1, G_L2, L1, L2)
        else:
            left_r = left_in.view(B, G_L1, G_L2, L1)

        if right_in is not None:
            if self.expand_R:
                right_r = right_in.view(B, G_R1, G_R2, R1, R2)
            else:
                right_r = right_in.view(B, G_R1, G_R2, R1)
        else:
            if self.expand_R:
                right_r = torch.zeros(B, G_R1, G_R2, R1, R2, device=left_in.device)
            else:
                right_r = torch.zeros(B, G_R1, G_R2, R1, device=left_in.device)

        if self.expand_L and self.expand_R:
            left_p = left_r.permute(0, 1, 3, 2, 4).reshape(B, G_L1, 1, L1, 1, G_L2, L2, 1, 1)
            right_p = right_r.permute(0, 1, 3, 2, 4).reshape(B, 1, G_R1, 1, R1, 1, 1, G_R2, R2)
            child2 = left_p + right_p
            temp = child2.unsqueeze(5).unsqueeze(6) + lw2.view(
                1, 1, 1, 1, 1, G2, U2, G_L2, L2, G_R2, R2
            )
            temp = safe_logsumexp(temp, dim=(7, 8, 9, 10))
            temp = temp.permute(0, 5, 6, 1, 3, 2, 4)
            out = temp.unsqueeze(3).unsqueeze(4) + lw1.view(1, 1, 1, G1, U1, G_L1, L1, G_R1, R1)
            out = safe_logsumexp(out, dim=(5, 6, 7, 8))
            out = out.permute(0, 3, 1, 4, 2)

        elif self.expand_L and not self.expand_R:
            left_p = left_r.permute(0, 1, 3, 2, 4).reshape(B, G_L1, 1, L1, G_L2, L2, 1, 1)
            right_p = right_r.reshape(B, 1, G_R1, 1, 1, 1, G_R2, R1)
            child2 = left_p + right_p
            temp = child2.unsqueeze(4).unsqueeze(5) + lw2.view(
                1, 1, 1, 1, G2, U2, G_L2, L2, G_R2, R1
            )
            temp = safe_logsumexp(temp, dim=(6, 7, 8))
            temp = temp.permute(0, 4, 5, 1, 3, 2, 6)
            out = temp.unsqueeze(3).unsqueeze(4) + lw1.view(1, 1, 1, G1, U1, G_L1, L1, G_R1, R1)
            out = safe_logsumexp(out, dim=(5, 6, 7, 8))
            out = out.permute(0, 3, 1, 4, 2)

        elif not self.expand_L and self.expand_R:
            left_p = left_r.reshape(B, G_L1, 1, 1, G_L2, 1, 1, L1)
            right_p = right_r.permute(0, 1, 3, 2, 4).reshape(B, 1, G_R1, R1, 1, G_R2, R2, 1)
            child2 = left_p + right_p
            temp = child2.unsqueeze(4).unsqueeze(5) + lw2.permute(0, 1, 2, 4, 5, 3).reshape(
                1, 1, 1, 1, G2, U2, G_L2, G_R2, R2, L1
            )
            temp = safe_logsumexp(temp, dim=(6, 7, 8))
            temp = temp.permute(0, 4, 5, 1, 6, 2, 3)
            out = temp.unsqueeze(3).unsqueeze(4) + lw1.view(1, 1, 1, G1, U1, G_L1, L1, G_R1, R1)
            out = safe_logsumexp(out, dim=(5, 6, 7, 8))
            out = out.permute(0, 3, 1, 4, 2)

        else:
            left_p = left_r.reshape(B, G_L1, 1, G_L2, 1, L1, 1)
            right_p = right_r.reshape(B, 1, G_R1, 1, G_R2, 1, R1)
            child2 = left_p + right_p
            temp = child2.unsqueeze(3).unsqueeze(4) + lw2.permute(0, 1, 2, 4, 3, 5).reshape(
                1, 1, 1, G2, U2, G_L2, G_R2, L1, R1
            )
            temp = safe_logsumexp(temp, dim=(5, 6))
            temp = temp.permute(0, 3, 4, 1, 5, 2, 6)
            out = temp.unsqueeze(3).unsqueeze(4) + lw1.view(1, 1, 1, G1, U1, G_L1, L1, G_R1, R1)
            out = safe_logsumexp(out, dim=(5, 6, 7, 8))
            out = out.permute(0, 3, 1, 4, 2)

        if self.expand_U:
            out = out.reshape(B, G1 * G2, U1 * U2)
        elif U1 == 1 and U2 == 1:
            out = out.reshape(B, G1 * G2, 1)
        elif U1 == 1:
            out = out.reshape(B, G1 * G2, U2)
        elif U2 == 1:
            out = out.reshape(B, G1 * G2, U1)
        else:
            out = out.diagonal(dim1=3, dim2=4)
            out = out.reshape(B, G1 * G2, min(U1, U2))

        return out

    def detach(self) -> "ProductWeights":
        return ProductWeights(
            self.w1.detach(),
            self.w2.detach(),
            expand_L=self.expand_L,
            expand_R=self.expand_R,
            expand_U=self.expand_U,
        )

    def requires_grad_(self, requires_grad: bool) -> "ProductWeights":
        self.w1.requires_grad_(requires_grad)
        self.w2.requires_grad_(requires_grad)
        return self

    def update_params(self, step_size: float = 1.0, smoothing: float = 1e-4) -> None:
        self.w1.update_params(step_size, smoothing)
        self.w2.update_params(step_size, smoothing)

    def __str__(self) -> str:
        return f"ProductWeights(expand_U={self.expand_U}, L={self.expand_L}, R={self.expand_R}\n{self.w1.__str__()},\n{self.w2.__str__()})"
