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
            dims = tuple(d if d >= 0 else d + input.ndim for d in dims)
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


def _blocked_broadcast_lse(
    prod: torch.Tensor, w: torch.Tensor, dim, max_elems: Optional[int] = None
) -> torch.Tensor:
    """``safe_logsumexp(prod + w, dim=dim)`` without materializing the full broadcast.

    ``prod`` and ``w`` must have the same rank with aligned axes (typically both
    already-unsqueezed views of a layer contraction), and axis 0 must be a real,
    independent batch dimension of ``prod`` that is *not* part of ``dim``. The
    materialized broadcast ``prod + w`` has the elementwise-max shape of the two
    operands; the batch axis is processed in chunks so the temporary stays under
    the element budget (default 256 MB in fp32 on CPU, 1 GB on CUDA), and the
    per-row results are concatenated. Chunking the batch axis never splits the
    reduction, so each row's result is bit-identical to the unchunked op (no
    ``logaddexp`` accumulation chain). When a single chunk covers the batch,
    the original op sequence runs unchanged. Only transient memory is bounded:
    under autograd each chunk input is still saved for backward by
    ``safe_logsumexp``.
    """
    dims = tuple(
        d if d >= 0 else d + prod.dim() for d in (dim if isinstance(dim, (tuple, list)) else (dim,))
    )
    if 0 in dims:
        raise ValueError("batch axis 0 must not be part of the reduction dim")
    bshape = [max(p, q) for p, q in zip(prod.shape, w.shape)]
    budget = max_elems if max_elems is not None else (1 << 28 if prod.is_cuda else 1 << 26)
    per_slice = max(1, _prod(bshape[1:]))
    rows_per_chunk = max(1, budget // per_slice)
    if rows_per_chunk >= bshape[0]:
        return safe_logsumexp(prod + w, dim=dim)

    parts = [
        safe_logsumexp(
            prod.narrow(0, start, min(start + rows_per_chunk, bshape[0]) - start) + w, dim=dim
        )
        for start in range(0, bshape[0], rows_per_chunk)
    ]
    return torch.cat(parts, dim=0)


def _prod(sizes) -> int:
    out = 1
    for s in sizes:
        out *= s
    return out


def _batch_chunk_rows(batch: int, per_row_elems: int, is_cuda: bool) -> int:
    """Largest chunk of batch rows whose [rows, per_row_elems] fp32 transient
    stays within the element budget (1 GiB on CUDA, 256 MB on CPU).

    ``_blocked_broadcast_lse`` bounds the logsumexp materialization, but the
    child outer-product that feeds it (``left + right``) is materialized by the
    caller first — those materializations must be chunked here.
    """
    if per_row_elems <= 0:
        return batch
    budget = 1 << 28 if is_cuda else 1 << 26
    return max(1, min(batch, budget // per_row_elems))


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

    def num_parameters(self) -> int:
        """Number of trainable scalar parameters.

        The default materializes ``log_weights`` (for lazy subclasses such as
        ``ProductWeights``/``MixingCondWeights`` this realizes the logical
        tensor) and counts its elements.  ``SparseWeights`` overrides this to
        count only mask-nonzero entries.
        """
        return int(self.log_weights.numel())

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
        raise NotImplementedError("Dense weights are never uniformized.")

    def forward(
        self, left_out: torch.Tensor, right_out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        w = self.log_weights
        G, U, G_L, L, G_R, R = w.shape
        B = left_out.shape[0]

        # Chunk the batch so the [B, G_L, L, G_R, R] product materialization
        # below stays within the memory budget (the follow-up logsumexp is
        # already chunked inside _blocked_broadcast_lse).
        rows = _batch_chunk_rows(B, G_L * L * G_R * R, left_out.is_cuda)
        if rows < B:
            parts = [
                self.forward(
                    left_out[i : i + rows],
                    right_out[i : i + rows] if right_out is not None else None,
                )
                for i in range(0, B, rows)
            ]
            return torch.cat(parts, dim=0)

        if right_out is None:
            # right_out is absent, effectively 0 in log space since we don't multiply it
            prod = left_out.view(B, G_L, L, 1, 1)
        else:
            prod = left_out.view(B, G_L, L, 1, 1) + right_out.view(B, 1, 1, G_R, R)

        out = _blocked_broadcast_lse(
            prod.unsqueeze(1).unsqueeze(2), w.unsqueeze(0), dim=(3, 4, 5, 6)
        )
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
    """Masked weights that enforce structural zeros.

    The canonical representation remains the dense ``log_weights`` tensor plus a
    boolean ``mask`` (so ``ProductWeights`` and the query compiler are
    unaffected).  Additionally, when the mask is genuinely sparse, we precompute
    coordinate indices of the non-zero entries so ``forward`` can gather only
    those child combinations instead of materializing the full dense product.
    """

    # Only use the gather-based forward when the mask density is below this.
    _SPARSE_FORWARD_MAX_DENSITY = 0.5

    def __init__(self, log_weights: torch.Tensor, mask: torch.Tensor = None):
        self.log_weights = log_weights
        if mask is None:
            mask = (log_weights.data > -20.0).float()
        self.mask = mask
        self._build_sparse_index()

    def _build_sparse_index(self) -> None:
        """Precompute coordinate indices of non-zero mask entries."""
        self._sparse_indices = None
        self._sparse_num = None
        self._sparse_widx = None

        mask_bool = self.mask > 0
        G, U, G_L, L, G_R, R = mask_bool.shape
        total = mask_bool.numel()
        nnz = int(mask_bool.sum().item())
        if nnz == 0 or nnz / total > self._SPARSE_FORWARD_MAX_DENSITY:
            return

        device = self.mask.device
        counts = mask_bool.sum(dim=(2, 3, 4, 5)).to(torch.long)  # [G, U]
        K = int(counts.max().item())

        indices = torch.zeros(G, U, K, 4, dtype=torch.long, device=device)
        counter = torch.zeros(G, U, dtype=torch.long, device=device)
        nz = mask_bool.nonzero(as_tuple=False)
        # Sort by (g, u) so entries for the same parent unit are contiguous.
        key = nz[:, 0] * U + nz[:, 1]
        nz = nz[torch.argsort(key)]
        for row in nz:
            g, u, gl, li, gr, r = row.tolist()
            k = int(counter[g, u].item())
            indices[g, u, k, 0] = gl
            indices[g, u, k, 1] = li
            indices[g, u, k, 2] = gr
            indices[g, u, k, 3] = r
            counter[g, u] += 1

        # Flat index into the [G_L, L, G_R, R] weight dimensions.
        widx = (indices[..., 0] * L + indices[..., 1]) * G_R + indices[..., 2]
        widx = widx * R + indices[..., 3]

        self._sparse_indices = indices
        self._sparse_num = counts
        self._sparse_widx = widx

    def to(self, device: torch.device) -> "Weights":
        if isinstance(self.log_weights, torch.Tensor):
            self.log_weights = self.log_weights.to(device)
            if not self.log_weights.is_leaf:
                self.log_weights = self.log_weights.detach().requires_grad_(
                    self.log_weights.requires_grad
                )
        if isinstance(self.mask, torch.Tensor):
            self.mask = self.mask.to(device)
        for attr in ("_sparse_indices", "_sparse_num", "_sparse_widx"):
            t = getattr(self, attr, None)
            if isinstance(t, torch.Tensor):
                setattr(self, attr, t.to(device))
        return self

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.log_weights.shape)

    def num_parameters(self) -> int:
        """Only mask-nonzero entries are trainable structural parameters."""
        return int(self.mask.sum().item())

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

        # Sparse gather path: only evaluate the non-masked child combinations.
        if self._sparse_indices is not None and right_out is not None:
            idx = self._sparse_indices  # [G, U, K, 4]
            K = idx.shape[2]
            gl, li, gr, r = idx[..., 0], idx[..., 1], idx[..., 2], idx[..., 3]
            left_sel = left_out[:, gl, li]  # [B, G, U, K]
            right_sel = right_out[:, gr, r]  # [B, G, U, K]
            w_sel = torch.gather(w.view(G, U, -1), 2, self._sparse_widx)  # [G, U, K]
            vals = left_sel + right_sel + w_sel.unsqueeze(0)  # [B, G, U, K]
            pad = torch.arange(K, device=w.device).view(1, 1, 1, K) >= self._sparse_num.view(
                1, G, U, 1
            )
            vals = vals.masked_fill(pad, float("-inf"))
            return safe_logsumexp(vals, dim=-1)  # [B, G, U]

        if right_out is None:
            child = left_out.view(B, G_L, L, 1, 1)
        else:
            child = left_out.view(B, G_L, L, 1, 1) + right_out.view(B, 1, 1, G_R, R)

        # Chunk the batch: the masked broadcast below materializes
        # [B, G, U, G_L, L, G_R, R], which is unbounded in B.
        rows = _batch_chunk_rows(B, G * U * G_L * L * G_R * R, left_out.is_cuda)
        if rows < B:
            parts = [
                self.forward(
                    left_out[i : i + rows],
                    right_out[i : i + rows] if right_out is not None else None,
                )
                for i in range(0, B, rows)
            ]
            return torch.cat(parts, dim=0)

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
            new_probs.clamp(min=1e-20),
            torch.tensor(1e-20, dtype=new_probs.dtype, device=new_probs.device),
        )

        self.log_weights.data.copy_(torch.log(new_probs))
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

    def num_parameters(self) -> int:
        """Degrees of freedom of the factors, not the materialized Kronecker product.

        The logical ``log_weights`` of a product is a derived broadcast of
        ``w1`` and ``w2`` — counting its elements would report the size of the
        expansion, not trainable parameters.
        """
        return self.w1.num_parameters() + self.w2.num_parameters()

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

        # Chunk the batch so the child outer-product materialization below
        # (child2 = left_p + right_p) stays within the memory budget.
        if self.expand_L and self.expand_R:
            per_row = G_L1 * G_R1 * L1 * R1 * G_L2 * L2 * G_R2 * R2
        elif self.expand_L:
            per_row = G_L1 * G_R1 * L1 * G_L2 * L2 * G_R2 * R1
        elif self.expand_R:
            per_row = G_L1 * G_R1 * R1 * G_L2 * L1 * G_R2 * R2
        else:
            per_row = G_L1 * G_R1 * G_L2 * G_R2 * L1 * R1
        rows = _batch_chunk_rows(B, per_row, left_in.is_cuda)
        if rows < B:
            parts = [
                self.forward(
                    left_in[i : i + rows],
                    right_in[i : i + rows] if right_in is not None else None,
                )
                for i in range(0, B, rows)
            ]
            return torch.cat(parts, dim=0)

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
            temp = _blocked_broadcast_lse(
                child2.unsqueeze(5).unsqueeze(6),
                lw2.view(1, 1, 1, 1, 1, G2, U2, G_L2, L2, G_R2, R2),
                dim=(7, 8, 9, 10),
            )
            temp = temp.permute(0, 5, 6, 1, 3, 2, 4)
            out = _blocked_broadcast_lse(
                temp.unsqueeze(3).unsqueeze(4),
                lw1.view(1, 1, 1, G1, U1, G_L1, L1, G_R1, R1),
                dim=(5, 6, 7, 8),
            )
            out = out.permute(0, 3, 1, 4, 2)

        elif self.expand_L and not self.expand_R:
            left_p = left_r.permute(0, 1, 3, 2, 4).reshape(B, G_L1, 1, L1, G_L2, L2, 1, 1)
            right_p = right_r.reshape(B, 1, G_R1, 1, 1, 1, G_R2, R1)
            child2 = left_p + right_p
            temp = _blocked_broadcast_lse(
                child2.unsqueeze(4).unsqueeze(5),
                lw2.view(1, 1, 1, 1, G2, U2, G_L2, L2, G_R2, R1),
                dim=(6, 7, 8),
            )
            temp = temp.permute(0, 4, 5, 1, 3, 2, 6)
            out = _blocked_broadcast_lse(
                temp.unsqueeze(3).unsqueeze(4),
                lw1.view(1, 1, 1, G1, U1, G_L1, L1, G_R1, R1),
                dim=(5, 6, 7, 8),
            )
            out = out.permute(0, 3, 1, 4, 2)

        elif not self.expand_L and self.expand_R:
            left_p = left_r.reshape(B, G_L1, 1, 1, G_L2, 1, 1, L1)
            right_p = right_r.permute(0, 1, 3, 2, 4).reshape(B, 1, G_R1, R1, 1, G_R2, R2, 1)
            child2 = left_p + right_p
            temp = _blocked_broadcast_lse(
                child2.unsqueeze(4).unsqueeze(5),
                lw2.permute(0, 1, 2, 4, 5, 3).reshape(1, 1, 1, 1, G2, U2, G_L2, G_R2, R2, L1),
                dim=(6, 7, 8),
            )
            temp = temp.permute(0, 4, 5, 1, 6, 2, 3)
            out = _blocked_broadcast_lse(
                temp.unsqueeze(3).unsqueeze(4),
                lw1.view(1, 1, 1, G1, U1, G_L1, L1, G_R1, R1),
                dim=(5, 6, 7, 8),
            )
            out = out.permute(0, 3, 1, 4, 2)

        else:
            left_p = left_r.reshape(B, G_L1, 1, G_L2, 1, L1, 1)
            right_p = right_r.reshape(B, 1, G_R1, 1, G_R2, 1, R1)
            child2 = left_p + right_p
            temp = _blocked_broadcast_lse(
                child2.unsqueeze(3).unsqueeze(4),
                lw2.permute(0, 1, 2, 4, 3, 5).reshape(1, 1, 1, G2, U2, G_L2, G_R2, L1, R1),
                dim=(5, 6),
            )
            temp = temp.permute(0, 3, 4, 1, 5, 2, 6)
            out = _blocked_broadcast_lse(
                temp.unsqueeze(3).unsqueeze(4),
                lw1.view(1, 1, 1, G1, U1, G_L1, L1, G_R1, R1),
                dim=(5, 6, 7, 8),
            )
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
        raise NotImplementedError(
            "ProductWeights does not support update_params; update the base weights instead."
        )

    def __str__(self) -> str:
        return f"ProductWeights(expand_U={self.expand_U}, L={self.expand_L}, R={self.expand_R}\n{self.w1.__str__()},\n{self.w2.__str__()})"


class MixingCondWeights(Weights):
    """Lazy view of base weights normalized over the non-MD child axis.

    Used by the conditional-circuit compiler (Wang's algorithm) for
    left/right-mixing layers that are MD with respect to the conditioning set:
    the conditional weights are ``base - logsumexp(base, over the mixing
    axis)``.  Unlike ``uniformize(other_child_axis=...)`` — which bakes the
    current values into a new tensor at compile time — this wrapper recomputes
    the normalization from the live base tensor at every forward pass, so a
    query circuit compiled before training keeps tracking EM updates of the
    base circuit.
    """

    def __init__(self, base: Weights, other_child_axis: str):
        assert other_child_axis in ("left", "right")
        self.base = base
        self.other_child_axis = other_child_axis

    def _normalized_log_weights(self) -> torch.Tensor:
        lw = self.base.log_weights
        total = safe_logsumexp(
            lw,
            dim=(4, 5) if self.other_child_axis == "right" else (2, 3),
            keepdim=True,
        )
        lw = lw - total
        mask = getattr(self.base, "mask", None)
        if mask is not None:
            lw = torch.where(mask == 0, float("-inf"), lw)
        return lw

    @property
    def log_weights(self) -> torch.Tensor:
        """Normalized log-weights, recomputed from the live base tensor."""
        return self._normalized_log_weights()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.base.shape

    def forward(
        self, left_out: torch.Tensor, right_out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return DenseWeights(self._normalized_log_weights()).forward(left_out, right_out)

    def to(self, device: torch.device) -> "Weights":
        # The base weights object is shared with the base circuit and is moved
        # by its own ``to``; nothing device-resident lives here.
        return self

    def detach(self) -> "Weights":
        return self.base.detach()

    def requires_grad_(self, requires_grad: bool) -> "Weights":
        self.base.requires_grad_(requires_grad)
        return self

    def update_params(self, step_size: float = 1.0, smoothing: float = 1e-4) -> None:
        raise NotImplementedError(
            "MixingCondWeights does not support update_params; update the base weights instead."
        )

    def num_parameters(self) -> int:
        """The normalized view adds no parameters; delegate to the base weights."""
        return self.base.num_parameters()

    def uniformize(self) -> "Weights":
        return MixingCondWeights(self.base.uniformize(), self.other_child_axis)

    def __str__(self) -> str:
        return f"MixingCondWeights(axis={self.other_child_axis}, base={self.base})"
