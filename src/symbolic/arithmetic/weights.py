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
    stays within the element budget (1 GiB on CUDA, 4 GiB on CPU).

    ``_blocked_broadcast_lse`` bounds the logsumexp materialization, but the
    child outer-product that feeds it (``left + right``) is materialized by the
    caller first — those materializations must be chunked here.
    """
    if per_row_elems <= 0:
        return batch
    budget = 1 << 32 if is_cuda else 1 << 30
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
        """Number of non-zero weights in the logical weight tensor.

        This is a circuit-size measure, not a degrees-of-freedom count: for a
        derived broadcast (``ProductWeights``) every replicated non-zero entry
        counts. The default counts every element of ``log_weights`` (dense
        tensors have no structural zeros). ``SparseWeights`` overrides this to
        count only mask-nonzero entries.
        """
        return int(self.log_weights.numel())

    def _zero_pattern(self) -> Optional[torch.Tensor]:
        """Bool tensor matching ``self.shape`` with True = structurally zero
        weight, or None if the logical tensor is fully dense (no zeros).

        Used by ``ProductWeights`` to count the non-zero entries of a derived
        broadcast without materializing it.
        """
        return None

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
        """Only mask-nonzero entries are counted."""
        return int(self.mask.sum().item())

    def _zero_pattern(self) -> Optional[torch.Tensor]:
        """Structural zeros are exactly the mask-off entries."""
        return self.mask == 0

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

        # Single materialization: add the weights, then mask in place. (The
        # naive ``where(mask, child, -inf) + w`` would materialize the full
        # [B, G, U, G_L, L, G_R, R] broadcast twice.)
        val = child + w
        val.masked_fill_(mask == 0, float("-inf"))

        out = safe_logsumexp(val, dim=(3, 4, 5, 6))
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
        expand_GL: Optional[bool] = None,
        expand_GR: Optional[bool] = None,
        expand_Lu: Optional[bool] = None,
        expand_Ru: Optional[bool] = None,
    ):
        self.w1: Weights = w1
        self.w2: Weights = w2

        self.expand_U = expand_U
        self.expand_L = expand_L
        self.expand_R = expand_R
        # Group and unit expansion of the child axes are independent (a child
        # may enumerate group pairs while sharing units); default both to the
        # legacy per-side flag.
        self.expand_GL = expand_L if expand_GL is None else expand_GL
        self.expand_GR = expand_R if expand_GR is None else expand_GR
        self.expand_Lu = expand_L if expand_Lu is None else expand_Lu
        self.expand_Ru = expand_R if expand_Ru is None else expand_Ru

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
        return self._combine(self.w1.log_weights, self.w2.log_weights, torch.Tensor.__add__)

    def _combine(self, a: torch.Tensor, b: torch.Tensor, op) -> torch.Tensor:
        """Broadcast-combine two aligned weight tensors per the expansion flags.

        ``op`` is addition for log-weights and boolean OR for zero-patterns —
        the layout is identical for both (a product entry is zero iff any
        factor entry is zero there).
        """
        G1, U1, G_L1, L1, G_R1, R1 = a.shape
        G2, U2, G_L2, L2, G_R2, R2 = b.shape

        new_G = G1 * G2
        # Shared (non-expanded) child axes are tied between the operands, so
        # they occupy a single broadcast slot and do not multiply; a size-1
        # side broadcasts (max). Expanded axes get two adjacent slots so the
        # pair index is side-1 major in the merged axis.
        new_G_L = (G_L1 * G_L2) if self.expand_GL else max(G_L1, G_L2)
        new_G_R = (G_R1 * G_R2) if self.expand_GR else max(G_R1, G_R2)
        new_U = (U1 * U2) if self.expand_U else max(U1, U2)
        new_L = (L1 * L2) if self.expand_Lu else max(L1, L2)
        new_R = (R1 * R2) if self.expand_Ru else max(R1, R2)

        # Layout: [G1, G2, U1, U2, G_L1, G_L2, L1, L2, G_R1, G_R2, R1, R2].
        w1_shape = (G1, 1, U1, 1, G_L1, 1, L1, 1, G_R1, 1, R1, 1)
        w2_shape = (
            1,
            G2,
            U2 if not self.expand_U else 1,
            U2 if self.expand_U else 1,
            G_L2 if not self.expand_GL else 1,
            G_L2 if self.expand_GL else 1,
            L2 if not self.expand_Lu else 1,
            L2 if self.expand_Lu else 1,
            G_R2 if not self.expand_GR else 1,
            G_R2 if self.expand_GR else 1,
            R2 if not self.expand_Ru else 1,
            R2 if self.expand_Ru else 1,
        )

        w_out = op(a.view(w1_shape), b.view(w2_shape))

        # Shared U already coincides in slot 2 (broadcast); expanded U merges
        # slots 2,3 as u = u1 * U2 + u2. Reshape handles both.
        return w_out.reshape(new_G, new_U, new_G_L, new_L, new_G_R, new_R)

    def _zero_pattern(self) -> Optional[torch.Tensor]:
        """Bool tensor of the logical shape with True = structurally zero weight.

        A product entry is zero iff any factor entry is zero, so the factors'
        zero-patterns are combined with OR, following the same expansion
        layout. None means the logical tensor is fully dense (no zeros).
        """
        z1 = self.w1._zero_pattern() if hasattr(self.w1, "_zero_pattern") else None
        z2 = self.w2._zero_pattern() if hasattr(self.w2, "_zero_pattern") else None
        if z1 is None and z2 is None:
            return None
        if z1 is None:
            z1 = torch.zeros(self.w1.shape, dtype=torch.bool)
        if z2 is None:
            z2 = torch.zeros(self.w2.shape, dtype=torch.bool)
        return self._combine(z1, z2, torch.Tensor.__or__)

    def num_parameters(self) -> int:
        """Number of non-zero weights in the logical (materialized) weight tensor.

        The logical tensor of a product is a derived broadcast of the factors,
        so its non-zero entries are counted from the factors' zero-patterns
        without materializing the broadcast.
        """
        z = self._zero_pattern()
        if z is None:
            return int(_prod(self.shape))
        return int((~z).sum())

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
        # Unified contraction through the materialized combined weight: the
        # combined tensor encodes the expand/shared layout (see ``_combine``),
        # and the child outputs must match its child axes. Batch-chunked to
        # bound the [B, G, U, G_L, L, G_R, R] materialization.
        w = self._combine(
            getattr(self.w1, "log_weights", self.w1),
            getattr(self.w2, "log_weights", self.w2),
            torch.Tensor.__add__,
        )
        G, U, G_L, L, G_R, R = w.shape
        B = left_in.shape[0]

        rows = _batch_chunk_rows(B, G_L * L * G_R * R, left_in.is_cuda)
        if rows < B:
            parts = [
                self.forward(
                    left_in[i : i + rows],
                    right_in[i : i + rows] if right_in is not None else None,
                )
                for i in range(0, B, rows)
            ]
            return torch.cat(parts, dim=0)

        left_r = left_in.reshape(B, G_L, L)
        if right_in is not None:
            right_r = right_in.reshape(B, G_R, R)
        else:
            right_r = torch.zeros(B, G_R, R, device=left_in.device, dtype=left_in.dtype)
        child = left_r[:, None, None, :, :, None, None] + right_r[:, None, None, None, None, :, :]
        return (w.unsqueeze(0) + child).logsumexp(dim=(3, 4, 5, 6))

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
        return f"ProductWeights(expand_U={self.expand_U}, G_L={self.expand_G_L}, L={self.expand_L}, G_R={self.expand_G_R}, R={self.expand_R}\n{self.w1.__str__()},\n{self.w2.__str__()})"


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

    def _zero_pattern(self) -> Optional[torch.Tensor]:
        """The normalization preserves the base factor's zero set."""
        return self.base._zero_pattern()

    def uniformize(self) -> "Weights":
        return MixingCondWeights(self.base.uniformize(), self.other_child_axis)

    def __str__(self) -> str:
        return f"MixingCondWeights(axis={self.other_child_axis}, base={self.base})"
