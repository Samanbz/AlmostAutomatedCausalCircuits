"""Unit tests for the lazy weights classes."""

import torch

from src.symbolic.arithmetic.weights import (
    ProductWeights,
    SparseWeights,
)


def test_product_weights_count_nnz_of_sparse_factors():
    """nnz of a ProductWeights of two SparseWeights follows the OR-combined
    zero-pattern: nnz(A x B) = nnz(A) * nnz(B) for the full expansion."""
    torch.manual_seed(0)
    shape = (2, 2, 2, 3, 3, 2)
    m1 = (torch.rand(*shape) > 0.5).float()
    m2 = (torch.rand(*shape) > 0.5).float()
    w1 = SparseWeights(torch.randn(*shape) * 0.5, mask=m1)
    w2 = SparseWeights(torch.randn(*shape) * 0.5, mask=m2)

    pw = ProductWeights(w1, w2, expand_U=True, expand_L=True, expand_R=True)
    assert pw.num_parameters() == int(m1.sum()) * int(m2.sum())
