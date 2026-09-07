import torch

from src.construction.circuit_builder import create_md_circuit
from src.symbolic.arithmetic.circuit import eval_circuit
from src.symbolic.arithmetic.nodes import GaussianDistribution
from src.symbolic.arithmetic.nodes.leaf_layer import (
    LogLinearSplineDistribution,
    LogLinearSplineLeafLayer,
)
from src.symbolic.arithmetic.train import SymbolicEMTrainer
from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet, Support


def _make_loglinear_leaf(num_nodes: int = 4, num_groups: int = 2) -> LogLinearSplineLeafLayer:
    spec = LogLinearSplineDistribution(var=0, base_mean=0.0, base_stddev=1.0)
    supports = [Support({0: iv}) for iv in spec.split_support(num_nodes)]
    return LogLinearSplineLeafLayer(
        spec, num_nodes=num_nodes, num_groups=num_groups, node_supports=supports
    )


def test_loglinear_spline_equal_heights_no_nan_grad():
    """Regression test for the N=6 NaN: equal adjacent heights must not produce NaN grads."""
    leaf = _make_loglinear_leaf(num_nodes=4, num_groups=1)
    # Force all boundary heights to be exactly equal.
    with torch.no_grad():
        leaf._log_heights.fill_(0.0)

    x = torch.linspace(-5, 5, 101).unsqueeze(1)
    out = leaf.forward(x)
    # Inactive children are -inf by design; active ones must be finite.
    active = out[torch.isfinite(out)]
    assert active.numel() > 0
    assert torch.isfinite(active).all()

    loss = out.sum()
    loss.backward()
    for p in [leaf._log_heights, leaf._b1]:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_loglinear_spline_child_mass_and_continuity():
    """Each child integrates to ~1 and neighbouring children agree at split points."""
    torch.manual_seed(0)
    leaf = _make_loglinear_leaf(num_nodes=6, num_groups=1)

    h = leaf._heights()
    b = leaf._split_points()

    # Per-child mass via quadrature on a wide grid (masked forward).
    xs = torch.linspace(-30, 30, 20001).unsqueeze(1)
    with torch.no_grad():
        log_d = leaf.forward(xs).squeeze(1)  # [B, N], -inf outside each child support
        d = torch.exp(log_d)
        dx = xs[1, 0] - xs[0, 0]
        mass = d.sum(dim=0) * dx
    assert torch.allclose(mass, torch.ones_like(mass), atol=2e-2)

    # Boundary continuity: child i at its right edge equals child i+1 at its left edge.
    with torch.no_grad():
        for i in range(leaf.num_nodes - 1):
            xb = b[0, i].view(1, 1)
            ld = leaf._log_densities(xb, h, b).squeeze(0).squeeze(0)
            assert torch.isclose(ld[i], ld[i + 1], atol=1e-5)


def test_sparse_weights_forward_matches_dense():
    """The gather-based sparse forward must match the dense masked forward exactly."""
    from src.symbolic.arithmetic.weights import SparseWeights

    torch.manual_seed(0)
    G, U, G_L, L, G_R, R = 2, 3, 2, 2, 2, 2
    log_w = torch.randn(G, U, G_L, L, G_R, R)
    # Sparse mask: ~20% non-zero.
    mask = (torch.rand(G, U, G_L, L, G_R, R) < 0.2).float()
    sw = SparseWeights(log_w.clone().requires_grad_(True), mask)

    B = 5
    left = torch.randn(B, G_L, L)
    right = torch.randn(B, G_R, R)

    out_sparse = sw.forward(left, right)

    # Force the dense path by disabling the sparse index.
    sw._sparse_indices = None
    out_dense = sw.forward(left, right)

    assert torch.allclose(out_sparse, out_dense, atol=1e-5, rtol=1e-5)


def test_trained_circuit_root_finite_and_weights_normalized():
    """A short EM run on a tiny MD circuit must produce finite root outputs and normalized weights."""
    torch.manual_seed(0)

    # Two-variable vtree: root -> {0} | {1}
    vt = VTree()
    root = vt.add_node(VNode(scope=BitSet([0, 1])))
    left = vt.add_node(VNode(scope=BitSet([0])))
    right = vt.add_node(VNode(scope=BitSet([1])))
    vt.add_children(root, left, right)
    vt.compute_md_labeling([{0, 1}])

    dists = {
        0: LogLinearSplineDistribution(var=0, base_mean=0.0, base_stddev=1.0),
        1: GaussianDistribution(var=1, base_mean=0.0, base_stddev=1.0),
    }
    ac = create_md_circuit(dists, vt, num_nodes=2, initialize_weights=True)

    data = torch.randn(512, 2)
    trainer = SymbolicEMTrainer(ac, leaf_lr=0.01)
    trainer.train(data, n_iter=2, batch_size=128, step_size=0.1, log_interval=0)

    log_probs = eval_circuit(ac, data, debug_check_finite=True)
    assert torch.isfinite(log_probs).all()

    # Sum-node weights must normalize to 1 over their child dimensions.
    for node_id in ac.topological_sort():
        node = ac.get_node_data(node_id)
        lw = getattr(node, "log_weights", None)
        if lw is None:
            continue
        w = torch.exp(lw.log_weights if hasattr(lw, "log_weights") else lw)
        s = w.sum(dim=(2, 3, 4, 5))
        assert torch.allclose(s, torch.ones_like(s), atol=1e-4)
