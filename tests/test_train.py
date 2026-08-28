"""Diagnostic tests for NaNs during SymbolicEMTrainer training.

These tests document the two root causes of NaNs in N=16 MD-circuit training
and verify the fixes:

1. PyTorch's native ``torch.logsumexp`` produces NaN gradients when all inputs
   to a slice are ``-inf``. We use ``safe_logsumexp`` in structural weight
   contractions.
2. ``GaussianLeafLayer`` previously stored unconstrained standard deviations.
   Adam can push a raw stddev to (or below) zero, making ``log(stddev)`` and
   density division produce NaN in the forward pass. We now store an
   unconstrained log-stddev parameter and expose strictly positive stddevs via
   ``exp``.
"""

import torch

from src.construction.circuit_builder import create_md_circuit
from src.symbolic.arithmetic.circuit import SymbolicArithmeticCircuit, eval_circuit
from src.symbolic.arithmetic.nodes import GaussianDistribution
from src.symbolic.arithmetic.nodes.leaf_layer import GaussianLeafLayer
from src.symbolic.arithmetic.train import SymbolicEMTrainer
from src.symbolic.arithmetic.weights import safe_logsumexp
from src.symbolic.scm import build_synthetic_continuous_scm
from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet


def _make_vtree_xz():
    """VTree over {Z1, Z2, X, Y} with {Z1, Z2, X} determinism."""
    vt = VTree()
    v0 = vt.add_node(VNode(BitSet([0]), md_set=BitSet([0])))
    v1 = vt.add_node(VNode(BitSet([1]), md_set=BitSet([1])))
    v2 = vt.add_node(VNode(BitSet([2]), md_set=BitSet([2])))
    v3 = vt.add_node(VNode(BitSet([3]), md_set=BitSet.universal()))
    v01 = vt.add_node(VNode(BitSet([0, 1]), md_set=BitSet([0, 1])))
    vt.add_children(v01, v0, v1)
    v23 = vt.add_node(VNode(BitSet([2, 3]), md_set=BitSet([2])))
    vt.add_children(v23, v2, v3)
    v0123 = vt.add_node(VNode(BitSet([0, 1, 2, 3]), md_set=BitSet([0, 1, 2])))
    vt.add_children(v0123, v01, v23)
    return vt


def _build_ac_from_data(vtree, df, vars=(0, 1, 2, 3), num_nodes=4):
    """Build an MD circuit with Gaussian leaves initialized from empirical marginals."""
    dists = {}
    for i in vars:
        col = df.iloc[:, i] if hasattr(df, "iloc") else df[:, i]
        vals = col.values if hasattr(col, "values") else col
        mean = float(vals.mean())
        std = float(vals.std())
        if std < 1e-6:
            std = 1.0
        dists[i] = GaussianDistribution(var=i, base_mean=mean, base_stddev=std)
    return create_md_circuit(dists, vtree, num_nodes=num_nodes, initialize_weights=True)


def test_native_logsumexp_nan_backward():
    """PyTorch native logsumexp backward produces NaN on all -inf slices."""
    x = torch.tensor([[-float("inf"), -float("inf")]], requires_grad=True)
    out = torch.logsumexp(x, dim=1)
    loss = out.sum()
    loss.backward()
    assert torch.isnan(x.grad).any(), "Expected native logsumexp to produce NaN gradients"


def test_safe_logsumexp_no_nan_backward():
    """safe_logsumexp returns -inf forward but zero (not NaN) gradients."""
    x = torch.tensor([[-float("inf"), -float("inf")]], requires_grad=True)
    out = safe_logsumexp(x, dim=1)
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(x.grad).any(), "safe_logsumexp produced NaN gradients"
    assert (x.grad == 0.0).all(), "safe_logsumexp should give zero gradients for all -inf input"


def test_gaussian_leaf_stddevs_are_positive():
    """GaussianLeafLayer exposes strictly positive stddevs via a log-parameter."""
    spec = GaussianDistribution(var=0, base_mean=0.0, base_stddev=1.0)
    leaf = GaussianLeafLayer(spec, num_nodes=4, num_groups=2)

    assert hasattr(leaf, "_raw_stddevs"), "Expected log-stddev parameter"
    assert (leaf.stddevs > 0).all(), "stddevs property must be strictly positive"
    assert torch.allclose(leaf.stddevs, torch.exp(leaf._raw_stddevs)), (
        "stddevs must be exp of the raw parameter"
    )


def test_non_positive_stddev_produces_nan():
    """Directly demonstrate that a non-positive stddev corrupts the density."""
    spec = GaussianDistribution(var=0, base_mean=0.0, base_stddev=1.0)
    leaf = GaussianLeafLayer(spec, num_nodes=1, num_groups=1)

    data = torch.tensor([[0.0]])
    # Simulate the old broken behaviour: raw stddev == 0 -> log(0) == -inf and
    # division by zero gives NaN for some inputs.
    leaf._raw_stddevs.data.fill_(-1e6)  # exp(-1e6) is effectively zero
    out = leaf.forward(data)
    assert torch.isnan(out).any() or torch.isinf(out).any(), (
        "Zero stddev should produce NaN/inf densities"
    )


def _find_first_nan_node(ac: SymbolicArithmeticCircuit, data: torch.Tensor):
    """Evaluate circuit node-by-node and return the first node id that emits NaN."""
    topo = list(ac.topological_sort(reverse=True))
    outputs = {}
    for node_id in topo:
        node = ac.get_node_data(node_id)
        children = ac.get_children(node_id)
        child_outputs = [outputs[c] for c in children]
        out = node.forward(data, child_outputs)
        outputs[node_id] = out
        if torch.isnan(out).any():
            return node_id, node, out
    return None, None, None


def test_n16_training_stays_finite():
    """N=16 MD-circuit training stays finite with safe_logsumexp and positive stddevs."""
    scm = build_synthetic_continuous_scm(2)
    df_train = scm.sample(10000)
    vtree = _make_vtree_xz()
    ac = _build_ac_from_data(vtree, df_train, num_nodes=16)

    pts_t = torch.tensor(df_train.values, dtype=torch.float32)
    trainer = SymbolicEMTrainer(ac)

    for step in range(80):
        batch = pts_t[torch.randperm(len(pts_t))[:500]]
        nll = trainer.em_step(batch, step_size=1.0)
        assert not torch.isnan(torch.tensor(nll)), f"NaN NLL at step {step}"

        for _, node in ac._nodes.items():
            if isinstance(node, GaussianLeafLayer):
                assert not torch.isnan(node.means).any(), f"NaN in means at step {step}"
                assert not torch.isnan(node.stddevs).any(), f"NaN in stddevs at step {step}"
                assert (node.stddevs > 0).all(), f"Non-positive stddev at step {step}"

        nan_id, _, _ = _find_first_nan_node(ac, batch)
        assert nan_id is None, f"NaN forward output at node {nan_id}, step {step}"
