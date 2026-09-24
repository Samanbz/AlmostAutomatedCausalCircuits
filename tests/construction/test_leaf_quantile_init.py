"""Empirical-quantile initialization of spline leaf split points.

``create_md_circuit`` accepts per-variable sample arrays (``leaf_quantile_data``);
spline leaves then initialize their split points at the empirical 1/H-quantiles
with matching boundary heights (``SplineLeafLayer(init_split_points=...)``)
instead of the Gaussian quantiles of the spec.  Degenerate data (too few
samples, tied quantiles) must fall back to the spec's ``split_support``.
"""

import numpy as np
import pytest

from src.construction.circuit_builder import create_md_circuit
from src.symbolic.arithmetic.nodes.leaf_layer import (
    LogLinearSplineDistribution,
    LogLinearSplineLeafLayer,
    SplineLeafLayer,
)
from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet


def _make_vtree():
    """Two-variable vtree with universal md_set (unconstrained leaves)."""
    vt = VTree()
    v0 = vt.add_node(VNode(BitSet([0]), md_set=BitSet.universal()))
    v1 = vt.add_node(VNode(BitSet([1]), md_set=BitSet.universal()))
    v01 = vt.add_node(VNode(BitSet([0, 1]), md_set=BitSet.universal()))
    vt.add_children(v01, v0, v1)
    return vt


def _spline_leaf(ac):
    """The (mixture-wrapped) spline leaf of the circuit."""
    for node_id in ac.get_leaves():
        node = ac.get_node_data(node_id)
        base = getattr(node, "base_dist", node)
        if isinstance(base, SplineLeafLayer):
            return base
    raise AssertionError("no spline leaf found")


def _build(samples, leaf_quantile_data=None, num_nodes=4):
    dists = {
        0: LogLinearSplineDistribution(var=0, base_mean=0.0, base_stddev=1.0),
        1: LogLinearSplineDistribution(var=1, base_mean=0.0, base_stddev=1.0),
    }
    return create_md_circuit(
        dists,
        _make_vtree(),
        num_nodes=num_nodes,
        initialize_weights=True,
        leaf_quantile_data=leaf_quantile_data,
    )


def test_splits_initialized_at_empirical_quantiles():
    rng = np.random.default_rng(0)
    samples = rng.standard_gamma(2.0, size=20000)  # skewed: Gaussian quantiles are wrong
    ac = _build(samples, leaf_quantile_data={0: samples})
    leaf = _spline_leaf(ac)

    b = leaf._split_points().detach().numpy().reshape(-1)
    expected = np.quantile(samples, np.linspace(0.0, 1.0, leaf.num_nodes + 1)[1:-1])
    assert np.max(np.abs(b - expected)) < 0.1 * float(np.std(samples))


def test_without_data_falls_back_to_gaussian_splits():
    rng = np.random.default_rng(0)
    samples = rng.standard_gamma(2.0, size=20000)
    ac = _build(samples)
    leaf = _spline_leaf(ac)

    b = leaf._split_points().detach().numpy().reshape(-1)
    expected = np.quantile(samples, np.linspace(0.0, 1.0, leaf.num_nodes + 1)[1:-1])
    # Gaussian-quantile init sits far from the skewed empirical quantiles.
    assert np.max(np.abs(b - expected)) > 0.5 * float(np.std(samples))


@pytest.mark.parametrize("bad_samples", [np.zeros(500), np.full(500, 3.14)])
def test_degenerate_data_falls_back_to_split_support(bad_samples):
    # Tied empirical quantiles: must not raise, must fall back to the spec.
    ac = _build(bad_samples, leaf_quantile_data={0: bad_samples})
    leaf = _spline_leaf(ac)
    assert leaf._split_points().detach().numpy().size == leaf.num_nodes - 1


def test_init_split_points_validation():
    dist = LogLinearSplineDistribution(var=0, base_mean=0.0, base_stddev=1.0)
    supports = dist.split_support(4)
    with pytest.raises(ValueError, match="strictly increasing"):
        LogLinearSplineLeafLayer(
            dist,
            num_nodes=4,
            num_groups=1,
            node_supports=supports,
            init_split_points=np.array([1.0, 0.0, 2.0]),
        )
    with pytest.raises(ValueError, match="shape"):
        LogLinearSplineLeafLayer(
            dist,
            num_nodes=4,
            num_groups=1,
            node_supports=supports,
            init_split_points=np.array([0.0, 1.0]),
        )
