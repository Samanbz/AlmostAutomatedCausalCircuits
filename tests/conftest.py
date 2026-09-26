import random

import numpy as np
import pytest
import torch

from src.construction.random_scm import generate_random_scm
from src.symbolic import (
    VNode,
    VTree,
)
from src.utils import BitSet


@pytest.fixture
def synthetic_data():
    """A fixture returning a basic 2D tensor of random variables generated from an SCM."""
    scm = generate_random_scm(n_nodes=4, expected_degree=2.0)
    df = scm.sample(100)
    return torch.from_numpy(df.values.copy()).float()


@pytest.fixture
def complex_vtree():
    """A 4-variable VTree with nested MD-sets."""
    vt = VTree()
    n0_id = vt.add_node(VNode(scope=BitSet({0, 1, 2, 3}), md_set=BitSet({0, 1})))

    n1_id = vt.add_node(VNode(scope=BitSet({0, 1}), md_set=BitSet({0, 1})))
    n2_id = vt.add_node(VNode(scope=BitSet({0}), md_set=BitSet({0})))
    n3_id = vt.add_node(VNode(scope=BitSet({1}), md_set=BitSet({1})))
    vt.add_children(n1_id, n2_id, n3_id)

    n4_id = vt.add_node(VNode(scope=BitSet({2, 3}), md_set=BitSet()))
    n5_id = vt.add_node(VNode(scope=BitSet({2}), md_set=BitSet()))
    n6_id = vt.add_node(VNode(scope=BitSet({3}), md_set=BitSet()))
    vt.add_children(n4_id, n5_id, n6_id)

    vt.add_children(n0_id, n1_id, n4_id)
    return vt


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
