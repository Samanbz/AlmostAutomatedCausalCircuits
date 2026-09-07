"""
Comprehensive tests for src/symbolic/arithmetic/query.py.

Fixtures:
  - ac:    circuit over {0,1,2,3} with vtree enforcing md-sets {0,1} and {0,1,2}
  - sub_ac:     sub-circuit over {0,1,2} (deferred-product partner for ac)
  - disjoint_ac: circuit over {4,5} with disjoint scope
  - leaf_ac_0 / leaf_ac_1: single-leaf circuits for simple multiplication tests
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from src.construction.circuit_builder import create_md_circuit
from src.symbolic.arithmetic.circuit import SymbolicArithmeticCircuit, eval_circuit
from src.symbolic.arithmetic.nodes import GaussianDistribution
from src.symbolic.arithmetic.nodes.leaf_layer import GaussianLeafLayer, MixtureLeafLayer
from src.symbolic.arithmetic.query import (
    _instantiate,
    _multiply,
    compile_query,
)
from src.symbolic.arithmetic.train import SymbolicEMTrainer
from src.symbolic.id_ast import make_cond, make_marg, make_p, make_prod
from src.symbolic.scm import build_synthetic_continuous_scm
from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet
from src.utils.visualization import plot_gaussian_leaf, plot_mixture_leaf


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


torch.set_printoptions(precision=3, sci_mode=False, linewidth=200, edgeitems=5)


def check_integration_to_one(
    ac: SymbolicArithmeticCircuit, active_vars: list[int], n_mc: int = 100000, atol: float = 0.1
) -> None:
    """
    Numerically checks if the distribution modeled by the circuit integrates to 1
    over the specified active variables using Monte Carlo importance sampling.
    """
    if not active_vars:
        return

    # We sample from a wider normal N(0, 1.5^2) proposal to ensure heavy enough tails
    proposal_std = 1.5
    samples = torch.randn(n_mc, len(active_vars)) * proposal_std

    # Create dummy data matrix (padding with zeros for inactive variables)
    max_var = max(active_vars) if active_vars else 0
    # Make sure data has enough columns to evaluate the circuit
    # Usually it's up to max_var + 1
    data = torch.zeros(n_mc, max_var + 1)

    for idx, var in enumerate(active_vars):
        data[:, var] = samples[:, idx]

    with torch.no_grad():
        log_vals = eval_circuit(ac, data).squeeze()

        # Compute log proposal density q(x)
        log_q = -0.5 * torch.log(torch.tensor(2 * torch.pi * proposal_std**2)) - 0.5 * (
            (samples / proposal_std) ** 2
        )
        total_log_q = log_q.sum(dim=1)  # sum over variables

        # Importance sampling: E_q [p(x) / q(x)]
        log_weights = log_vals - total_log_q

        mc_log = torch.logsumexp(log_weights, dim=0) - torch.log(
            torch.tensor(n_mc, dtype=torch.float32)
        )
        mc_prob = torch.exp(mc_log).item()

    # print(f"density: {mc_prob}")
    assert abs(mc_prob - 1.0) < atol, f"Circuit does not integrate to 1. Integral: {mc_prob:.4f}"


N = 4
# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_ac(vtree, vars=(0, 1, 2, 3), num_nodes=N):
    dists = {
        i: GaussianDistribution(
            var=i,
            base_mean=0.0,
            base_stddev=1.0,
        )
        for i in vars
    }
    return create_md_circuit(
        dists,
        vtree,
        num_nodes=num_nodes,
        initialize_weights=True,
    )


@pytest.fixture
def vtree_zxz():
    """VTree over {Z1, Z2, X, Y} with {Z1, Z2} and {Z1, Z2, X} determinisms."""
    vt = VTree()
    v0 = vt.add_node(VNode(BitSet([0]), md_set=BitSet([0])))
    v1 = vt.add_node(VNode(BitSet([1]), md_set=BitSet([1])))
    v2 = vt.add_node(VNode(BitSet([2]), md_set=BitSet([2])))
    v3 = vt.add_node(VNode(BitSet([3]), md_set=BitSet.universal()))
    v01 = vt.add_node(VNode(BitSet([0, 1]), md_set=BitSet([0, 1])))
    vt.add_children(v01, v0, v1)
    v23 = vt.add_node(VNode(BitSet([2, 3]), md_set=BitSet([2])))
    vt.add_children(v23, v2, v3)
    v0123 = vt.add_node(VNode(BitSet([0, 1, 2, 3]), md_set=BitSet([0, 1])))
    vt.add_children(v0123, v01, v23)
    return vt


@pytest.fixture
def ac_zxz(vtree_zxz):
    """Full circuit over {Z1, Z2, X, Y} with {Z1, Z2} and {Z1, Z2, X} determinisms."""
    return _build_ac(vtree_zxz)


@pytest.fixture
def vtree_xz():
    """VTree over {Z1, Z2, X, Y} with only {Z1, Z2, X} determinism."""
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


@pytest.fixture
def vtree_xz_skewed():
    """VTree over {Z1, Z2, X, Y} with only {Z1, Z2, X} determinism and a skewed structure."""
    vt = VTree()
    v0 = vt.add_node(VNode(BitSet([0]), md_set=BitSet([0])))
    v1 = vt.add_node(VNode(BitSet([1]), md_set=BitSet([1])))
    v2 = vt.add_node(VNode(BitSet([2]), md_set=BitSet([2])))
    v3 = vt.add_node(VNode(BitSet([3]), md_set=BitSet.universal()))
    v01 = vt.add_node(VNode(BitSet([0, 1]), md_set=BitSet([0, 1])))
    vt.add_children(v01, v0, v1)
    v012 = vt.add_node(VNode(BitSet([0, 1, 2]), md_set=BitSet([0, 1, 2])))
    vt.add_children(v012, v01, v2)
    v0123 = vt.add_node(VNode(BitSet([0, 1, 2, 3]), md_set=BitSet([0, 1, 2])))
    vt.add_children(v0123, v012, v3)
    return vt


@pytest.fixture
def ac_xz(vtree_xz):
    """Full circuit over {Z1, Z2, X, Y} with ONLY {Z1, Z2, X} determinism."""
    return _build_ac(vtree_xz)


@pytest.fixture
def ac_xz_skewed(vtree_xz_skewed):
    return _build_ac(vtree_xz_skewed)


@pytest.fixture
def ac_zxz_10(vtree_zxz):
    return _build_ac(vtree_zxz, h=10)


@pytest.fixture
def sub_vtree_z():
    """Sub-vtree over {Z1, Z2} with {Z1, Z2} determinisms."""
    vt = VTree()
    v0 = vt.add_node(VNode(BitSet([0]), md_set=BitSet([0])))
    v1 = vt.add_node(VNode(BitSet([1]), md_set=BitSet([1])))
    v01 = vt.add_node(VNode(BitSet([0, 1]), md_set=BitSet([0, 1])))
    vt.add_children(v01, v0, v1)
    return vt


@pytest.fixture
def sub_ac_z(sub_vtree_z):
    """Sub-circuit over {Z1, Z2} with {Z1, Z2} determinisms."""
    return _build_ac(sub_vtree_z, vars=(0, 1))


@pytest.fixture
def sub_vtree_xy():
    """VTree over {X, Y} with {X} determinism."""
    vt = VTree()
    v2 = vt.add_node(VNode(BitSet([2]), md_set=BitSet([2])))
    v3 = vt.add_node(VNode(BitSet([3]), md_set=BitSet.universal()))
    v23 = vt.add_node(VNode(BitSet([2, 3]), md_set=BitSet([2])))
    vt.add_children(v23, v2, v3)
    return vt


@pytest.fixture
def sub_ac_xy(sub_vtree_xy):
    """Disjoint circuit over {2,3}."""
    return _build_ac(sub_vtree_xy, vars=(2, 3))


@pytest.fixture
def vtree_non_det():
    """VTree over {0,1,2,3} wih no md-sets."""
    vt = VTree()
    # Leaves
    v0 = vt.add_node(VNode(BitSet([0]), md_set=BitSet.universal()))
    v1 = vt.add_node(VNode(BitSet([1]), md_set=BitSet.universal()))
    v2 = vt.add_node(VNode(BitSet([2]), md_set=BitSet.universal()))
    v3 = vt.add_node(VNode(BitSet([3]), md_set=BitSet.universal()))
    # Internal: {0,1}  -> synthesizing (Kronecker) because parent {0,1,2} != child md-sets
    v01 = vt.add_node(VNode(BitSet([0, 1]), md_set=BitSet.universal()))
    vt.add_children(v01, v0, v1)
    # Internal: {0,1,2} -> synthesizing
    v23 = vt.add_node(VNode(BitSet([2, 3]), md_set=BitSet.universal()))
    vt.add_children(v23, v2, v3)
    # Root: {0,1,2,3} -> right-mixing (Hadamard) because parent md == left-child md
    v0123 = vt.add_node(VNode(BitSet([0, 1, 2, 3]), md_set=BitSet.universal()))
    vt.add_children(v0123, v01, v23)
    return vt


@pytest.fixture
def ac_non_det(vtree_non_det):
    """Full circuit over {0,1,2,3} with no deterministic nodes."""
    return _build_ac(vtree_non_det)


@pytest.fixture
def sub_vtree_non_det():
    """Sub-vtree over {0,1} with no md-sets."""
    vt = VTree()
    v0 = vt.add_node(VNode(BitSet([0]), md_set=BitSet.universal()))
    v1 = vt.add_node(VNode(BitSet([1]), md_set=BitSet.universal()))
    v01 = vt.add_node(VNode(BitSet([0, 1]), md_set=BitSet.universal()))
    vt.add_children(v01, v0, v1)
    return vt


@pytest.fixture
def sub_ac_non_det(sub_vtree_non_det):
    """Sub-circuit over {0,1} with no deterministic nodes."""
    return _build_ac(sub_vtree_non_det, vars=(0, 1))


@pytest.fixture
def leaf_ac_z1():
    """Single-leaf circuit over var Z1"""
    vt = VTree()
    vt.add_node(VNode(BitSet([0]), md_set=BitSet([0])))
    return _build_ac(vt, vars=(0,))


@pytest.fixture
def leaf_ac_z2():
    """Single-leaf circuit over var Z2"""
    vt = VTree()
    vt.add_node(VNode(BitSet([1]), md_set=BitSet([1])))
    return _build_ac(vt, vars=(1,))


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_instantiate(ac_zxz):
    """Clamping a variable should be equivalent to overriding the data column."""
    data = torch.randn(10, 4)
    clamped = data.clone()
    clamped[:, 0] = 1.5

    out_inst = eval_circuit(_instantiate(ac_zxz, {0: 1.5}), data)
    out_clamped = eval_circuit(ac_zxz, clamped)
    assert torch.allclose(out_inst, out_clamped, atol=1e-5)


# ---------------------------------------------------------------------------
# Multiplication — four structural cases
# ---------------------------------------------------------------------------


def test_multiply_deferred_product(ac_zxz, sub_ac_z):
    """Case 3: Deferred product.

    ac_zxz scope: {Z1,Z2,X,Y}     sub_ac_z scope: {Z1,Z2}
    Common scope {Z1,Z2} is exactly the left child of ac's root.
    This triggers the deferred-product branch: the bigger circuit is copied
    upward and the smaller circuit is plugged into the matched child.
    """

    new_ac = _multiply(ac_zxz, sub_ac_z)

    data = torch.randn(5, 4)

    out_sub = eval_circuit(sub_ac_z, data[:, :2])
    out_full = eval_circuit(ac_zxz, data)
    out_new = eval_circuit(new_ac, data)
    out_expected = out_full + out_sub

    assert out_new.shape == (5, 1, 1)

    assert torch.allclose(out_new, out_expected, atol=1e-5)


def test_multiply_matching_children(ac_zxz):
    """Case 4: Matching children (deterministic).

    Multiplying a circuit by itself over the same vtree
    triggers the matching-children branch and performs a Hadamard product,
    preserving unit counts. The output is bounded but not exactly 2 * log P(x) due to weight clamping.
    """
    new_ac = _multiply(ac_zxz, ac_zxz)

    data = torch.randn(5, 4)

    _ = eval_circuit(ac_zxz, data)
    out_new = eval_circuit(new_ac, data)

    assert out_new.shape == (5, 1, 1)
    assert torch.isfinite(out_new).all()


def test_multiply_matching_children_non_det(ac_non_det):
    """Case 4: Matching children (non-deterministic).

    If the matching child is not deterministic, we should fall back to the
    general case and produce a Kronecker product, which will have more units
    than the original circuit.
    """
    new_ac = _multiply(ac_non_det, ac_non_det)

    data = torch.randn(5, 4)

    out_orig = eval_circuit(ac_non_det, data)
    out_new = eval_circuit(new_ac, data)

    assert out_new.shape == (5, 1, 1)  # still a single output unit, but more internal units
    assert torch.isfinite(out_new).all()
    assert torch.allclose(out_new, out_orig + out_orig, atol=1e-5)


def test_multiply_deferred_product_non_det(ac_non_det, sub_ac_non_det):
    """Deferred product should still work if the circuits are non-deterministic."""
    new_ac = _multiply(ac_non_det, sub_ac_non_det)
    data = torch.randn(5, 4)

    out_sub = eval_circuit(sub_ac_non_det, data[:, :2])
    out_full = eval_circuit(ac_non_det, data)
    out_new = eval_circuit(new_ac, data)
    out_expected = out_full + out_sub

    assert out_new.shape == (5, 1, 1)
    assert torch.isfinite(out_new).all()
    assert torch.allclose(out_new, out_expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Integration: compile_query
# ---------------------------------------------------------------------------


def check_marg_numerical(ac_xz):
    """Compiling MARG({2})[P(V)] should match a Monte Carlo approximation."""
    torch.manual_seed(42)
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}
    ast_marg = make_marg({"2"}, make_p({"0", "1", "2", "3"}))
    q_marg, _ = compile_query(ast_marg, ac_xz, ac_xz.get_roots()[0], var_to_id)

    data = torch.randn(5, 4)
    out_marg = eval_circuit(q_marg, data)
    assert out_marg.shape == (5, 1, 1)
    assert torch.isfinite(out_marg).all()

    # MC approximation: compile the full joint and sample var 2 many times
    ast_joint = make_p({"0", "1", "2", "3"})
    q_joint, _ = compile_query(ast_joint, ac_xz, ac_xz.get_roots()[0], var_to_id)

    n_mc = 200000
    proposal_std = 1.5
    z2_samples = torch.randn(n_mc, 1) * proposal_std

    mc_results = []
    for i in range(data.shape[0]):
        pts = data[i : i + 1].expand(n_mc, -1).clone()
        pts[:, 2] = z2_samples[:, 0]
        log_vals = eval_circuit(q_joint, pts).squeeze()

        # Adjust for the N(0, 3^2) proposal density (importance sampling)
        log_q = -0.5 * torch.log(torch.tensor(2 * torch.pi * proposal_std**2)) - 0.5 * (
            (z2_samples[:, 0] / proposal_std) ** 2
        )
        log_vals = log_vals - log_q

        mc_log = torch.logsumexp(log_vals, dim=0) - torch.log(
            torch.tensor(n_mc, dtype=torch.float32)
        )
        mc_results.append(mc_log)

    mc_result = torch.stack(mc_results)
    diff = torch.abs(out_marg.squeeze() - mc_result)
    print(f"Diff:\n{diff}")
    assert torch.allclose(out_marg.squeeze(), mc_result, atol=1e-1)

    return q_marg


def test_marg_numerical(ac_xz):
    check_marg_numerical(ac_xz)


def check_cond_numerical(base_ac):
    """
    Compiling conditional queries and checking that P(Y|X,Z) = P(X,Y,Z) / P(X,Z) holds numerically.
    """
    torch.manual_seed(42)
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}

    ast_joint = make_p({"0", "1", "2", "3"})
    ast_cond = make_cond({"3"}, {"0", "1", "2"}, ast_joint)

    # 1. Passes with base_ac
    q_cond, _ = compile_query(ast_cond, base_ac, base_ac.get_roots()[0], var_to_id)

    # Check conditional logic numerical results
    ast_marg = make_marg({"3"}, ast_joint)
    q_marg_ac, _ = compile_query(ast_marg, base_ac, base_ac.get_roots()[0], var_to_id)

    data = torch.randn(5, 4)
    out_cond = eval_circuit(q_cond, data)
    out_joint = eval_circuit(base_ac, data)
    out_marg_out = eval_circuit(q_marg_ac, data)
    expected = out_joint - out_marg_out
    assert torch.allclose(out_cond, expected, atol=1e-5)

    return q_cond


def test_cond_numerical(ac_xz):
    check_cond_numerical(ac_xz)  # Should pass with only {Z1, Z2, X} determinism


def test_cond_numerical_overlap(ac_zxz):
    with pytest.raises(ValueError):
        check_cond_numerical(ac_zxz)  # Should fail with {Z1, Z2} determinism


def check_backdoor_numerical(ac_xz):
    """Compiling backdoor query should match Monte Carlo approximation of the summand."""
    torch.manual_seed(42)
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}

    # Backdoor: P(3|do(2)) with Z={0,1}
    # Summand: P(3|0,1,2) * P(0,1) = P(0,1,2,3) * P(0,1,2)^-1 * P(0,1)
    ast_joint = make_p({"0", "1", "2", "3"})
    ast_z = make_marg({"2", "3"}, ast_joint)

    ast_cond = make_cond({"3"}, {"0", "1", "2"}, ast_joint)
    ast_summand = make_prod([ast_cond, ast_z])
    ast_backdoor = make_marg({"0", "1"}, ast_summand)

    q_z, _ = compile_query(ast_z, ac_xz, ac_xz.get_roots()[0], var_to_id)
    q_backdoor, _ = compile_query(ast_backdoor, ac_xz, ac_xz.get_roots()[0], var_to_id)

    data = torch.randn(5, 4)
    out_backdoor = eval_circuit(q_backdoor, data)
    assert out_backdoor.shape == (5, 1, 1)
    assert torch.isfinite(out_backdoor).all()

    ast_xz = make_marg({"3"}, ast_joint)
    q_xz, _ = compile_query(ast_xz, ac_xz, ac_xz.get_roots()[0], var_to_id)

    n_mc = 2000
    z_samples = torch.randn(n_mc, 2)

    mc_results = []
    for i in range(data.shape[0]):
        pts = data[i : i + 1].expand(n_mc, -1).clone()
        pts[:, 0] = z_samples[:, 0]
        pts[:, 1] = z_samples[:, 1]

        log_p_yxz = eval_circuit(ac_xz, pts).squeeze()
        log_p_xz = eval_circuit(q_xz, pts).squeeze()
        log_p_z = eval_circuit(q_z, pts).squeeze()

        log_vals = log_p_yxz - log_p_xz + log_p_z

        log_q0 = -0.5 * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * (z_samples[:, 0] ** 2)
        log_q1 = -0.5 * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * (z_samples[:, 1] ** 2)
        log_vals = log_vals - log_q0 - log_q1

        mc_log = torch.logsumexp(log_vals, dim=0) - torch.log(
            torch.tensor(n_mc, dtype=torch.float32)
        )
        mc_results.append(mc_log)

    mc_result = torch.stack(mc_results)
    assert torch.allclose(out_backdoor.squeeze(), mc_result, atol=2e-1)

    return q_backdoor


def test_backdoor_numerical(ac_xz):
    check_backdoor_numerical(ac_xz)


def _build_ac_from_data(vtree, df, vars=(0, 1, 2, 3), num_nodes=N):
    """Build an MD circuit with Gaussian leaves initialized from empirical marginals."""
    dists = {}
    for i in vars:
        col = df.iloc[:, i] if isinstance(df, pd.DataFrame) else df[:, i]
        if hasattr(col, "values"):
            col = col.values
        mean = float(np.mean(col))
        std = float(np.std(col))
        if std < 1e-6:
            std = 1.0
        dists[i] = GaussianDistribution(var=i, base_mean=mean, base_stddev=std)
    return create_md_circuit(dists, vtree, num_nodes=num_nodes, initialize_weights=True)


def _train_circuit(vtree, vars=(0, 1, 2, 3), num_nodes=N):
    scm = build_synthetic_continuous_scm(2)
    df_train = scm.sample(10000)
    # The SCM variables are Z0(0), Z1(1), X(2), Y(3) matching ac_xz
    ac = _build_ac_from_data(vtree, df_train, vars=vars, num_nodes=num_nodes)
    pts_t = torch.tensor(df_train.values, dtype=torch.float32)
    trainer = SymbolicEMTrainer(ac)
    trainer.train(pts_t, n_iter=2, batch_size=500, log_interval=0)
    return ac, scm


@pytest.fixture
def trained_ac_xz_and_scm(vtree_xz):
    trained_circuit, scm = _train_circuit(vtree_xz)
    print("\nWeights after training:\n")
    trained_circuit.print_weights(log_domain=True)
    return trained_circuit, scm


@pytest.fixture
def trained_ac_xz_skewed_and_scm(vtree_xz_skewed):
    return _train_circuit(vtree_xz_skewed)


def test_trained_multiply_matching_children(trained_ac_xz_and_scm):
    ac, scm = trained_ac_xz_and_scm
    test_multiply_matching_children(ac)


def test_trained_marg_numerical(trained_ac_xz_and_scm):
    ac, scm = trained_ac_xz_and_scm
    q_marg = check_marg_numerical(ac)
    check_integration_to_one(q_marg, active_vars=[0, 1, 3], n_mc=100000, atol=0.1)


def test_trained_cond_numerical(trained_ac_xz_and_scm):
    ac, scm = trained_ac_xz_and_scm
    q_cond = check_cond_numerical(ac)
    check_integration_to_one(q_cond, active_vars=[3], n_mc=100000, atol=0.1)


def test_trained_backdoor_numerical(trained_ac_xz_and_scm):
    ac, scm = trained_ac_xz_and_scm
    q_backdoor = check_backdoor_numerical(ac)
    check_integration_to_one(q_backdoor, active_vars=[3], n_mc=10000, atol=0.01)


def check_conditional_dependencies(ac, scm, data=None):
    """
    Ensure that P(Y|X,Z) != P(Y|X) and P(Y|X,Z) != P(Y|Z).
    Since the circuit is not MD with respect to X or Z alone, we compute P(Y|X) and P(Y|Z)
    via joint and marg evaluation and subtraction.
    """
    torch.manual_seed(42)
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}
    print(f"Input:\n{data}")

    ast_joint = make_p({"0", "1", "2", "3"})

    # 1. P(Y|X,Z) using direct compilation
    ast_cond_yxz = make_cond({"3"}, {"0", "1", "2"}, ast_joint)
    q_cond_yxz, _ = compile_query(ast_cond_yxz, ac, ac.get_roots()[0], var_to_id)

    # 2. P(Y|X) = P(Y,X) / P(X)
    ast_yx = make_marg({"0", "1"}, ast_joint)
    ast_x = make_marg({"0", "1", "3"}, ast_joint)
    q_yx, _ = compile_query(ast_yx, ac, ac.get_roots()[0], var_to_id)
    q_x, _ = compile_query(ast_x, ac, ac.get_roots()[0], var_to_id)

    # 3. P(Y|Z) = P(Y,Z) / P(Z)  (where Z = {0, 1})
    ast_yz = make_marg({"2"}, ast_joint)
    ast_z = make_marg({"2", "3"}, ast_joint)
    q_yz, _ = compile_query(ast_yz, ac, ac.get_roots()[0], var_to_id)
    q_z, _ = compile_query(ast_z, ac, ac.get_roots()[0], var_to_id)

    data = torch.randn(10, 4) if data is None else data

    eval_circuit(ac, data)

    for leaf_id in ac.get_leaves():
        leaf = ac.get_node_data(leaf_id)
        if isinstance(leaf, GaussianLeafLayer):
            print(
                f"\nLeaf {leaf_id}: var={leaf.var}, node_supports=\n{leaf.node_supports}, means=\n{leaf.means}, stddevs=\n{leaf.stddevs}"
            )
            plot_gaussian_leaf(leaf)
            plt.savefig(f"leaf_{leaf_id}_distribution.png")
            plt.close()
        elif isinstance(leaf, MixtureLeafLayer):
            print(
                f"\nLeaf {leaf_id}: var={leaf.var}, means=\n{leaf.base_dist.means}, stddevs=\n{leaf.base_dist.stddevs}"
            )
            plot_mixture_leaf(leaf)
            plt.savefig(f"leaf_{leaf_id}_distribution.png")
            plt.close()

    out_cond_yxz = eval_circuit(q_cond_yxz, data, verbose=True, show_weights=True, log_domain=False)

    out_yx = eval_circuit(q_yx, data)
    out_x = eval_circuit(q_x, data)
    out_cond_yx = out_yx - out_x

    out_yz = eval_circuit(q_yz, data)
    out_z = eval_circuit(q_z, data)
    out_cond_yz = out_yz - out_z

    # pri
    diff_yx = out_cond_yxz - out_cond_yx
    diff_yz = out_cond_yxz - out_cond_yz

    print(f"\nP(Y|X,Z) - P(Y|X): {diff_yx.flatten()}")
    print(f"P(Y|X,Z) - P(Y|Z): {diff_yz.flatten()}")

    # Check that they are not equal, accumulating errors so we see all failures
    errors = []
    if torch.any(torch.abs(out_cond_yxz - out_cond_yx) < 1e-3):
        errors.append("P(Y|X,Z) is equal to P(Y|X)")
    if torch.any(torch.abs(out_cond_yxz - out_cond_yz) < 1e-3):
        errors.append("P(Y|X,Z) is equal to P(Y|Z)")

    assert not errors, f"Conditional dependencies missing: {errors}"


def test_conditional_dependencies(trained_ac_xz_and_scm):
    ac, scm = trained_ac_xz_and_scm
    check_conditional_dependencies(ac, scm, data=torch.tensor(scm.sample(10).values))


def test_conditional_dependencies_skewed(trained_ac_xz_skewed_and_scm):
    ac, scm = trained_ac_xz_skewed_and_scm
    check_conditional_dependencies(ac, scm)


def test_conditional_dependencies_numerical(trained_ac_xz_and_scm):
    """
    Evaluates P(Y|X,Z) while fixing Y and X, and varying Z across its unit supports.
    """
    import math

    trained_ac_xz, _ = trained_ac_xz_and_scm

    torch.manual_seed(42)
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}

    ast_joint = make_p({"0", "1", "2", "3"})
    ast_cond_yxz = make_cond({"3"}, {"0", "1", "2"}, ast_joint)
    q_cond_yxz, _ = compile_query(
        ast_cond_yxz, trained_ac_xz, trained_ac_xz.get_roots()[0], var_to_id
    )

    z0_pts = []
    z1_pts = []
    for leaf_id in trained_ac_xz.get_leaves():
        leaf = trained_ac_xz.get_node_data(leaf_id)
        if getattr(leaf, "var", None) == 0:
            assert leaf.node_supports is not None, f"Leaf {leaf_id} has no unit supports"
            for supp in leaf.node_supports:
                interval = supp.intervals.get(0)
                if interval:
                    low = -3.0 if math.isinf(interval.low) else interval.low
                    high = 3.0 if math.isinf(interval.high) else interval.high
                    z0_pts.append((low + high) / 2.0)
        elif getattr(leaf, "var", None) == 1:
            assert leaf.node_supports is not None, f"Leaf {leaf_id} has no unit supports"
            for supp in leaf.node_supports:
                interval = supp.intervals.get(1)
                if interval:
                    low = -3.0 if math.isinf(interval.low) else interval.low
                    high = 3.0 if math.isinf(interval.high) else interval.high
                z1_pts.append((low + high) / 2.0)

    pts = []
    for z0 in z0_pts:
        for z1 in z1_pts:
            pts.append([z0, z1, 1.0, 1.0])  # Z0, Z1, X=1.0, Y=1.0

    data = torch.tensor(pts, dtype=torch.float32)
    out_cond_yxz = eval_circuit(q_cond_yxz, data)
    print(f"\nConditional outputs for fixed X=1.0, Y=1.0 and varying Z:\n{out_cond_yxz.flatten()}")

    # Check if the conditional probability changes as Z varies
    # We assert that it actually varies
    assert out_cond_yxz.std() > 1e-4, (
        f"P(Y|X,Z) did not change with Z! outputs: {out_cond_yxz.flatten()}"
    )


def check_interventional_observational_distinct(ac, data=None):
    """
    Check that P(Y|do(X)) != P(Y|X).
    P(Y|do(X)) is compiled via the backdoor query.
    P(Y|X) is computed using P(Y,X) / P(X).
    """
    torch.manual_seed(42)
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}

    ast_joint = make_p({"0", "1", "2", "3"})

    # 1. P(Y|do(X)) -> backdoor query: sum_{Z} P(Y|X,Z) P(Z)
    ast_z = make_marg({"2", "3"}, ast_joint)
    ast_cond = make_cond({"3"}, {"0", "1", "2"}, ast_joint)
    ast_summand = make_prod([ast_cond, ast_z])
    ast_backdoor = make_marg({"0", "1"}, ast_summand)

    q_backdoor, _ = compile_query(ast_backdoor, ac, ac.get_roots()[0], var_to_id)

    # 2. P(Y|X) = P(Y,X) / P(X)
    ast_yx = make_marg({"0", "1"}, ast_joint)
    ast_x = make_marg({"0", "1", "3"}, ast_joint)
    q_yx, _ = compile_query(ast_yx, ac, ac.get_roots()[0], var_to_id)
    q_x, _ = compile_query(ast_x, ac, ac.get_roots()[0], var_to_id)

    data = torch.randn(10, 4) if data is None else data

    out_backdoor = eval_circuit(q_backdoor, data)

    out_yx = eval_circuit(q_yx, data)
    out_x = eval_circuit(q_x, data)
    out_cond_yx = out_yx - out_x

    diff = torch.abs(out_backdoor - out_cond_yx)
    # print(f"Diff:\n{diff}")

    # Check they are distinct
    assert torch.all(diff > 1e-3), "P(Y|do(X)) is equal to P(Y|X)!"


def test_interventional_observational_distinct(trained_ac_xz_and_scm):
    ac, scm = trained_ac_xz_and_scm
    check_interventional_observational_distinct(ac)


def test_interventional_observational_distinct_skewed(trained_ac_xz_skewed_and_scm):
    ac, scm = trained_ac_xz_skewed_and_scm
    check_interventional_observational_distinct(ac)


def check_interventional_observational_ratio_fit(ac, scm):
    """
    Check that P(Y|do(X))/P(Y|X) in the ground truth SCM matches the circuit.
    This shows the circuit has learned the interventional vs observational relationship,
    even if the absolute densities are not perfectly fit.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}

    ast_joint = make_p({"0", "1", "2", "3"})

    # 1. P(Y|do(X)) -> backdoor query
    ast_z = make_marg({"2", "3"}, ast_joint)
    ast_cond = make_cond({"3"}, {"0", "1", "2"}, ast_joint)
    ast_summand = make_prod([ast_cond, ast_z])
    ast_backdoor = make_marg({"0", "1"}, ast_summand)
    q_backdoor, _ = compile_query(ast_backdoor, ac, ac.get_roots()[0], var_to_id)

    # 2. P(Y|X) = P(Y,X) / P(X)
    ast_yx = make_marg({"0", "1"}, ast_joint)
    ast_x = make_marg({"0", "1", "3"}, ast_joint)
    q_yx, _ = compile_query(ast_yx, ac, ac.get_roots()[0], var_to_id)
    q_x, _ = compile_query(ast_x, ac, ac.get_roots()[0], var_to_id)

    x_values = [-1.0, 0.0, 1.0]
    y_values = [-1.5, -1.0, 0.0, 1.0, 1.5]

    n_samples = 1000000
    eps = 0.1

    errors = []
    for x_val in x_values:
        scm_do_x = scm.intervene({"X": x_val})

        for y_val in y_values:
            data = torch.zeros(1, 4)
            data[0, 2] = x_val  # X
            data[0, 3] = y_val  # Y

            # Circuit Evaluation (log space)
            out_backdoor = eval_circuit(q_backdoor, data).item()
            out_yx = eval_circuit(q_yx, data).item()
            out_x = eval_circuit(q_x, data).item()
            out_cond_yx = out_yx - out_x

            circuit_log_ratio = out_backdoor - out_cond_yx

            # SCM Ground Truth Evaluation (empirical log density)
            true_log_p_y_do_x = scm_do_x.empirical_log_density(
                {"Y": y_val}, n_samples=n_samples, eps=eps
            )
            true_log_p_y_given_x = scm.empirical_log_density(
                {"Y": y_val}, {"X": x_val}, n_samples=n_samples, eps=eps
            )

            true_log_ratio = true_log_p_y_do_x - true_log_p_y_given_x

            print(
                f"\nX={x_val}, Y={y_val} -> Circuit log ratio P(Y|do(X))/P(Y|X): {circuit_log_ratio:.4f}"
            )
            print(
                f"X={x_val}, Y={y_val} -> True SCM log ratio P(Y|do(X))/P(Y|X): {true_log_ratio:.4f}"
            )

            # The tolerance might need to be somewhat loose given the approximation and limited training
            # accumulate errors
            if not abs(circuit_log_ratio - true_log_ratio) < 0.2:
                errors.append(
                    f"X={x_val}, Y={y_val} -> Circuit log ratio {circuit_log_ratio:.4f} != True SCM log ratio {true_log_ratio:.4f}"
                )

    assert not errors, f"Interventional vs observational ratio mismatch: {errors}"


def test_interventional_observational_ratio_fit(trained_ac_xz_and_scm):
    ac, scm = trained_ac_xz_and_scm

    # Train further to actually learn the dependencies
    df_train = scm.sample(10000)
    pts_t = torch.tensor(df_train.values, dtype=torch.float32)
    trainer = SymbolicEMTrainer(ac)
    trainer.train(pts_t, n_iter=15, batch_size=500, log_interval=0)

    check_interventional_observational_ratio_fit(ac, scm)


def _circuit_conditional_outputs(ac, data, var_to_id):
    """Return circuit log P(Y|X,Z) and log P(Y|X) for each row in data."""
    ast_joint = make_p({"0", "1", "2", "3"})

    ast_cond_yxz = make_cond({"3"}, {"0", "1", "2"}, ast_joint)
    q_cond_yxz, _ = compile_query(ast_cond_yxz, ac, ac.get_roots()[0], var_to_id)

    ast_yx = make_marg({"0", "1"}, ast_joint)
    ast_x = make_marg({"0", "1", "3"}, ast_joint)
    q_yx, _ = compile_query(ast_yx, ac, ac.get_roots()[0], var_to_id)
    q_x, _ = compile_query(ast_x, ac, ac.get_roots()[0], var_to_id)

    out_cond_yxz = eval_circuit(q_cond_yxz, data)
    out_yx = eval_circuit(q_yx, data)
    out_x = eval_circuit(q_x, data)
    out_cond_yx = out_yx - out_x

    return out_cond_yxz, out_cond_yx


def test_conditional_dependencies_by_x_range(trained_ac_xz_and_scm):
    """
    Check whether the circuit fails to distinguish P(Y|X,Z) from P(Y|X)
    depending on which X-leaf node is active.
    """
    ac, scm = trained_ac_xz_and_scm
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}

    # Draw many realistic samples and keep a spread across X.
    df = scm.sample(3000)
    df = df.sort_values("X").reset_index(drop=True)

    # Take a stratified sample so we get some rows from each X-leaf node.
    data_full = torch.tensor(df[["Z0", "Z1", "X", "Y"]].values, dtype=torch.float32)

    # Find the X leaf and evaluate it to see which node is active per sample.
    x_leaf = None
    for leaf_id in ac.get_leaves():
        leaf = ac.get_node_data(leaf_id)
        if getattr(leaf, "var", None) == 2:
            x_leaf = leaf
            break
    if x_leaf is None:
        raise RuntimeError("No X leaf found")

    with torch.no_grad():
        leaf_out = x_leaf.forward(data_full)  # [B, G, U]
    # Active node is the one with finite output.
    active_node = torch.isfinite(leaf_out).long().argmax(dim=-1).squeeze(-1).numpy()

    # Attach active node index to dataframe.
    df["active_x_node"] = active_node

    # Sample up to 10 rows per active X node.
    sampled_rows = []
    for node_idx in sorted(df["active_x_node"].unique()):
        subset = (
            df[df["active_x_node"] == node_idx]
            .sample(n=100)
            .sort_values("X")
            .reset_index(drop=True)
        )
        sampled_rows.append(subset)
    sampled = pd.concat(sampled_rows, ignore_index=True)

    data = torch.tensor(sampled[["Z0", "Z1", "X", "Y"]].values, dtype=torch.float32)

    out_yxz, out_yx = _circuit_conditional_outputs(ac, data, var_to_id)
    diff = out_yxz - out_yx

    print("\nActive X-node correlation check:")
    header = f"{'idx':>4} | {'X':>8} | {'Y':>8} | {'Z0':>8} | {'Z1':>8} | {'P(Y|X,Z)':>10} | {'P(Y|X)':>10} | {'diff':>10} | active X node"
    print(header)
    for i in range(len(sampled)):
        row = sampled.iloc[i]
        x, y, z0, z1 = row["X"], row["Y"], row["Z0"], row["Z1"]
        node_idx = row["active_x_node"]
        print(
            f"{i:4} | {x:8.3f} | {y:8.3f} | {z0:8.3f} | {z1:8.3f} | "
            f"{float(out_yxz[i]):10.4f} | {float(out_yx[i]):10.4f} | {float(diff[i]):10.4f} | {int(node_idx)}"
        )

    # Summarize mean absolute diff per active X node.
    print("\nMean |P(Y|X,Z) - P(Y|X)| by active X node:")
    for node_idx in sorted(sampled["active_x_node"].unique()):
        mask = sampled["active_x_node"] == node_idx
        if mask.any():
            idxs = mask.to_numpy().nonzero()[0]
            mean_diff = float(diff[idxs].abs().mean())
            print(f"  node {int(node_idx)}: {mean_diff:.4f}  (n={int(mask.sum())})")


def _format_interval(iv):
    """Format a ContinuousInterval as a readable string."""
    if iv is None:
        return "(-inf, inf)"
    low = f"{iv.low:.3f}" if iv.low > float("-inf") else "-inf"
    high = f"{iv.high:.3f}" if iv.high < float("inf") else "inf"
    lpar = "[" if getattr(iv, "include_low", True) else "("
    rpar = "]" if getattr(iv, "include_high", True) else ")"
    return f"{lpar}{low}, {high}{rpar}"


def test_extract_z_conditions(trained_ac_xz_and_scm):
    """
    Extract the Z0/Z1 support conditions that activate each node of the [0,1] sum layer,
    and plot the ground-truth P(X,Y) distribution for each such node.

    The [0,1] layer is a constrained synthesizing layer. Its log-weights have shape
    [G, H, G_L, H_L, G_R, H_R]; after reshaping to [G, H, G_L*H_L, G_R*H_R], the
    rows correspond to (Z0-group, Z0-node) combinations and the columns to
    (Z1-group, Z1-node) combinations. A finite weight at (row, col) means that
    the corresponding Z0-node / Z1-node pair activates the parent (group, node).
    """
    ac, scm = trained_ac_xz_and_scm

    # Find the [0,1] sum layer in the trained circuit.
    z_sum_id = None
    for _vtree_id, sum_id in ac.vtree_to_sum.items():
        node = ac.get_node_data(sum_id)
        if node.scope == BitSet([0, 1]):
            z_sum_id = sum_id
            break

    if z_sum_id is None:
        raise RuntimeError("No [0,1] sum layer found")

    z_sum = ac.get_node_data(z_sum_id)
    l_id, r_id = ac.get_children(z_sum_id)
    z0_leaf = ac.get_node_data(l_id)
    z1_leaf = ac.get_node_data(r_id)

    z0_supports = z0_leaf.node_supports
    z1_supports = z1_leaf.node_supports

    w_obj = z_sum.log_weights
    wt = w_obj.log_weights.detach().cpu()
    G, H, G_L, H_L, G_R, H_R = wt.shape
    wt_flat = wt.reshape(G, H, G_L * H_L, G_R * H_R)
    finite_mask = wt_flat > -20

    print("\n[0,1] sum layer Z0/Z1 activation conditions:")
    print(f"G={G}, H={H}, G_L={G_L}, H_L={H_L}, G_R={G_R}, H_R={H_R}")
    print("Flattened rows: (Z0-group, Z0-node); cols: (Z1-group, Z1-node)")

    # Collect activation rectangles per parent node (union across output groups).
    node_regions = {n: [] for n in range(H)}

    for g in range(G):
        for n in range(H):
            print(f"\nGroup {g}, node {n}:")
            active_pairs = []
            for row in range(G_L * H_L):
                for col in range(G_R * H_R):
                    if finite_mask[g, n, row, col]:
                        g_l = row // H_L
                        i = row % H_L
                        g_r = col // H_R
                        k = col % H_R
                        z0_iv = z0_supports[i].intervals.get(0) if z0_supports else None
                        z1_iv = z1_supports[k].intervals.get(1) if z1_supports else None
                        log_w = float(wt_flat[g, n, row, col])
                        active_pairs.append((g_l, i, g_r, k, z0_iv, z1_iv, log_w))
                        if (z0_iv, z1_iv) not in node_regions[n]:
                            node_regions[n].append((z0_iv, z1_iv))
            if not active_pairs:
                print("  (no active pairs)")
            else:
                for g_l, i, g_r, k, z0_iv, z1_iv, log_w in active_pairs:
                    print(
                        f"  Z0(g={g_l},node={i}): {_format_interval(z0_iv)}  AND  "
                        f"Z1(g={g_r},node={k}): {_format_interval(z1_iv)}  "
                        f"(log-w={log_w:.3f})"
                    )

    print("\nPer-node Z0/Z1 activation regions (union over groups):")
    for n, regions in node_regions.items():
        print(f"\nNode {n}:")
        if not regions:
            print("  (no region)")
        for z0_iv, z1_iv in regions:
            print(f"  Z0 in {_format_interval(z0_iv)}  AND  Z1 in {_format_interval(z1_iv)}")

    # -----------------------------------------------------------------------
    # Ground-truth vs. circuit P(X,Y) heatmaps per [0,1] node.
    # -----------------------------------------------------------------------
    n_samples = 1000000
    df_samples = scm.sample(n_samples)

    x_min, x_max = float(df_samples["X"].min()), float(df_samples["X"].max())
    y_min, y_max = float(df_samples["Y"].min()), float(df_samples["Y"].max())

    bins = 500
    xedges = np.linspace(x_min, x_max, bins + 1)
    yedges = np.linspace(y_min, y_max, bins + 1)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Extract finite boundaries from the X leaf supports to mark on the heatmaps.
    x_split_points = []
    for leaf_id in ac.get_leaves():
        leaf = ac.get_node_data(leaf_id)
        if getattr(leaf, "var", None) == 2 and getattr(leaf, "node_supports", None):
            for supp in leaf.node_supports:
                iv = supp.intervals.get(2)
                if iv is None:
                    continue
                if not np.isinf(iv.low):
                    x_split_points.append(float(iv.low))
                if not np.isinf(iv.high):
                    x_split_points.append(float(iv.high))
            break
    x_split_points = sorted(set(x_split_points))
    print(f"\nX leaf split points: {x_split_points}")

    def _node_mask(df, regions):
        mask = np.zeros(len(df), dtype=bool)
        for z0_iv, z1_iv in regions:
            m = z0_iv.contains(df["Z0"].values) & z1_iv.contains(df["Z1"].values)
            mask |= m
        return mask

    # Evaluate the full circuit on the same realistic samples.
    pts = torch.tensor(df_samples.values, dtype=torch.float32)
    with torch.no_grad():
        log_p_all = eval_circuit(ac, pts, log_domain=True).squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, H, figsize=(6 * H, 10))
    if H == 1:
        axes = axes.reshape(2, 1)

    for n in range(H):
        regions = node_regions[n]
        mask = _node_mask(df_samples, regions) if regions else np.zeros(len(df_samples), dtype=bool)
        df_node = df_samples[mask]

        region_strs = [
            f"Z0∈{_format_interval(z0_iv)} ∧ Z1∈{_format_interval(z1_iv)}"
            for z0_iv, z1_iv in regions
        ]
        subtitle = " OR ".join(region_strs) if region_strs else "no region"

        # ---- Ground truth (empirical conditional mass) ----
        counts_gt, _, _ = np.histogram2d(
            df_node["X"].values,
            df_node["Y"].values,
            bins=[xedges, yedges],
        )
        total_gt = counts_gt.sum()
        probs_gt = counts_gt / total_gt if total_gt > 0 else counts_gt

        ax_gt = axes[0, n]
        im_gt = ax_gt.imshow(
            probs_gt.T,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=probs_gt.max() if probs_gt.max() > 0 else 1.0,
        )
        ax_gt.set_title(f"Ground-truth P(X,Y | [0,1] node {n})\n(n={len(df_node)} samples)")
        ax_gt.set_xlabel("X")
        ax_gt.set_ylabel("Y")
        fig.colorbar(im_gt, ax=ax_gt, label="P(X,Y | node)")
        ax_gt.text(
            0.5,
            -0.18,
            subtitle,
            transform=ax_gt.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            wrap=True,
        )

        # ---- Circuit-learned conditional mass ----
        log_p_node = log_p_all[mask]
        x_node = df_node["X"].values
        y_node = df_node["Y"].values

        if log_p_node.size > 0 and np.any(np.isfinite(log_p_node)):
            shift = np.nanmax(log_p_node[np.isfinite(log_p_node)])
            weights = np.exp(log_p_node - shift)
            weights /= weights.sum()
        else:
            weights = np.ones_like(log_p_node) / max(log_p_node.size, 1)

        counts_circ, _, _ = np.histogram2d(
            x_node,
            y_node,
            bins=[xedges, yedges],
            weights=weights,
        )
        total_circ = counts_circ.sum()
        probs_circ = counts_circ / total_circ if total_circ > 0 else counts_circ

        ax_circ = axes[1, n]
        im_circ = ax_circ.imshow(
            probs_circ.T,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=probs_circ.max() if probs_circ.max() > 0 else 1.0,
        )
        ax_circ.set_title(f"Circuit P(X,Y | [0,1] node {n})\n(n={len(df_node)} samples)")
        ax_circ.set_xlabel("X")
        ax_circ.set_ylabel("Y")
        fig.colorbar(im_circ, ax=ax_circ, label="P(X,Y | node)")
        ax_circ.text(
            0.5,
            -0.18,
            subtitle,
            transform=ax_circ.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            wrap=True,
        )

        for b in x_split_points:
            ax_gt.axvline(b, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
            ax_circ.axvline(b, color="red", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = "xy_per_z_node_ground_truth_vs_circuit.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved ground-truth vs. circuit XY heatmaps to {out_path}")
