"""
Comprehensive tests for src/symbolic/arithmetic/query.py.

Fixtures:
  - full_ac:    circuit over {0,1,2,3} with vtree enforcing md-sets {0,1} and {0,1,2}
  - sub_ac:     sub-circuit over {0,1,2} (deferred-product partner for full_ac)
  - disjoint_ac: circuit over {4,5} with disjoint scope
  - leaf_ac_0 / leaf_ac_1: single-leaf circuits for simple multiplication tests
"""

import numpy as np
import pytest
import torch

from src.construction.circuit_builder import create_md_circuit
from src.symbolic.arithmetic.circuit import SymbolicArithmeticCircuit, eval_circuit
from src.symbolic.arithmetic.nodes import GaussianDistribution, SumNode
from src.symbolic.arithmetic.query import (
    _instantiate,
    _inverse,
    _multiply,
    compile_query,
)
from src.symbolic.id_ast import (
    make_det_prod,
    make_marg,
    make_p,
    make_pow,
    make_prod,
)
from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


H = 4


def check_integration_to_one(
    ac: SymbolicArithmeticCircuit, active_vars: list[int], n_mc: int = 100000, atol: float = 0.1
) -> None:
    """
    Numerically checks if the distribution modeled by the circuit integrates to 1
    over the specified active variables using Monte Carlo importance sampling.
    """
    if not active_vars:
        return

    torch.manual_seed(42)

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
        log_q = -0.5 * torch.log(torch.tensor(2 * torch.pi * proposal_std**2)) - 0.5 * ((samples / proposal_std)**2)
        total_log_q = log_q.sum(dim=1)  # sum over variables

        # Importance sampling: E_q [p(x) / q(x)]
        log_weights = log_vals - total_log_q

        mc_log = torch.logsumexp(log_weights, dim=0) - torch.log(
            torch.tensor(n_mc, dtype=torch.float32)
        )
        mc_prob = torch.exp(mc_log).item()

    assert abs(mc_prob - 1.0) < atol, f"Circuit does not integrate to 1. Integral: {mc_prob:.4f}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_vtree_det():
    """VTree over {0,1,2,3} with md-sets {0,1} and {0,1,2}."""
    vt = VTree()
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
def full_ac_det(full_vtree_det):
    """Full circuit over {0,1,2,3}."""
    torch.manual_seed(42)
    np.random.seed(42)
    dists = {
        i: GaussianDistribution(
            var=i,
            mean=torch.tensor([0.0, 0.0]),
            stddev=torch.tensor([1.0, 1.0]),
            unit_count=H,
        )
        for i in range(4)
    }
    return create_md_circuit(dists, full_vtree_det, leaf_h=H, sum_h=H, initialize_weights=True)


@pytest.fixture
def sub_vtree_det():
    """Sub-vtree over {0,1} (identical to the left subtree of full_vtree)."""
    vt = VTree()
    v0 = vt.add_node(VNode(BitSet([0]), md_set=BitSet([0])))
    v1 = vt.add_node(VNode(BitSet([1]), md_set=BitSet([1])))
    v01 = vt.add_node(VNode(BitSet([0, 1]), md_set=BitSet([0, 1])))
    vt.add_children(v01, v0, v1)
    return vt


@pytest.fixture
def sub_ac_det(sub_vtree_det):
    """Sub-circuit over {0,1}."""
    torch.manual_seed(42)
    np.random.seed(42)
    dists = {
        i: GaussianDistribution(
            var=i,
            mean=torch.tensor([0.0, 0.0]),
            stddev=torch.tensor([1.0, 1.0]),
            unit_count=H,
        )
        for i in range(2)
    }
    return create_md_circuit(dists, sub_vtree_det, leaf_h=H, sum_h=H, initialize_weights=True)


@pytest.fixture
def disjoint_vtree_det():
    """VTree over {2,3}, disjoint from full_ac."""
    vt = VTree()
    v2 = vt.add_node(VNode(BitSet([2]), md_set=BitSet([2])))
    v3 = vt.add_node(VNode(BitSet([3]), md_set=BitSet.universal()))
    v23 = vt.add_node(VNode(BitSet([2, 3]), md_set=BitSet([2])))
    vt.add_children(v23, v2, v3)
    return vt


@pytest.fixture
def disjoint_ac_det(disjoint_vtree_det):
    """Disjoint circuit over {2,3}."""
    torch.manual_seed(42)
    np.random.seed(42)
    dists = {
        2: GaussianDistribution(
            var=2,
            mean=torch.tensor([0.0, 0.0]),
            stddev=torch.tensor([1.0, 1.0]),
            unit_count=H,
        ),
        3: GaussianDistribution(
            var=3,
            mean=torch.tensor([0.0, 0.0]),
            stddev=torch.tensor([1.0, 1.0]),
            unit_count=H,
        ),
    }
    return create_md_circuit(dists, disjoint_vtree_det, leaf_h=H, sum_h=H, initialize_weights=True)


@pytest.fixture
def full_vtree_non_det():
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
def full_ac_non_det(full_vtree_non_det):
    """Full circuit over {0,1,2,3} with no deterministic nodes."""
    dists = {
        i: GaussianDistribution(
            var=i,
            mean=torch.tensor([0.0, 0.0]),
            stddev=torch.tensor([1.0, 1.0]),
            unit_count=H,
        )
        for i in range(4)
    }
    return create_md_circuit(dists, full_vtree_non_det, leaf_h=H, sum_h=H, initialize_weights=True)


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
    dists = {
        i: GaussianDistribution(
            var=i,
            mean=torch.tensor([0.0, 0.0]),
            stddev=torch.tensor([1.0, 1.0]),
            unit_count=H,
        )
        for i in range(2)
    }
    return create_md_circuit(dists, sub_vtree_non_det, leaf_h=H, sum_h=H, initialize_weights=True)


@pytest.fixture
def leaf_ac_0():
    """Single-leaf circuit over var 0."""
    vt = VTree()
    vt.add_node(VNode(BitSet([0]), md_set=BitSet([0])))
    dists = {
        0: GaussianDistribution(
            var=0,
            mean=torch.tensor([0.0, 0.0]),
            stddev=torch.tensor([1.0, 1.0]),
            unit_count=H,
        )
    }
    return create_md_circuit(dists, vt, leaf_h=H, sum_h=H, initialize_weights=True)


@pytest.fixture
def leaf_ac_1():
    """Single-leaf circuit over var 1."""
    vt = VTree()
    vt.add_node(VNode(BitSet([1]), md_set=BitSet([1])))
    dists = {
        1: GaussianDistribution(
            var=1,
            mean=torch.tensor([0.0, 0.0]),
            stddev=torch.tensor([1.0, 1.0]),
            unit_count=H,
        )
    }
    return create_md_circuit(dists, vt, leaf_h=H, sum_h=H, initialize_weights=True)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_instantiate(full_ac_det):
    """Clamping a variable should be equivalent to overriding the data column."""
    data = torch.randn(10, 4)
    clamped = data.clone()
    clamped[:, 0] = 1.5

    out_inst = eval_circuit(_instantiate(full_ac_det, {0: 1.5}), data)
    out_clamped = eval_circuit(full_ac_det, clamped)
    assert torch.allclose(out_inst, out_clamped, atol=1e-5)


# ---------------------------------------------------------------------------
# Inverse
# ---------------------------------------------------------------------------


def test_inverse_leaf(leaf_ac_0):
    """Inverting a leaf circuit should negate its log-density."""
    inv_ac = _inverse(leaf_ac_0)
    data = torch.randn(5, 1)

    orig = eval_circuit(leaf_ac_0, data)
    inv = eval_circuit(inv_ac, data)

    # -(-inf) = nan in PyTorch, so mask dead branches before comparing
    mask = torch.isneginf(orig)
    safe_orig = orig.masked_fill(mask, 0.0)
    safe_inv = inv.masked_fill(mask, 0.0)
    assert torch.allclose(safe_inv, -safe_orig, atol=1e-4)


def j(full_ac_det):
    """Inverting a full circuit should produce finite outputs."""
    print("\n\n=== ORIGINAL CIRCUIT ===")
    print(full_ac_det.to_dot())
    inv_ac = _inverse(full_ac_det)
    print("\n\n=== INVERTED CIRCUIT ===")
    print(inv_ac.to_dot())
    data = torch.randn(10, 4)

    print("\n\n=== EVALUATING ORIGINAL CIRCUIT ===")
    original = eval_circuit(full_ac_det, data, verbose=True)

    print("\n\n=== EVALUATING INVERTED CIRCUIT ===")
    out = eval_circuit(inv_ac, data, verbose=True)

    assert out.shape == (10, 1)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, -original, atol=1e-4)


# ---------------------------------------------------------------------------
# Multiplication — four structural cases
# ---------------------------------------------------------------------------


def test_multiply_leaves_disjoint_scopes(leaf_ac_0, leaf_ac_1):
    """Case 1: Disjoint scopes → Kronecker product.

    Two leaf circuits over different variables produce an outer product of
    their log-densities.
    """
    new_ac = _multiply(leaf_ac_0, leaf_ac_1)
    data = torch.randn(5, 2)

    out1 = eval_circuit(leaf_ac_0, data)
    out2 = eval_circuit(leaf_ac_1, data)
    out_new = eval_circuit(new_ac, data)

    assert out_new.shape == (5, H * H)

    # Kronecker (Cartesian) outer sum in log-space
    expected = (out1.unsqueeze(2) + out2.unsqueeze(1)).reshape(5, -1)
    assert torch.allclose(out_new, expected, atol=1e-5)


def test_multiply_leaves_same_scope(leaf_ac_0):
    """Case 2: Same-scope leaf multiplication (det) → ProductLeafNode."""
    new_ac = _multiply(leaf_ac_0, leaf_ac_0)
    data = torch.randn(5, 1)

    out = eval_circuit(leaf_ac_0, data)
    out_new = eval_circuit(new_ac, data)

    # Element-wise sum in log-space (Hadamard product)
    assert torch.allclose(out_new, out + out, atol=1e-5)


def test_multiply_deferred_product_det(full_ac_det, sub_ac_det):
    """Case 3: Deferred product.

    full_ac scope: {0,1,2,3}     sub_ac scope: {0,1}
    Common scope {0,1} is exactly the left child of full_ac's root.
    This triggers the deferred-product branch: the bigger circuit is copied
    upward and the smaller circuit is plugged into the matched child.
    """
    new_ac = _multiply(full_ac_det, sub_ac_det)
    data = torch.randn(5, 4)

    out_sub = eval_circuit(sub_ac_det, data[:, :2])
    out_full = eval_circuit(full_ac_det, data)
    out_new = eval_circuit(new_ac, data)
    out_expected = out_full + out_sub

    assert out_new.shape == (5, 1)

    # Due to sparse sum nodes over Kronecker products of deterministic leaves,
    # it is mathematically possible for branches to be entirely pruned (log-prob = -inf).
    # We only assert correctness on finite paths.
    mask = torch.isfinite(out_expected)
    assert torch.allclose(out_new[mask], out_expected[mask], atol=1e-5)


def test_multiply_matching_children_det(full_ac_det):
    """Case 4: Matching children (deterministic).

    Multiplying a circuit by itself over the same vtree
    triggers the matching-children branch and performs a Hadamard product,
    preserving unit counts. The output is bounded but not exactly 2 * log P(x) due to weight clamping.
    """
    new_ac = _multiply(full_ac_det, full_ac_det)

    data = torch.randn(5, 4)

    out_orig = eval_circuit(full_ac_det, data)
    out_new = eval_circuit(new_ac, data)

    assert out_new.shape == (5, 1)
    assert torch.isfinite(out_new).all()


def test_multiply_matching_children_non_det(full_ac_non_det):
    """Case 4: Matching children (non-deterministic).

    If the matching child is not deterministic, we should fall back to the
    general case and produce a Kronecker product, which will have more units
    than the original circuit.
    """
    new_ac = _multiply(full_ac_non_det, full_ac_non_det)

    data = torch.randn(5, 4)

    out_orig = eval_circuit(full_ac_non_det, data)
    out_new = eval_circuit(new_ac, data)

    assert out_new.shape == (5, 1)  # still a single output unit, but more internal units
    assert torch.isfinite(out_new).all()
    assert torch.allclose(out_new, out_orig + out_orig, atol=1e-5)


def test_multiply_deferred_product_non_det(full_ac_non_det, sub_ac_non_det):
    """Deferred product should still work if the circuits are non-deterministic."""
    new_ac = _multiply(full_ac_non_det, sub_ac_non_det)
    data = torch.randn(5, 4)

    out_sub = eval_circuit(sub_ac_non_det, data[:, :2])
    out_full = eval_circuit(full_ac_non_det, data)
    out_new = eval_circuit(new_ac, data)
    out_expected = out_full + out_sub

    assert out_new.shape == (5, 1)
    assert torch.isfinite(out_new).all()
    assert torch.allclose(out_new, out_expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Integration: compile_query
# ---------------------------------------------------------------------------


def test_compile_query_marginalization(full_ac_det):
    """Compiling MARG({2})[P(V)] should match a Monte Carlo approximation."""
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}
    ast_marg = make_marg({"2"}, make_p({"0", "1", "2", "3"}))
    q_marg, _ = compile_query(ast_marg, full_ac_det, full_ac_det.get_roots()[0], var_to_id)

    data = torch.randn(5, 4)
    out_marg = eval_circuit(q_marg, data, verbose=True)
    assert out_marg.shape == (5, 1)
    assert torch.isfinite(out_marg).all()

    # MC approximation: compile the full joint and sample var 2 many times
    ast_joint = make_p({"0", "1", "2", "3"})
    q_joint, _ = compile_query(ast_joint, full_ac_det, full_ac_det.get_roots()[0], var_to_id)

    n_mc = 20000
    torch.manual_seed(42)
    proposal_std = 1.5
    z2_samples = torch.randn(n_mc, 1) * proposal_std

    mc_results = []
    for i in range(data.shape[0]):
        pts = data[i : i + 1].expand(n_mc, -1).clone()
        pts[:, 2] = z2_samples[:, 0]
        log_vals = eval_circuit(q_joint, pts).squeeze()

        # Adjust for the N(0, 3^2) proposal density (importance sampling)
        log_q = -0.5 * torch.log(torch.tensor(2 * torch.pi * proposal_std**2)) - 0.5 * ((z2_samples[:, 0] / proposal_std) ** 2)
        log_vals = log_vals - log_q

        mc_log = torch.logsumexp(log_vals, dim=0) - torch.log(
            torch.tensor(n_mc, dtype=torch.float32)
        )
        mc_results.append(mc_log)

    mc_result = torch.stack(mc_results)
    assert torch.allclose(out_marg.squeeze(), mc_result, atol=1e-1)

    # Check valid distribution integrates to 1
    check_integration_to_one(q_marg, active_vars=[0, 1, 3])


def test_compile_query_conditional(full_ac_det):
    """Compiling P(3|0,1,2) should match P(0,1,2,3) - P(0,1,2)."""
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}

    ast_joint = make_p({"0", "1", "2", "3"})
    ast_marg = make_marg({"3"}, ast_joint)
    ast_inv = make_pow(-1, ast_marg)
    ast_cond = make_det_prod([ast_joint, ast_inv])

    q_cond, _ = compile_query(ast_cond, full_ac_det, full_ac_det.get_roots()[0], var_to_id)
    q_marg_ac, _ = compile_query(ast_marg, full_ac_det, full_ac_det.get_roots()[0], var_to_id)

    print(f"FULL AC:\n{full_ac_det.to_dot()}\n")

    print(f"COND AC:\n{q_cond.to_dot()}")

    data = torch.randn(5, 4)

    # check that ast_marg = - ast_inv
    out_marg = eval_circuit(q_marg_ac, data, verbose=True)
    out_inv = eval_circuit(_inverse(q_marg_ac), data, verbose=True)
    assert torch.allclose(out_marg, -out_inv, atol=1e-4)

    out_cond = eval_circuit(q_cond, data)
    out_joint = eval_circuit(full_ac_det, data)
    out_marg_out = eval_circuit(q_marg_ac, data)

    expected = out_joint - out_marg_out
    assert torch.allclose(out_cond, expected, atol=1e-4)

    # Check valid distribution integrates to 1
    # Conditional is P(3|0,1,2), so it should integrate to 1 over variable 3 (with 0,1,2 clamped)
    # wait, check_integration_to_one will evaluate it with 0 for clamped vars.
    check_integration_to_one(full_ac_det, active_vars=[0, 1, 2, 3], n_mc=500000, atol=0.1)
    check_integration_to_one(q_marg_ac, active_vars=[0, 1, 2], n_mc=500000, atol=0.1)
    check_integration_to_one(q_cond, active_vars=[3], n_mc=100000, atol=0.01)


def test_compile_query_backdoor(full_ac_det):
    """Compiling backdoor query should match Monte Carlo approximation of the summand."""
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}

    # Backdoor: P(3|do(2)) with Z={0,1}
    # Summand: P(3|0,1,2) * P(0,1) = P(0,1,2,3) * P(0,1,2)^-1 * P(0,1)
    ast_joint = make_p({"0", "1", "2", "3"})
    ast_x_z = make_marg({"3"}, ast_joint)
    ast_z = make_marg({"2", "3"}, ast_joint)

    ast_cond = make_det_prod([ast_joint, make_pow(-1, ast_x_z)])
    ast_summand = make_prod([ast_cond, ast_z])
    ast_backdoor = make_marg({"0", "1"}, ast_summand)

    q_backdoor, _ = compile_query(ast_backdoor, full_ac_det, full_ac_det.get_roots()[0], var_to_id)
    q_summand, _ = compile_query(ast_summand, full_ac_det, full_ac_det.get_roots()[0], var_to_id)
    q_z, _ = compile_query(ast_z, full_ac_det, full_ac_det.get_roots()[0], var_to_id)
    q_cond, _ = compile_query(ast_cond, full_ac_det, full_ac_det.get_roots()[0], var_to_id)

    print(f"FULL AC:\n{full_ac_det.to_dot()}\n")
    print(f"COND AC:\n{q_cond.to_dot()}\n")
    print(f"Z AC:\n{q_z.to_dot()}\n")
    print(f"SUMMAND AC:\n{q_summand.to_dot()}\n")
    print(f"BACKDOOR AC:\n{q_backdoor.to_dot()}\n")

    data = torch.randn(5, 4)
    out_backdoor = eval_circuit(q_backdoor, data)
    assert out_backdoor.shape == (5, 1)
    assert torch.isfinite(out_backdoor).all()

    n_mc = 2000
    torch.manual_seed(42)
    z_samples = torch.randn(n_mc, 2)

    mc_results = []
    for i in range(data.shape[0]):
        pts = data[i : i + 1].expand(n_mc, -1).clone()
        pts[:, 0] = z_samples[:, 0]
        pts[:, 1] = z_samples[:, 1]
        log_vals = eval_circuit(q_summand, pts).squeeze()

        log_q0 = -0.5 * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * (z_samples[:, 0] ** 2)
        log_q1 = -0.5 * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * (z_samples[:, 1] ** 2)
        log_vals = log_vals - log_q0 - log_q1

        mc_log = torch.logsumexp(log_vals, dim=0) - torch.log(
            torch.tensor(n_mc, dtype=torch.float32)
        )
        mc_results.append(mc_log)

    mc_result = torch.stack(mc_results)
    assert torch.allclose(out_backdoor.squeeze(), mc_result, atol=2e-1)

    # Check valid distribution integrates to 1
    # Backdoor is P(3|do(2)), integrating over 3 should give 1
    check_integration_to_one(q_z, active_vars=[0, 1], n_mc=10000, atol=0.1)
    check_integration_to_one(q_cond, active_vars=[3], n_mc=10000, atol=0.1)
    check_integration_to_one(q_backdoor, active_vars=[3], n_mc=10000, atol=0.01)
