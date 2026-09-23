"""Structural product (_multiply) tests: the four structural cases."""

import torch
from conftest import eval_circuit

from src.symbolic.arithmetic.query import _multiply, compile_query
from src.symbolic.id_ast import make_cond, make_marg, make_p, make_prod


def test_prod_operand_order_commutes(ac_z_xy):
    """PROD is mathematically commutative: the frontdoor outer-sum estimand

        int_M [ P(M|X) * int_X{P(Y|X,M) * P(X)} ]

    must compile to the same circuit regardless of the inner PROD's operand
    order (conditional-first vs marginal-first, as emitted by a hand-written
    AST vs T-ID's COAST). Regression canary for the deferred-product (Case 3)
    layout convention.
    """
    torch.manual_seed(42)
    var_to_id = {"X": 0, "M": 1, "Y": 2}

    def build(order):
        ast_joint = make_p({"X", "M", "Y"})
        ast_cond_y = make_cond({"Y"}, {"X", "M"}, ast_joint)  # P(Y|X,M)
        ast_px = make_marg({"Y", "M"}, ast_joint)  # P(X)
        if order == "cond_first":
            ast_prod_inner = make_prod([ast_cond_y, ast_px])
        else:
            ast_prod_inner = make_prod([ast_px, ast_cond_y])
        ast_inner = make_marg({"X"}, ast_prod_inner)
        ast_xm = make_marg({"Y"}, ast_joint)  # P(X,M)
        ast_cond_m = make_cond({"M"}, {"X"}, ast_xm)  # P(M|X)
        ast_prod = make_prod([ast_cond_m, ast_inner])
        return make_marg({"M"}, ast_prod)

    q_cond_first = compile_query(build("cond_first"), ac_z_xy, var_to_id)
    q_marg_first = compile_query(build("marg_first"), ac_z_xy, var_to_id)

    data = torch.randn(20, 3)  # rows of (X, M, Y)
    out_a = eval_circuit(q_cond_first, data)
    out_b = eval_circuit(q_marg_first, data)
    assert torch.allclose(out_a, out_b, atol=1e-5), (
        f"PROD operand order changed the compiled product: "
        f"max|diff| = {(out_a - out_b).abs().max()}"
    )


def test_multiply_z1(ac_z_xy):
    torch.manual_seed(42)
    var_to_id = {"Z": 0, "X": 1, "Y": 2}

    ast_joint = make_p({"Z", "X", "Y"})
    ast_pz = make_marg({"Y", "X"}, ast_joint)  # P(Z)
    ast_prod = make_prod([ast_joint, ast_pz])  # P(Y,X,Z) * P(Z)

    q_prod = compile_query(ast_prod, ac_z_xy, var_to_id)
    q_joint = compile_query(ast_joint, ac_z_xy, var_to_id)
    q_z = compile_query(ast_pz, ac_z_xy, var_to_id)

    data = torch.randn(20, 3)  # rows of (Z, X, Y)
    print("\nJOINT\n")
    out_joint = eval_circuit(q_joint, data)
    print("\nPROD\n")
    out_prod = eval_circuit(q_prod, data, verbose=True, show_weights=True, log_domain=False)
    out_z = eval_circuit(q_z, data)

    expected = out_joint + out_z

    diff = out_prod - expected
    print(f"diff:\n{diff}")
    assert out_prod.shape == (20, 1, 1)
    assert torch.allclose(out_prod, expected, atol=1e-5)


def check_deferred(ac, sub_ac_z):
    """Case 3: Deferred product.

    ac_xz scope: {Z1,Z2,X,Y}     sub_ac_z scope: {Z1,Z2}
    Common scope {Z1,Z2} is exactly the left child of ac's root.
    This triggers the deferred-product branch: the bigger circuit is copied
    upward and the smaller circuit is plugged into the matched child.
    """

    new_ac = _multiply(ac, sub_ac_z)

    data = torch.randn(5, 4)

    out_sub = eval_circuit(sub_ac_z, data[:, :2])
    out_full = eval_circuit(ac, data)
    out_new = eval_circuit(new_ac, data)
    out_expected = out_full + out_sub

    assert out_new.shape == (5, 1, 1)

    assert torch.allclose(out_new, out_expected, atol=1e-5)


def test_multiply_deferred_xz(ac_xz, sub_ac_z_det):
    check_deferred(ac_xz, sub_ac_z_det)


def test_multiply_deferred_zxz(ac_zxz, sub_ac_z_det):
    check_deferred(ac_zxz, sub_ac_z_det)


def test_multiply_deferred_xz_skewed(ac_xz_skewed, sub_ac_z_det):
    check_deferred(ac_xz_skewed, sub_ac_z_det)


def test_multiply_deferred_non_det(ac_non_det, sub_ac_z_non_det):
    check_deferred(ac_non_det, sub_ac_z_non_det)


def check_matching_children(ac):
    """Case 4: Matching children (deterministic).

    Multiplying a circuit by itself over the same vtree
    triggers the matching-children branch and performs a Hadamard product,
    preserving unit counts. The output is bounded but not exactly 2 * log P(x) due to weight clamping.
    """
    new_ac = _multiply(ac, ac)

    data = torch.randn(5, 4)

    out_orign = eval_circuit(ac, data)
    out_new = eval_circuit(new_ac, data)
    expected = out_orign + out_orign

    assert out_new.shape == (5, 1, 1)
    assert torch.isfinite(out_new).all()
    assert torch.allclose(out_new, expected, atol=1e-5)


def test_multiply_matching_children_xz(ac_xz):
    check_matching_children(ac_xz)


def test_multiply_matching_children_zxz(ac_zxz):
    check_matching_children(ac_zxz)


def test_multiply_matching_children_zxz_skewed(ac_xz_skewed):
    check_matching_children(ac_xz_skewed)


def test_multiply_matching_children_non_det(ac_non_det):
    check_matching_children(ac_non_det)


def test_trained_multiply_matching_children(trained_ac_xz_and_scm):
    check_matching_children(trained_ac_xz_and_scm[0])
