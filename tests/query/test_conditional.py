"""Conditional compilation tests: ratio identities and the backdoor estimand."""

import math

import torch
from conftest import eval_circuit

from src.symbolic.arithmetic.query import compile_query
from src.symbolic.id_ast import make_cond, make_marg, make_p


def test_conditional_identities(ac_zxz):
    """
    Ratio identities compiled from the vtree_xz circuit (which represents P(Z1,Z2,X,Y)):

      1. P(Y|X,Z1,Z2) == P(Z1,Z2,X,Y) / P(X,Z1,Z2)
      2. P(Y,X|Z1,Z2) == P(Z1,Z2,X,Y) / P(Z1,Z2)

    The joint and both marginals are compiled as query circuits from the same
    base circuit, then compared pointwise on random data.
    """
    torch.manual_seed(42)
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}

    ast_joint = make_p({"0", "1", "2", "3"})

    # Joint P(Z1,Z2,X,Y) and marginals P(X,Z1,Z2), P(Z1,Z2)
    q_joint = compile_query(ast_joint, ac_zxz, var_to_id)
    q_xz = compile_query(make_marg({"3"}, ast_joint), ac_zxz, var_to_id)
    q_z = compile_query(make_marg({"2", "3"}, ast_joint), ac_zxz, var_to_id)

    # Compiled conditional queries
    print("\nCompiling P(Y|X,Z1,Z2)")

    q_cond_yxz = compile_query(make_cond({"3"}, {"0", "1", "2"}, ast_joint), ac_zxz, var_to_id)
    print("\nCompiling P(Y,X|Z1,Z2)")
    q_cond_yx_z = compile_query(make_cond({"2", "3"}, {"0", "1"}, ast_joint), ac_zxz, var_to_id)

    data = torch.randn(5, 4)
    out_joint = eval_circuit(q_joint, data)
    out_xz = eval_circuit(q_xz, data)
    out_z = eval_circuit(q_z, data)

    out_cond_yxz = eval_circuit(q_cond_yxz, data)
    out_cond_yx_z = eval_circuit(q_cond_yx_z, data)

    expected_yxz = out_joint - out_xz
    expected_yx_z = out_joint - out_z

    diff_yxz = torch.abs(out_cond_yxz - expected_yxz)
    diff_yx_z = torch.abs(out_cond_yx_z - expected_yx_z)
    print(f"Diff P(Y|X,Z1,Z2):\n{diff_yxz}\nDiff P(Y,X|Z1,Z2):\n{diff_yx_z}")

    assert torch.allclose(out_cond_yxz, out_joint - out_xz, atol=1e-5)
    assert torch.allclose(out_cond_yx_z, out_joint - out_z, atol=1e-5)


def check_conditional_dependencies(ac, data):
    """
    Ensure that P(Y|X,Z) != P(Y|X) and P(Y|X,Z) != P(Y|Z).
    Since the circuit is not MD with respect to X or Z alone, we compute P(Y|X) and P(Y|Z)
    via joint and marg evaluation and subtraction.
    """
    torch.manual_seed(42)
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}
    # print(f"Input:\n{data}")

    ast_joint = make_p({"0", "1", "2", "3"})

    # 1. P(Y|X,Z) using direct compilation
    ast_cond_yxz = make_cond({"3"}, {"0", "1", "2"}, ast_joint)
    q_cond_yxz = compile_query(ast_cond_yxz, ac, var_to_id)

    # 2. P(Y|X) = P(Y,X) / P(X)
    ast_yx = make_marg({"0", "1"}, ast_joint)
    ast_x = make_marg({"0", "1", "3"}, ast_joint)
    q_yx = compile_query(ast_yx, ac, var_to_id)
    q_x = compile_query(ast_x, ac, var_to_id)

    # 3. P(Y|Z) = P(Y,Z) / P(Z)  (where Z = {0, 1})
    ast_yz = make_marg({"2"}, ast_joint)
    ast_z = make_marg({"2", "3"}, ast_joint)
    q_yz = compile_query(ast_yz, ac, var_to_id)
    q_z = compile_query(ast_z, ac, var_to_id)

    eval_circuit(ac, data)

    out_cond_yxz = eval_circuit(q_cond_yxz, data, verbose=True, show_weights=True, log_domain=False)

    out_yx = eval_circuit(q_yx, data)
    out_x = eval_circuit(q_x, data)
    out_cond_yx = out_yx - out_x

    out_yz = eval_circuit(q_yz, data)
    out_z = eval_circuit(q_z, data)
    out_cond_yz = out_yz - out_z

    # Check that they are not equal, accumulating errors so we see all failures
    errors = []
    if torch.all(torch.abs(out_cond_yxz - out_cond_yx) < 1e-3):
        errors.append("P(Y|X,Z) is equal to P(Y|X)")
    if torch.all(torch.abs(out_cond_yxz - out_cond_yz) < 1e-3):
        errors.append("P(Y|X,Z) is equal to P(Y|Z)")

    assert not errors, f"Conditional dependencies missing: {errors}"


def test_conditional_dependencies(trained_ac_xz_and_scm):
    ac, scm = trained_ac_xz_and_scm
    check_conditional_dependencies(ac, data=torch.tensor(scm.sample(10).values))


def test_conditional_dependencies_skewed(trained_ac_xz_skewed_and_scm):
    ac, scm = trained_ac_xz_skewed_and_scm
    check_conditional_dependencies(ac, data=torch.tensor(scm.sample(10).values))


def test_conditional_dependencies_numerical(trained_ac_xz_and_scm):
    """
    Evaluates P(Y|X,Z) while fixing Y and X, and varying Z across its unit supports.
    """

    trained_ac_xz, _ = trained_ac_xz_and_scm

    torch.manual_seed(42)
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}

    ast_joint = make_p({"0", "1", "2", "3"})
    ast_cond_yxz = make_cond({"3"}, {"0", "1", "2"}, ast_joint)
    q_cond_yxz = compile_query(ast_cond_yxz, trained_ac_xz, var_to_id)

    z0_pts = []
    z1_pts = []
    for leaf_id in trained_ac_xz.get_leaves():
        leaf = trained_ac_xz.get_node_data(leaf_id)
        var = getattr(leaf, "var", None)
        if var not in (0, 1):
            continue
        assert leaf.node_supports is not None, f"Leaf {leaf_id} has no unit supports"
        pts = z0_pts if var == 0 else z1_pts
        for iv in leaf.node_supports:
            if iv is None:
                continue
            low = -3.0 if math.isinf(iv.low) else iv.low
            high = 3.0 if math.isinf(iv.high) else iv.high
            mid = (low + high) / 2.0
            if mid not in pts:
                pts.append(mid)

    pts = []
    for z0 in z0_pts:
        for z1 in z1_pts:
            pts.append([z0, z1, 1.0, 1.0])  # Z0, Z1, X=1.0, Y=1.0

    data = torch.tensor(pts, dtype=torch.float32)
    out_cond_yxz = eval_circuit(q_cond_yxz, data)

    # Check if the conditional probability changes as Z varies
    # We assert that it actually varies
    assert out_cond_yxz.std() > 1e-4, (
        f"P(Y|X,Z) did not change with Z! outputs: {out_cond_yxz.flatten()}"
    )
