"""Compositional query tests: products of queries, dependency structure,
interventional vs observational distinctness, and diagnostic (skipped) tests."""

import torch
from conftest import check_integration_to_one, eval_circuit

from src.symbolic.arithmetic.query import compile_query
from src.symbolic.id_ast import make_cond, make_marg, make_p, make_prod


def test_backdoor_inner_product(ac_z_xy):
    torch.manual_seed(42)
    var_to_id = {"Z": 0, "X": 1, "Y": 2}

    ast_joint = make_p({"Z", "X", "Y"})  # P(Z,X,Y)
    ast_cond = make_cond({"Y"}, {"Z", "X"}, ast_joint)  # P(Y|Z,X)
    ast_marg = make_marg({"Y", "X"}, ast_joint)  # P(Z)
    ast_prod = make_prod([ast_cond, ast_marg])  # P(Y|Z,X) * P(Z)

    q_cond = compile_query(ast_cond, ac_z_xy, var_to_id)
    q_marg = compile_query(ast_marg, ac_z_xy, var_to_id)
    q_prod = compile_query(ast_prod, ac_z_xy, var_to_id)

    data = torch.randn(20, 3)  # rows of (Z, X, Y)
    out_cond = eval_circuit(q_cond, data)
    out_marg = eval_circuit(q_marg, data)
    out_prod = eval_circuit(q_prod, data)

    expected = out_cond + out_marg

    # diff = out_prod - expected
    assert torch.allclose(out_prod, expected, atol=1e-5)


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

    q_z = compile_query(ast_z, ac_xz, var_to_id)
    q_backdoor = compile_query(ast_backdoor, ac_xz, var_to_id)

    data = torch.randn(10, 4)
    out_backdoor = eval_circuit(q_backdoor, data)
    assert out_backdoor.shape == (10, 1, 1)
    assert torch.isfinite(out_backdoor).all()

    ast_xz = make_marg({"3"}, ast_joint)
    q_xz = compile_query(ast_xz, ac_xz, var_to_id)

    n_mc = 20000
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
    print(f"Diff:\n{torch.abs(out_backdoor.squeeze() - mc_result)}")
    assert torch.allclose(out_backdoor.squeeze(), mc_result, atol=2e-1)

    return q_backdoor


def test_backdoor_numerical(ac_xz):
    check_backdoor_numerical(ac_xz)


def test_trained_backdoor_numerical(trained_ac_xz_and_scm):
    ac, scm = trained_ac_xz_and_scm
    q_backdoor = check_backdoor_numerical(ac)
    check_integration_to_one(q_backdoor, active_vars=[3], n_mc=100000, atol=0.01)


def test_backdoor_do_x_dependence(trained_ac_xz_and_scm):
    """P(Y|do(X)) must depend on X: at fixed Y, different X values must give
    different densities. A previous bug made the compiled do-density
    bit-exactly independent of X."""
    torch.manual_seed(42)
    ac, scm = trained_ac_xz_and_scm
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}

    ast_joint = make_p({"0", "1", "2", "3"})
    ast_z = make_marg({"2", "3"}, ast_joint)  # P(Z)
    ast_cond = make_cond({"3"}, {"0", "1", "2"}, ast_joint)  # P(Y|Z,X)
    ast_summand = make_prod([ast_cond, ast_z])
    ast_backdoor = make_marg({"0", "1"}, ast_summand)  # P(Y|do(X))

    q_backdoor = compile_query(ast_backdoor, ac, var_to_id)

    # Sweep X (col 2) at a few fixed Y values (col 3).
    n_x = 8
    x_vals = torch.linspace(-2.0, 2.0, n_x)
    outs = []
    for y_val in (-1.0, 0.0, 1.0):
        data = torch.zeros(n_x, 4)
        data[:, 2] = x_vals
        data[:, 3] = y_val
        out = eval_circuit(q_backdoor, data).squeeze()
        assert torch.isfinite(out).all()
        outs.append(out)

    for y_idx, out in enumerate(outs):
        spread = out.max() - out.min()
        assert spread > 1e-3, f"P(Y|do(X)) is independent of X at Y index {y_idx}: {out}"
