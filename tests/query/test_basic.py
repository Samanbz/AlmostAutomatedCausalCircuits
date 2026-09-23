"""Basic query-compiler operations: instantiation and marginal compilation."""

import torch
from conftest import eval_circuit

from src.symbolic.arithmetic.query import _instantiate, compile_query
from src.symbolic.id_ast import make_marg, make_p


def test_instantiate(ac_xz):
    """Clamping a variable should be equivalent to overriding the data column."""
    data = torch.randn(10, 4)
    clamped = data.clone()
    clamped[:, 0] = 1.5

    out_inst = eval_circuit(_instantiate(ac_xz, {0: 1.5}), data)
    out_clamped = eval_circuit(ac_xz, clamped)
    assert torch.allclose(out_inst, out_clamped, atol=1e-5)


def test_marg_z1_numerical(ac_z_xy):
    """Compiling P(Z) should match a Monte Carlo approximation."""
    torch.manual_seed(42)
    var_to_id = {"Z": 0, "X": 1, "Y": 2}
    ast_joint = make_p({"Z", "X", "Y"})
    ast_marg = make_marg({"Z"}, ast_joint)

    q_marg = compile_query(ast_marg, ac_z_xy, var_to_id)
    q_joint = compile_query(ast_joint, ac_z_xy, var_to_id)

    data = torch.randn(5, 3)  # rows of (Z, X, Y)
    out_marg = eval_circuit(q_marg, data)
    assert out_marg.shape == (5, 1, 1)

    # MC approximation: compile the full joint and sample var Z many times
    n_mc = 2000000
    proposal_std = 1.5
    z_samples = torch.randn(n_mc, 1) * proposal_std

    mc_results = []
    for i in range(data.shape[0]):
        pts = data[i : i + 1].expand(n_mc, -1).clone()
        pts[:, 0] = z_samples[:, 0]
        log_vals = eval_circuit(q_joint, pts).squeeze()

        # Adjust for the N(0, 1.5^2) proposal density (importance sampling)
        log_q = -0.5 * torch.log(torch.tensor(2 * torch.pi * proposal_std**2)) - 0.5 * (
            (z_samples[:, 0] / proposal_std) ** 2
        )
        log_vals = log_vals - log_q

        mc_log = torch.logsumexp(log_vals, dim=0) - torch.log(
            torch.tensor(n_mc, dtype=torch.float32)
        )
        mc_results.append(mc_log)

    mc_result = torch.stack(mc_results)
    diff = torch.abs(out_marg.squeeze() - mc_result)
    print(f"Diff:\n{diff}")
    assert torch.allclose(out_marg.squeeze(), mc_result, atol=1e-3)


def test_marg_numerical(ac_xz):
    """Compiling MARG({2})[P(V)] should match a Monte Carlo approximation."""
    torch.manual_seed(42)
    var_to_id = {"0": 0, "1": 1, "2": 2, "3": 3}
    ast_marg = make_marg({"2"}, make_p({"0", "1", "2", "3"}))
    q_marg = compile_query(ast_marg, ac_xz, var_to_id)

    data = torch.randn(5, 4)
    out_marg = eval_circuit(q_marg, data)
    assert out_marg.shape == (5, 1, 1)
    assert torch.isfinite(out_marg).all()

    # MC approximation: compile the full joint and sample var 2 many times
    ast_joint = make_p({"0", "1", "2", "3"})
    q_joint = compile_query(ast_joint, ac_xz, var_to_id)

    # 400k samples leaves IS error right at the atol=1e-3 boundary (the exact
    # marginal is verified: error scales as 1/sqrt(n_mc)); use enough samples
    # for a stable margin.
    n_mc = 3_200_000
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
    assert torch.allclose(out_marg.squeeze(), mc_result, atol=1e-3)
