"""Compositional query tests: products of queries and marginalization over them.

These cover the COAST shape that appears in identified interventional
estimands — a product of a conditional and a marginal, with a further
marginalization applied on top — and check it against numerical integration
over the marginalized variable.
"""

import pytest
import torch
from conftest import check_integration_to_one, eval_circuit

from src.symbolic.arithmetic.query import compile_query
from src.symbolic.id_ast import make_cond, make_marg, make_p, make_prod


def test_frontdoor_inner_product_numerical(ac_z_xy):
    # test
    torch.manual_seed(42)
    var_to_id = {"X": 0, "M": 1, "Y": 2}

    ast_joint = make_p({"X", "M", "Y"})
    ast_cond = make_cond({"Y"}, {"X", "M"}, ast_joint)  # P(Y|X,M)
    ast_px = make_marg({"Y", "M"}, ast_joint)  # P(X)
    ast_prod = make_prod([ast_cond, ast_px])  # P(Y|X,M) * P(X)

    q_prod = compile_query(ast_prod, ac_z_xy, var_to_id)
    q_cond = compile_query(ast_cond, ac_z_xy, var_to_id)
    q_x = compile_query(ast_px, ac_z_xy, var_to_id)

    data = torch.randn(5, 3)  # rows of (Z, X, Y)
    out_prod = eval_circuit(q_prod, data)
    out_cond = eval_circuit(q_cond, data)
    out_x = eval_circuit(q_x, data)

    expected = out_cond + out_x
    diff = torch.abs(out_prod - expected)
    print(f"Diff P(Y|X,M)P(X) vs P(Y|X,M) + P(X):\n{diff}")
    assert torch.allclose(out_prod, expected, atol=1e-5), (
        f"Compiled product does not match P(Y|X,M)P(X) = P(Y|X,M) + P(X). Diff: {diff}"
    )


def test_frontdoor_inner_sum_numerical(ac_z_xy):
    torch.manual_seed(42)
    var_to_id = {"X": 0, "M": 1, "Y": 2}

    ast_joint = make_p({"X", "M", "Y"})
    ast_cond = make_cond({"Y"}, {"X", "M"}, ast_joint)  # P(Y|X,M)
    ast_px = make_marg({"Y", "M"}, ast_joint)  # P(X)
    ast_prod = make_prod([ast_cond, ast_px])  # P(Y|X,M) * P(X)
    ast_marg = make_marg({"X"}, ast_prod)  # sum_X P(Y|X,M) * P(X)

    q_prod = compile_query(ast_prod, ac_z_xy, var_to_id)
    q_marg = compile_query(ast_marg, ac_z_xy, var_to_id)

    data = torch.randn(5, 3)  # rows of (Z, X, Y)

    # --- Check 1: the compiled marginal is invariant to the X column. ---
    shifted = data.clone()
    shifted[:, 0] += 3.14159
    out_marg = eval_circuit(q_marg, data)
    out_marg_shifted = eval_circuit(q_marg, shifted)
    assert torch.allclose(out_marg, out_marg_shifted, atol=1e-5), (
        "MARG(X)[...] still depends on the X column — X was not marginalized."
    )

    # --- Check 2: match numerical integration over X. ---
    n_mc = 10000
    proposal_std = 1.5
    x_samples = torch.randn(n_mc, 1) * proposal_std
    log_q = -0.5 * torch.log(torch.tensor(2 * torch.pi * proposal_std**2)) - 0.5 * (
        (x_samples[:, 0] / proposal_std) ** 2
    )

    mc_results = []
    for i in range(data.shape[0]):
        pts = data[i : i + 1].expand(n_mc, -1).clone()
        pts[:, 0] = x_samples[:, 0]
        log_vals = eval_circuit(q_prod, pts).squeeze() - log_q
        mc_log = torch.logsumexp(log_vals, dim=0) - torch.log(
            torch.tensor(n_mc, dtype=torch.float32)
        )
        mc_results.append(mc_log)
    mc_result = torch.stack(mc_results)

    diff = torch.abs(out_marg.squeeze() - mc_result)
    print(f"mc_result:\n{mc_result}")
    print(f"out_marg:\n{out_marg.squeeze()}")
    print(f"Diff MARG(X)[P(Y|X,Z)P(Z)] vs numerical integration:\n{diff}")
    assert torch.allclose(out_marg.squeeze(), mc_result, atol=1e-1), (
        f"Compiled marginal does not match numerical integration over X. Diff: {diff}"
    )

    # --- Check 3: normalized in Y for a fixed M ---
    # int_Y [ int_X P(Y|X,M) P(X) dX ] dY = 1 for every M.
    check_integration_to_one(q_marg, active_vars=[2])


def test_frontdoor_outer_sum_numerical(ac_z_xy):
    """
    The full frontdoor summation

        int_M [ P(M|X) * int_X{P(Y|X,M) * P(X)} ]

    must equal a Monte-Carlo integration over M of the product of the two
    separately compiled (and previously verified) factor circuits
    q_cond = P(M|X) and q_inner = int_X{P(Y|X,M) * P(X)}:

        compiled(x, y)  ==  logsumexp_m [ log q_cond(x, m) + log q_inner(m, y) - log q(m) ] - log n

    This exercises the outer MARG(M)[PROD(...)] composition — the shape of the
    identified frontdoor estimand P(Y|do(X)).
    """
    torch.manual_seed(42)
    var_to_id = {"X": 0, "M": 1, "Y": 2}

    ast_joint = make_p({"X", "M", "Y"})
    ast_cond_y = make_cond({"Y"}, {"X", "M"}, ast_joint)  # P(Y|X,M)
    ast_px = make_marg({"Y", "M"}, ast_joint)  # P(X)
    ast_prod_inner = make_prod([ast_cond_y, ast_px])  # P(Y|X,M) * P(X)
    ast_inner = make_marg({"X"}, ast_prod_inner)  # int_X P(Y|X,M) * P(X)
    ast_xm = make_marg({"Y"}, ast_joint)  # P(X,M)
    ast_cond_m = make_cond({"M"}, {"X"}, ast_xm)  # P(M|X)
    ast_prod = make_prod([ast_cond_m, ast_inner])  # P(M|X) * int_X{...}
    ast_outer = make_marg({"M"}, ast_prod)  # int_M P(M|X) * int_X{...}

    q_inner = compile_query(ast_inner, ac_z_xy, var_to_id)
    q_cond = compile_query(ast_cond_m, ac_z_xy, var_to_id)
    q_outer = compile_query(ast_outer, ac_z_xy, var_to_id)

    data = torch.randn(5, 3)  # rows of (X, M, Y)

    # --- Check 1: the compiled outer sum is invariant to the M column. ---
    shifted = data.clone()
    shifted[:, 1] += 3.14159
    out_outer = eval_circuit(q_outer, data)
    out_outer_shifted = eval_circuit(q_outer, shifted)
    assert torch.allclose(out_outer, out_outer_shifted, atol=1e-5), (
        "MARG(M)[...] still depends on the M column — M was not marginalized."
    )

    # --- Check 2: match numerical integration over M of the product factors. ---
    n_mc = 10000
    proposal_std = 1.5
    m_samples = torch.randn(n_mc, 1) * proposal_std
    log_q = -0.5 * torch.log(torch.tensor(2 * torch.pi * proposal_std**2)) - 0.5 * (
        (m_samples[:, 0] / proposal_std) ** 2
    )

    mc_results = []
    for i in range(data.shape[0]):
        pts = data[i : i + 1].expand(n_mc, -1).clone()
        pts[:, 1] = m_samples[:, 0]
        log_vals = (
            eval_circuit(q_cond, pts).squeeze() + eval_circuit(q_inner, pts).squeeze() - log_q
        )
        mc_log = torch.logsumexp(log_vals, dim=0) - torch.log(
            torch.tensor(n_mc, dtype=torch.float32)
        )
        mc_results.append(mc_log)
    mc_result = torch.stack(mc_results)

    diff = torch.abs(out_outer.squeeze() - mc_result)
    print(f"mc_result:\n{mc_result}")
    print(f"out_outer:\n{out_outer.squeeze()}")
    print(f"Diff int_M[P(M|X) * int_X{{P(Y|X,M)P(X)}}] vs numerical integration:\n{diff}")
    assert torch.allclose(out_outer.squeeze(), mc_result, atol=1e-1), (
        f"Compiled outer sum does not match numerical integration over M. Diff: {diff}"
    )

    # --- Check 3: normalized in Y for a fixed X — this IS P(Y|do(X)) ---
    # int_Y P(Y|do(X)) dY = 1 for every X.
    check_integration_to_one(q_outer, active_vars=[2])


def test_m_cond_x(ac_z_xy):
    torch.manual_seed(42)
    var_to_id = {"X": 0, "M": 1, "Y": 2}

    ast_joint = make_p({"X", "M", "Y"})
    ast_xm = make_marg({"Y"}, ast_joint)  # P(X,M)
    ast_x = make_marg({"M", "Y"}, ast_joint)  # P(X)
    ast_cond = make_cond({"M"}, {"X"}, ast_xm)  # P(M|X)

    q_cond = compile_query(ast_cond, ac_z_xy, var_to_id)
    q_xm = compile_query(ast_xm, ac_z_xy, var_to_id)
    q_x = compile_query(ast_x, ac_z_xy, var_to_id)

    data = torch.randn(5, 3)  # rows of (Z, X, Y)
    out_cond = eval_circuit(q_cond, data)
    out_xm = eval_circuit(q_xm, data)
    out_x = eval_circuit(q_x, data)

    out_expected = out_xm - out_x
    diff = torch.abs(out_cond - out_expected)
    print(f"Diff P(M|X) vs P(X,M)/P(X):\n{diff}")
    assert torch.allclose(out_cond, out_expected, atol=1e-5), (
        f"Compiled conditional does not match P(M|X) = P(X,M)/P(X). Diff: {diff}"
    )


def _sweep_spreads(q, sweep_col: int, fixed_cols: list[int], n: int = 8) -> list[float]:
    """Evaluate q on a grid sweeping one column over [-2, 2] at a few fixed
    values of the other columns; return the per-(fixed-values) spread
    (max - min) of the log-output."""
    vals = torch.linspace(-2.0, 2.0, n)
    spreads = []
    for fixed_val in (-1.0, 0.0, 1.0):
        data = torch.zeros(n, 3)
        data[:, sweep_col] = vals
        for c in fixed_cols:
            data[:, c] = fixed_val
        out = eval_circuit(q, data).squeeze()
        assert torch.isfinite(out).all()
        spreads.append((out.max() - out.min()).item())
    return spreads


def test_frontdoor_component_dependence(trained_ac_z_xy_and_scm):
    """Every intermediate component of the frontdoor estimand

        int_M P(M|X) * int_X' P(Y|X',M) P(X')

    must retain the dependence structure of the true frontdoor SCM (hidden U;
    X -> M -> Y; U -> X, Y) through query compilation:

      - P(Y|X,M):            X-dependent, M-dependent
      - P(Y|X,M)*P(X):       X-dependent, M-dependent
      - int_X P(Y|X,M)P(X):  X-invariant, M-dependent
      - P(M|X):              X-dependent, M-dependent
      - P(M|X)*inner:        X-dependent, M-dependent
      - outer_sum:           X-dependent, M-invariant

    All components are evaluated even if some checks fail; the test fails at
    the end with the full list. The M-dependence of the compiled P(Y|X,M) is
    a known open issue and is reported but not counted as a failure.
    """
    torch.manual_seed(42)
    ac, scm = trained_ac_z_xy_and_scm
    var_to_id = {"X": 0, "M": 1, "Y": 2}

    ast_joint = make_p({"X", "M", "Y"})
    ast_cond_y = make_cond({"Y"}, {"X", "M"}, ast_joint)  # P(Y|X,M)
    ast_px = make_marg({"Y", "M"}, ast_joint)  # P(X)
    ast_prod_inner = make_prod([ast_cond_y, ast_px])  # P(Y|X,M) * P(X)
    ast_inner = make_marg({"X"}, ast_prod_inner)  # int_X P(Y|X,M) P(X)
    ast_xm = make_marg({"Y"}, ast_joint)  # P(X,M)
    ast_cond_m = make_cond({"M"}, {"X"}, ast_xm)  # P(M|X)
    ast_prod = make_prod([ast_cond_m, ast_inner])  # P(M|X) * int_X{...}
    ast_outer = make_marg({"M"}, ast_prod)  # int_M P(M|X) * int_X{...}

    X, M, Y = 0, 1, 2
    # (name, query, sweep_col, fixed_cols, expect_dependent, known_issue)
    checks = [
        ("P(Y|X,M) ~ X", compile_query(ast_cond_y, ac, var_to_id), X, [M, Y], True, False),
        ("P(Y|X,M) ~ M", compile_query(ast_cond_y, ac, var_to_id), M, [X, Y], True, True),
        (
            "P(Y|X,M)*P(X) ~ X",
            compile_query(ast_prod_inner, ac, var_to_id),
            X,
            [M, Y],
            True,
            False,
        ),
        (
            "P(Y|X,M)*P(X) ~ M",
            compile_query(ast_prod_inner, ac, var_to_id),
            M,
            [X, Y],
            True,
            False,
        ),
        (
            "int_X P(Y|X,M)*P(X) ~ X",
            compile_query(ast_inner, ac, var_to_id),
            X,
            [M, Y],
            False,
            False,
        ),
        (
            "int_X P(Y|X,M)*P(X) ~ M",
            compile_query(ast_inner, ac, var_to_id),
            M,
            [X, Y],
            True,
            False,
        ),
        ("P(M|X) ~ X", compile_query(ast_cond_m, ac, var_to_id), X, [M], True, False),
        ("P(M|X) ~ M", compile_query(ast_cond_m, ac, var_to_id), M, [X], True, False),
        (
            "P(M|X)*inner ~ X",
            compile_query(ast_prod, ac, var_to_id),
            X,
            [M, Y],
            True,
            False,
        ),
        (
            "P(M|X)*inner ~ M",
            compile_query(ast_prod, ac, var_to_id),
            M,
            [X, Y],
            True,
            False,
        ),
        ("outer_sum ~ X", compile_query(ast_outer, ac, var_to_id), X, [Y], True, False),
        ("outer_sum ~ M", compile_query(ast_outer, ac, var_to_id), M, [X, Y], False, False),
    ]

    failures = []
    known_issues = []
    for name, q, sweep_col, fixed_cols, expect_dependent, known_issue in checks:
        spreads = _sweep_spreads(q, sweep_col, fixed_cols)
        print(f"spreads of {name}: {spreads}")
        if expect_dependent:
            ok = all(s > 1e-3 for s in spreads)
            msg = f"{name} lost its dependence (spreads: {spreads})"
        else:
            ok = all(s < 1e-5 for s in spreads)
            msg = f"{name} still depends on the marginalized variable (spreads: {spreads})"
        if not ok:
            (known_issues if known_issue else failures).append(msg)

    for msg in known_issues:
        print(f"KNOWN ISSUE: {msg}")
    assert not failures, "Component dependence failures:\n" + "\n".join(failures)


def test_frontdoor_do_x_dependence(trained_ac_z_xy_and_scm):
    """P(Y|do(X)) = int_M P(M|X) int_X'{P(Y|X',M) P(X')} must depend on X: at
    fixed Y, different X values must give different densities. A previous bug
    made the compiled do-density bit-exactly independent of X."""
    torch.manual_seed(42)
    ac, scm = trained_ac_z_xy_and_scm
    var_to_id = {"X": 0, "M": 1, "Y": 2}

    ast_joint = make_p({"X", "M", "Y"})
    ast_cond_y = make_cond({"Y"}, {"X", "M"}, ast_joint)  # P(Y|X,M)
    ast_px = make_marg({"Y", "M"}, ast_joint)  # P(X)
    ast_prod_inner = make_prod([ast_cond_y, ast_px])  # P(Y|X,M) * P(X)
    ast_inner = make_marg({"X"}, ast_prod_inner)  # int_X P(Y|X,M) * P(X)
    ast_xm = make_marg({"Y"}, ast_joint)  # P(X,M)
    ast_cond_m = make_cond({"M"}, {"X"}, ast_xm)  # P(M|X)
    ast_prod = make_prod([ast_cond_m, ast_inner])  # P(M|X) * int_X{...}
    ast_outer = make_marg({"M"}, ast_prod)  # int_M P(M|X) * int_X{...} = P(Y|do(X))

    q_outer = compile_query(ast_outer, ac, var_to_id)

    # Sweep X (col 0) at a few fixed Y values (col 2); M (col 1) is marginalized.
    n_x = 8
    x_vals = torch.linspace(-2.0, 2.0, n_x)
    for y_val in (-1.0, 0.0, 1.0):
        data = torch.zeros(n_x, 3)
        data[:, 0] = x_vals
        data[:, 2] = y_val
        out = eval_circuit(q_outer, data).squeeze()
        assert torch.isfinite(out).all()
        spread = out.max() - out.min()
        assert spread > 1e-3, f"P(Y|do(X)) is independent of X at Y={y_val}: {out}"


def _trapz_integral(q, integ_col: int, fixed: dict[int, float], lo=-8.0, hi=8.0, n=801) -> float:
    """Numerically integrate a compiled log-density over ``integ_col`` (trapezoid)."""
    grid = torch.linspace(lo, hi, n)
    data = torch.zeros(n, 3)
    for col, val in fixed.items():
        data[:, col] = val
    data[:, integ_col] = grid
    with torch.no_grad():
        log_p = eval_circuit(q, data).squeeze()
    p = torch.exp(log_p)
    return torch.trapezoid(p, grid).item()


REPRO_PLOT_DIR = "scratch/frontdoor_repro"


def _grid_log_density(
    q,
    row_var,
    col_var,
    fixed,
    n_row=121,
    n_col=61,
    lo_row=-8.0,
    hi_row=8.0,
    lo_col=-3.0,
    hi_col=3.0,
):
    """Evaluate a compiled log-density on a 2-D grid: rows = row_var, cols = col_var."""
    rows = torch.linspace(lo_row, hi_row, n_row)
    cols = torch.linspace(lo_col, hi_col, n_col)
    data = torch.zeros(n_row * n_col, 3)
    data[:, row_var] = rows.repeat_interleave(n_col)
    data[:, col_var] = cols.repeat(n_row)
    for c, v in fixed.items():
        data[:, c] = v
    with torch.no_grad():
        log_p = eval_circuit(q, data).squeeze()
    return log_p.reshape(n_row, n_col).numpy(), rows.numpy(), cols.numpy()


def _plot_shared_colorscale(panels, path, xlabel, ylabel, vmin, vmax, row_lo=-8.0, row_hi=8.0):
    """3-panel log-density heatmaps on one shared, fixed colorscale.

    ``panels`` is a list of (title, grid) with grid indexed [row, col]. All
    panels share [vmin, vmax]; values outside are clipped so heavy tails in one
    panel don't flatten the structure of the others. The colorbar sits in a
    dedicated axis on the far right so no panel is squeezed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2), sharey=True)
    im = None
    for ax, (title, grid) in zip(axes, panels):
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            extent=[-3.0, 3.0, row_lo, row_hi],
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    fig.subplots_adjust(left=0.05, right=0.91, bottom=0.11, top=0.90, wspace=0.12)
    cax = fig.add_axes([0.93, 0.15, 0.012, 0.72])
    fig.colorbar(im, cax=cax, label=f"log density (clipped to [{vmin:.1f}, {vmax:.1f}])")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


@pytest.mark.xfail(
    reason="Known limitation: pseudo-mixing layers share support across parent "
    "units, so conditionals that traverse them (y_mx) are not normalized. "
    "Construct vtrees with learn_liang_ps_md_vtree to avoid this.",
    strict=False,
)
def test_frontdoor_pm_given_x_normalization(
    trained_ac_z_xy_long_and_scm,
    trained_ac_x_yz_long_and_scm,
    trained_ac_y_xz_long_and_scm,
):
    """int_M P(M|X) dM must equal 1 for every X.

    Repro handoff: conditioning on X inside a (pseudo) mixing layer — required
    for the m_xy and y_mx vtrees — yields piecewise-garbage P(M|X). On y_mx it
    also doubles the mass (integral ~2). The x_my vtree (conditioning happens
    at a synthesizing layer) is the control.
    """
    import os

    torch.manual_seed(42)
    var_to_id = {"X": 0, "M": 1, "Y": 2}
    circuits = {
        "x_my": trained_ac_z_xy_long_and_scm,
        "m_xy": trained_ac_x_yz_long_and_scm,
        "y_mx": trained_ac_y_xz_long_and_scm,
    }
    X, M = 0, 1

    failures = []
    panels = []
    for name, (ac, _scm) in circuits.items():
        joint = make_p({"X", "M", "Y"})
        q = compile_query(make_cond({"M"}, {"X"}, make_marg({"Y"}, joint)), ac, var_to_id)
        for x_val in (-1.0, 0.0, 1.0):
            integral = _trapz_integral(q, M, fixed={X: x_val})
            print(f"P(M|X) vtree={name} X={x_val}: int_M = {integral:.4f}")
            if abs(integral - 1.0) > 0.1:
                failures.append(f"vtree {name}: int_M P(M|X={x_val}) dM = {integral:.4f}")
        grid, _, _ = _grid_log_density(q, M, X, fixed={}, lo_row=-6.0, hi_row=6.0)
        panels.append((f"{name}\nP(M|X)", grid))

    os.makedirs(REPRO_PLOT_DIR, exist_ok=True)
    _plot_shared_colorscale(
        panels,
        f"{REPRO_PLOT_DIR}/pm_given_x_heatmaps.png",
        xlabel="X",
        ylabel="M",
        vmin=-10.0,
        vmax=0.0,
        row_lo=-6.0,
        row_hi=6.0,
    )

    assert not failures, "P(M|X) normalization failures:\n" + "\n".join(failures)


@pytest.mark.xfail(
    reason="Known limitation: pseudo-mixing layers share support across parent "
    "units, so the frontdoor estimand compiled on the y_mx vtree is not "
    "normalized. Construct vtrees with learn_liang_ps_md_vtree to avoid this.",
    strict=False,
)
def test_frontdoor_do_normalization(
    trained_ac_z_xy_long_and_scm,
    trained_ac_x_yz_long_and_scm,
    trained_ac_y_xz_long_and_scm,
):
    """int_Y P(Y|do(X)) dY must equal 1 for every X.

    Repro handoff: on the y_mx vtree the compiled frontdoor estimand is not
    normalized in Y (x_my and m_mx are the passing controls).
    """
    import os

    torch.manual_seed(42)
    var_to_id = {"X": 0, "M": 1, "Y": 2}
    circuits = {
        "x_my": trained_ac_z_xy_long_and_scm,
        "m_xy": trained_ac_x_yz_long_and_scm,
        "y_mx": trained_ac_y_xz_long_and_scm,
    }
    X, Y = 0, 2

    failures = []
    panels = []
    for name, (ac, _scm) in circuits.items():
        joint = make_p({"X", "M", "Y"})
        ast_cond_y = make_cond({"Y"}, {"X", "M"}, joint)  # P(Y|X,M)
        ast_px = make_marg({"Y", "M"}, joint)  # P(X)
        ast_prod_inner = make_prod([ast_cond_y, ast_px])  # P(Y|X,M) * P(X)
        ast_inner = make_marg({"X"}, ast_prod_inner)  # int_X P(Y|X,M) P(X)
        ast_xm = make_marg({"Y"}, joint)  # P(X,M)
        ast_cond_m = make_cond({"M"}, {"X"}, ast_xm)  # P(M|X)
        ast_prod = make_prod([ast_cond_m, ast_inner])  # P(M|X) * int_X{...}
        ast_outer = make_marg({"M"}, ast_prod)  # int_M P(M|X) * int_X{...} = P(Y|do(X))
        q_outer = compile_query(ast_outer, ac, var_to_id)

        for x_val in (-1.0, 0.0, 1.0):
            integral = _trapz_integral(q_outer, Y, fixed={X: x_val})
            print(f"P(Y|do(X)) vtree={name} X={x_val}: int_Y = {integral:.4f}")
            if abs(integral - 1.0) > 0.1:
                failures.append(f"vtree {name}: int_Y P(Y|do(X={x_val})) dY = {integral:.4f}")
        grid, _, _ = _grid_log_density(q_outer, Y, X, fixed={})
        panels.append((f"{name}\nP(Y|do(X))", grid))

    os.makedirs(REPRO_PLOT_DIR, exist_ok=True)
    _plot_shared_colorscale(
        panels, f"{REPRO_PLOT_DIR}/do_heatmaps.png", xlabel="X", ylabel="Y", vmin=-7.0, vmax=-1.0
    )

    assert not failures, "P(Y|do(X)) normalization failures:\n" + "\n".join(failures)


def test_frontdoor_pm_given_x_y_mx_forward_pass(trained_ac_y_xz_long_and_scm):
    """Hand over the compiled P(M|X) circuit for the y_mx vtree: forward pass
    on an (X, M) grid for manual analysis. No correctness assertions."""
    torch.manual_seed(42)
    ac, scm = trained_ac_y_xz_long_and_scm
    var_to_id = {"X": 0, "M": 1, "Y": 2}

    joint = make_p({"X", "M", "Y"})
    q = compile_query(make_cond({"M"}, {"X"}, make_marg({"Y"}, joint)), ac, var_to_id)

    X, M = 0, 1
    x_vals = torch.linspace(-1.5, 1.5, 4)
    data = torch.zeros(4, 3)
    data[:, X] = x_vals
    data[:, M] = -0.5

    out = eval_circuit(q, data, verbose=True, show_weights=True, log_domain=False).squeeze()
    assert torch.isfinite(out).all()
    print("\nP(M|X) y_mx forward pass at M=-0.5, log-density:")
    print(out)
    print(f"X values: {x_vals.tolist()}")


@pytest.mark.xfail(
    reason="Known limitation: pseudo-mixing layers share support across parent "
    "units, so with the richer PS construction (H up to H_L*H_R parents) the "
    "compiled P(Y,M|X) multi-fires and integrates to ~2 instead of 1.",
    strict=False,
)
def test_frontdoor_pym_given_x_y_mx(trained_ac_y_xz_long_and_scm):
    """int_M int_Y P(Y,M|X) dY dM must equal 1, and the compiled COND must match
    the quotient P(X,M,Y)/P(X) — exercises two nested mixing conditionals."""
    torch.manual_seed(42)
    ac, scm = trained_ac_y_xz_long_and_scm
    var_to_id = {"X": 0, "M": 1, "Y": 2}

    joint = make_p({"X", "M", "Y"})
    q_cond = compile_query(make_cond({"Y", "M"}, {"X"}, joint), ac, var_to_id)
    q_num = compile_query(joint, ac, var_to_id)
    q_den = compile_query(make_marg({"M", "Y"}, joint), ac, var_to_id)

    X, M, Y = 0, 1, 2

    # Fine grid for the 2-D trapezoid: the conditional has staircase ridges in
    # M, so coarse grids under-integrate (201^2 over [-8,8] gives ~0.89).
    m_grid = torch.linspace(-12.0, 12.0, 601)
    y_grid = torch.linspace(-12.0, 12.0, 601)
    mm, yy = torch.meshgrid(m_grid, y_grid, indexing="ij")
    n = mm.numel()

    # Coarser grid suffices for the COND-vs-quotient comparison.
    m_grid_c = torch.linspace(-8.0, 8.0, 201)
    y_grid_c = torch.linspace(-8.0, 8.0, 201)
    mm_c, yy_c = torch.meshgrid(m_grid_c, y_grid_c, indexing="ij")
    n_c = mm_c.numel()

    for x_val in (-1.0, 0.0, 1.0):
        data = torch.zeros(n, 3)
        data[:, X] = x_val
        data[:, M] = mm.reshape(-1)
        data[:, Y] = yy.reshape(-1)
        with torch.no_grad():
            log_cond = eval_circuit(q_cond, data).squeeze()
        p_cond = torch.exp(log_cond).reshape(601, 601)
        integral = torch.trapezoid(torch.trapezoid(p_cond, y_grid, dim=1), m_grid).item()

        data_c = torch.zeros(n_c, 3)
        data_c[:, X] = x_val
        data_c[:, M] = mm_c.reshape(-1)
        data_c[:, Y] = yy_c.reshape(-1)
        with torch.no_grad():
            log_cond_c = eval_circuit(q_cond, data_c).squeeze()
            log_quot = eval_circuit(q_num, data_c).squeeze() - eval_circuit(q_den, data_c).squeeze()
        max_diff = (log_cond_c - log_quot).abs().max().item()
        print(
            f"X={x_val}: int int P(Y,M|X) = {integral:.4f} | max|COND - quotient| = {max_diff:.2e}"
        )
        assert abs(integral - 1.0) < 0.02, f"int int P(Y,M|X={x_val}) = {integral:.4f}"
        assert max_diff < 1e-4, f"COND != quotient at X={x_val}: max diff {max_diff:.2e}"
