"""
End-to-end verification of causal backdoor adjustment via eval_estimand.

SCM structure (classical confounded graph):
    Z -> X -> Y
    Z -> Y
where all variables are linear-Gaussian.

The circuit is built with (X∪Z)-determinism (md_sets=[{0, 1}]), ensuring
disjoint bins for Z and X.  The estimand AST is produced by the identification
algorithm and evaluated algebraically through eval_estimand.

Tests:
1. Backdoor formula — eval_estimand on the identified AST matches the
   intervened SCM's ground-truth log P(Y | do(X)).

2. Observational ≠ causal — the circuit distinguishes P(Y|X) from P(Y|do(X))
   at a query point where confounding creates a measurable gap.
"""

import numpy as np
import pytest
import torch

from src.compilation.estimand_eval import eval_estimand
from src.compilation.folded_circuit import CircuitFolder
from src.compilation.fused_circuit import CircuitFuser
from src.compilation.tensorized_circuit import TensorizedCircuit
from src.construction import MDCircuitBuilder, MDRegionGraphBuilder, construct_optimal_md_vtree
from src.symbolic import GaussianDistribution
from src.symbolic.id_ast import ast_to_str, make_det_prod, make_marg, make_p, make_pow, simplify_ast
from src.symbolic.identification import identify, minimize_determinisms, required_determinisms
from src.symbolic.io_utils import plot_dag
from src.symbolic.scm import (
    AdditiveNoiseMechanism,
    GaussianNoise,
    LinearLogic,
    StructuralCausalModel,
)


# ── Fixed query points ────────────────────────────────────────────────────────
_X_VAL = 0.5
_Y_VAL = 1.0

# Contrast point: at x=0.5 the observational mean E[Y|X=0.5] = 1.3,
# while the causal mean E[Y|do(X=0.5)] = 1.0.  Querying at y=1.3
# (the observational mode) maximises the density gap.
_X_CONTRAST = 0.5
_Y_CONTRAST = 1.3


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def confounded_scm():
    """Z → X → Y, Z → Y (classical backdoor graph).

    Mechanisms:
        Z  ~ N(0, 1)
        X  = 2·Z + eps_X,           eps_X ~ N(0, 1)
        Y  = 2·X + 1.5·Z + eps_Y,  eps_Y ~ N(0, 1)

    Variable names are strings matching the SCM API; the topological
    sampling order determines tensor columns: col 0 = Z, col 1 = X, col 2 = Y.
    """
    scm = StructuralCausalModel()
    scm.add_variable(
        "Z",
        mechanism=AdditiveNoiseMechanism(logic=None, noise_dist=GaussianNoise(0.0, 1.0)),
    )
    scm.add_variable(
        "X",
        mechanism=AdditiveNoiseMechanism(
            logic=LinearLogic({"Z": 2.0}), noise_dist=GaussianNoise(0.0, 1.0)
        ),
        parents=["Z"],
    )
    scm.add_variable(
        "Y",
        mechanism=AdditiveNoiseMechanism(
            logic=LinearLogic({"X": 2.0, "Z": 1.5}), noise_dist=GaussianNoise(0.0, 1.0)
        ),
        parents=["X", "Z"],
    )
    return scm


@pytest.fixture(scope="module")
def var_map():
    """Variable name → tensor column index (topological order: Z=0, X=1, Y=2)."""
    return {"Z": 0, "X": 1, "Y": 2}


@pytest.fixture(scope="module")
def trained_circuit(confounded_scm):
    """TensorizedCircuit with (X∪Z)-determinism, trained on the confounded SCM."""
    torch.manual_seed(0)
    np.random.seed(0)

    pilot = confounded_scm.sample(2**14)
    pilot_tensor = torch.from_numpy(pilot.values.copy()).float()

    # (X∪Z)-determinism: cols 0 (Z) and 1 (X) get disjoint bins
    md_vtree = construct_optimal_md_vtree(pilot_tensor, md_sets=[{0}, {0, 1}])
    input_dists = {v: GaussianDistribution(var=v, mean=0.0, stddev=1.0) for v in range(3)}
    rg = MDRegionGraphBuilder(md_vtree=md_vtree, input_dists=input_dists).build()
    spn = MDCircuitBuilder(rg=rg, h=4, input_dists=input_dists).build()
    circuit = TensorizedCircuit(CircuitFuser(CircuitFolder(spn).build()).build())

    plot_dag(md_vtree).render(filename="backdoor_test_md_vtree", format="svg")
    plot_dag(spn, node_config=spn.get_node_config(show_unit_supports=True)).render(
        filename="backdoor_test_spn", format="svg"
    )
    train_data = torch.from_numpy(confounded_scm.sample(2**17).values.copy()).float()
    for _ in range(25):
        for batch in torch.split(train_data, 2**12):
            circuit.em_step(batch, step_size=0.15)

    return circuit


@pytest.fixture(scope="module")
def backdoor_ast(confounded_scm, var_map):
    """Simplified AST for P(Y | do(X)) via the identification algorithm."""
    scm = confounded_scm
    req = required_determinisms({"Y"}, {"X"}, scm)
    min_d = minimize_determinisms(req)
    P = make_p(scm.observable_variables)
    ast, root_id, _, _ = identify({"Y"}, {"X"}, P, scm, D=min_d)
    sast, sroot = simplify_ast(ast, root_id)
    print(f"Backdoor AST: {ast_to_str(sast, sroot)}")
    return sast, sroot


@pytest.fixture(scope="module")
def conditional_ast():
    """AST for P(Y | X) = P(X,Y) / P(X) via DetProd."""
    p_xy = make_marg({"Z"}, *make_p({"X", "Y", "Z"}))
    p_x_inv = make_pow(-1, *make_marg({"Y", "Z"}, *make_p({"X", "Y", "Z"})))
    return make_det_prod([p_xy, p_x_inv])


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_backdoor_matches_scm(confounded_scm, trained_circuit, var_map, backdoor_ast):
    """eval_estimand on the backdoor AST ≈ SCM ground-truth log P(Y | do(X))."""
    sast, sroot = backdoor_ast
    sample = torch.tensor([[0.0, _X_VAL, _Y_VAL]])  # Z column value is irrelevant

    with torch.no_grad():
        circuit_result = eval_estimand(trained_circuit, sast, sample, var_map).item()

    # SCM ground truth
    intervened = confounded_scm.intervene({"X": _X_VAL})
    gt = intervened.empirical_log_density(query={"Y": _Y_VAL}, n_samples=2_000_000, eps=0.05)

    diff = abs(circuit_result - gt)
    print(f"\n[backdoor] circuit={circuit_result:.4f}  gt={gt:.4f}  diff={diff:.4f}")
    assert diff < 2.0, (
        f"Backdoor result {circuit_result:.4f} deviates from SCM ground "
        f"truth {gt:.4f} by {diff:.4f}"
    )


def test_observational_differs_from_causal(
    confounded_scm, trained_circuit, var_map, backdoor_ast, conditional_ast
):
    """P(Y | X=x) ≠ P(Y | do(X=x)) — the circuit captures the confounding gap.

    At x=0.5, y=1.3 (the observational mode):
        log P(Y=1.3 | X=0.5)     ≈ −1.10  (near the observational peak)
        log P(Y=1.3 | do(X=0.5)) ≈ −1.52  (off-centre from the causal peak)
    """
    cond_ast, cond_root = conditional_ast
    bd_ast, bd_root = backdoor_ast
    sample = torch.tensor([[0.0, _X_CONTRAST, _Y_CONTRAST]])

    # ── SCM ground truth ──
    scm_obs = confounded_scm.empirical_log_density(
        query={"Y": _Y_CONTRAST},
        evidence={"X": _X_CONTRAST},
        n_samples=2_000_000,
        eps=0.05,
    )
    intervened = confounded_scm.intervene({"X": _X_CONTRAST})
    scm_causal = intervened.empirical_log_density(
        query={"Y": _Y_CONTRAST},
        n_samples=2_000_000,
        eps=0.05,
    )
    scm_gap = scm_obs - scm_causal
    print(f"\n[SCM]     obs={scm_obs:.4f}  causal={scm_causal:.4f}  gap={scm_gap:.4f}")
    assert scm_gap > 0.2, f"SCM gap too small: {scm_gap:.4f}"

    # ── Circuit estimates ──
    with torch.no_grad():
        circuit_obs = eval_estimand(trained_circuit, cond_ast, sample, var_map).item()
        circuit_causal = eval_estimand(trained_circuit, bd_ast, sample, var_map).item()

    circuit_gap = circuit_obs - circuit_causal
    print(f"[circuit] obs={circuit_obs:.4f}  causal={circuit_causal:.4f}  gap={circuit_gap:.4f}")

    assert circuit_gap > 0, (
        f"Circuit should find log P(Y|X) > log P(Y|do(X)) at this query point, "
        f"but got obs={circuit_obs:.4f}, causal={circuit_causal:.4f}"
    )
