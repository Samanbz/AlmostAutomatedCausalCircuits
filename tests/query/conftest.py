import numpy as np
import pandas as pd
import pytest
import torch

from src.construction.circuit_builder import create_md_circuit
from src.construction.random_mechanisms import randomize_mechanisms
from src.construction.skeleton import backdoor_skeleton, frontdoor_skeleton
from src.symbolic.arithmetic.circuit import SymbolicArithmeticCircuit, eval_circuit
from src.symbolic.arithmetic.nodes import GaussianDistribution
from src.symbolic.arithmetic.train import SymbolicEMTrainer
from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet


torch.set_printoptions(precision=2, sci_mode=False, linewidth=200, edgeitems=5)


N = 8


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
        dists, vtree, num_nodes=num_nodes, initialize_weights=True, fairness_temperature=10.0
    )


@pytest.fixture
def vtree_xz():
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
def vtree_zxz():
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
def vtree_non_det():
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
def vtree_z_xy():
    vt = VTree()
    v0 = vt.add_node(VNode(BitSet([0]), md_set=BitSet([0])))
    v1 = vt.add_node(VNode(BitSet([1]), md_set=BitSet([1])))
    v2 = vt.add_node(VNode(BitSet([2]), md_set=BitSet.universal()))
    v12 = vt.add_node(VNode(BitSet([1, 2]), md_set=BitSet([1])))
    vt.add_children(v12, v1, v2)
    v012 = vt.add_node(VNode(BitSet([0, 1, 2]), md_set=BitSet([0])))
    vt.add_children(v012, v0, v12)
    return vt


@pytest.fixture
def vtree_x_yz():
    vt = VTree()
    v1 = vt.add_node(VNode(BitSet([1]), md_set=BitSet([1])))
    v0 = vt.add_node(VNode(BitSet([0]), md_set=BitSet([0])))
    v2 = vt.add_node(VNode(BitSet([2]), md_set=BitSet.universal()))
    v02 = vt.add_node(VNode(BitSet([0, 2]), md_set=BitSet([0])))
    vt.add_children(v02, v0, v2)
    v012 = vt.add_node(VNode(BitSet([0, 1, 2]), md_set=BitSet([0])))
    vt.add_children(v012, v1, v02)
    return vt


@pytest.fixture
def vtree_y_xz():
    vt = VTree()
    v2 = vt.add_node(VNode(BitSet([2]), md_set=BitSet.universal()))
    v0 = vt.add_node(VNode(BitSet([0]), md_set=BitSet([0])))
    v1 = vt.add_node(VNode(BitSet([1]), md_set=BitSet([1])))
    v01 = vt.add_node(VNode(BitSet([0, 1]), md_set=BitSet([0])))
    vt.add_children(v01, v0, v1)
    v012 = vt.add_node(VNode(BitSet([0, 1, 2]), md_set=BitSet([0])))
    vt.add_children(v012, v2, v01)
    return vt


@pytest.fixture
def sub_vtree_z_det():
    vt = VTree()
    v0 = vt.add_node(VNode(BitSet([0]), md_set=BitSet([0])))
    v1 = vt.add_node(VNode(BitSet([1]), md_set=BitSet([1])))
    v01 = vt.add_node(VNode(BitSet([0, 1]), md_set=BitSet([0, 1])))
    vt.add_children(v01, v0, v1)
    return vt


@pytest.fixture
def sub_vtree_z_no_det():
    vt = VTree()
    v0 = vt.add_node(VNode(BitSet([0]), md_set=BitSet.universal()))
    v1 = vt.add_node(VNode(BitSet([1]), md_set=BitSet.universal()))
    v01 = vt.add_node(VNode(BitSet([0, 1]), md_set=BitSet.universal()))
    vt.add_children(v01, v0, v1)
    return vt


@pytest.fixture
def ac_xz(vtree_xz):
    return _build_ac(vtree_xz)


@pytest.fixture
def ac_zxz(vtree_zxz):
    return _build_ac(vtree_zxz)


@pytest.fixture
def ac_xz_skewed(vtree_xz_skewed):
    return _build_ac(vtree_xz_skewed)


@pytest.fixture
def ac_non_det(vtree_non_det):
    return _build_ac(vtree_non_det)


@pytest.fixture
def ac_z_xy(vtree_z_xy):
    return _build_ac(vtree_z_xy)


@pytest.fixture
def sub_ac_z_det(sub_vtree_z_det):
    return _build_ac(sub_vtree_z_det, vars=(0, 1))


@pytest.fixture
def sub_ac_z_non_det(sub_vtree_z_no_det):
    return _build_ac(sub_vtree_z_no_det, vars=(0, 1))


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
    # The legacy builder this replaced left np.random/torch unseeded, making the
    # trained-* tests flaky (outcome depends on the RNG stream state); pin both.
    np.random.seed(4)
    torch.manual_seed(4)
    # Reproduce the retired legacy builder's default
    # build_synthetic_continuous_scm(2): Z_i ~ N(0,1),
    # X = Z0 + Z1 + N(0,1),  Y = X + 1.5*(Z0 + Z1) + N(0,1).
    scm = randomize_mechanisms(
        backdoor_skeleton(n_confounders=2, kind="continuous", cardinality=None),
        rng=5,
        gmm_components=(1,),
        gmm_mean_range=(0.0, 0.0),
        intercept_range=(0.0, 0.0),
        sigma2_range=(1.0, 1.0),
        edge_coefs={
            "Z_0->X": 1.0,
            "Z_1->X": 1.0,
            "X->Y": 1.0,
            "Z_0->Y": 1.5,
            "Z_1->Y": 1.5,
        },
    )
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
    return trained_circuit, scm


@pytest.fixture
def trained_ac_xz_skewed_and_scm(vtree_xz_skewed):
    return _train_circuit(vtree_xz_skewed)


@pytest.fixture
def trained_ac_zxz_and_scm(vtree_zxz):
    trained_circuit, scm = _train_circuit(vtree_zxz)
    return trained_circuit, scm


def _train_backdoor3_circuit(vtree, num_nodes=N, n_iter=2):
    """3-variable backdoor SCM (columns Z, X, Y): Z -> X, Z -> Y, X -> Y.

    Same mechanism params as ``_train_circuit`` but with a single confounder,
    matching the 3-leaf vtree fixtures (vtree_z_xy / vtree_x_yz / vtree_y_xz).
    """
    np.random.seed(4)
    torch.manual_seed(4)
    scm = randomize_mechanisms(
        backdoor_skeleton(n_confounders=1, kind="continuous", cardinality=None),
        rng=5,
        gmm_components=(1,),
        gmm_mean_range=(0.0, 0.0),
        intercept_range=(0.0, 0.0),
        sigma2_range=(1.0, 1.0),
        edge_coefs={"Z_0->X": 1.0, "X->Y": 1.0, "Z_0->Y": 1.5},
    )
    df_train = scm.sample(10000)
    ac = _build_ac_from_data(vtree, df_train, vars=(0, 1, 2), num_nodes=num_nodes)
    pts_t = torch.tensor(df_train.values, dtype=torch.float32)
    trainer = SymbolicEMTrainer(ac)
    trainer.train(pts_t, n_iter=n_iter, batch_size=500, log_interval=0)
    return ac, scm


@pytest.fixture
def trained_ac_y_xz_backdoor_and_scm(vtree_y_xz):
    return _train_backdoor3_circuit(vtree_y_xz)


@pytest.fixture
def trained_ac_z_xy_backdoor_and_scm(vtree_z_xy):
    return _train_backdoor3_circuit(vtree_z_xy)


def _train_frontdoor_circuit(vtree, vars=(0, 1, 2), num_nodes=N, n_iter=2):
    """Frontdoor SCM (hidden U; X -> M -> Y; U -> X, Y) matching the
    (X, M, Y) column layout of vtree_z_xy."""
    np.random.seed(4)
    torch.manual_seed(4)
    scm = randomize_mechanisms(
        frontdoor_skeleton(
            n_confounders=1,
            n_treatments=1,
            n_mediators=1,
            n_outcomes=1,
            kind="continuous",
            cardinality=None,
            confounder_kind="continuous",
        ),
        rng=5,
        gmm_components=(1,),
        gmm_mean_range=(0.0, 0.0),
        intercept_range=(0.0, 0.0),
        sigma2_range=(1.0, 1.0),
        edge_coefs={
            "U->X": 1.0,
            "X->M": 1.5,
            "M->Y": 1.0,
            "U->Y": 2.0,
        },
    )
    df_train = scm.sample_dataset(10000)  # drops hidden U -> columns (X, M, Y)
    ac = _build_ac_from_data(vtree, df_train, vars=vars, num_nodes=num_nodes)
    pts_t = torch.tensor(df_train.values, dtype=torch.float32)
    trainer = SymbolicEMTrainer(ac)
    trainer.train(pts_t, n_iter=n_iter, batch_size=500, log_interval=0)
    return ac, scm


@pytest.fixture
def trained_ac_z_xy_and_scm(vtree_z_xy):
    trained_circuit, scm = _train_frontdoor_circuit(vtree_z_xy)
    return trained_circuit, scm


@pytest.fixture
def trained_ac_x_yz_and_scm(vtree_x_yz):
    trained_circuit, scm = _train_frontdoor_circuit(vtree_x_yz)
    return trained_circuit, scm


@pytest.fixture
def trained_ac_y_xz_and_scm(vtree_y_xz):
    trained_circuit, scm = _train_frontdoor_circuit(vtree_y_xz)
    return trained_circuit, scm


# Longer-trained variants (n_iter=50) for reproducing conditioning-on-mixing-layer
# artifacts that only appear once EM has converged.
@pytest.fixture
def trained_ac_z_xy_long_and_scm(vtree_z_xy):
    return _train_frontdoor_circuit(vtree_z_xy, n_iter=50)


@pytest.fixture
def trained_ac_x_yz_long_and_scm(vtree_x_yz):
    return _train_frontdoor_circuit(vtree_x_yz, n_iter=50)


@pytest.fixture
def trained_ac_y_xz_long_and_scm(vtree_y_xz):
    return _train_frontdoor_circuit(vtree_y_xz, n_iter=50)
