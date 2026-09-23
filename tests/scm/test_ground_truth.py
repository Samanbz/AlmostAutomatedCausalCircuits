"""Tests for the unified exact ground-truth engine (src/symbolic/scm/ground_truth.py)."""

import itertools

import numpy as np
import pytest

from src.symbolic.scm import (
    AdditiveNoiseMechanism,
    BinaryMechanism,
    CLGMechanism,
    ConstantMechanism,
    DirichletCPTMechanism,
    GaussianMixtureNoise,
    LinearGMMMechanism,
    LogisticLogic,
    RegionalDiscreteMechanism,
    StructuralCausalModel,
    UniformNoise,
)
from src.symbolic.scm.ground_truth import GaussianMixture, GroundTruth


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# GaussianMixture unit behavior
# ---------------------------------------------------------------------------


def test_gaussian_mixture_marginal_condition_pdf():
    gm = GaussianMixture(
        [0.25, 0.75],
        [[0.0, 0.0], [1.0, 2.0]],
        [np.eye(2), 2.0 * np.eye(2)],
        ("a", "b"),
    )
    marg = gm.marginal(["b"])
    assert marg.var_names == ("b",)
    assert np.allclose(marg.mean, [0.25 * 0.0 + 0.75 * 2.0])

    cond = gm.condition({"a": 0.0})
    # Independent components: conditioning on a leaves b's mean/var untouched.
    d0 = np.exp(-0.5 * (0.0 - 0.0) ** 2 / 1.0) / np.sqrt(2 * np.pi * 1.0)
    d1 = np.exp(-0.5 * (0.0 - 1.0) ** 2 / 2.0) / np.sqrt(2 * np.pi * 2.0)
    w = np.array([0.25 * d0, 0.75 * d1])
    w /= w.sum()
    assert cond.var_names == ("b",)
    assert np.allclose(cond.weights, w)
    assert np.allclose(cond.means[:, 0], [0.0, 2.0])
    assert np.allclose(cond.covs[:, 0, 0], [1.0, 2.0])

    # pdf at a point equals the manual 2-component evaluation.
    point = np.array([0.0, 2.0])
    expected = 0.25 * np.exp(-0.5 * point @ point) / (2 * np.pi)
    expected += 0.75 * np.exp(-0.25 * ((point - [1.0, 2.0]) ** 2).sum()) / (4 * np.pi)
    assert np.isclose(gm.pdf(point), expected)
    batch = gm.pdf(np.stack([point, point]))
    assert batch.shape == (2,) and np.allclose(batch, expected)
    assert np.isclose(gm.evidence_pdf({"a": 0.0}), 0.25 * d0 + 0.75 * d1)


# ---------------------------------------------------------------------------
# Pure discrete SCMs
# ---------------------------------------------------------------------------


def _discrete_scm():
    rng = np.random.default_rng(7)
    scm = StructuralCausalModel()
    scm.add_variable("A", DirichletCPTMechanism(2, {}, np.array([0.4, 0.6])))
    scm.add_variable("B", DirichletCPTMechanism(3, {}, np.array([0.2, 0.3, 0.5])))
    scm.add_variable(
        "X",
        RegionalDiscreteMechanism(
            2, {"A": 2}, np.array([0.0, 0.5, 1.0]), np.array([[0, 1], [1, 0]])
        ),
        parents=["A"],
    )
    table = rng.dirichlet(np.full(3, 0.8), size=6).reshape(2, 3, 3)
    scm.add_variable("Y", DirichletCPTMechanism(3, {"X": 2, "B": 3}, table), parents=["X", "B"])
    return scm


def _brute_force_joint(scm, do=None):
    """Full-enumeration (truncated-factorization) joint over the discrete vars."""
    do = do or {}
    gt = GroundTruth(scm)
    joint = {}
    for config in itertools.product(*[range(gt.cards[v]) for v in gt.discrete_vars]):
        assignment = dict(zip(gt.discrete_vars, config))
        p = 1.0
        for v in gt.discrete_vars:
            if v in do:
                if assignment[v] != do[v]:
                    p = 0.0
                    break
                continue
            mech = scm.get_node_data(v)
            parents = scm.get_parents(v)
            cpt = mech.cpt(parent_names=parents)
            p *= cpt[tuple(assignment[pn] for pn in parents) + (assignment[v],)]
        joint[config] = p
    return joint


def test_discrete_classification():
    gt = GroundTruth(_discrete_scm())
    assert gt.classify() == {"discrete": ["A", "B", "X", "Y"], "continuous": []}
    assert gt.cards == {"A": 2, "B": 3, "X": 2, "Y": 3}


def test_marginal_prob_matches_brute_force():
    scm = _discrete_scm()
    gt = GroundTruth(scm)
    joint = _brute_force_joint(scm)
    order = list(gt.discrete_vars)

    for x in range(2):
        for y in range(3):
            expected = sum(
                p for c, p in joint.items() if c[order.index("X")] == x and c[order.index("Y")] == y
            )
            assert np.isclose(gt.marginal_prob({"X": x, "Y": y}), expected)

    for b in range(3):
        expected = sum(p for c, p in joint.items() if c[order.index("B")] == b)
        assert np.isclose(gt.marginal_prob({"B": b}), expected)

    # The full joint via discrete_joint matches enumeration too.
    assert gt.discrete_joint() == pytest.approx(joint)


def test_do_matches_truncated_enumeration():
    scm = _discrete_scm()
    gt = GroundTruth(scm)
    joint_do = _brute_force_joint(scm, do={"X": 1})
    order = list(gt.discrete_vars)
    assert np.isclose(sum(joint_do.values()), 1.0)
    for y in range(3):
        expected = sum(p for c, p in joint_do.items() if c[order.index("Y")] == y)
        assert np.isclose(gt.marginal_prob({"Y": y}, do={"X": 1}), expected)


def test_do_matches_intervention_samples():
    scm = _discrete_scm()
    gt = GroundTruth(scm)
    np.random.seed(3)
    n = 200_000
    df = scm.intervene({"X": 1}).sample(n)
    for y in range(3):
        p = gt.marginal_prob({"Y": y}, do={"X": 1})
        freq = float((df["Y"] == y).mean())
        tol = max(5.0 * np.sqrt(p * (1 - p) / n), 1e-3)
        assert abs(freq - p) < tol


def test_evidence_matches_enumeration():
    scm = _discrete_scm()
    gt = GroundTruth(scm)
    joint = _brute_force_joint(scm)
    order = list(gt.discrete_vars)
    b = 2
    denom = sum(p for c, p in joint.items() if c[order.index("B")] == b)
    for y in range(3):
        expected = (
            sum(
                p for c, p in joint.items() if c[order.index("B")] == b and c[order.index("Y")] == y
            )
            / denom
        )
        assert np.isclose(gt.marginal_prob({"Y": y}, evidence={"B": b}), expected)


def test_hidden_variable_marginalized():
    # X <- U -> Y with U hidden; P(Y=1 | do(X=1)) = sum_u P(u) P(Y=1 | X=1, u).
    scm = StructuralCausalModel()
    scm.add_variable("U", DirichletCPTMechanism(2, {}, np.array([0.3, 0.7])), hidden=True)
    scm.add_variable("X", BinaryMechanism(LogisticLogic({"U": 2.0}, intercept=-0.5)), parents=["U"])
    scm.add_variable(
        "Y",
        BinaryMechanism(LogisticLogic({"X": 1.5, "U": -1.0}, intercept=0.2)),
        parents=["X", "U"],
    )
    assert "U" in scm.hidden_variables
    gt = GroundTruth(scm)

    p_u = [0.3, 0.7]
    expected = sum(p_u[u] * _sigmoid(0.2 + 1.5 * 1 - 1.0 * u) for u in (0, 1))
    assert np.isclose(gt.marginal_prob({"Y": 1}, do={"X": 1}), expected)

    # Observational P(Y=1) by hand.
    p_y1 = 0.0
    for u in (0, 1):
        for x in (0, 1):
            p_x = _sigmoid(-0.5 + 2.0 * u) if x == 1 else 1 - _sigmoid(-0.5 + 2.0 * u)
            p_y1 += p_u[u] * p_x * _sigmoid(0.2 + 1.5 * x - 1.0 * u)
    assert np.isclose(gt.marginal_prob({"Y": 1}), p_y1)


# ---------------------------------------------------------------------------
# Pure continuous SCMs
# ---------------------------------------------------------------------------


def _chain_scm():
    scm = StructuralCausalModel()
    scm.add_variable("X", LinearGMMMechanism({}, 0.5, GaussianMixtureNoise.single(0.0, 1.0)))
    scm.add_variable(
        "Y",
        LinearGMMMechanism({"X": 2.0}, -1.0, GaussianMixtureNoise.single(0.0, 0.5)),
        parents=["X"],
    )
    scm.add_variable(
        "Z",
        LinearGMMMechanism({"Y": -0.5}, 0.3, GaussianMixtureNoise.single(0.0, 0.7)),
        parents=["Y"],
    )
    return scm


# Hand-computed moments for the chain X -> Y -> Z:
# X ~ N(0.5, 1);  Y = -1 + 2X + N(0, 0.25);  Z = 0.3 - 0.5Y + N(0, 0.49)
_MEAN_X, _VAR_X = 0.5, 1.0
_MEAN_Y, _VAR_Y = 0.0, 4.0 * 1.0 + 0.25
_MEAN_Z, _VAR_Z = 0.3, 0.25 * 4.25 + 0.49
_COV_XZ = -0.5 * 2.0 * _VAR_X


def test_chain_marginal_density():
    gt = GroundTruth(_chain_scm())
    gm = gt.marginal_density(["Z"])
    assert gm.var_names == ("Z",)
    assert gm.n_components == 1
    assert np.isclose(gm.mean[0], _MEAN_Z)
    assert np.isclose(gm.cov[0, 0], _VAR_Z)


def test_chain_do_continuous():
    gt = GroundTruth(_chain_scm())
    gm = gt.marginal_density(["Z"], do={"X": 1.5})
    # Z | do(X=1.5): Y = -1 + 2*1.5 + N(0, .25) = N(2, .25); Z = 0.3 - 0.5Y + N(0, .49)
    assert np.isclose(gm.mean[0], 0.3 - 0.5 * 2.0)
    assert np.isclose(gm.cov[0, 0], 0.25 * 0.25 + 0.49)


def test_chain_condition_matches_bivariate_normal():
    gt = GroundTruth(_chain_scm())
    gm = gt.marginal_density(["X", "Z"])
    assert np.allclose(gm.mean, [_MEAN_X, _MEAN_Z])
    assert np.allclose(gm.cov, [[_VAR_X, _COV_XZ], [_COV_XZ, _VAR_Z]], atol=1e-12)

    x0 = 1.0
    cond = gm.condition({"X": x0})
    expected_mean = _MEAN_Z + _COV_XZ / _VAR_X * (x0 - _MEAN_X)
    expected_var = _VAR_Z - _COV_XZ**2 / _VAR_X
    assert cond.var_names == ("Z",)
    assert np.isclose(cond.mean[0], expected_mean)
    assert np.isclose(cond.cov[0, 0], expected_var)

    # Evidence density equals the analytic 1-D normal pdf of X at x0.
    expected_pdf = np.exp(-0.5 * (x0 - _MEAN_X) ** 2 / _VAR_X) / np.sqrt(2 * np.pi * _VAR_X)
    assert np.isclose(gm.evidence_pdf({"X": x0}), expected_pdf)

    # Conditioning via the public API on evidence gives the same law.
    gm_ev = gt.marginal_density(["Z"], evidence={"X": x0})
    assert np.isclose(gm_ev.mean[0], expected_mean)
    assert np.isclose(gm_ev.cov[0, 0], expected_var)


def test_gmm_noise_moments():
    scm = StructuralCausalModel()
    scm.add_variable(
        "X",
        LinearGMMMechanism({}, 0.0, GaussianMixtureNoise([0.3, 0.7], [-2.0, 1.0], [0.5, 1.0])),
    )
    scm.add_variable(
        "Y",
        LinearGMMMechanism({"X": 1.5}, 0.2, GaussianMixtureNoise.single(0.0, 0.8)),
        parents=["X"],
    )
    gt = GroundTruth(scm)

    gm_x = gt.marginal_density(["X"])
    assert np.allclose(gm_x.weights, [0.3, 0.7])
    assert np.allclose(gm_x.means[:, 0], [-2.0, 1.0])

    # Law of total expectation/variance.
    e_x = 0.3 * -2.0 + 0.7 * 1.0
    var_x = 0.3 * (0.25 + 4.0) + 0.7 * (1.0 + 1.0) - e_x**2
    assert np.isclose(gm_x.mean[0], e_x)
    assert np.isclose(gm_x.cov[0, 0], var_x)

    gm_y = gt.marginal_density(["Y"])
    assert np.isclose(gm_y.mean[0], 0.2 + 1.5 * e_x)
    assert np.isclose(gm_y.cov[0, 0], 1.5**2 * var_x + 0.64)


# ---------------------------------------------------------------------------
# Mixed CLG SCMs
# ---------------------------------------------------------------------------


def _clg_scm():
    scm = StructuralCausalModel()
    scm.add_variable("Z", DirichletCPTMechanism(2, {}, np.array([0.6, 0.4])))
    scm.add_variable(
        "X",
        CLGMechanism({"Z": 2}, [], np.array([0.0, 2.0]), np.zeros((2, 0)), np.array([1.0, 0.5])),
        parents=["Z"],
    )
    scm.add_variable(
        "Y",
        CLGMechanism(
            {"Z": 2},
            ["X"],
            np.array([1.0, -1.0]),
            np.array([[0.5], [-0.8]]),
            np.array([0.3, 0.6]),
        ),
        parents=["Z", "X"],
    )
    return scm


# Per-regime analytics: Y | Z=z ~ N(int_y + coef * mu_Xz, coef^2 * var_Xz + std_y^2)
_CLG_MEANS = [1.0 + 0.5 * 0.0, -1.0 + (-0.8) * 2.0]  # [1.0, -2.6]
_CLG_VARS = [0.25 * 1.0 + 0.09, 0.64 * 0.25 + 0.36]  # [0.34, 0.52]


def test_clg_marginal_density():
    gt = GroundTruth(_clg_scm())
    assert gt.classify() == {"discrete": ["Z"], "continuous": ["X", "Y"]}
    gm = gt.marginal_density(["Y"])
    assert gm.n_components == 2
    assert np.allclose(gm.weights, [0.6, 0.4])
    assert np.allclose(gm.means[:, 0], _CLG_MEANS)
    assert np.allclose(gm.covs[:, 0, 0], _CLG_VARS)


def test_clg_density_vs_samples():
    scm = _clg_scm()
    gt = GroundTruth(scm)
    gm = gt.marginal_density(["Y"])
    np.random.seed(11)
    df = scm.sample(100_000)
    counts, edges = np.histogram(df["Y"], bins=60, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    assert np.allclose(gm.pdf(centers), counts, atol=0.08)


def test_clg_do_discrete():
    gt = GroundTruth(_clg_scm())
    gm = gt.marginal_density(["Y"], do={"Z": 1})
    assert gm.n_components == 1
    assert np.isclose(gm.mean[0], _CLG_MEANS[1])
    assert np.isclose(gm.cov[0, 0], _CLG_VARS[1])


def test_clg_do_continuous():
    gt = GroundTruth(_clg_scm())
    gm = gt.marginal_density(["Y"], do={"X": 0.5})
    # Continuous surgery leaves the discrete mixture weights untouched (CLG).
    assert np.allclose(gm.weights, [0.6, 0.4])
    assert np.allclose(gm.means[:, 0], [1.0 + 0.5 * 0.5, -1.0 + (-0.8) * 0.5])
    assert np.allclose(gm.covs[:, 0, 0], [0.09, 0.36])


def test_clg_evidence_discrete():
    gt = GroundTruth(_clg_scm())
    gm = gt.marginal_density(["Y"], evidence={"Z": 1})
    assert gm.n_components == 1
    assert np.isclose(gm.mean[0], _CLG_MEANS[1])
    assert np.isclose(gm.cov[0, 0], _CLG_VARS[1])


def test_clg_evidence_continuous():
    gt = GroundTruth(_clg_scm())
    gm = gt.marginal_density(["Y"], evidence={"X": 0.5})

    # Posterior weights: w_z ∝ P(Z=z) * N(0.5; mu_Xz, std_Xz).
    def _norm_pdf(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    w = np.array([0.6 * _norm_pdf(0.5, 0.0, 1.0), 0.4 * _norm_pdf(0.5, 2.0, 0.5)])
    w /= w.sum()
    assert np.allclose(gm.weights, w)
    # X is observed exactly, so Y | Z=z, X=0.5 = N(int + coef * 0.5, std_y^2).
    assert np.allclose(gm.means[:, 0], [1.25, -1.4])
    assert np.allclose(gm.covs[:, 0, 0], [0.09, 0.36])


def test_clg_mixed_do_and_evidence():
    gt = GroundTruth(_clg_scm())
    # do(Z=1) plus continuous evidence pins the single remaining regime.
    gm = gt.marginal_density(["Y"], do={"Z": 1}, evidence={"X": 0.5})
    assert gm.n_components == 1
    assert np.isclose(gm.mean[0], -1.4)
    assert np.isclose(gm.cov[0, 0], 0.36)


# ---------------------------------------------------------------------------
# Unsupported mechanisms
# ---------------------------------------------------------------------------


def test_constant_mechanism_unsupported():
    scm = StructuralCausalModel()
    scm.add_variable("C", ConstantMechanism(1.0))
    with pytest.raises(ValueError, match="No analytical ground truth"):
        GroundTruth(scm)


def test_uniform_noise_unsupported():
    scm = StructuralCausalModel()
    scm.add_variable("X", AdditiveNoiseMechanism(None, UniformNoise(0.0, 1.0)))
    gt = GroundTruth(scm)  # classified as continuous; failure surfaces at query time
    assert gt.classify()["continuous"] == ["X"]
    with pytest.raises(ValueError, match="No exact ground truth"):
        gt.marginal_density(["X"])


def test_unknown_query_variable_raises():
    gt = GroundTruth(_chain_scm())
    with pytest.raises(ValueError, match="Unknown variable"):
        gt.marginal_density(["Z"], do={"W": 1.0})
    with pytest.raises(ValueError, match="not continuous"):
        gt.marginal_density(["X", "nope"])
