"""Tests for src/construction/scm_analysis.py (validation/reporting, App. F style)."""

import numpy as np

from src.construction.random_mechanisms import randomize_mechanisms
from src.construction.scm_analysis import (
    distribution_stats,
    effect_gap,
    mechanism_stats,
    positivity_check,
)
from src.construction.skeleton import SCMSkeleton, VariableSpec, backdoor_skeleton
from src.symbolic.scm import (
    BinaryMechanism,
    DirichletCPTMechanism,
    GaussianMixtureNoise,
    LinearGMMMechanism,
    StructuralCausalModel,
)
from src.symbolic.scm.ground_truth import GroundTruth


def _uniform_binary_scm():
    """A -> B with every CPT entry exactly 1/2."""
    scm = StructuralCausalModel()
    scm.add_variable("A", BinaryMechanism(base_p=0.5))
    scm.add_variable(
        "B",
        DirichletCPTMechanism(2, {"A": 2}, np.full((2, 2), 0.5)),
        parents=["A"],
    )
    return scm


# ---------------------------------------------------------------------------
# mechanism_stats
# ---------------------------------------------------------------------------


def test_mechanism_stats_regions_entropy():
    """R=1 regional mechanisms are deterministic (H=0); higher R is strictly stochastic."""
    skeleton = backdoor_skeleton(n_confounders=1, kind="discrete", cardinality=2)
    scm_det = randomize_mechanisms(skeleton, 0, regions=1)
    scm_stoch = randomize_mechanisms(skeleton, 0, regions=10)

    stats_det = mechanism_stats(scm_det)
    stats_stoch = mechanism_stats(scm_stoch)

    assert list(stats_det.index) == ["Z_0", "X", "Y"]
    for col in ("conditional_entropy", "pearson_noise", "spearman_noise"):
        assert col in stats_det.columns
    assert "pearson_Z_0" in stats_det.columns  # X and Y both have Z_0 as parent

    assert np.allclose(stats_det["conditional_entropy"], 0.0, atol=1e-12)

    assert stats_stoch["conditional_entropy"].sum() > 0.0
    assert (stats_stoch["conditional_entropy"] > 0.0).all()


def test_mechanism_stats_noise_correlation_threshold_mechanism():
    """Root X = 1[u < 0.6]: the output is a decreasing step of its noise.

    Both correlations are strongly negative; ties in the binary output keep
    them shy of -1.
    """
    scm = StructuralCausalModel()
    scm.add_variable("X", BinaryMechanism(base_p=0.6))
    stats = mechanism_stats(scm)
    assert stats.loc["X", "spearman_noise"] < -0.8
    assert stats.loc["X", "pearson_noise"] < -0.8
    assert stats.loc["X", "conditional_entropy"] > 0.0


def test_mechanism_stats_continuous_strong_parent():
    """A child with a strong linear continuous parent shows high |rho| against it."""
    skeleton = SCMSkeleton(
        [VariableSpec("X", "continuous"), VariableSpec("Y", "continuous")],
        [("X", "Y")],
    )
    scm = randomize_mechanisms(
        skeleton,
        1,
        coef_range=(2.0, 3.0),
        sigma2_range=(0.01, 0.02),
        gmm_components=(1,),
    )
    stats = mechanism_stats(scm)
    assert stats.loc["Y", "kind"] == "continuous"
    assert abs(stats.loc["Y", "pearson_X"]) > 0.5
    assert abs(stats.loc["Y", "spearman_X"]) > 0.5
    # With a tiny noise scale the noise correlation is comparatively weak.
    assert abs(stats.loc["Y", "pearson_noise"]) < abs(stats.loc["Y", "pearson_X"])
    # Root variable has no parent columns beyond noise.
    assert stats.loc["X", "n_parents"] == 0


# ---------------------------------------------------------------------------
# distribution_stats
# ---------------------------------------------------------------------------


def test_distribution_stats_uniform():
    scm = _uniform_binary_scm()
    stats = distribution_stats(scm)["discrete"]
    assert stats["min_joint_prob"] == 0.25
    assert stats["min_marginal_prob"] == 0.5
    assert stats["zero_cell_fraction"] == 0.0
    assert np.isclose(stats["joint_l1_to_uniform"], 0.0)
    assert all(np.isclose(v, 0.0) for v in stats["marginal_l1_to_uniform"].values())
    assert np.isclose(stats["joint_entropy"], np.log(4.0))


def test_distribution_stats_matches_ground_truth():
    rng = np.random.default_rng(5)
    skeleton = backdoor_skeleton(n_confounders=1, kind="discrete", cardinality=3)
    scm = randomize_mechanisms(skeleton, rng, discrete_strategy="dirichlet", dirichlet_alpha=0.7)

    stats = distribution_stats(scm)
    joint = GroundTruth(scm).discrete_joint()
    probs = np.array(list(joint.values()))
    assert np.isclose(stats["discrete"]["min_joint_prob"], probs.min())
    assert np.isclose(stats["discrete"]["joint_entropy"], -(probs * np.log(probs)).sum())
    assert stats["continuous"] is None

    # Hand-computed tiny example: root with P = [0.4, 0.6].
    scm2 = StructuralCausalModel()
    scm2.add_variable("A", DirichletCPTMechanism(2, {}, np.array([0.4, 0.6])))
    stats2 = distribution_stats(scm2)["discrete"]
    assert np.isclose(stats2["min_marginal_prob"], 0.4)
    assert np.isclose(stats2["min_joint_prob"], 0.4)


def test_distribution_stats_continuous_part():
    scm = StructuralCausalModel()
    scm.add_variable("X", LinearGMMMechanism({}, 1.0, GaussianMixtureNoise.single(0.0, 2.0)))
    stats = distribution_stats(scm)
    assert stats["discrete"] is None
    assert np.isclose(stats["continuous"]["marginals"]["X"]["mean"], 1.0)
    assert np.isclose(stats["continuous"]["marginals"]["X"]["std"], 2.0)


# ---------------------------------------------------------------------------
# positivity_check
# ---------------------------------------------------------------------------


def test_positivity_check_flags_degenerate():
    scm = StructuralCausalModel()
    scm.add_variable("A", DirichletCPTMechanism(2, {}, np.array([0.5, 0.5])))
    scm.add_variable(
        "B",
        DirichletCPTMechanism(2, {"A": 2}, np.array([[0.9999, 0.0001], [0.5, 0.5]])),
        parents=["A"],
    )
    assert not positivity_check(scm, min_prob=1e-3)
    assert positivity_check(_uniform_binary_scm(), min_prob=1e-3)


def test_positivity_check_pure_continuous_is_true():
    scm = StructuralCausalModel()
    scm.add_variable("X", LinearGMMMechanism({}, 0.0, GaussianMixtureNoise.single(0.0, 1.0)))
    assert positivity_check(scm)


# ---------------------------------------------------------------------------
# effect_gap
# ---------------------------------------------------------------------------


def _confounded_discrete_scm():
    """Z -> X, Z -> Y; Y ignores X entirely, so all association is confounding."""
    scm = StructuralCausalModel()
    scm.add_variable("Z", DirichletCPTMechanism(2, {}, np.array([0.5, 0.5])))
    scm.add_variable(
        "X",
        DirichletCPTMechanism(2, {"Z": 2}, np.array([[0.9, 0.1], [0.1, 0.9]])),
        parents=["Z"],
    )
    scm.add_variable(
        "Y",
        DirichletCPTMechanism(2, {"Z": 2, "X": 2}, np.array([[[0.8, 0.2]] * 2, [[0.2, 0.8]] * 2])),
        parents=["Z", "X"],
    )
    return scm


def test_effect_gap_discrete_confounded():
    scm = _confounded_discrete_scm()
    gap = effect_gap(scm, "X", "Y")
    assert gap["outcome_kind"] == "discrete"
    # P(Y=1|do(X)) = 0.5 but P(Y=1|X=1) = 0.9*0.8 + 0.1*0.2 = 0.74.
    assert np.isclose(gap["per_value"][1], abs(0.74 - 0.5) + abs(0.26 - 0.5))
    assert gap["max_l1"] > 0.3
    assert gap["mean_l1"] > 0.0


def test_effect_gap_discrete_unconfounded_chain():
    """X -> Y with no confounder: P(y|do(x)) == P(y|x) exactly."""
    scm = StructuralCausalModel()
    scm.add_variable("X", DirichletCPTMechanism(2, {}, np.array([0.5, 0.5])))
    scm.add_variable(
        "Y",
        DirichletCPTMechanism(2, {"X": 2}, np.array([[0.7, 0.3], [0.2, 0.8]])),
        parents=["X"],
    )
    gap = effect_gap(scm, "X", "Y")
    assert gap["max_l1"] < 1e-12


def test_effect_gap_continuous_unconfounded_chain():
    scm = StructuralCausalModel()
    scm.add_variable("X", LinearGMMMechanism({}, 0.0, GaussianMixtureNoise.single(0.0, 1.0)))
    scm.add_variable(
        "Y",
        LinearGMMMechanism({"X": 1.5}, 0.0, GaussianMixtureNoise.single(0.0, 0.5)),
        parents=["X"],
    )
    gap = effect_gap(scm, "X", "Y")
    assert gap["outcome_kind"] == "continuous"
    assert gap["max_l1"] < 1e-6


def test_effect_gap_continuous_confounded():
    """Z -> X, Z -> Y with no X -> Y edge: do(X) leaves Y's marginal unchanged."""
    scm = StructuralCausalModel()
    scm.add_variable("Z", LinearGMMMechanism({}, 0.0, GaussianMixtureNoise.single(0.0, 1.0)))
    scm.add_variable(
        "X",
        LinearGMMMechanism({"Z": 1.0}, 0.0, GaussianMixtureNoise.single(0.0, 0.3)),
        parents=["Z"],
    )
    scm.add_variable(
        "Y",
        LinearGMMMechanism({"Z": 1.0}, 0.0, GaussianMixtureNoise.single(0.0, 0.3)),
        parents=["Z"],
    )
    gap = effect_gap(scm, "X", "Y", x_grid=np.linspace(-2.0, 2.0, 9))
    # P(Y|do(X=x)) = N(0, 1.09) for all x, while P(Y|X=x) = N(x/1.09, ~0.17):
    # the conditional is much narrower, so the gap is large everywhere.
    assert gap["mean_l1"] > 0.05
    # Symmetric in x and minimized at x = 0 (means coincide; only the variance
    # mismatch contributes there), growing as the conditional mean shifts.
    assert np.isclose(gap["per_value"][2.0], gap["per_value"][-2.0])
    assert gap["per_value"][0.0] == min(gap["per_value"].values())
    assert gap["per_value"][0.0] > 0.3
    assert gap["per_value"][2.0] > gap["per_value"][0.0]
