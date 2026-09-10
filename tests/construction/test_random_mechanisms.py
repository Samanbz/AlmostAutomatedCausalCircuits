import numpy as np
import pytest

from src.construction.random_mechanisms import randomize_mechanisms, sample_dataset
from src.construction.skeleton import (
    SCMSkeleton,
    VariableSpec,
    backdoor_skeleton,
    frontdoor_skeleton,
)
from src.symbolic.scm import (
    CLGMechanism,
    DirichletCPTMechanism,
    LinearGMMMechanism,
    RegionalDiscreteMechanism,
)


def _mixed_skeleton() -> SCMSkeleton:
    """Exercises all four mechanism families (regional/dirichlet, CLG, linear-GMM)."""
    return SCMSkeleton(
        [
            VariableSpec("A", "discrete", 3),
            VariableSpec("B", "discrete", 2),
            VariableSpec("E", "continuous"),
            VariableSpec("C", "continuous"),
            VariableSpec("D", "continuous"),
            VariableSpec("F", "continuous"),
        ],
        [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("B", "F"), ("D", "F")],
    )


def _mechanism_params(mechanism) -> dict:
    if isinstance(mechanism, RegionalDiscreteMechanism):
        return {"cut_points": mechanism.cut_points, "mappings": mechanism.mappings}
    if isinstance(mechanism, DirichletCPTMechanism):
        return {"table": mechanism.table}
    if isinstance(mechanism, LinearGMMMechanism):
        return {
            "intercept": np.asarray(mechanism.intercept),
            "coefficients": np.asarray(list(mechanism.coefficients.values())),
            "weights": mechanism.noise.weights,
            "means": mechanism.noise.means,
            "stds": mechanism.noise.stds,
        }
    if isinstance(mechanism, CLGMechanism):
        return {
            "intercepts": mechanism.intercepts,
            "coefficients": mechanism.coefficients,
            "stds": mechanism.stds,
        }
    raise AssertionError(f"Unexpected mechanism type {type(mechanism)}.")


# Reproducibility


@pytest.mark.parametrize("strategy", ["regional", "dirichlet"])
def test_same_seed_identical_mechanisms(strategy):
    skeleton = _mixed_skeleton()
    scm1 = randomize_mechanisms(skeleton, 42, discrete_strategy=strategy)
    scm2 = randomize_mechanisms(skeleton, np.random.default_rng(42), discrete_strategy=strategy)
    for name in skeleton.topological_order():
        params1 = _mechanism_params(scm1.get_node_data(name))
        params2 = _mechanism_params(scm2.get_node_data(name))
        assert params1.keys() == params2.keys()
        for key in params1:
            np.testing.assert_array_equal(params1[key], params2[key])


def test_different_seeds_differ():
    skeleton = _mixed_skeleton()
    scm1 = randomize_mechanisms(skeleton, 0)
    scm2 = randomize_mechanisms(skeleton, 1)
    any_difference = any(
        not np.array_equal(p1, p2)
        for name in skeleton.topological_order()
        for p1, p2 in zip(
            _mechanism_params(scm1.get_node_data(name)).values(),
            _mechanism_params(scm2.get_node_data(name)).values(),
        )
    )
    assert any_difference


def test_mechanism_types_and_hidden_flag():
    scm = randomize_mechanisms(_mixed_skeleton(), 0)
    assert isinstance(scm.get_node_data("A"), RegionalDiscreteMechanism)
    assert isinstance(scm.get_node_data("B"), RegionalDiscreteMechanism)
    assert isinstance(scm.get_node_data("C"), CLGMechanism)  # discrete parents only
    assert isinstance(scm.get_node_data("D"), LinearGMMMechanism)  # continuous parent only
    assert isinstance(scm.get_node_data("F"), CLGMechanism)  # mixed parents

    scm_fd = randomize_mechanisms(frontdoor_skeleton(), 0)
    assert "U" in scm_fd.hidden_variables
    assert scm_fd.get_node_data("U").cardinality == 2


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="discrete_strategy"):
        randomize_mechanisms(backdoor_skeleton(), 0, discrete_strategy="nope")


# Sampling vs analytical conditionals


def test_discrete_frequencies_match_cpt():
    skeleton = SCMSkeleton(
        [VariableSpec("X", "discrete", 3), VariableSpec("Y", "discrete", 2)],
        [("X", "Y")],
    )
    scm = randomize_mechanisms(skeleton, 7, regions=3)

    np.random.seed(123)
    n = 100_000
    df = scm.sample(n)
    x = df["X"].to_numpy(dtype=int)
    y = df["Y"].to_numpy(dtype=int)

    cpt = scm.get_node_data("Y").cpt()  # shape (3, 2)
    for config in range(3):
        mask = x == config
        n_config = mask.sum()
        empirical = np.bincount(y[mask], minlength=2) / n_config
        for value in range(2):
            p = cpt[config, value]
            sigma = np.sqrt(p * (1 - p) / n_config)
            assert abs(empirical[value] - p) < max(5 * sigma, 1e-3)


def test_continuous_moments_match_gaussian_params():
    skeleton = SCMSkeleton(
        [VariableSpec("X", "continuous"), VariableSpec("Y", "continuous")],
        [("X", "Y")],
    )
    scm = randomize_mechanisms(skeleton, 11)

    np.random.seed(321)
    n = 100_000
    df = scm.sample(n)

    def noise_moments(noise):
        mean = float(np.sum(noise.weights * noise.means))
        var = float(np.sum(noise.weights * (noise.stds**2 + noise.means**2)) - mean**2)
        return mean, var

    params_x = scm.get_node_data("X").gaussian_params()
    noise_mean_x, noise_var_x = noise_moments(params_x.noise)
    mean_x = params_x.intercept + noise_mean_x
    var_x = noise_var_x

    params_y = scm.get_node_data("Y").gaussian_params()
    beta = params_y.coefficients["X"]
    noise_mean_y, noise_var_y = noise_moments(params_y.noise)
    # Law of total expectation/variance for Y = intercept + beta * X + noise.
    mean_y = params_y.intercept + beta * mean_x + noise_mean_y
    var_y = beta**2 * var_x + noise_var_y

    assert abs(df["X"].mean() - mean_x) < 0.05
    assert abs(df["Y"].mean() - mean_y) < 0.05
    assert abs(df["X"].var() - var_x) / var_x < 0.05
    assert abs(df["Y"].var() - var_y) / var_y < 0.05


def test_clg_regimes_visible():
    skeleton = SCMSkeleton(
        [VariableSpec("D", "discrete", 2), VariableSpec("Y", "continuous")],
        [("D", "Y")],
    )
    scm = randomize_mechanisms(skeleton, 5)
    mechanism = scm.get_node_data("Y")
    assert isinstance(mechanism, CLGMechanism)

    n = 100_000
    for config in (0, 1):
        np.random.seed(1000 + config)
        samples = mechanism(n, D=np.full(n, config))
        params = mechanism.gaussian_params((config,))
        sigma = params.noise.stds[0]
        assert abs(samples.mean() - params.intercept) < 5 * sigma / np.sqrt(n) + 1e-3
        assert abs(samples.std() - sigma) / sigma < 0.02


def test_sample_dataset_drops_hidden():
    scm = randomize_mechanisms(frontdoor_skeleton(), 3)
    np.random.seed(0)
    df = sample_dataset(scm, 1_000)
    assert set(df.columns) == {"X", "M", "Y"}
    df_full = sample_dataset(scm, 1_000, drop_hidden=False)
    assert "U" in df_full.columns


# Randomization knobs


def test_regions_int_and_cap():
    # Binary X with one binary parent: max_regions = 2^(2^1) = 4.
    skeleton = SCMSkeleton(
        [VariableSpec("Z", "discrete", 2), VariableSpec("X", "discrete", 2)],
        [("Z", "X")],
    )
    scm = randomize_mechanisms(skeleton, 0, regions=3)
    assert scm.get_node_data("X").n_regions == 3
    scm = randomize_mechanisms(skeleton, 0, regions=10)
    assert scm.get_node_data("X").n_regions == 4  # capped at the number of mappings


def test_regions_per_node_dict():
    skeleton = backdoor_skeleton(n_confounders=1)
    scm = randomize_mechanisms(skeleton, 0, regions={"X": 1, "Y": 5})
    assert scm.get_node_data("X").n_regions == 1  # deterministic
    assert scm.get_node_data("Y").n_regions == 5
    # Node missing from the dict falls back to the default (2 regions here).
    assert scm.get_node_data("Z_0").n_regions == 2


def test_dirichlet_alpha_controls_concentration():
    def mean_row_max(alpha):
        skeleton = SCMSkeleton([VariableSpec(f"V_{i}", "discrete", 4) for i in range(8)], [])
        scm = randomize_mechanisms(
            skeleton, 0, discrete_strategy="dirichlet", dirichlet_alpha=alpha
        )
        row_maxima = [scm.get_node_data(f"V_{i}").cpt().max(axis=-1).mean() for i in range(8)]
        return float(np.mean(row_maxima))

    low = mean_row_max(0.3)  # near-deterministic rows
    high = mean_row_max(3.0)  # closer to uniform
    assert low > 0.6
    assert high < low - 0.1


def test_gmm_and_range_knobs_respected():
    skeleton = SCMSkeleton(
        [VariableSpec("X", "continuous"), VariableSpec("Y", "continuous")],
        [("X", "Y")],
    )
    coef_range = (0.7, 1.3)
    intercept_range = (-0.5, 0.5)
    sigma2_range = (0.4, 0.9)
    mean_range = (-2.0, 2.0)
    scm = randomize_mechanisms(
        skeleton,
        9,
        gmm_components=(2,),
        coef_range=coef_range,
        intercept_range=intercept_range,
        sigma2_range=sigma2_range,
        gmm_mean_range=mean_range,
        gmm_min_mean_gap=0.8,
    )
    for name in ("X", "Y"):
        mechanism = scm.get_node_data(name)
        assert mechanism.noise.n_components == 2
        assert intercept_range[0] <= mechanism.intercept <= intercept_range[1]
        assert np.all(np.abs(mechanism.noise.means) <= mean_range[1])
        assert abs(np.diff(np.sort(mechanism.noise.means))[0]) >= 0.8
        assert np.all(sigma2_range[0] <= mechanism.noise.stds**2)
        assert np.all(mechanism.noise.stds**2 <= sigma2_range[1])
    coef = scm.get_node_data("Y").coefficients["X"]
    assert coef_range[0] <= abs(coef) <= coef_range[1]


def test_gmm_mean_placement_raises_when_infeasible():
    skeleton = SCMSkeleton([VariableSpec("X", "continuous")], [])
    with pytest.raises(ValueError, match="Could not place"):
        randomize_mechanisms(
            skeleton,
            0,
            gmm_components=(4,),
            gmm_mean_range=(0.0, 1.0),
            gmm_min_mean_gap=2.0,
        )


# Effect engineering (--direct_effect / --confounding_strength)


def _central_window(scm, treatment):
    """Mean +/-1 std of the treatment under the exact ground truth."""
    gm = scm.ground_truth().marginal_density([treatment])
    mu = float(gm.mean[0])
    std = float(np.sqrt(gm.cov[0, 0]))
    return mu - std, mu + std


def _observational_slope(scm, treatment, outcome):
    """Exact regression slope dE[outcome | treatment=x]/dx over the central +/-1 std."""
    x0, x1 = _central_window(scm, treatment)
    gt = scm.ground_truth()
    m0 = gt.marginal_density([outcome], evidence={treatment: x0})
    m1 = gt.marginal_density([outcome], evidence={treatment: x1})
    return float((m1.mean[0] - m0.mean[0]) / (x1 - x0))


def test_direct_effect_pins_treatment_outcome_edges():
    skeleton = backdoor_skeleton(n_confounders=2, kind="continuous", cardinality=None)
    scm = randomize_mechanisms(skeleton, 0, direct_effect=-2.0)
    assert scm.get_node_data("Y").coefficients["X"] == -2.0


def test_confounding_strength_cancels_direct_effect():
    skeleton = backdoor_skeleton(n_confounders=2, kind="continuous", cardinality=None)
    scm = randomize_mechanisms(skeleton, 0, direct_effect=-2.0, confounding_strength=2.0)
    assert _observational_slope(scm, "X", "Y") == pytest.approx(0.0, abs=1e-6)

    scm = randomize_mechanisms(skeleton, 0, direct_effect=1.5, confounding_strength=0.0)
    assert _observational_slope(scm, "X", "Y") == pytest.approx(1.5, abs=1e-6)

    scm = randomize_mechanisms(skeleton, 0, direct_effect=1.0, confounding_strength=1.0)
    assert _observational_slope(scm, "X", "Y") == pytest.approx(2.0, abs=1e-6)


def test_confounding_strength_is_excess_slope_over_do():
    # S is the excess of the observational slope over the interventional slope,
    # measured over the same central +/-1 std window the engineer targets.
    skeleton = backdoor_skeleton(n_confounders=1, kind="continuous", cardinality=None)

    def slopes(scm):
        gt = scm.ground_truth()
        x0, x1 = _central_window(scm, "X")
        m = lambda g: gt.marginal_density(["Y"], **g).mean[0]  # noqa: E731
        obs = (m({"evidence": {"X": x1}}) - m({"evidence": {"X": x0}})) / (x1 - x0)
        do = (m({"do": {"X": x1}}) - m({"do": {"X": x0}})) / (x1 - x0)
        return obs, do

    obs, do = slopes(randomize_mechanisms(skeleton, 0, confounding_strength=1.25))
    assert obs - do == pytest.approx(1.25, abs=1e-6)

    obs, do = slopes(
        randomize_mechanisms(skeleton, 0, direct_effect=0.5, confounding_strength=-2.0)
    )
    assert do == pytest.approx(0.5, abs=1e-6)
    assert obs == pytest.approx(0.5 - 2.0, abs=1e-6)


def test_effect_engineering_rejects_unsupported_structures():
    discrete_skel = backdoor_skeleton(n_confounders=1, kind="discrete", cardinality=2)
    with pytest.raises(ValueError, match="linear"):
        randomize_mechanisms(discrete_skel, 0, confounding_strength=1.0)

    frontdoor = frontdoor_skeleton(kind="continuous", cardinality=None)
    with pytest.raises(ValueError, match="no treatment->outcome"):
        randomize_mechanisms(frontdoor, 0, direct_effect=1.0)


def test_engineered_coefficients_drive_sampling():
    # Regression guard: sampling reads `mechanism.logic`, ground truth reads
    # `mechanism.coefficients` — they must not desync after in-place edits.
    skeleton = backdoor_skeleton(n_confounders=1, kind="continuous", cardinality=None)
    scm = randomize_mechanisms(skeleton, 0, direct_effect=-2.0, confounding_strength=1.0)
    df = scm.sample_dataset(20000, drop_hidden=False)
    design = np.column_stack([df["X"], df["Z_0"], np.ones(len(df))])
    coef, *_ = np.linalg.lstsq(design, df["Y"], rcond=None)
    y_mech = scm.get_node_data("Y")
    assert coef[0] == pytest.approx(y_mech.coefficients["X"], abs=0.05)
    assert coef[1] == pytest.approx(y_mech.coefficients["Z_0"], abs=0.05)
