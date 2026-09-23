"""Tests for src/symbolic/scm/continuous.py and GaussianMixtureNoise."""

import numpy as np
import pytest

from src.symbolic.scm.continuous import (
    AdditiveNoiseMechanism,
    CLGMechanism,
    LinearGMMMechanism,
)
from src.symbolic.scm.mechanisms import (
    GaussianMixtureNoise,
    GaussianNoise,
    LinearLogic,
    LogisticLogic,
    UniformNoise,
)


def _gmm_moments(weights, means, stds):
    """Exact mean/variance/4th central moment of a 1-D Gaussian mixture."""
    w = np.asarray(weights, dtype=np.float64)
    m = np.asarray(means, dtype=np.float64)
    s = np.asarray(stds, dtype=np.float64)
    mean = float(w @ m)
    var = float(w @ (s**2 + m**2) - mean**2)
    a = m - mean
    central4 = float(w @ (a**4 + 6 * a**2 * s**2 + 3 * s**4))
    return mean, var, central4


# ---------------------------------------------------------------------------
# GaussianMixtureNoise
# ---------------------------------------------------------------------------


def test_gmm_noise_constructor_validation():
    with pytest.raises(ValueError, match="same length"):
        GaussianMixtureNoise([0.5, 0.5], [0.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        GaussianMixtureNoise([1.0], [0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="sum to 1"):
        GaussianMixtureNoise([0.5, 0.6], [0.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        GaussianMixtureNoise([1.2, -0.2], [0.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        GaussianMixtureNoise([1.0], [0.0], [0.0])
    with pytest.raises(ValueError, match="positive"):
        GaussianMixtureNoise([0.5, 0.5], [0.0, 1.0], [1.0, -1.0])
    with pytest.raises(ValueError, match="1-D"):
        GaussianMixtureNoise([[0.5, 0.5]], [[0.0, 1.0]], [[1.0, 1.0]])


def test_gmm_noise_single():
    noise = GaussianMixtureNoise.single(mean=1.5, std=0.5)
    assert noise.n_components == 1
    np.testing.assert_array_equal(noise.weights, [1.0])
    np.testing.assert_array_equal(noise.means, [1.5])
    np.testing.assert_array_equal(noise.stds, [0.5])


def test_gmm_noise_moments_match_theory():
    weights = [0.3, 0.7]
    means = [-2.0, 3.0]
    stds = [0.5, 1.0]
    noise = GaussianMixtureNoise(weights, means, stds)

    np.random.seed(0)
    n = 200_000
    samples = noise(n)
    assert samples.shape == (n,)

    mean, var, central4 = _gmm_moments(weights, means, stds)
    mean_tol = 6 * np.sqrt(var / n)
    # Var of the sample variance is (central4 - var^2) / n.
    var_tol = 6 * np.sqrt((central4 - var**2) / n)
    assert abs(samples.mean() - mean) < mean_tol
    assert abs(samples.var() - var) < var_tol


# ---------------------------------------------------------------------------
# LinearGMMMechanism
# ---------------------------------------------------------------------------


def _linear_gmm():
    noise = GaussianMixtureNoise([0.4, 0.6], [0.0, 1.0], [0.5, 2.0])
    return LinearGMMMechanism(coefficients={"X": 2.0}, intercept=1.0, noise=noise)


def test_linear_gmm_gaussian_params_returns_constructor_values():
    mech = _linear_gmm()
    params = mech.gaussian_params()
    assert params.intercept == 1.0
    assert params.coefficients == {"X": 2.0}
    assert params.noise is mech.noise
    assert mech.discrete_parent_cardinalities == {}
    assert mech.continuous_parent_names == ["X"]


def test_linear_gmm_gaussian_params_rejects_discrete_config():
    mech = _linear_gmm()
    with pytest.raises(ValueError, match="no discrete parents"):
        mech.gaussian_params((0,))


def test_linear_gmm_sampling_moments():
    mech = _linear_gmm()
    noise_mean, noise_var, _ = _gmm_moments(mech.noise.weights, mech.noise.means, mech.noise.stds)

    np.random.seed(1)
    n = 200_000
    x = np.tile([0.0, 1.0], n // 2)
    samples = mech(n, X=x)

    for x_val in (0.0, 1.0):
        subset = samples[x == x_val]
        expected_mean = 1.0 + 2.0 * x_val + noise_mean
        mean_tol = 6 * np.sqrt(noise_var / len(subset))
        assert abs(subset.mean() - expected_mean) < mean_tol
        # Var of the sample variance of a Gaussian ~ 2 var^2 / n; the mixture is
        # heavier-tailed, so allow a generous factor.
        var_tol = 6 * np.sqrt(2 * noise_var**2 / len(subset)) * 3
        assert abs(subset.var() - noise_var) < var_tol


def test_linear_gmm_abduction_round_trip():
    mech = _linear_gmm()
    u = np.linspace(-3.0, 3.0, 11)
    x = np.full(11, 0.7)
    values = mech.evaluate(u, X=x)
    np.testing.assert_allclose(mech.abduct(values, X=x), u, atol=1e-12)


# ---------------------------------------------------------------------------
# CLGMechanism
# ---------------------------------------------------------------------------


def _clg():
    return CLGMechanism(
        discrete_parent_cardinalities={"D": 3},
        continuous_parent_names=["X"],
        intercepts=np.array([1.0, -2.0, 0.5]),
        coefficients=np.array([[0.5], [-1.0], [2.0]]),
        stds=np.array([0.5, 1.0, 2.0]),
    )


def test_clg_gaussian_params_per_config():
    mech = _clg()
    expected_intercepts = [1.0, -2.0, 0.5]
    expected_coefs = [0.5, -1.0, 2.0]
    expected_stds = [0.5, 1.0, 2.0]
    for d in range(3):
        params = mech.gaussian_params((d,))
        assert params.intercept == expected_intercepts[d]
        assert params.coefficients == {"X": expected_coefs[d]}
        assert params.noise.n_components == 1
        np.testing.assert_array_equal(params.noise.weights, [1.0])
        np.testing.assert_array_equal(params.noise.means, [0.0])
        np.testing.assert_array_equal(params.noise.stds, [expected_stds[d]])

    assert mech.discrete_parent_cardinalities == {"D": 3}
    assert mech.continuous_parent_names == ["X"]


def test_clg_gaussian_params_out_of_range_config_raises():
    mech = _clg()
    with pytest.raises(ValueError, match="out of range"):
        mech.gaussian_params((3,))
    with pytest.raises(ValueError, match="out of range"):
        mech.gaussian_params((-1,))
    with pytest.raises(ValueError, match="Expected a config"):
        mech.gaussian_params(())
    with pytest.raises(ValueError, match="Expected a config"):
        mech.gaussian_params((0, 1))


def test_clg_regime_switching():
    """With X fixed at 0, each discrete config d gives N(intercepts[d], stds[d]^2)."""
    mech = _clg()
    intercepts = [1.0, -2.0, 0.5]
    stds = [0.5, 1.0, 2.0]

    np.random.seed(2)
    n = 50_000
    for d in range(3):
        samples = mech(n, D=np.full(n, d), X=np.zeros(n))
        mean_tol = 6 * stds[d] / np.sqrt(n)
        std_tol = 6 * stds[d] / np.sqrt(2 * n)
        assert abs(samples.mean() - intercepts[d]) < mean_tol
        assert abs(samples.std() - stds[d]) < std_tol


def test_clg_evaluate_deterministic_with_zero_noise():
    mech = _clg()
    noise = np.zeros(4)
    d = np.array([0, 1, 2, 2])
    x = np.array([2.0, 1.0, -1.0, 3.0])
    # mean = intercepts[d] + coefs[d] * x
    expected = np.array([1.0 + 0.5 * 2.0, -2.0 - 1.0 * 1.0, 0.5 + 2.0 * -1.0, 0.5 + 2.0 * 3.0])
    np.testing.assert_allclose(mech.evaluate(noise, D=d, X=x), expected)


def test_clg_abduction_round_trip():
    mech = _clg()
    rng = np.random.default_rng(5)
    n = 1000
    u = rng.normal(size=n)
    d = rng.integers(0, 3, size=n)
    x = rng.normal(size=n)
    values = mech.evaluate(u, D=d, X=x)
    np.testing.assert_allclose(mech.abduct(values, D=d, X=x), u, atol=1e-10)


def test_clg_constructor_validation():
    cards = {"D": 3}
    names = ["X"]
    with pytest.raises(ValueError, match="intercepts"):
        CLGMechanism(
            cards,
            names,
            np.array([1.0, 2.0]),
            np.array([[0.5], [0.5], [0.5]]),
            np.array([1.0, 1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="coefficients"):
        CLGMechanism(
            cards,
            names,
            np.array([1.0, 2.0, 3.0]),
            np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
            np.array([1.0, 1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="stds"):
        CLGMechanism(
            cards,
            names,
            np.array([1.0, 2.0, 3.0]),
            np.array([[0.5], [0.5], [0.5]]),
            np.array([1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="stds"):
        CLGMechanism(
            cards,
            names,
            np.array([1.0, 2.0, 3.0]),
            np.array([[0.5], [0.5], [0.5]]),
            np.array([1.0, 0.0, 1.0]),
        )
    with pytest.raises(ValueError, match="stds"):
        CLGMechanism(
            cards,
            names,
            np.array([1.0, 2.0, 3.0]),
            np.array([[0.5], [0.5], [0.5]]),
            np.array([1.0, -1.0, 1.0]),
        )


# ---------------------------------------------------------------------------
# AdditiveNoiseMechanism.gaussian_params
# ---------------------------------------------------------------------------


def test_additive_gaussian_params_linear_gaussian():
    mech = AdditiveNoiseMechanism(
        logic=LinearLogic({"X": 2.0, "Z": -1.0}, intercept=0.5),
        noise_dist=GaussianNoise(loc=0.25, scale=1.5),
    )
    params = mech.gaussian_params()
    assert params.intercept == 0.5
    assert params.coefficients == {"X": 2.0, "Z": -1.0}
    # GaussianNoise is wrapped as a K=1 mixture.
    assert params.noise.n_components == 1
    np.testing.assert_array_equal(params.noise.weights, [1.0])
    np.testing.assert_array_equal(params.noise.means, [0.25])
    np.testing.assert_array_equal(params.noise.stds, [1.5])


def test_additive_gaussian_params_linear_gmm_noise():
    noise = GaussianMixtureNoise([0.5, 0.5], [0.0, 2.0], [1.0, 0.5])
    mech = AdditiveNoiseMechanism(logic=LinearLogic({"X": 1.0}, intercept=0.0), noise_dist=noise)
    params = mech.gaussian_params()
    assert params.noise is noise


def test_additive_gaussian_params_no_logic():
    mech = AdditiveNoiseMechanism(logic=None, noise_dist=GaussianNoise(loc=1.0, scale=2.0))
    params = mech.gaussian_params()
    assert params.intercept == 0.0
    assert params.coefficients == {}
    np.testing.assert_array_equal(params.noise.means, [1.0])


def test_additive_gaussian_params_nonlinear_logic_raises():
    mech = AdditiveNoiseMechanism(
        logic=LogisticLogic({"X": 1.0}), noise_dist=GaussianNoise(loc=0.0, scale=1.0)
    )
    with pytest.raises(ValueError, match="non-linear logic"):
        mech.gaussian_params()


def test_additive_gaussian_params_uniform_noise_raises():
    mech = AdditiveNoiseMechanism(
        logic=LinearLogic({"X": 1.0}), noise_dist=UniformNoise(low=-1.0, high=1.0)
    )
    with pytest.raises(ValueError, match="noise of type"):
        mech.gaussian_params()


def test_additive_gaussian_params_rejects_discrete_config():
    mech = AdditiveNoiseMechanism(
        logic=LinearLogic({"X": 1.0}), noise_dist=GaussianNoise(loc=0.0, scale=1.0)
    )
    with pytest.raises(ValueError, match="no discrete parents"):
        mech.gaussian_params((0,))
