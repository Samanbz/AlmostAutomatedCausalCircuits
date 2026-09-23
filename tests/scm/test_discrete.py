"""Tests for src/symbolic/scm/discrete.py and the shared helpers it uses."""

import numpy as np
import pytest

from src.symbolic.scm.discrete import (
    BinaryMechanism,
    DirichletCPTMechanism,
    RegionalDiscreteMechanism,
)
from src.symbolic.scm.mechanisms import LogisticLogic, mixed_radix_index


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# mixed_radix_index
# ---------------------------------------------------------------------------


def test_mixed_radix_index_hand_computed():
    # First cardinality entry is the most significant digit: idx = A * 3 + B.
    idx = mixed_radix_index(
        {"A": np.array([1, 0, 1]), "B": np.array([2, 1, 0])},
        {"A": 2, "B": 3},
    )
    np.testing.assert_array_equal(idx, [5, 1, 3])


def test_mixed_radix_index_single_parent():
    idx = mixed_radix_index({"A": np.array([0, 2, 1])}, {"A": 3})
    np.testing.assert_array_equal(idx, [0, 2, 1])


def test_mixed_radix_index_no_parents():
    idx = mixed_radix_index({}, {})
    np.testing.assert_array_equal(idx, [0])


def test_mixed_radix_index_out_of_range_raises():
    with pytest.raises(ValueError, match="outside"):
        mixed_radix_index({"A": np.array([2])}, {"A": 2})
    with pytest.raises(ValueError, match="outside"):
        mixed_radix_index({"A": np.array([-1])}, {"A": 2})


def test_mixed_radix_index_missing_parent_raises():
    with pytest.raises(KeyError, match="Missing"):
        mixed_radix_index({"A": np.array([0])}, {"A": 2, "B": 3})


# ---------------------------------------------------------------------------
# RegionalDiscreteMechanism
# ---------------------------------------------------------------------------


def _tiny_regional():
    """One binary parent, C=2, two regions with swapped mappings."""
    return RegionalDiscreteMechanism(
        cardinality=2,
        parent_cardinalities={"A": 2},
        cut_points=np.array([0.0, 0.4, 1.0]),
        mappings=np.array([[1, 0], [0, 1]]),
    )


def test_regional_r1_deterministic():
    """R=1: a single region spanning [0,1], so noise cannot change the output."""
    mech = RegionalDiscreteMechanism(
        cardinality=3,
        parent_cardinalities={"A": 2},
        cut_points=np.array([0.0, 1.0]),
        mappings=np.array([[2, 1]]),
    )
    noise = np.array([0.01, 0.5, 0.99, 0.01, 0.99])
    parents = {"A": np.array([0, 0, 0, 1, 1])}
    out = mech.evaluate(noise, **parents)
    np.testing.assert_array_equal(out, [2, 2, 2, 1, 1])

    # Same via the random constructor.
    rng = np.random.default_rng(0)
    mech_r = RegionalDiscreteMechanism.random(3, {"A": 2}, n_regions=1, rng=rng)
    assert mech_r.n_regions == 1
    np.testing.assert_array_equal(mech_r.cut_points, [0.0, 1.0])
    outs = mech_r.evaluate(np.linspace(0.0, 1.0, 11), A=np.zeros(11, dtype=np.int64))
    assert np.all(outs == outs[0])


def test_regional_mappings_pairwise_distinct():
    rng = np.random.default_rng(42)
    mech = RegionalDiscreteMechanism.random(2, {"A": 2, "B": 2}, n_regions=5, rng=rng)
    assert mech.n_regions == 5
    for r in range(mech.n_regions):
        for s in range(r + 1, mech.n_regions):
            assert not np.array_equal(mech.mappings[r], mech.mappings[s])


def test_regional_n_regions_capped_at_max():
    # C=2, 2 binary parents -> 2^4 = 16 distinct mappings max.
    rng = np.random.default_rng(1)
    mech = RegionalDiscreteMechanism.random(2, {"A": 2, "B": 2}, n_regions=1000, rng=rng)
    assert mech.n_regions == mech.max_regions == 16
    # Even at the cap, all mappings remain pairwise distinct.
    for r in range(mech.n_regions):
        for s in range(r + 1, mech.n_regions):
            assert not np.array_equal(mech.mappings[r], mech.mappings[s])


def test_regional_cpt_rows_sum_to_one():
    rng = np.random.default_rng(2)
    mech = RegionalDiscreteMechanism.random(3, {"A": 2, "B": 3}, n_regions=4, rng=rng)
    table = mech.cpt()
    assert table.shape == (2, 3, 3)
    np.testing.assert_allclose(table.sum(axis=-1), 1.0)
    assert np.all(table >= 0.0)


def test_regional_evaluate_hand_computed():
    mech = _tiny_regional()
    # Regions: [0, 0.4) -> mapping [1, 0]; [0.4, 1] -> mapping [0, 1].
    # noise 0.2 (region 0): A=0 -> 1, A=1 -> 0.
    # noise 0.6 (region 1): A=0 -> 0, A=1 -> 1.
    # noise exactly 0.4 falls in region 1 (searchsorted side="right").
    noise = np.array([0.2, 0.2, 0.6, 0.6, 0.4, 0.0, 1.0])
    parents = {"A": np.array([0, 1, 0, 1, 0, 1, 1])}
    out = mech.evaluate(noise, **parents)
    np.testing.assert_array_equal(out, [1, 0, 0, 1, 0, 0, 1])


def test_regional_cpt_hand_computed():
    mech = _tiny_regional()
    # Region widths 0.4 and 0.6. A=0: value 1 in region 0 -> P=[0.6, 0.4].
    # A=1: value 0 in region 0 -> P=[0.4, 0.6].
    np.testing.assert_allclose(mech.cpt(), [[0.6, 0.4], [0.4, 0.6]])


def test_regional_cpt_matches_monte_carlo():
    rng = np.random.default_rng(3)
    mech = RegionalDiscreteMechanism.random(3, {"A": 2}, n_regions=3, rng=rng)
    table = mech.cpt()  # shape (2, 3)

    np.random.seed(0)
    n = 200_000
    parents = {"A": np.random.randint(0, 2, size=n)}
    samples = mech(n, **parents)
    for a in range(2):
        mask = parents["A"] == a
        counts = np.bincount(samples[mask].astype(np.int64), minlength=3)
        freqs = counts / counts.sum()
        sigma = np.sqrt(table[a] * (1 - table[a]) / counts.sum())
        assert np.all(np.abs(freqs - table[a]) <= 5 * sigma + 1e-12)


def test_regional_cpt_parent_axis_reordering():
    mech = RegionalDiscreteMechanism(
        cardinality=2,
        parent_cardinalities={"A": 2, "B": 3},
        cut_points=np.array([0.0, 0.25, 1.0]),
        # config idx = A * 3 + B (A most significant)
        mappings=np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]]),
    )
    table = mech.cpt()
    assert table.shape == (2, 3, 2)
    reordered = mech.cpt(parent_names=["B", "A"])
    assert reordered.shape == (3, 2, 2)
    np.testing.assert_array_equal(reordered, np.transpose(table, (1, 0, 2)))
    # Spot-check a hand-computed entry: config A=1, B=2 -> idx 5.
    # Region 0 (width 0.25) maps to 1, region 1 (width 0.75) maps to 0.
    np.testing.assert_allclose(table[1, 2], [0.75, 0.25])
    np.testing.assert_allclose(reordered[2, 1], [0.75, 0.25])


def test_regional_cpt_parent_names_mismatch_raises():
    mech = _tiny_regional()
    with pytest.raises(ValueError, match="do not match"):
        mech.cpt(parent_names=["B"])
    with pytest.raises(ValueError, match="do not match"):
        mech.cpt(parent_names=["A", "B"])


def test_regional_constructor_validation():
    cards = {"A": 2}
    good_cuts = np.array([0.0, 0.5, 1.0])
    good_maps = np.array([[0, 1], [1, 0]])

    with pytest.raises(ValueError, match="shape"):
        RegionalDiscreteMechanism(2, cards, good_cuts, np.array([[0, 1]]))  # wrong n_regions
    with pytest.raises(ValueError, match="shape"):
        RegionalDiscreteMechanism(2, cards, good_cuts, np.array([[0, 1, 0], [1, 0, 1]]))
    with pytest.raises(ValueError, match="start at 0 and end at 1"):
        RegionalDiscreteMechanism(2, cards, np.array([0.1, 0.5, 1.0]), good_maps)
    with pytest.raises(ValueError, match="start at 0 and end at 1"):
        RegionalDiscreteMechanism(2, cards, np.array([0.0, 0.5, 0.9]), good_maps)
    with pytest.raises(ValueError, match="strictly increasing"):
        RegionalDiscreteMechanism(
            2, cards, np.array([0.0, 0.7, 0.7, 1.0]), np.array([[0, 1], [1, 0], [0, 0]])
        )
    with pytest.raises(ValueError, match="outside"):
        RegionalDiscreteMechanism(2, cards, good_cuts, np.array([[0, 2], [1, 0]]))
    with pytest.raises(ValueError, match="outside"):
        RegionalDiscreteMechanism(2, cards, good_cuts, np.array([[0, -1], [1, 0]]))
    with pytest.raises(ValueError, match=">= 1"):
        RegionalDiscreteMechanism.random(2, cards, n_regions=0, rng=np.random.default_rng(0))


def test_regional_abduct_raises():
    with pytest.raises(NotImplementedError):
        _tiny_regional().abduct(np.array([0.0]), A=np.array([0.0]))


# ---------------------------------------------------------------------------
# DirichletCPTMechanism
# ---------------------------------------------------------------------------


def _tiny_dirichlet():
    """One binary parent, C=3; row 1 is degenerate (all mass on value 2)."""
    return DirichletCPTMechanism(
        cardinality=3,
        parent_cardinalities={"A": 2},
        table=np.array([[0.2, 0.3, 0.5], [0.0, 0.0, 1.0]]),
    )


def test_dirichlet_random_table_shape_and_validity():
    rng = np.random.default_rng(7)
    mech = DirichletCPTMechanism.random(3, {"A": 2, "B": 4}, alpha=1.5, rng=rng)
    assert mech.table.shape == (2, 4, 3)
    assert np.all(mech.table > 0.0)  # Dirichlet rows are strictly positive
    np.testing.assert_allclose(mech.table.sum(axis=-1), 1.0)
    assert mech.parent_cardinalities() == {"A": 2, "B": 4}


def test_dirichlet_cpt_returns_table():
    mech = _tiny_dirichlet()
    np.testing.assert_array_equal(mech.cpt(), mech.table)
    reordered = mech.cpt(parent_names=["A"])
    np.testing.assert_array_equal(reordered, mech.table)


def test_dirichlet_cpt_parent_axis_reordering():
    table = np.array(
        [
            [[0.5, 0.5], [0.1, 0.9], [0.3, 0.7]],
            [[0.8, 0.2], [0.4, 0.6], [0.0, 1.0]],
        ]
    )
    mech = DirichletCPTMechanism(2, {"A": 2, "B": 3}, table)
    reordered = mech.cpt(parent_names=["B", "A"])
    assert reordered.shape == (3, 2, 2)
    np.testing.assert_array_equal(reordered, np.transpose(table, (1, 0, 2)))
    with pytest.raises(ValueError, match="do not match"):
        mech.cpt(parent_names=["A", "C"])


def test_dirichlet_evaluate_inverse_cdf_hand_computed():
    mech = _tiny_dirichlet()
    # Row 0: cumprobs [0.2, 0.5, 1.0]; value = #{cumprob <= noise}.
    #   noise 0.10 -> 0; 0.20 -> 1 (boundary counts); 0.49 -> 1; 0.50 -> 2; 0.99 -> 2.
    # Row 1 (degenerate [0, 0, 1]): any noise in [0, 1] -> 2.
    noise = np.array([0.10, 0.20, 0.49, 0.50, 0.99, 0.0, 0.5, 0.999])
    parents = {"A": np.array([0, 0, 0, 0, 0, 1, 1, 1])}
    out = mech.evaluate(noise, **parents)
    np.testing.assert_array_equal(out, [0, 1, 1, 2, 2, 2, 2, 2])


def test_dirichlet_frequencies_match_table():
    rng = np.random.default_rng(11)
    mech = DirichletCPTMechanism.random(4, {"A": 3}, alpha=2.0, rng=rng)
    table = mech.cpt()  # shape (3, 4)

    np.random.seed(1)
    n = 200_000
    parents = {"A": np.random.randint(0, 3, size=n)}
    samples = mech(n, **parents).astype(np.int64)
    for a in range(3):
        mask = parents["A"] == a
        counts = np.bincount(samples[mask], minlength=4)
        freqs = counts / counts.sum()
        sigma = np.sqrt(table[a] * (1 - table[a]) / counts.sum())
        assert np.all(np.abs(freqs - table[a]) <= 5 * sigma + 1e-12)


def test_dirichlet_constructor_validation():
    with pytest.raises(ValueError, match="shape"):
        DirichletCPTMechanism(3, {"A": 2}, np.array([[0.5, 0.5, 0.0]]))  # wrong n_configs
    with pytest.raises(ValueError, match="sum to 1"):
        DirichletCPTMechanism(2, {"A": 2}, np.array([[0.5, 0.6], [0.5, 0.5]]))
    with pytest.raises(ValueError, match="sum to 1"):
        DirichletCPTMechanism(2, {"A": 2}, np.array([[1.5, -0.5], [0.5, 0.5]]))


def test_dirichlet_abduct_raises():
    with pytest.raises(NotImplementedError):
        _tiny_dirichlet().abduct(np.array([0.0]), A=np.array([0.0]))


# ---------------------------------------------------------------------------
# BinaryMechanism (legacy)
# ---------------------------------------------------------------------------


def test_binary_cpt_matches_logistic_grid():
    logic = LogisticLogic({"Z0": 1.5, "Z1": -0.5}, intercept=0.25)
    mech = BinaryMechanism(logic=logic)
    table = mech.cpt()
    assert table.shape == (2, 2, 2)

    # Evaluate the logistic rule on the full parent grid by hand.
    for z0 in range(2):
        for z1 in range(2):
            p1 = _sigmoid(0.25 + 1.5 * z0 - 0.5 * z1)
            np.testing.assert_allclose(table[z0, z1], [1.0 - p1, p1])

    # Reordered axes must be the transpose of the default order.
    np.testing.assert_allclose(mech.cpt(parent_names=["Z1", "Z0"]), np.transpose(table, (1, 0, 2)))


def test_binary_cpt_parentless_base_p():
    mech = BinaryMechanism(base_p=0.3)
    assert mech.parent_cardinalities() == {}
    np.testing.assert_allclose(mech.cpt(), [0.7, 0.3])

    # evaluate applies the same threshold rule.
    noise = np.array([0.0, 0.29, 0.3, 0.9])
    np.testing.assert_array_equal(mech.evaluate(noise), [1.0, 1.0, 0.0, 0.0])


def test_binary_evaluate_matches_logic():
    logic = LogisticLogic({"Z0": 2.0}, intercept=-1.0)
    mech = BinaryMechanism(logic=logic)
    z0 = np.array([0.0, 1.0, 0.0, 1.0])
    noise = np.array([0.1, 0.5, 0.9, 0.95])
    probs = logic(Z0=z0)
    np.testing.assert_array_equal(mech.evaluate(noise, Z0=z0), (noise < probs).astype(np.float64))


def test_binary_cpt_parent_names_mismatch_raises():
    mech = BinaryMechanism(logic=LogisticLogic({"Z0": 1.0}))
    with pytest.raises(ValueError, match="do not match"):
        mech.cpt(parent_names=["Z1"])


def test_binary_abduct_raises():
    mech = BinaryMechanism(base_p=0.5)
    with pytest.raises(NotImplementedError):
        mech.abduct(np.array([1.0]))
