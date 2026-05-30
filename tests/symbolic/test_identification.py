import numpy as np
import pytest

from src.symbolic.id_ast import ast_to_str, make_p
from src.symbolic.identification import (
    IdentificationError,
    TractabilityError,
    _is_det,
    identify,
    minimize_determinisms,
    required_determinisms,
    tractable_queries,
)
from src.symbolic.scm import Mechanism, StructuralCausalModel


class DummyMechanism(Mechanism):
    def __call__(self, n_samples: int, **parents: np.ndarray) -> np.ndarray:
        return np.zeros(n_samples)

    def evaluate(self, noise: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        return np.zeros_like(noise)

    def abduct(self, value: np.ndarray, **parents: np.ndarray) -> np.ndarray:
        return np.zeros_like(value)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backdoor_scm():
    """Z -> X, Z -> Y, X -> Y (no unobserved confounders)."""
    scm = StructuralCausalModel()
    mech = DummyMechanism()
    scm.add_variable("Z", mech)
    scm.add_variable("X", mech, parents=["Z"])
    scm.add_variable("Y", mech, parents=["X", "Z"])
    return scm


@pytest.fixture
def frontdoor_scm():
    """X -> Z -> Y, U -> X, U -> Y (U unobserved)."""
    scm = StructuralCausalModel()
    mech = DummyMechanism()
    scm.add_variable("U", mech, is_exogenous=True)
    scm.add_variable("X", mech, parents=["U"])
    scm.add_variable("Z", mech, parents=["X"])
    scm.add_variable("Y", mech, parents=["Z", "U"])
    return scm


@pytest.fixture
def bow_arc_scm():
    """X -> Y, U -> X, U -> Y (U unobserved) — unidentifiable."""
    scm = StructuralCausalModel()
    mech = DummyMechanism()
    scm.add_variable("U", mech, is_exogenous=True)
    scm.add_variable("X", mech, parents=["U"])
    scm.add_variable("Y", mech, parents=["X", "U"])
    return scm


# ---------------------------------------------------------------------------
# identify() — existing behaviour
# ---------------------------------------------------------------------------


def test_frontdoor_criterion(frontdoor_scm):
    scm = frontdoor_scm
    V = scm.observable_variables
    P_ast = make_p(V)

    # No D: tractability checking disabled
    ast, D_out, K = identify({"Y"}, {"X"}, P_ast, scm)

    assert ast is not None
    assert D_out is None
    assert K is None

    expected = "MARG(Z)[PROD[MARG(X)[PROD[MARG(Y,Z)[P(X,Y,Z)], PROD[P(X,Y,Z), POW(-1)[MARG(Y)[P(X,Y,Z)]]]]], PROD[MARG(Y)[P(X,Y,Z)], POW(-1)[MARG(Y,Z)[P(X,Y,Z)]]]]]"
    assert ast_to_str(ast) == expected


def test_bow_arc_unidentifiable(bow_arc_scm):
    scm = bow_arc_scm
    V = scm.observable_variables
    P_ast = make_p(V)

    with pytest.raises(IdentificationError, match="Hedge discovered"):
        identify({"Y"}, {"X"}, P_ast, scm)


def test_backdoor_criterion(backdoor_scm):
    scm = backdoor_scm
    V = scm.observable_variables
    P_ast = make_p(V)

    # No D provided: tractability checking disabled, K and D_out are None
    ast, D_out, K = identify({"Y"}, {"X"}, P_ast, scm)

    assert ast is not None
    assert D_out is None
    assert K is None

    expected = "MARG(Z)[PROD[MARG(X,Y)[P(X,Y,Z)], PROD[P(X,Y,Z), POW(-1)[MARG(Y)[P(X,Y,Z)]]]]]"
    assert ast_to_str(ast) == expected


def test_backdoor_with_minimal_D_is_linear(backdoor_scm):
    """Minimal D = {frozenset({'Z'})}; by Prop 2, {Z,X}-determinism comes for free → K=1."""
    scm = backdoor_scm
    V = scm.observable_variables
    P_ast = make_p(V)

    D = {frozenset({"Z"})}
    _, _, K = identify({"Y"}, {"X"}, P_ast, scm, D)
    assert K == 1


def test_tractability_error_no_determinism(backdoor_scm):
    """Empty D: POW(-1) over predecessors={Z} is not covered → TractabilityError."""
    scm = backdoor_scm
    V = scm.observable_variables
    P_ast = make_p(V)

    with pytest.raises(TractabilityError, match="not deterministic"):
        identify({"Y"}, {"X"}, P_ast, scm, D=set())


def test_no_determinism_no_error_single_variable():
    """Single observable, no intervention → Line 1 base case, no POW needed."""
    scm = StructuralCausalModel()
    scm.add_variable("Y", DummyMechanism())
    V = scm.observable_variables
    P_ast = make_p(V)

    ast, D_out, K = identify({"Y"}, set(), P_ast, scm, D=set())
    assert ast is not None
    assert K == 1


# ---------------------------------------------------------------------------
# _is_det() — Proposition 2 awareness
# ---------------------------------------------------------------------------


def test_is_det_exact_match():
    D = {frozenset({"Z", "X"})}
    assert _is_det(D, frozenset({"Z", "X"})) is True


def test_is_det_subset_covers():
    # frozenset({'Z'}) ⊆ {'Z','X'} → Prop 2 implies {'Z','X'} is covered
    D = {frozenset({"Z"})}
    assert _is_det(D, frozenset({"Z", "X"})) is True


def test_is_det_superset_does_not_cover():
    # {'Z','X'} ⊄ {'Z'} → not covered
    D = {frozenset({"Z", "X"})}
    assert _is_det(D, frozenset({"Z"})) is False


def test_is_det_empty_D():
    assert _is_det(set(), frozenset({"Z"})) is False


def test_is_det_empty_scope_with_empty_Q():
    D = {frozenset()}
    assert _is_det(D, frozenset()) is True


def test_is_det_empty_scope_without_empty_Q():
    D = {frozenset({"Z"})}
    assert _is_det(D, frozenset()) is False


# ---------------------------------------------------------------------------
# minimize_determinisms()
# ---------------------------------------------------------------------------


def test_minimize_removes_supersets():
    D = {frozenset({"Z"}), frozenset({"Z", "X"}), frozenset({"Z", "X", "Y"})}
    assert minimize_determinisms(D) == {frozenset({"Z"})}


def test_minimize_antichain_unchanged():
    D = {frozenset({"Z"}), frozenset({"X", "Y"})}
    assert minimize_determinisms(D) == D


def test_minimize_empty():
    assert minimize_determinisms(set()) == set()


def test_minimize_single():
    D = {frozenset({"Z"})}
    assert minimize_determinisms(D) == D


# ---------------------------------------------------------------------------
# required_determinisms()
# ---------------------------------------------------------------------------


def test_required_determinisms_backdoor(backdoor_scm):
    """Backdoor (Z->X->Y, Z->Y): the only POW site is preds(Y) = {Z, X}.
    Sub-call 1 (identify {Z}) goes through Line 2 base-case with empty x, so no POW there.
    """
    required = required_determinisms({"Y"}, {"X"}, backdoor_scm)
    assert required == {frozenset({"Z", "X"})}


def test_required_determinisms_minimize_backdoor(backdoor_scm):
    """required is already minimal: {frozenset({'Z','X'})} has no strict subset in required."""
    required = required_determinisms({"Y"}, {"X"}, backdoor_scm)
    minimal = minimize_determinisms(required)
    assert minimal == {frozenset({"Z", "X"})}


def test_required_determinisms_frontdoor(frontdoor_scm):
    """Frontdoor query should be identifiable; required set is non-trivially populated."""
    required = required_determinisms({"Y"}, {"X"}, frontdoor_scm)
    assert required is not None
    # Must have collected at least one POW requirement from the inner conditional products
    assert len(required) > 0


def test_required_determinisms_unidentifiable(bow_arc_scm):
    with pytest.raises(IdentificationError):
        required_determinisms({"Y"}, {"X"}, bow_arc_scm)


def test_required_determinisms_base_case():
    """No intervention: x=∅ → Line 1 fires, no POW, required = ∅."""
    scm = StructuralCausalModel()
    scm.add_variable("Y", DummyMechanism())
    assert required_determinisms({"Y"}, set(), scm) == set()


def test_required_and_identify_consistent(backdoor_scm):
    """required_determinisms produces the minimum D to avoid TractabilityError.

    Note: K=1 (linear) may require a *stronger* D (smaller frozensets that survive
    marginalizations) than what required_determinisms returns. Here required = {{'Z','X'}},
    which is sufficient for tractability (K ∈ {1,2}), but gives K=2 because {Z,X}-determinism
    is destroyed when sub-call 1 marginalizes {X,Y}. Using D = {{'Z'}} gives K=1 because
    Z-determinism survives that marginalization.
    """
    scm = backdoor_scm
    V = scm.observable_variables
    P_ast = make_p(V)

    required = required_determinisms({"Y"}, {"X"}, scm)

    # required D makes the query tractable (no TractabilityError), K is defined
    ast, _, K = identify({"Y"}, {"X"}, P_ast, scm, required)
    assert ast is not None
    assert K is not None

    # A stronger D (smaller frozenset, survives marginalization) achieves K=1
    stronger_D = {frozenset({"Z"})}
    ast2, _, K2 = identify({"Y"}, {"X"}, P_ast, scm, stronger_D)
    assert ast2 is not None
    assert K2 == 1


# ---------------------------------------------------------------------------
# tractable_queries()
# ---------------------------------------------------------------------------


def test_tractable_queries_includes_base_case(backdoor_scm):
    """P(Y) with no intervention is always tractable (K=1)."""
    results = tractable_queries(backdoor_scm, set())
    result_pairs = {(y, x) for y, x, _ in results}
    assert (frozenset({"Y"}), frozenset()) in result_pairs


def test_tractable_queries_backdoor_with_D(backdoor_scm):
    """With minimal D, the backdoor query should appear with K=1."""
    D = {frozenset({"Z"})}
    results = tractable_queries(backdoor_scm, D)
    found = [(y, x, K) for y, x, K in results if y == frozenset({"Y"}) and x == frozenset({"X"})]
    assert len(found) == 1
    assert found[0][2] == 1


def test_tractable_queries_excludes_intractable(backdoor_scm):
    """With empty D, the backdoor query (which needs POW) should be excluded."""
    results = tractable_queries(backdoor_scm, set())
    result_pairs = {(y, x) for y, x, _ in results}
    assert (frozenset({"Y"}), frozenset({"X"})) not in result_pairs


def test_tractable_queries_excludes_unidentifiable(bow_arc_scm):
    """Unidentifiable queries (hedge) should not appear regardless of D."""
    D = {frozenset({"X"}), frozenset({"Y"}), frozenset({"X", "Y"})}
    results = tractable_queries(bow_arc_scm, D)
    result_pairs = {(y, x) for y, x, _ in results}
    assert (frozenset({"Y"}), frozenset({"X"})) not in result_pairs
