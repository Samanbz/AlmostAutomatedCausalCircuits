import pytest

from src.construction.skeleton import (
    SCMSkeleton,
    VariableSpec,
    backdoor_skeleton,
    frontdoor_skeleton,
)


# Construction validation


def test_valid_construction_and_views():
    skeleton = SCMSkeleton(
        [
            VariableSpec("U", "discrete", 2, hidden=True),
            VariableSpec("Z", "discrete", 3),
            VariableSpec("X", "continuous"),
            VariableSpec("Y", "continuous"),
        ],
        [("U", "Z"), ("U", "X"), ("Z", "X"), ("X", "Y")],
    )
    assert skeleton.parents_of("X") == ["U", "Z"]
    assert skeleton.parents_of("U") == []
    assert [v.name for v in skeleton.discrete_variables] == ["U", "Z"]
    assert [v.name for v in skeleton.continuous_variables] == ["X", "Y"]
    assert [v.name for v in skeleton.hidden_variables] == ["U"]
    # Topological order respects all edges.
    order = skeleton.topological_order()
    position = {name: i for i, name in enumerate(order)}
    for a, b in skeleton.edges:
        assert position[a] < position[b]


def test_duplicate_names_raise():
    with pytest.raises(ValueError, match="Duplicate"):
        SCMSkeleton([VariableSpec("X", "discrete", 2), VariableSpec("X", "continuous")], [])


def test_unknown_edge_endpoint_raises():
    with pytest.raises(ValueError, match="unknown variable"):
        SCMSkeleton([VariableSpec("X", "discrete", 2)], [("X", "Y")])


def test_cycle_raises():
    variables = [VariableSpec(n, "discrete", 2) for n in ("A", "B", "C")]
    with pytest.raises(ValueError, match="cycle"):
        SCMSkeleton(variables, [("A", "B"), ("B", "C"), ("C", "A")])


def test_discrete_with_continuous_parent_raises():
    variables = [VariableSpec("X", "continuous"), VariableSpec("Y", "discrete", 2)]
    with pytest.raises(ValueError, match="CLG"):
        SCMSkeleton(variables, [("X", "Y")])


def test_cardinality_rules():
    with pytest.raises(ValueError, match="requires a cardinality"):
        VariableSpec("X", "discrete")
    with pytest.raises(ValueError, match=">= 2"):
        VariableSpec("X", "discrete", 1)
    with pytest.raises(ValueError, match="must not set a cardinality"):
        VariableSpec("X", "continuous", 2)


def test_parents_of_unknown_raises():
    skeleton = backdoor_skeleton()
    with pytest.raises(KeyError):
        skeleton.parents_of("nope")


# Named factories


def test_backdoor_skeleton_edges_and_hidden():
    skeleton = backdoor_skeleton(n_confounders=2, kind="discrete", cardinality=3)
    assert set(skeleton.edges) == {
        ("Z_0", "X"),
        ("Z_0", "Y"),
        ("Z_1", "X"),
        ("Z_1", "Y"),
        ("X", "Y"),
    }
    assert skeleton.hidden_variables == []
    assert [v.name for v in skeleton.discrete_variables] == ["Z_0", "Z_1", "X", "Y"]
    assert all(v.cardinality == 3 for v in skeleton.variables)


def test_backdoor_skeleton_set_sizes():
    skeleton = backdoor_skeleton(n_confounders=2, n_treatments=2, n_outcomes=2, n_bystanders=3)
    names = [v.name for v in skeleton.variables]
    assert names == ["Z_0", "Z_1", "X_0", "X_1", "Y_0", "Y_1", "W_0", "W_1", "W_2"]
    # Every confounder points at every treatment and outcome; every treatment at
    # every outcome; bystanders are disconnected.
    for z in ("Z_0", "Z_1"):
        for t in ("X_0", "X_1", "Y_0", "Y_1"):
            assert (z, t) in skeleton.edges
    for x in ("X_0", "X_1"):
        for y in ("Y_0", "Y_1"):
            assert (x, y) in skeleton.edges
    assert not any("W" in a or "W" in b for a, b in skeleton.edges)
    assert len(skeleton.edges) == 2 * 4 + 2 * 2


def test_backdoor_skeleton_invalid_sizes():
    with pytest.raises(ValueError, match="set sizes"):
        backdoor_skeleton(n_confounders=0)
    with pytest.raises(ValueError, match="set sizes"):
        backdoor_skeleton(n_bystanders=-1)


def test_backdoor_skeleton_hidden_confounders():
    skeleton = backdoor_skeleton(n_confounders=2, hidden_confounders=True)
    assert [v.name for v in skeleton.hidden_variables] == ["Z_0", "Z_1"]


def test_backdoor_skeleton_continuous():
    skeleton = backdoor_skeleton(n_confounders=1, kind="continuous", cardinality=None)
    assert [v.name for v in skeleton.continuous_variables] == ["Z_0", "X", "Y"]


def test_frontdoor_skeleton():
    skeleton = frontdoor_skeleton()
    assert set(skeleton.edges) == {("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y")}
    assert [v.name for v in skeleton.hidden_variables] == ["U"]
    assert skeleton.spec_of("U").kind == "discrete"


def test_frontdoor_skeleton_set_sizes():
    skeleton = frontdoor_skeleton(
        n_confounders=2, n_treatments=2, n_mediators=2, n_outcomes=1, n_bystanders=1
    )
    names = [v.name for v in skeleton.variables]
    assert names == ["U_0", "U_1", "X_0", "X_1", "M_0", "M_1", "Y", "W_0"]
    assert [v.name for v in skeleton.hidden_variables] == ["U_0", "U_1"]
    for x in ("X_0", "X_1"):
        for m in ("M_0", "M_1"):
            assert (x, m) in skeleton.edges
    for m in ("M_0", "M_1"):
        assert (m, "Y") in skeleton.edges
    for u in ("U_0", "U_1"):
        for t in ("X_0", "X_1", "Y"):
            assert (u, t) in skeleton.edges
    # CLG constraint: with discrete observed kind, confounders stay discrete.
    assert all(skeleton.spec_of(u).kind == "discrete" for u in ("U_0", "U_1"))


def test_frontdoor_skeleton_continuous():
    skeleton = frontdoor_skeleton(kind="continuous", cardinality=None)
    assert [v.name for v in skeleton.continuous_variables] == ["X", "M", "Y"]
