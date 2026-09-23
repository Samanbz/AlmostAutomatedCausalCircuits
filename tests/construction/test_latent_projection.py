"""Tests for Verma's latent projection (``src/construction/latent_projection.py``)."""

from src.construction.latent_projection import latent_projection
from src.symbolic.causal_graph import CausalGraph
from src.symbolic.scm import ConstantMechanism, StructuralCausalModel


def _scm(nodes, edges, hidden=frozenset(), exogenous=frozenset()):
    """Build a tiny SCM with placeholder mechanisms.

    ``nodes``: declaration order. ``edges``: list of (parent, child) pairs.
    """
    scm = StructuralCausalModel()
    parents = {n: [] for n in nodes}
    for a, b in edges:
        parents[b].append(a)
    for n in nodes:
        scm.add_variable(
            name=n,
            mechanism=ConstantMechanism(0.0),
            parents=parents[n],
            is_exogenous=n in exogenous,
            hidden=n in hidden,
        )
    return scm


def test_confounder_gives_bidirected_edge():
    # U -> X, U -> Y with U hidden  =>  X <-> Y, no directed edges.
    scm = _scm(["U", "X", "Y"], [("U", "X"), ("U", "Y")], hidden={"U"})
    admg = latent_projection(scm)

    assert set(admg.nodes) == {"X", "Y"}
    assert admg.directed == set()
    assert admg.has_bidirected("X", "Y")
    assert admg.has_bidirected("Y", "X")  # order-insensitive
    assert len(admg.bidirected) == 1


def test_latent_chain_gives_directed_edge():
    # A -> H -> B with H hidden  =>  A -> B only.
    scm = _scm(["A", "H", "B"], [("A", "H"), ("H", "B")], hidden={"H"})
    admg = latent_projection(scm)

    assert admg.directed == {("A", "B")}
    assert admg.has_edge("A", "B")
    assert not admg.has_edge("B", "A")
    assert admg.bidirected == set()


def test_instrument_gives_no_bidirected_edge():
    # U -> X -> Y with U hidden  =>  X -> Y, but NO X <-> Y:
    # the trek U -> X -> Y has the observed intermediate X.
    scm = _scm(["U", "X", "Y"], [("U", "X"), ("X", "Y")], hidden={"U"})
    admg = latent_projection(scm)

    assert admg.directed == {("X", "Y")}
    assert admg.bidirected == set()


def test_napkin_shape():
    # Hidden: U1 -> X, U1 -> Y and U2 -> Z, U2 -> Y.
    # Observed chain: W -> Z -> X -> Y.
    nodes = ["U1", "U2", "W", "Z", "X", "Y"]
    edges = [
        ("U1", "X"),
        ("U1", "Y"),
        ("U2", "Z"),
        ("U2", "Y"),
        ("W", "Z"),
        ("Z", "X"),
        ("X", "Y"),
    ]
    scm = _scm(nodes, edges, hidden={"U1", "U2"})
    admg = latent_projection(scm)

    # Direct observed-observed edges are preserved as-is; W -> X via the observed
    # intermediate Z is NOT a projection edge.
    assert admg.directed == {("W", "Z"), ("Z", "X"), ("X", "Y")}
    assert not admg.has_edge("W", "X")

    assert admg.has_bidirected("X", "Y")  # trek via U1
    assert admg.has_bidirected("Z", "Y")  # trek via U2
    assert len(admg.bidirected) == 2


def test_default_hidden_picks_up_scm_hidden_variables():
    # hidden=None must default to scm.hidden_variables.
    scm = _scm(["U", "X", "Y"], [("U", "X"), ("U", "Y")], hidden={"U"})
    admg_default = latent_projection(scm)
    admg_explicit = latent_projection(scm, hidden={"U"})

    assert admg_default.nodes == admg_explicit.nodes
    assert admg_default.directed == admg_explicit.directed
    assert admg_default.bidirected == admg_explicit.bidirected
    assert admg_default.has_bidirected("X", "Y")


def test_shared_exogenous_parent_gives_bidirected_edge():
    # Noise-style exogenous nodes are latent for path traversal: observed nodes
    # sharing an exogenous parent (a c-component) are confounded.
    scm = _scm(["E", "X", "Y"], [("E", "X"), ("E", "Y")], exogenous={"E"})
    admg = latent_projection(scm)

    assert set(admg.nodes) == {"X", "Y"}  # exogenous node excluded from V_O
    assert admg.directed == set()
    assert admg.has_bidirected("X", "Y")


def test_no_hidden_returns_observed_subgraph():
    # Nothing hidden  =>  projection keeps the observed DAG as-is.
    scm = _scm(["A", "B", "C"], [("A", "B"), ("B", "C")])
    admg = latent_projection(scm)

    assert admg.directed == {("A", "B"), ("B", "C")}
    assert admg.bidirected == set()


def test_bidirected_stored_canonically_and_str():
    scm = _scm(["U", "X", "Y"], [("U", "X"), ("U", "Y")], hidden={"U"})
    admg = latent_projection(scm)

    (pair,) = admg.bidirected
    assert pair == tuple(sorted(("X", "Y")))
    assert "X <-> Y" in str(admg)
    assert isinstance(admg, CausalGraph)
