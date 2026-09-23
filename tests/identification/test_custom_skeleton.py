"""Custom skeleton from ``generate_custom_skeleton.py``: tractability semantics.

SCM (all continuous):

    U -> X, U -> Y   hidden confounder; latent projection -> bidirected X <-> Y
    Z -> X, Z -> M   observed parent of treatment and mediator
    X -> M -> Y      mediated treatment effect

Observed graph fed to ID (U hidden):

    Z -> X,  Z -> M,  X -> M,  M -> Y,   X <-> Y

Query:  P(Y | do(X)).

Determinism semantics (see src/symbolic/identification.py):
  * Conditioning P(N|D) is tractable iff D is an EXACT member of the
    determinism family (subset coverage is not sufficient).
  * Marginalizing a set X wipes every determinism with a non-empty
    intersection with X.

With D = {{X,Z,M}, {X,Z}, {Z}} the derived COAST contains the kernel
P(Y|M,X) whose marginal context is {Z} — wiping all three determinisms — so
{M,X} is not an exact member and identification MUST fail with
TractabilityError.  Adding {M,X} to D makes it succeed (K = 4).

Run:  pytest tests/identification/test_custom_skeleton.py -s
"""

import pytest

from src.symbolic.causal_graph import CausalGraph
from src.symbolic.id_ast import EstimandAST, make_p
from src.symbolic.identification import ID, TractabilityError, UncompiledCircuit


def _graph() -> CausalGraph:
    nodes = ["Z", "X", "M", "Y"]
    return CausalGraph(
        nodes,
        directed_edges=[("Z", "X"), ("Z", "M"), ("X", "M"), ("M", "Y")],
        bidirected_edges=[("X", "Y")],
    )


def _base(determinisms) -> UncompiledCircuit:
    return UncompiledCircuit(
        ast=make_p({"Z", "X", "M", "Y"}),
        determinisms={frozenset(d) for d in determinisms},
        complexity=1.0,
        is_base=True,
    )


def _print_ast(ast: EstimandAST) -> None:
    def rec(node_id: int, indent: int = 0) -> None:
        node = ast.get_node_data(node_id)
        print("    " * indent + str(node))
        for child_id, _ in ast.get_outgoing_edges(node_id):
            rec(child_id, indent + 1)

    rec(ast.get_root())


def test_custom_skeleton_requires_exact_mx_determinism():
    """D without {M,X}: the P(Y|M,X) kernel is not tractable -> must fail."""
    determinisms = {
        frozenset({"X", "Z", "M"}),
        frozenset({"X", "Z"}),
        frozenset({"Z"}),
    }
    with pytest.raises(TractabilityError, match="conditioning on .* is not tractable"):
        ID({"Y"}, {"X"}, _base(determinisms), _graph())


def test_custom_skeleton_identifies_with_mx_determinism():
    """D with {M,X} added: identification succeeds, COAST + K = 4."""
    determinisms = {
        frozenset({"X", "Z", "M"}),
        frozenset({"X", "Z"}),
        frozenset({"Z"}),
        frozenset({"M", "X"}),
    }
    result = ID({"Y"}, {"X"}, _base(determinisms), _graph())

    print("\n=== Custom skeleton (U->X, U->Y | Z->X, Z->M | X->M->Y):  P(Y | do(X)) ===")
    print("COAST:")
    _print_ast(result.ast)
    print(f"complexity K = {result.complexity}")
    print(f"determinisms = {result.determinisms}")
    print(f"variables    = {sorted(map(str, result.variables))}")


def test_required_determinisms_custom():
    """Collection mode returns exactly the kernel denominators of the COAST."""
    from src.symbolic.identification import required_determinisms

    assert required_determinisms({"Y"}, {"X"}, _graph()) == {
        frozenset({"X", "Z"}),
        frozenset({"Z"}),
        frozenset({"M", "X"}),
    }


def test_required_determinisms_frontdoor():
    """Frontdoor graph: the classic {{X,M},{X}} family."""
    from src.symbolic.identification import required_determinisms

    g = CausalGraph(
        ["X", "M", "Y"],
        directed_edges=[("X", "M"), ("M", "Y")],
        bidirected_edges=[("X", "Y")],
    )
    assert required_determinisms({"Y"}, {"X"}, g) == {
        frozenset({"X", "M"}),
        frozenset({"X"}),
    }


def test_required_determinisms_suffice_for_id():
    """Feeding the collected family back into ID makes the query tractable."""
    from src.symbolic.identification import required_determinisms

    req = required_determinisms({"Y"}, {"X"}, _graph())
    ID({"Y"}, {"X"}, _base(req), _graph())
