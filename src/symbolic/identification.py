"""T-ID: tractable identification of interventional queries (revised algorithm).

Implements the user's unified T-ID framework: Shpitser's complete identification
algorithm interleaved with marginal-determinism tracking (``D``) and structural
complexity accounting (``K``), emitting a COAST (:class:`~src.symbolic.id_ast.EstimandAST`).

Graph conventions: identification runs on the single mixed-graph class
:class:`~src.symbolic.causal_graph.CausalGraph` (ordered nodes, ``directed`` and
bidirected edge sets), obtained from an SCM via
:func:`~src.construction.latent_projection.latent_projection`.  All graph
operations are immutable (return new graphs).

Determinisms ``D`` are a set of scopes (frozensets of variables) on which the
base circuit is marginally deterministic; ``IsDET(D, S)`` holds iff ``S`` is an
exact member of ``D`` (subset coverage is NOT sufficient).  Marginalizing a set
``X`` wipes every determinism with a non-empty intersection with ``X``.
``K`` is the circuit-size exponent: deterministic products take ``max``,
non-deterministic products take ``+``.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set

from .causal_graph import CausalGraph
from .id_ast import (
    EstimandAST,
    get_vars,
    make_cond,
    make_marg,
    make_p,
    make_prod,
)


__all__ = [
    "IdentificationError",
    "TractabilityError",
    "Factor",
    "UncompiledCircuit",
    "CausalGraph",
    "ID",
    "required_determinisms",
    "simplify_factors",
    "multiply_factors",
    "prod_determinism",
]


class IdentificationError(ValueError):
    """The query is not identifiable from the observational distribution."""


class TractabilityError(ValueError):
    """Identification is valid but requires a non-deterministic conditioning scope."""


@dataclass
class Factor:
    """A single atomic factor ``P(N | D)`` with numerator and denominator scopes.

    This is the currency of ``simplify_factors``: a pure scope descriptor with
    no AST.  Circuits that are exactly this kernel record their Factor on
    :attr:`UncompiledCircuit.factor` at construction time, so ``ID`` never has
    to recover scopes by digging through ASTs.
    """

    num: Set[Any]
    den: Set[Any] = field(default_factory=set)

    def __post_init__(self):
        assert self.num, "numerator set must be non-empty"
        assert self.num.isdisjoint(self.den), "numerator and denominator must be disjoint"

    @property
    def variables(self) -> Set[Any]:
        return self.num | self.den

    def __str__(self):
        n = ",".join(map(str, sorted(self.num, key=repr)))
        d = ",".join(map(str, sorted(self.den, key=repr)))
        return f"P({n}|{d})"


@dataclass
class UncompiledCircuit:
    """Output of T-ID: the COAST plus tracked determinisms and complexity.

    ``factor`` is set iff this circuit is exactly the atomic kernel for that
    factor (recorded at construction by ``_cond``); it lets line-4-style call
    sites feed ``simplify_factors`` directly.  Composite products carry
    ``factor=None``.

    ``is_base`` marks circuits that still *are* the base law P(V) (or a plain
    marginal of it).  Only base marginals coincide with base kernels; a
    marginal of a rebased estimand is an *interventional* marginal
    (e.g. the frontdoor sum_x' P(x') P(y|x',m)) and must not be re-expressed
    as a single kernel of P.
    """

    ast: EstimandAST
    determinisms: Set[frozenset]
    complexity: float
    factor: Optional[Factor] = None
    is_base: bool = False

    @property
    def variables(self) -> Set[Any]:
        return get_vars(self.ast)


def _marg(
    p: UncompiledCircuit, keep: Optional[Set[Any]] = None, remove: Optional[Set[Any]] = None
) -> UncompiledCircuit:
    keep = keep or set()
    remove = remove or set()
    assert keep.isdisjoint(remove), "keep and remove sets must be disjoint"
    if keep:
        remove = (p.variables - keep) | remove
    factor = p.factor
    if factor is not None and factor.variables & remove:
        factor = None
    return UncompiledCircuit(
        ast=make_marg(remove, p.ast),
        # Marginalizing `remove` wipes every determinism that intersects it.
        determinisms={d for d in p.determinisms if d.isdisjoint(remove)},
        complexity=p.complexity,
        factor=factor,
        is_base=p.is_base,  # a marginal of the base law is still the base law
    )


def _cond(
    p: UncompiledCircuit,
    factor: Factor,
    required: Optional[Set[frozenset]] = None,
) -> UncompiledCircuit:
    """Materialize the kernel ``P(factor.num | factor.den)`` from ``p``.

    This is the single place where a :class:`Factor` becomes a circuit; the
    result is tagged with ``factor`` so callers can feed it to
    ``simplify_factors`` without inspecting the AST.

    When ``required`` is given (collection mode), the determinism check is
    skipped and ``factor.den`` is recorded into ``required`` instead — this is
    how :func:`required_determinisms` derives the determinism family a base
    circuit must provide.
    """
    marg = p.variables - factor.variables
    if not factor.den:
        result = _marg(p, remove=marg)
        return UncompiledCircuit(
            ast=result.ast,
            determinisms=result.determinisms,
            complexity=result.complexity,
            factor=Factor(factor.num, set()),
        )
    ast = make_cond(factor.num, factor.den, make_marg(marg, p.ast))
    # Marginalizing `marg` wipes every determinism that intersects it; the
    # conditioning set must then be an exact member of the surviving family.
    determinisms = {d for d in p.determinisms if d.isdisjoint(marg)}
    if required is not None:
        required.add(frozenset(factor.den))
    elif factor.den not in determinisms:
        raise TractabilityError(
            f"conditioning on {factor.den} is not tractable; "
            "it is not a determinism of the marginalized circuit"
        )
    return UncompiledCircuit(
        ast=ast, determinisms=determinisms, complexity=p.complexity, factor=factor
    )


def simplify_factors(factors: List[Factor], graph: CausalGraph) -> List[Factor]:
    """Merge atomic factors pairwise into coarser kernels with the same semantics."""
    factors = list(factors)

    def _check_mergeable(num_i, den_i, num_j, den_j):
        # Check if i can be merged into j: P(n_j|d_j) * P(n_i|d_i) = P(n_j∪n_i|d')
        den_i_new = set(den_i)
        den_j_new = set(den_j)
        # Extend d_j by n_i:  P(n_j|d_j) = P(n_j | d_j∪n_i)  iff  n_j ⊥ n_i\d_j | d_j
        if graph.d_separated(num_j, num_i - den_j, den_j):
            den_j_new |= num_i
        # Extend d_j by d_i:  iff  n_j ⊥ d_i\d_j | d_j
        if graph.d_separated(num_j, den_i - den_j, den_j):
            den_j_new |= den_i
        # Extend d_i by d_j\n_i:  P(n_i|d_i) = P(n_i | d_i∪(d_j\n_i))
        #   iff  n_i ⊥ (d_j\n_i)\d_i | d_i
        if graph.d_separated(num_i, (den_j - num_i) - den_i, den_i):
            den_i_new |= den_j - num_i

        if num_j & den_i_new:
            return None  # merged kernel would condition on its own numerator
        if den_j_new == num_i | den_i_new:
            new_num = num_j | num_i
            new_den = den_j_new - num_i
            return Factor(new_num, new_den)
        return None

    merged_any = True
    while merged_any and len(factors) > 1:
        merged_any = False
        for i in range(len(factors)):
            for j in range(len(factors)):
                if i == j:
                    continue
                fi = factors[i]
                fj = factors[j]
                merged = _check_mergeable(fi.num, fi.den, fj.num, fj.den)
                if merged is None:
                    continue
                factors[i] = merged
                del factors[j]
                merged_any = True
                break
            if merged_any:
                break

    return factors


def prod_determinism(q1: Set[Any], q2: Set[Any], shared_vars: Set[Any]) -> Optional[Set[Any]]:
    """Single-pair product determinism rule (T-ID UpdateMult).

    Returns the determinism scope carried by the product of two circuits with
    determinism scopes ``q1``/``q2`` whose scopes overlap in ``shared_vars``, or
    ``None`` when the product makes no determinism claim:

    - identical scopes inside the shared region are kept (``q1 == q2 ⊆ shared``);
    - a union is claimed only when both operands are deterministic over sets
      covering the whole shared scope (``q1, q2 ⊇ shared``).
    """
    if q1 == q2 and q1 <= shared_vars:
        return q1
    if q1 >= shared_vars and q2 >= shared_vars:
        return q1 | q2
    return None


def _get_prod_determinisms(
    dets1: Set[frozenset], dets2: Set[frozenset], shared_vars: Set[Any]
) -> Set[frozenset]:
    prod_determinisms = set()
    for q1 in dets1:
        for q2 in dets2:
            q = prod_determinism(q1, q2, shared_vars)
            if q is not None:
                prod_determinisms.add(frozenset(q))
    return prod_determinisms


def multiply_factors(factors: List[UncompiledCircuit]) -> UncompiledCircuit:
    if not factors:
        raise ValueError("Cannot multiply an empty list of factors")
    if len(factors) == 1:
        return factors[0]
    ast = make_prod([f.ast for f in factors])

    # Complexity K: support-compatible factors (the product retains a
    # determinism over the shared scope, per the UpdateMult rule) tie
    # unit-wise, so composing them multiplies no sizes and K folds to the
    # max; support-incompatible factors multiply sizes and K adds.
    curr_scope, curr_dets = factors[0].variables, factors[0].determinisms
    complexity = factors[0].complexity
    for f in factors[1:]:
        shared = curr_scope & f.variables
        compatible = bool(_get_prod_determinisms(curr_dets, f.determinisms, shared))
        complexity = max(complexity, f.complexity) if compatible else complexity + f.complexity
        curr_dets = _get_prod_determinisms(curr_dets, f.determinisms, shared)
        curr_scope |= f.variables

    return UncompiledCircuit(ast=ast, determinisms=curr_dets, complexity=complexity, factor=None)


def ID(
    y: Set[Any],
    x: Set[Any],
    p: UncompiledCircuit,
    g: CausalGraph,
    *,
    required: Optional[Set[frozenset]] = None,
) -> UncompiledCircuit:
    """Identify P(y | do(x)) from the base circuit P(V).

    Line-by-line Shpitser & Pearl (2006) ID over :class:`CausalGraph`, with
    BUILD_COND_SET-style minimal d-separating denominators and the
    Factor-based simplification of the user's T-ID framework.  Branch
    structure follows the reference implementation in ``y0``
    (``y0.algorithm.identify.id_std``): ``G \\ x`` removes nodes, the line-5
    check is "``C(G)`` is the single district over all of ``V``".

    With ``required`` given, determinism tractability checks are skipped and
    every kernel's denominator is recorded instead (collection mode — see
    :func:`required_determinisms`).
    """
    assert set(g.nodes) <= p.variables, (
        "graph nodes must be a subset of the base circuit's variables"
    )

    # Line 1: if x is empty, return P(y)
    if not x:
        result = _marg(p, remove=set(g.nodes) - y)
        factor = Factor(set(y), set()) if p.is_base else None
        return UncompiledCircuit(
            ast=result.ast,
            determinisms=result.determinisms,
            complexity=result.complexity,
            factor=factor,
        )

    # Line 2: if V != An(y)_G, restrict to the ancestors of y
    an_y = g.ancestors(y)
    if set(g.nodes) != an_y:
        remove = set(g.nodes) - an_y  # only graph nodes; kernel context stays free
        if remove:
            p = _marg(p, remove=remove)
        return ID(y, x & an_y, p, g.induced_subgraph(an_y), required=required)

    # Line 3: W = (V \ x) \ An(y)_{G_{\bar{x}}}; intervene on W as well
    an_y_bar_x = g.remove_outgoing(x).ancestors(y)
    w = set(g.nodes) - x - an_y_bar_x
    if w:
        return ID(y, x | w, p, g, required=required)

    # Line 4: c-component factorization over G \ x (node removal)
    s_comps = g.remove_nodes(x).districts()
    if len(s_comps) > 1:
        results = [ID(comp, set(g.nodes) - comp, p, g, required=required) for comp in s_comps]
        atomic, non_atomic = [], []
        for r in results:
            if r.factor is None:
                non_atomic.append(r)
                continue
            f = r.factor
            ctx = r.variables - f.num - f.den
            atomic.append(Factor(f.num, f.den | ctx))
        minimal = simplify_factors(atomic, g)
        circuits = [_cond(p, f, required=required) for f in minimal]
        product = multiply_factors(circuits + non_atomic)
        return _marg(product, remove=set(g.nodes) - (y | x))

    # C(G \ x) = {S}, single district
    (s,) = s_comps
    c_comps = g.districts()

    # Line 5: if C(G) = {V}, the query is not identifiable
    if len(c_comps) == 1 and next(iter(c_comps)) == set(g.nodes):
        raise IdentificationError(
            "The query is not identifiable from the observational distribution."
        )

    # Minimal d-separating predecessor sets (BUILD_COND_SET's pi_tilde)
    order = g.topological_order()
    rank = {v: i for i, v in enumerate(order)}

    def pi_tilde(v: Any) -> Set[Any]:
        preds = {u for u in g.nodes if rank[u] < rank[v]}
        return g.minimal_separator(v, preds) if preds else set()

    def build_cond_set(target: Set[Any]) -> UncompiledCircuit:
        factors = [Factor({v}, pi_tilde(v)) for v in order if v in target]
        minimal = simplify_factors(factors, g)
        return multiply_factors([_cond(p, f, required=required) for f in minimal])

    # Line 6: if S in C(G), return the product of kernels over S
    if s in c_comps:
        product = build_cond_set(s)
        return _marg(product, remove=s - y)

    # Line 7: if (∃ S') S ⊂ S' in C(G), identify the S'-factor and recurse
    for c in c_comps:
        if s < c:
            product = build_cond_set(c)
            return ID(y, x & c, product, g.induced_subgraph(c), required=required)

    raise IdentificationError("The query is not identifiable from the observational distribution.")


def required_determinisms(y: Set[Any], x: Set[Any], g: CausalGraph) -> Set[frozenset]:
    """Determinism family required to identify P(y | do(x)) on ``g``.

    Runs :func:`ID` in collection mode: every kernel denominator encountered
    along the derivation is recorded instead of being checked against the
    base circuit's determinisms.  The returned family is the minimal set of
    exact determinism scopes a base circuit must provide (supersets may of
    course be offered instead, per the circuit's own MD sets).

    Raises :class:`IdentificationError` when the query is not identifiable
    from the observational distribution at all (independent of determinisms).
    """
    required: Set[frozenset] = set()
    p = UncompiledCircuit(
        ast=make_p(set(g.nodes)),
        determinisms=set(),
        complexity=1.0,
        is_base=True,
    )
    ID(set(y), set(x), p, g, required=required)
    return required
