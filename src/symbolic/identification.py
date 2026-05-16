from itertools import combinations

from .id_ast import (
    DetProdNode,
    EstimandAST,
    PowNode,
    ProdNode,
    _copy_subtree_local,
    ast_to_str,
    get_vars,
    make_det_prod,
    make_marg,
    make_pow,
    make_prod,
)
from .scm import StructuralCausalModel


class IdentificationError(Exception):
    """Exception raised when an estimand is unidentifiable."""

    pass


class TractabilityError(Exception):
    """Exception raised when an identifiable query requires POW(-1) on a non-deterministic circuit."""

    pass


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------


def _is_det(D: set, scope: frozenset) -> bool:
    """
    Return True if `scope` is covered by any Q ∈ D, respecting Proposition 2:
    Q-determinism implies Q'-determinism for all Q ⊆ Q' (supersets come for free).
    """
    return any(Q.issubset(scope) for Q in D)


def _update_D_marg(D: set, W: set) -> set:
    """Remove determinisms whose scope intersects the marginalized variables W.

    Marginalization over W can break Q-determinism when Q ∩ W ≠ ∅, because the
    marginalized variables no longer constrain which component is active.
    Only Q-determinisms where Q is disjoint from W are preserved.
    """
    return {Q for Q in D if Q.isdisjoint(W)}


def _D_for_conditionals(D: set, S: set, topological_order: list) -> set:
    """
    Compute determinisms of PROD_{vi ∈ S} P(vi | preds_i).

    Each conditional P(vi | preds_i) has determinisms {Q ∈ D | Q ⊆ preds_i} because fixing
    preds_i (guaranteed tractable when preds_i is covered by D) uniquely determines the branch.
    The product's determinisms are the intersection across all factors.
    """
    D_result = D.copy()
    for vi in S:
        vi_idx = topological_order.index(vi)
        predecessors_i = set(topological_order[:vi_idx])
        scope_i = predecessors_i | {vi}
        # Keep Q only if it is entirely within the factor's scope (Q ⊆ {vi}∪preds_i)
        # or entirely outside it. Q's that partially overlap the scope are dropped
        # because they cannot be read off from the conditional alone.
        D_result = {Q for Q in D_result if Q.issubset(scope_i) or Q.isdisjoint(scope_i)}
    return D_result


def minimize_determinisms(D: set[frozenset]) -> set[frozenset]:
    """
    Return the antichain of D: remove any Q ∈ D for which a strict subset Q' ⊊ Q exists in D.

    By Proposition 2, Q'-determinism (for Q' ⊆ Q) implies Q-determinism, so the larger Q is
    redundant. Only the smallest (most restrictive) determinisms need to be kept.
    """
    D_list = list(D)
    return {Q for Q in D_list if not any(Q2 < Q for Q2 in D_list)}


# ---------------------------------------------------------------------------
# Algebraic Fraction Helpers
# ---------------------------------------------------------------------------


def _get_fraction_parts(ast: EstimandAST, root_id: str):
    node = ast.get_node_data(root_id)
    if isinstance(node, (ProdNode, DetProdNode)):
        edges = ast.get_outgoing_edges(root_id)
        if len(edges) == 2:
            c1, _ = edges[0]
            c2, _ = edges[1]
            n1 = ast.get_node_data(c1)
            n2 = ast.get_node_data(c2)
            if isinstance(n2, PowNode) and n2.power == -1:
                return c1, ast.get_outgoing_edges(c2)[0][0]
            if isinstance(n1, PowNode) and n1.power == -1:
                return c2, ast.get_outgoing_edges(c1)[0][0]
    return None


def _cancel_fractions(
    children: list, D_children: list, K_children: list, c_components: list, tracking: bool
):
    """
    ALGEBRAIC FRACTION CANCELLATION
    If we have P(A|B,C) * P(B|C) = P(A,B|C), we can merge them to reduce K.
    Returns: updated children, D_children, K_children, c_components
    """
    changed = True
    while changed:
        changed = False
        for i in range(len(children)):
            if children[i] is None:
                continue
            for j in range(len(children)):
                if i == j or children[j] is None:
                    continue

                parts_i = _get_fraction_parts(children[i][0], children[i][1])
                parts_j = _get_fraction_parts(children[j][0], children[j][1])

                if not parts_i or not parts_j:
                    continue

                num_i, den_i = parts_i
                num_j, den_j = parts_j

                str_den_i = ast_to_str(children[i][0], den_i)
                str_num_j = ast_to_str(children[j][0], num_j)

                if str_den_i == str_num_j:
                    num_ast, num_root = _copy_subtree_local(children[i][0], num_i)
                    den_ast, den_root = _copy_subtree_local(children[j][0], den_j)
                    pow_ast, pow_root = make_pow(-1, den_ast, den_root)

                    if tracking:
                        merged_ast, merged_root = make_det_prod(
                            [(num_ast, num_root), (pow_ast, pow_root)]
                        )
                        D_children[i] = minimize_determinisms(D_children[i] | D_children[j])
                        K_children[i] = max(K_children[i], K_children[j])
                    else:
                        merged_ast, merged_root = make_prod(
                            [(num_ast, num_root), (pow_ast, pow_root)]
                        )

                    children[i] = (merged_ast, merged_root)
                    children[j] = None
                    c_components[i] = c_components[i] | c_components[j]
                    c_components[j] = None
                    changed = True
                    break

    new_children, new_D_children, new_K_children, new_c_components = [], [], [], []
    for i in range(len(children)):
        if children[i] is not None:
            new_children.append(children[i])
            new_c_components.append(c_components[i])
            if tracking:
                new_D_children.append(D_children[i])
                new_K_children.append(K_children[i])

    return new_children, new_D_children, new_K_children, new_c_components


def _check_support_compatibility(children: list, D_children: list) -> bool:
    """
    Check pairwise support-compatibility: for each pair, the shared scope
    must be covered by determinisms of both factors.
    """
    child_scopes = []
    for child_ast, child_root_id in children:
        child_scopes.append(get_vars(child_ast, child_root_id))

    for i in range(len(children)):
        for j in range(i + 1, len(children)):
            shared = child_scopes[i] & child_scopes[j]
            if not shared:
                continue  # Disjoint scopes — trivially compatible
            shared_fs = frozenset(shared)
            if not (_is_det(D_children[i], shared_fs) and _is_det(D_children[j], shared_fs)):
                return False
    return True


def _merge_non_query_components(
    y: set,
    V: set,
    P: tuple[EstimandAST, str],
    D: set,
    children: list,
    D_children: list,
    K_children: list,
    c_components: list,
) -> tuple:
    r"""
    If full factorization isn't tractable, try merging non-query c-components
    into a single joint marginal: P(Z) = MARG(V\Z)[P] to keep adjustment set together.
    """

    query_indices = [i for i, S_i in enumerate(c_components) if S_i.intersection(y)]
    non_query_indices = [i for i in range(len(children)) if i not in query_indices]

    if non_query_indices and query_indices:
        non_query_vars = set()
        for idx in non_query_indices:
            non_query_vars |= c_components[idx]
        marg_joint_vars = V - non_query_vars
        merged_factor = make_marg(marg_joint_vars, *P)
        D_merged = _update_D_marg(D, marg_joint_vars)

        new_children = [children[i] for i in query_indices] + [merged_factor]
        new_D_children = [D_children[i] for i in query_indices] + [D_merged]
        new_K_children = [K_children[i] for i in query_indices] + [1]

        if _check_support_compatibility(new_children, new_D_children):
            return new_children, new_D_children, new_K_children, True

    return children, D_children, K_children, False


# ---------------------------------------------------------------------------
# Core identification algorithm (T-ID)
# ---------------------------------------------------------------------------


def identify(
    y: set,
    x: set,
    P: tuple[EstimandAST, str],
    G: StructuralCausalModel,
    D: set[frozenset] | None = None,
) -> tuple[EstimandAST, str, set[frozenset] | None, int | None]:
    """
    Shpitser's ID Algorithm, augmented with tractability tracking (T-ID).

    y: query variables
    x: intervention variables
    P: current probability distribution as an AST tuple (EstimandAST, root_id)
    G: current causal graph
    D: set of frozensets representing the marginal determinisms of P.
       Pass None (default) to disable tractability checking entirely.
       The check respects Proposition 2: if Q ∈ D and Q ⊆ scope, scope is covered.

    Returns: (ast, root_id, D_out, K) where:
        D_out: determinisms of the returned expression (None when D is None)
        K: complexity exponent, the estimand is computable in O(|C|^K).
           K=1 (linear) or K=2 (quadratic). None when D is None.

    Raises:
        IdentificationError: when the query is not identifiable (Hedge condition).
        TractabilityError: when D is provided and a POW(-1) would be applied to a
                           circuit whose scope is not covered by D.
    """
    tracking = D is not None
    if D is None:
        D = set()

    V = G.observable_variables

    # Line 1: Base Case
    if not x:
        marg_vars = V - y
        result = make_marg(marg_vars, *P)
        D_out = _update_D_marg(D, marg_vars) if tracking else None
        return *result, D_out, (1 if tracking else None)

    # Line 2: Ancestral Isolation
    ancestors_Y = G.get_ancestors(y).intersection(V)
    if V != ancestors_Y:
        marg_vars = V - ancestors_Y
        P_new = make_marg(marg_vars, *P)
        D_new = _update_D_marg(D, marg_vars) if tracking else None
        G_new = G.subgraph(ancestors_Y)
        return identify(y, x.intersection(ancestors_Y), P_new, G_new, D_new)

    # Line 3: Harmless Interventions
    G_xbar = G.remove_incoming_edges(x)
    ancestors_Y_G_xbar = G_xbar.get_ancestors(y).intersection(V)
    W = (V - x) - ancestors_Y_G_xbar
    if W:
        return identify(y, x.union(W), P, G, D if tracking else None)

    # Line 4: C-Component Factorization
    c_components_G_minus_X = G.subgraph(V - x).get_c_components(V - x)
    if len(c_components_G_minus_X) > 1:
        children = []
        D_children = []
        K_children = []
        for S_i in c_components_G_minus_X:
            child_ast, child_root_id, D_child, K_child = identify(
                S_i, V - S_i, P, G, D if tracking else None
            )
            children.append((child_ast, child_root_id))
            D_children.append(D_child)
            K_children.append(K_child)

        children, D_children, K_children, c_components_G_minus_X = _cancel_fractions(
            children, D_children, K_children, c_components_G_minus_X, tracking
        )

        if tracking:
            pairwise_tractable = _check_support_compatibility(children, D_children)

            all_scopes = {Q for D_c in D_children for Q in D_c}
            D_prod = {Q for Q in all_scopes if all(_is_det(D_c, Q) for D_c in D_children)}

            K_prod = 1 if pairwise_tractable else sum(K_children)
            K = max(K_children + [K_prod])

            prod_fn = make_det_prod if K_prod == 1 else make_prod
            prod_result = prod_fn(children)

            marg_vars = V - (y | x)
            result = make_marg(marg_vars, *prod_result)
            D_out = _update_D_marg(D_prod, marg_vars)
            return *result, D_out, K
        else:
            prod_result = make_prod(children)
            marg_vars = V - (y | x)
            result = make_marg(marg_vars, *prod_result)
            return *result, None, None

    # Line 5: The Hedge (Fail Condition)
    if len(c_components_G_minus_X) == 1:
        S = c_components_G_minus_X[0]
        c_components_G = G.get_c_components(V)

        if len(c_components_G) == 1 and c_components_G[0] == V:
            raise IdentificationError(f"Hedge discovered. Unidentifiable query. G: {V}, S: {S}")

        topological_order = [v for v in G.topological_sort() if v in V]

        def _build_conditionals(scope: set) -> list:
            """Build PROD_{vi ∈ scope} P(vi | preds_i), checking tractability at each POW."""
            prod_children = []
            for vi in scope:
                vi_idx = topological_order.index(vi)
                predecessors = set(topological_order[:vi_idx])

                joint_vars_to_marg = V - predecessors.union({vi})
                marg_joint = make_marg(joint_vars_to_marg, *P)

                if not predecessors:
                    # P(vi) = MARG(V\{vi})[P] — no denominator needed.
                    prod_children.append(marg_joint)
                    continue

                if tracking and not _is_det(D, frozenset(predecessors)):
                    raise TractabilityError(
                        f"Cannot compute POW(-1): circuit over {predecessors} is not "
                        f"deterministic. Current determinisms: {D}"
                    )

                marg_denom = make_marg({vi}, *marg_joint)
                pow_denom = make_pow(-1, *marg_denom)
                if tracking and _is_det(D, frozenset(predecessors)):
                    prod_children.append(make_det_prod([marg_joint, pow_denom]))
                else:
                    prod_children.append(make_prod([marg_joint, pow_denom]))
            return prod_children

        # Line 6: Identifying a Clean C-Component
        if S in c_components_G:
            # Under tracking with determinism, the conditionals P(vi|preds_i) are
            # all deterministic, so their product is support-compatible (DetProd).
            conds = _build_conditionals(S)
            s_prod = make_det_prod(conds) if tracking else make_prod(conds)

            if tracking:
                D_s = _D_for_conditionals(D, S, topological_order)
                marg_vars = S - y
                result = make_marg(marg_vars, *s_prod)
                D_out = _update_D_marg(D_s, marg_vars)
                return *result, D_out, 1
            else:
                marg_vars = S - y
                result = make_marg(marg_vars, *s_prod)
                return *result, None, None

        # Line 7: Sub-Component Isolation
        S_prime = None
        for c in c_components_G:
            if S.issubset(c):
                S_prime = c
                break

        if S_prime is not None:
            conds = _build_conditionals(S_prime)
            P_new = make_det_prod(conds) if tracking else make_prod(conds)

            D_new = _D_for_conditionals(D, S_prime, topological_order) if tracking else None
            G_new = G.subgraph(S_prime)
            ast, root_id, D_out, K = identify(y, x.intersection(S_prime), P_new, G_new, D_new)
            K_out = max(1, K) if K is not None else None
            return ast, root_id, D_out, K_out

    raise IdentificationError(
        "Algorithm failed to identify the causal query for an unknown reason."
    )


# ---------------------------------------------------------------------------
# Required determinisms query
# ---------------------------------------------------------------------------


def _collect_required(y: set, x: set, G: StructuralCausalModel, required: set[frozenset]) -> None:
    """
    Structural traversal mirroring identify(), collecting all POW(-1) scope requirements.

    P is not needed because the execution path depends only on (y, x, G).
    All collected frozensets are valid requirements on the top-level D because D is only
    filtered (never expanded) as we recurse deeper.
    """
    V = G.observable_variables

    # Line 1: Base Case — no POW involved
    if not x:
        return

    # Line 2: Ancestral Isolation
    ancestors_Y = G.get_ancestors(y).intersection(V)
    if V != ancestors_Y:
        G_new = G.subgraph(ancestors_Y)
        _collect_required(y, x.intersection(ancestors_Y), G_new, required)
        return

    # Line 3: Harmless Interventions
    G_xbar = G.remove_incoming_edges(x)
    ancestors_Y_G_xbar = G_xbar.get_ancestors(y).intersection(V)
    W = (V - x) - ancestors_Y_G_xbar
    if W:
        _collect_required(y, x.union(W), G, required)
        return

    # Line 4: C-Component Factorization
    c_components_G_minus_X = G.subgraph(V - x).get_c_components(V - x)
    if len(c_components_G_minus_X) > 1:
        for S_i in c_components_G_minus_X:
            _collect_required(S_i, V - S_i, G, required)
        return

    # Line 5: Hedge check
    if len(c_components_G_minus_X) == 1:
        S = c_components_G_minus_X[0]
        c_components_G = G.get_c_components(V)

        if len(c_components_G) == 1 and c_components_G[0] == V:
            raise IdentificationError(f"Hedge discovered. Unidentifiable query. G: {V}, S: {S}")

        topological_order = [v for v in G.topological_sort() if v in V]

        def _add_from_scope(scope: set) -> None:
            for vi in scope:
                vi_idx = topological_order.index(vi)
                predecessors = frozenset(topological_order[:vi_idx])
                if predecessors:
                    required.add(predecessors)

        # Line 6: Clean C-Component
        if S in c_components_G:
            _add_from_scope(S)
            return

        # Line 7: Sub-Component Isolation
        for c in c_components_G:
            if S.issubset(c):
                S_prime = c
                _add_from_scope(S_prime)
                _collect_required(y, x.intersection(S_prime), G.subgraph(S_prime), required)
                return

    raise IdentificationError(
        "Algorithm failed to identify the causal query for an unknown reason."
    )


def required_determinisms(y: set, x: set, G: StructuralCausalModel) -> set[frozenset]:
    """
    Return the set of marginal determinisms a circuit must fulfil to tractably compute P(y|do(x)).

    These are exactly the frozensets that must be in D for identify(y, x, P, G, D) to succeed
    without raising TractabilityError, for any circuit P. By Proposition 2, if Q ∈ result then
    any Q' ⊇ Q is automatically satisfied; call minimize_determinisms(result) to obtain the
    minimal antichain representation where only the essential (smallest) determinisms are retained.

    Raises IdentificationError if the query is not identifiable.
    """
    required: set[frozenset] = set()
    _collect_required(y, x, G, required)

    if not required:
        return set()

    # To satisfy all required POW(-1) operations simultaneously with a single md-set Q,
    # Q must be a subset of EVERY scope in `required`.
    # Therefore, Q must be a subset of their intersection.
    intersection = None
    for req in required:
        if intersection is None:
            intersection = set(req)
        else:
            intersection = intersection.intersection(req)

    return {frozenset(intersection)}


# ---------------------------------------------------------------------------
# Tractable query enumeration
# ---------------------------------------------------------------------------


def _compute_tractability(
    y: set,
    x: set,
    G: StructuralCausalModel,
    D: set[frozenset],
) -> tuple[set[frozenset], int] | None:
    """
    Fast version of identify() that tracks D and K without building any AST.

    Returns (D_out, K) if the query is identifiable and tractable under D, or None if
    intractable (instead of raising TractabilityError). Still raises IdentificationError
    for unidentifiable queries.
    """
    try:
        from .id_ast import make_p

        V = G.observable_variables
        ast, root_id, D_out, K = identify(y, x, make_p(V), G, D)
        return D_out, K
    except TractabilityError:
        return None


def tractable_queries(
    G: StructuralCausalModel,
    D: set[frozenset],
) -> list[tuple[frozenset, frozenset, int]]:
    """
    Enumerate all causal queries P(y|do(x)) on G that are identifiable and tractably computable
    given marginal determinisms D.

    Returns a list of (y, x, K) where y and x are frozensets of variable names and K is the
    complexity exponent (1 = linear, 2 = quadratic).

    Uses _compute_tractability instead of the full identify() to avoid AST construction, and
    special-cases x=∅ (always tractable, K=1) to skip graph traversal entirely.

    Note: enumeration is O(3^|V|) — use on small graphs only (|V| ≲ 8).
    """
    V = sorted(G.observable_variables)
    results = []

    for x_size in range(len(V) + 1):
        for x_tuple in combinations(V, x_size):
            x_set = set(x_tuple)
            remaining = [v for v in V if v not in x_set]

            # x = ∅: Line 1 always fires, K=1 regardless of D
            if not x_set:
                for y_size in range(1, len(remaining) + 1):
                    for y_tuple in combinations(remaining, y_size):
                        results.append((frozenset(y_tuple), frozenset(), 1))
                continue

            for y_size in range(1, len(remaining) + 1):
                for y_tuple in combinations(remaining, y_size):
                    y_set = set(y_tuple)
                    try:
                        result = _compute_tractability(y_set, x_set, G, D)
                    except IdentificationError:
                        continue
                    if result is not None:
                        _, K = result
                        results.append((frozenset(y_set), frozenset(x_set), K))

    return results
