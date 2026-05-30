from typing import Any, Optional

from .base import Tree


class ASTNode:
    """Base class for AST payloads."""

    pass


class PNode(ASTNode):
    """Represents the base joint distribution P(V)."""

    def __init__(self, variables: set):
        self.variables = set(variables)

    def __str__(self):
        return f"P({','.join(map(str, sorted(self.variables)))})"


class CondNode(ASTNode):
    """Represents a conditional probability distribution P(num | den)."""

    def __init__(self, num_vars: set, den_vars: set):
        self.num_vars = set(num_vars)
        self.den_vars = set(den_vars)

    def __str__(self):
        n = ",".join(map(str, sorted(self.num_vars)))
        d = ",".join(map(str, sorted(self.den_vars)))
        return f"P({n}|{d})"


class MargNode(ASTNode):
    """Represents marginalization over a set of variables."""

    def __init__(self, marginalize_vars: set):
        self.marginalize_vars = set(marginalize_vars)

    def __str__(self):
        return f"MARG({','.join(map(str, sorted(self.marginalize_vars)))})"


class ProdNode(ASTNode):
    """Binary non-deterministic product (Kronecker / quadratic-time circuit product)."""

    def __str__(self):
        return "PROD"


class DetProdNode(ASTNode):
    """Binary deterministic product (Hadamard / linear-time circuit product).

    One child is marginally deterministic over the shared scope, so the product
    can be evaluated in O(|C|) rather than O(|C_1| * |C_2|).
    """

    def __str__(self):
        return "DETPROD"


class PowNode(ASTNode):
    """Represents raising an expression to a power (e.g., POW(-1; ...))."""

    def __init__(self, power: int):
        self.power = power

    def __str__(self):
        return f"POW({self.power})"


class ConstantNode(ASTNode):
    """Represents a constant value (e.g., 1 from total marginalization)."""

    def __init__(self, value: float):
        self.value = value

    def __str__(self):
        return str(self.value)


class InstNode(ASTNode):
    """Instantiate a set of variables from data (evidence/clamping).

    Applied to a child circuit expression.  The concrete values come from the
    data tensor at evaluation time — this node records *which* variables are
    fixed, not the values themselves.
    """

    def __init__(self, variables: dict[str, Any]):
        self.variables = variables

    def __str__(self):
        return f"INST({','.join(f'{v}={val}' for v, val in sorted(self.variables.items()))})"


class EstimandAST(Tree[int, ASTNode, Any]):
    """
    Abstract Syntax Tree for representing causal estimands.
    Nodes are uniquely identified by integer IDs.
    """

    def merge(self, other: "EstimandAST") -> int:
        """Merges another EstimandAST into this one, returning the new root ID."""
        return _copy_subtree(other, other.get_root(), self)

    def _get_base_node_config(self):
        def _math_vars(vars_set):
            # Triple-emphasis for a heavy LaTeX math look (Bold + Italic)
            return ", ".join(f"<b><i>{v}</i></b>" for v in sorted(vars_set))

        def _tt(text):
            return f'<font face="Courier-Bold">{text}</font>'

        return {
            PNode: {
                "label": lambda n: "<<b>P</b>(V)>",
                "shape": "box",
                "style": "filled,rounded",
                "color": "#E1F5FE",  # Light Blue
                "fontcolor": "black",
                "fontname": "Times-BoldItalic",
            },
            CondNode: {
                "label": lambda n: f"<<b>P</b>({_math_vars(n.num_vars)} | {_math_vars(n.den_vars)})>",
                "shape": "box",
                "style": "filled,rounded",
                "color": "#E8F5E9",  # Light Green
                "fontcolor": "black",
                "fontname": "Times-BoldItalic",
            },
            MargNode: {
                "label": lambda n: f"<{_tt('MARG')}({_math_vars(n.marginalize_vars)})>",
                "shape": "box",
                "style": "filled,rounded",
                "color": "#FFF9C4",  # Light Yellow
                "fontcolor": "black",
                "fontname": "Times-Roman",
            },
            ProdNode: {
                "label": f"<{_tt('PROD')}>",
                "shape": "box",
                "style": "filled,rounded",
                "color": "#FFEBEE",  # Light Red
                "fontcolor": "black",
            },
            DetProdNode: {
                "label": f"<{_tt('DETPROD')}>",
                "shape": "box",
                "style": "filled,rounded",
                "color": "#F3E5F5",  # Light Purple
                "fontcolor": "black",
            },
            PowNode: {
                "label": lambda n: f"<{_tt('POW')}({n.power};)>",
                "shape": "box",
                "style": "filled,rounded",
                "color": "#EEEEEE",  # Light Grey
                "fontcolor": "black",
            },
            ConstantNode: {
                "label": lambda n: str(n),
                "shape": "plaintext",
                "fontname": "Times-Roman",
            },
            InstNode: {
                "label": lambda n: f"<{_tt('INST')}({','.join(f'<b><i>{v}</i></b>={val}' for v, val in sorted(n.variables.items()))})>",
                "shape": "diamond",
                "style": "filled",
                "color": "#FFF3E0",  # Light Orange
                "fontcolor": "black",
                "fontname": "Times-Roman",
            },
        }


# ---------------------------------------------------------------------------
# Helpers to build the AST
# ---------------------------------------------------------------------------


def _copy_subtree(
    source_ast: EstimandAST, source_node_id: int, target_ast: EstimandAST
) -> int:
    """Deep copy of a subtree from source_ast into target_ast (avoids shared-ID conflicts)."""
    new_id = target_ast.add_node(source_ast.get_node_data(source_node_id))
    for child_id, edge_data in source_ast.get_outgoing_edges(source_node_id):
        new_child_id = _copy_subtree(source_ast, child_id, target_ast)
        target_ast.add_edge(new_id, new_child_id, edge_data)
    return new_id


def make_p(variables: set) -> EstimandAST:
    ast = EstimandAST()
    ast.add_node(PNode(variables))
    return ast


def make_marg(marginalize_vars: set, child_ast: EstimandAST) -> EstimandAST:
    if not marginalize_vars:
        return child_ast

    child_root = child_ast.get_root()
    child_node = child_ast.get_node_data(child_root)

    # Merge nested MARGs
    if isinstance(child_node, MargNode):
        merged_vars = marginalize_vars.union(child_node.marginalize_vars)
        grandchildren = list(child_ast.get_outgoing_edges(child_root))

        if len(grandchildren) == 1:
            grandchild_id, _ = grandchildren[0]
            temp_ast = EstimandAST()
            _copy_subtree(child_ast, grandchild_id, temp_ast)
            return make_marg(merged_vars, temp_ast)

        clean_ast = EstimandAST()
        root_id = clean_ast.add_node(MargNode(merged_vars))
        for grandchild_id, edge_data in grandchildren:
            new_grandchild_id = _copy_subtree(child_ast, grandchild_id, clean_ast)
            clean_ast.add_edge(root_id, new_grandchild_id, edge_data)
        return clean_ast

    # Full marginalization of a PNode gives 1
    if isinstance(child_node, PNode):
        if marginalize_vars.issuperset(child_node.variables):
            ast = EstimandAST()
            ast.add_node(ConstantNode(1))
            return ast

    ast = EstimandAST()
    root_id = ast.add_node(MargNode(marginalize_vars))
    child_root_in_ast = _copy_subtree(child_ast, child_root, ast)
    ast.add_edge(root_id, child_root_in_ast)
    return ast


def _make_binary_prod(
    node_cls: type,
    children: list[EstimandAST],
) -> EstimandAST:
    """Left-associative binary fold of children into a tree of node_cls nodes."""
    filtered = []
    for child_ast in children:
        node = child_ast.get_node_data(child_ast.get_root())
        if isinstance(node, ConstantNode) and node.value == 1:
            continue
        filtered.append(child_ast)

    if not filtered:
        ast = EstimandAST()
        ast.add_node(ConstantNode(1))
        return ast

    if len(filtered) == 1:
        return filtered[0]

    # Left-associative fold: PROD[PROD[A, B], C]
    acc_ast = EstimandAST()
    _copy_subtree(filtered[0], filtered[0].get_root(), acc_ast)

    for child_ast in filtered[1:]:
        new_ast = EstimandAST()
        new_root = new_ast.add_node(node_cls())

        acc_root_in_new = _copy_subtree(acc_ast, acc_ast.get_root(), new_ast)
        child_root_in_new = _copy_subtree(child_ast, child_ast.get_root(), new_ast)

        new_ast.add_edge(new_root, acc_root_in_new)
        new_ast.add_edge(new_root, child_root_in_new)
        acc_ast = new_ast

    return acc_ast


def make_prod(children: list[EstimandAST]) -> EstimandAST:
    """Create a left-associative binary tree of ProdNodes."""
    return _make_binary_prod(ProdNode, children)


def make_det_prod(children: list[EstimandAST]) -> EstimandAST:
    """Create a left-associative binary tree of DetProdNodes."""
    return _make_binary_prod(DetProdNode, children)


def make_pow(power: int, child_ast: EstimandAST) -> EstimandAST:
    child_root = child_ast.get_root()
    child_node = child_ast.get_node_data(child_root)
    if isinstance(child_node, ConstantNode) and child_node.value == 1:
        return child_ast

    ast = EstimandAST()
    root_id = ast.add_node(PowNode(power))
    child_root_in_ast = _copy_subtree(child_ast, child_root, ast)
    ast.add_edge(root_id, child_root_in_ast)
    return ast


def make_inst(variables: set, child_ast: EstimandAST) -> EstimandAST:
    """Wrap a child expression with an InstNode for the given variables."""
    if not variables:
        return child_ast

    ast = EstimandAST()
    root_id = ast.add_node(InstNode(variables))
    child_root_in_ast = _copy_subtree(child_ast, child_ast.get_root(), ast)
    ast.add_edge(root_id, child_root_in_ast)
    return ast


def get_vars(ast: EstimandAST, node_id: Optional[int] = None) -> set:
    if node_id is None:
        node_id = ast.get_root()
    node = ast.get_node_data(node_id)
    if isinstance(node, PNode):
        return set(node.variables)
    elif isinstance(node, CondNode):
        return set(node.num_vars)
    elif isinstance(node, ConstantNode):
        return set()
    elif isinstance(node, MargNode):
        child_id, _ = ast.get_outgoing_edges(node_id)[0]
        return get_vars(ast, child_id) - node.marginalize_vars
    elif isinstance(node, PowNode):
        child_id, _ = ast.get_outgoing_edges(node_id)[0]
        return get_vars(ast, child_id)
    elif isinstance(node, (ProdNode, DetProdNode)):
        vars_set = set()
        for child_id, _ in ast.get_outgoing_edges(node_id):
            vars_set.update(get_vars(ast, child_id))
        return vars_set
    elif isinstance(node, InstNode):
        child_id, _ = ast.get_outgoing_edges(node_id)[0]
        return get_vars(ast, child_id)
    return set()


def ast_to_str(ast: EstimandAST, node_id: Optional[int] = None) -> str:
    if node_id is None:
        node_id = ast.get_root()
    node = ast.get_node_data(node_id)
    children = ast.get_outgoing_edges(node_id)
    if not children:
        return str(node)

    child_strs = []
    for child_id, _ in children:
        child_strs.append(ast_to_str(ast, child_id))

    return f"{node}[{', '.join(sorted(child_strs))}]"


def is_pnode_or_marg_pnode(ast: EstimandAST, node_id: int) -> bool:
    node = ast.get_node_data(node_id)
    if isinstance(node, PNode):
        return True
    if isinstance(node, MargNode):
        child_id, _ = ast.get_outgoing_edges(node_id)[0]
        return is_pnode_or_marg_pnode(ast, child_id)
    return False


def is_conditional(ast: EstimandAST, num_id: int, den_id: int):
    den_node = ast.get_node_data(den_id)
    if not (isinstance(den_node, PowNode) and den_node.power == -1):
        return False, None, None

    pow_child_edges = ast.get_outgoing_edges(den_id)
    if not pow_child_edges:
        return False, None, None
    pow_child_id, _ = pow_child_edges[0]
    pow_child_node = ast.get_node_data(pow_child_id)

    if not isinstance(pow_child_node, MargNode):
        return False, None, None

    marg_child_edges = ast.get_outgoing_edges(pow_child_id)
    if not marg_child_edges:
        return False, None, None
    marg_child_id, _ = marg_child_edges[0]

    if not is_pnode_or_marg_pnode(ast, marg_child_id):
        return False, None, None

    # Case 1
    if ast_to_str(ast, num_id) == ast_to_str(ast, marg_child_id):
        M = pow_child_node.marginalize_vars
        V_A = get_vars(ast, num_id)
        return True, M, V_A - M

    # Case 2
    num_node = ast.get_node_data(num_id)
    if isinstance(num_node, MargNode):
        num_child_edges = ast.get_outgoing_edges(num_id)
        if not num_child_edges:
            return False, None, None
        num_child_id, _ = num_child_edges[0]
        if ast_to_str(ast, num_child_id) == ast_to_str(ast, marg_child_id):
            M_A = num_node.marginalize_vars
            M_total = pow_child_node.marginalize_vars
            M = M_total - M_A
            if M_total == M_A.union(M):
                V_A = get_vars(ast, num_id)
                return True, M, V_A - M

    return False, None, None


def ast_to_latex(ast: EstimandAST, node_id: Optional[int] = None) -> str:
    if node_id is None:
        node_id = ast.get_root()
    node = ast.get_node_data(node_id)
    children = ast.get_outgoing_edges(node_id)

    if isinstance(node, PNode):
        return f"P({','.join(map(str, sorted(node.variables)))})"

    elif isinstance(node, ConstantNode):
        return str(node.value)

    elif isinstance(node, MargNode):
        child_id, _ = children[0]
        child_node = ast.get_node_data(child_id)

        # If child is PNode, display as the resulting marginal for readability
        if isinstance(child_node, PNode):
            remaining_vars = child_node.variables - node.marginalize_vars
            return f"P({','.join(map(str, sorted(remaining_vars)))})"

        child_latex = ast_to_latex(ast, child_id)
        if isinstance(child_node, (ProdNode, DetProdNode, MargNode)):
            child_latex = f"\\left( {child_latex} \\right)"

        vars_str = ",".join(map(str, sorted(node.marginalize_vars)))
        return f"\\sum_{{{vars_str}}} {child_latex}"

    elif isinstance(node, CondNode):
        child_latex = ast_to_latex(ast, children[0][0])
        n_str = ",".join(map(str, sorted(node.num_vars)))
        d_str = ",".join(map(str, sorted(node.den_vars)))
        return f"P({n_str} \\mid {d_str})" if d_str else f"P({n_str})"

    elif isinstance(node, PowNode):
        child_id, _ = children[0]
        child_latex = ast_to_latex(ast, child_id)
        child_node = ast.get_node_data(child_id)
        if isinstance(child_node, (ProdNode, DetProdNode, MargNode)):
            child_latex = f"\\left( {child_latex} \\right)"
        return f"{{{child_latex}}}^{{{node.power}}}"

    elif isinstance(node, (ProdNode, DetProdNode)):
        op = "\\otimes" if isinstance(node, ProdNode) else "\\odot"
        if len(children) == 2:
            child1_id, _ = children[0]
            child2_id, _ = children[1]

            is_cond, M, given = is_conditional(ast, child1_id, child2_id)
            if is_cond:
                m_str = ",".join(map(str, sorted(M)))
                given_str = ",".join(map(str, sorted(given)))
                return f"P({m_str} \\mid {given_str})" if given else f"P({m_str})"

            is_cond, M, given = is_conditional(ast, child2_id, child1_id)
            if is_cond:
                m_str = ",".join(map(str, sorted(M)))
                given_str = ",".join(map(str, sorted(given)))
                return f"P({m_str} \\mid {given_str})" if given else f"P({m_str})"

        terms = []
        for child_id, _ in children:
            c_node = ast.get_node_data(child_id)
            c_latex = ast_to_latex(ast, child_id)
            if isinstance(c_node, MargNode):
                c_latex = f"\\left( {c_latex} \\right)"
            terms.append(c_latex)

        return f" {op} ".join(terms)

    elif isinstance(node, InstNode):
        child_id, _ = children[0]
        child_latex = ast_to_latex(ast, child_id)
        vars_str = ",".join(f"{v}={v}" for v in sorted(node.variables))
        return f"\\left[ {child_latex} \\right]_{{\\substack{{{vars_str}}}}}"

    return ""


# ---------------------------------------------------------------------------
# SymExpr algebra (for simplified display)
# ---------------------------------------------------------------------------


class SymExpr:
    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def rename_var(self, old_var: str, new_var: str) -> "SymExpr":
        return self


class SymOne(SymExpr):
    def __str__(self):
        return "1"

    def to_latex(self):
        return "1"


class SymP(SymExpr):
    def __init__(self, num: set, den: set = None):
        self.num = set(num)
        self.den = set(den) if den else set()

    def __str__(self):
        n = ",".join(sorted(self.num))
        if self.den:
            d = ",".join(sorted(self.den))
            return f"P({n}|{d})"
        return f"P({n})"

    def to_latex(self):
        n = ",".join(sorted(self.num))
        if self.den:
            d = ",".join(sorted(self.den))
            return f"P({n} \\mid {d})"
        return f"P({n})"

    def rename_var(self, old_var: str, new_var: str) -> "SymExpr":
        new_num = {new_var if v == old_var else v for v in self.num}
        new_den = {new_var if v == old_var else v for v in self.den}
        return SymP(new_num, new_den)


class SymProd(SymExpr):
    def __init__(self, terms: list):
        self.terms = terms

    def __str__(self):
        return " * ".join(sorted([str(t) for t in self.terms]))

    def to_latex(self):
        return " \\times ".join(
            [
                t.to_latex() if not isinstance(t, SymSum) else f"\\left( {t.to_latex()} \\right)"
                for t in self.terms
            ]
        )

    def rename_var(self, old_var: str, new_var: str) -> "SymExpr":
        return SymProd([t.rename_var(old_var, new_var) for t in self.terms])


class SymFrac(SymExpr):
    def __init__(self, num: SymExpr, den: SymExpr):
        self.num = num
        self.den = den

    def __str__(self):
        return f"({self.num}) / ({self.den})"

    def to_latex(self):
        def _inv(t: "SymExpr") -> str:
            tex = t.to_latex()
            if isinstance(t, (SymSum, SymProd, SymFrac)):
                tex = f"\\left( {tex} \\right)"
            return f"{{{tex}}}^{{-1}}"

        if isinstance(self.den, SymProd):
            inv_parts = [_inv(t) for t in self.den.terms]
        else:
            inv_parts = [_inv(self.den)]

        if isinstance(self.num, SymOne):
            return " \\times ".join(inv_parts)

        if isinstance(self.num, SymProd):
            n_tex = self.num.to_latex()
        else:
            n_tex = self.num.to_latex()
            if isinstance(self.num, (SymSum, SymFrac)):
                n_tex = f"\\left( {n_tex} \\right)"

        return n_tex + " \\times " + " \\times ".join(inv_parts)

    def rename_var(self, old_var: str, new_var: str) -> "SymExpr":
        return SymFrac(self.num.rename_var(old_var, new_var), self.den.rename_var(old_var, new_var))


class SymSum(SymExpr):
    def __init__(self, vars: set, term: SymExpr):
        self.vars = set(vars)
        self.term = term

    def __str__(self):
        v = ",".join(sorted(self.vars))
        return f"SUM_{v}[{self.term}]"

    def to_latex(self):
        v = ",".join(sorted(self.vars))
        term_latex = self.term.to_latex()
        if isinstance(self.term, (SymSum, SymFrac)):
            term_latex = f"\\left( {term_latex} \\right)"
        return f"\\sum_{{{v}}} {term_latex}"

    def rename_var(self, old_var: str, new_var: str) -> "SymExpr":
        new_vars = {new_var if v == old_var else v for v in self.vars}
        return SymSum(new_vars, self.term.rename_var(old_var, new_var))


def ast_to_sym(ast: EstimandAST, node_id: Optional[int] = None) -> SymExpr:
    if node_id is None:
        node_id = ast.get_root()
    node = ast.get_node_data(node_id)
    children = ast.get_outgoing_edges(node_id)

    if isinstance(node, PNode):
        return SymP(node.variables)
    elif isinstance(node, CondNode):
        return SymP(node.num_vars, node.den_vars)
    elif isinstance(node, ConstantNode):
        if node.value == 1:
            return SymOne()
    elif isinstance(node, MargNode):
        child_sym = ast_to_sym(ast, children[0][0])
        return SymSum(node.marginalize_vars, child_sym)
    elif isinstance(node, PowNode):
        child_sym = ast_to_sym(ast, children[0][0])
        if node.power == -1:
            return SymFrac(SymOne(), child_sym)
    elif isinstance(node, (ProdNode, DetProdNode)):
        terms = [ast_to_sym(ast, c) for c, _ in children]
        return SymProd(terms)
    elif isinstance(node, InstNode):
        return ast_to_sym(ast, children[0][0])
    return SymOne()


def get_free_vars(expr: SymExpr) -> set:
    if isinstance(expr, SymOne):
        return set()
    if isinstance(expr, SymP):
        return expr.num.union(expr.den)
    if isinstance(expr, SymProd):
        v = set()
        for t in expr.terms:
            v.update(get_free_vars(t))
        return v
    if isinstance(expr, SymFrac):
        return get_free_vars(expr.num).union(get_free_vars(expr.den))
    if isinstance(expr, SymSum):
        return get_free_vars(expr.term) - expr.vars
    return set()


def resolve_name_clashes(expr: SymExpr, outer_free_vars: set = None) -> SymExpr:
    if outer_free_vars is None:
        outer_free_vars = get_free_vars(expr)

    if isinstance(expr, SymOne) or isinstance(expr, SymP):
        return expr

    if isinstance(expr, SymProd):
        return SymProd([resolve_name_clashes(t, outer_free_vars) for t in expr.terms])

    if isinstance(expr, SymFrac):
        return SymFrac(
            resolve_name_clashes(expr.num, outer_free_vars),
            resolve_name_clashes(expr.den, outer_free_vars),
        )

    if isinstance(expr, SymSum):
        new_vars = set()
        new_term = expr.term
        for v in expr.vars:
            if v in outer_free_vars:
                new_v = v + "'"
                while new_v in outer_free_vars:
                    new_v += "'"
                new_term = new_term.rename_var(v, new_v)
                new_vars.add(new_v)
            else:
                new_vars.add(v)

        inner_free_vars = outer_free_vars.union(new_vars)
        new_term = resolve_name_clashes(new_term, inner_free_vars)

        return SymSum(new_vars, new_term)


def simplify_sym(expr: SymExpr) -> SymExpr:
    if isinstance(expr, SymP):
        return expr
    if isinstance(expr, SymOne):
        return expr
    if isinstance(expr, SymSum):
        term = simplify_sym(expr.term)
        if isinstance(term, SymP):
            eff_v = expr.vars.intersection(term.num)
            rem = term.num - eff_v
            if not rem:
                return SymOne()
            if eff_v:
                return SymP(rem, term.den)
        if isinstance(term, SymSum):
            return simplify_sym(SymSum(expr.vars | term.vars, term.term))
        if isinstance(term, SymProd):
            sum_vars = expr.vars.copy()
            final_terms = term.terms.copy()

            vars_to_remove = set()
            for v in list(sum_vars):
                containing = [t for t in final_terms if v in get_free_vars(t)]
                not_containing = [t for t in final_terms if v not in get_free_vars(t)]
                if not_containing:
                    inner_sum = simplify_sym(SymSum({v}, SymProd(containing)))
                    final_terms = not_containing + [inner_sum]
                    vars_to_remove.add(v)
            sum_vars -= vars_to_remove

            if not sum_vars:
                return simplify_sym(SymProd(final_terms))
            return SymSum(sum_vars, simplify_sym(SymProd(final_terms)))

        return SymSum(expr.vars, term)

    if isinstance(expr, SymFrac):
        num = simplify_sym(expr.num)
        den = simplify_sym(expr.den)
        if isinstance(num, SymP) and isinstance(den, SymP):
            if not num.den and not den.den and den.num.issubset(num.num):
                rem = num.num - den.num
                if not rem:
                    return SymOne()
                return SymP(rem, den.num)

        if str(num) == str(den):
            return SymOne()

        return SymFrac(num, den)

    if isinstance(expr, SymProd):
        terms = []
        for t in expr.terms:
            st = simplify_sym(t)
            if isinstance(st, SymProd):
                terms.extend(st.terms)
            elif not isinstance(st, SymOne):
                terms.append(st)

        changed = True
        while changed:
            changed = False
            for i in range(len(terms)):
                for j in range(len(terms)):
                    if i != j and isinstance(terms[i], SymP) and isinstance(terms[j], SymP):
                        ti = terms[i]
                        tj = terms[j]
                        if ti.den == tj.num.union(tj.den):
                            merged = SymP(ti.num.union(tj.num), tj.den)
                            terms[j] = merged
                            terms.pop(i)
                            changed = True
                            break
                if changed:
                    break

        nums = []
        dens = []
        for t in terms:
            if isinstance(t, SymFrac):
                nums.append(t.num)
                dens.append(t.den)
            else:
                nums.append(t)

        if dens:
            n_expr = simplify_sym(SymProd(nums)) if nums else SymOne()
            d_expr = simplify_sym(SymProd(dens)) if dens else SymOne()

            if isinstance(n_expr, SymProd) and isinstance(d_expr, SymProd):
                n_terms = n_expr.terms[:]
                d_terms = d_expr.terms[:]
                for nt in n_terms[:]:
                    for dt in d_terms[:]:
                        if str(nt) == str(dt):
                            n_terms.remove(nt)
                            d_terms.remove(dt)
                            break
                n_expr = simplify_sym(SymProd(n_terms)) if n_terms else SymOne()
                d_expr = simplify_sym(SymProd(d_terms)) if d_terms else SymOne()

            if isinstance(n_expr, SymP) and isinstance(d_expr, SymP):
                if not n_expr.den and not d_expr.den and d_expr.num.issubset(n_expr.num):
                    rem = n_expr.num - d_expr.num
                    if not rem:
                        return SymOne()
                    return SymP(rem, d_expr.num)

            if str(n_expr) == str(d_expr):
                return SymOne()

            return SymFrac(n_expr, d_expr)

        if not terms:
            return SymOne()
        if len(terms) == 1:
            return terms[0]
        return SymProd(terms)


def get_simplified_latex(ast: EstimandAST) -> str:
    sym_expr = ast_to_sym(ast)
    simplified_expr = simplify_sym(sym_expr)
    clean_expr = resolve_name_clashes(simplified_expr)
    return clean_expr.to_latex()


# ---------------------------------------------------------------------------
# AST canonicalization
# ---------------------------------------------------------------------------


def _get_ast_free_vars(ast: "EstimandAST", node_id: int) -> set:
    """Return free (non-integrated) variables in an AST subtree."""
    node = ast.get_node_data(node_id)
    if isinstance(node, PNode):
        return set(node.variables)
    if isinstance(node, CondNode):
        return set(node.num_vars) | set(node.den_vars)
    if isinstance(node, ConstantNode):
        return set()
    if isinstance(node, MargNode):
        child_id = ast.get_outgoing_edges(node_id)[0][0]
        return _get_ast_free_vars(ast, child_id) - node.marginalize_vars
    if isinstance(node, PowNode):
        child_id = ast.get_outgoing_edges(node_id)[0][0]
        return _get_ast_free_vars(ast, child_id)
    if isinstance(node, (ProdNode, DetProdNode)):
        v: set = set()
        for child_id, _ in ast.get_outgoing_edges(node_id):
            v |= _get_ast_free_vars(ast, child_id)
        return v
    if isinstance(node, InstNode):
        child_id = ast.get_outgoing_edges(node_id)[0][0]
        return _get_ast_free_vars(ast, child_id)
    return set()


def _rename_var_in_subtree(
    ast: "EstimandAST", node_id: int, old: str, new: str
) -> "EstimandAST":
    """Deep copy of a subtree with every occurrence of ``old`` renamed to ``new``."""
    result = EstimandAST()
    _rename_var_in_subtree_recursive(ast, node_id, old, new, result)
    return result


def _rename_var_in_subtree_recursive(
    ast: "EstimandAST", node_id: int, old: str, new: str, target_ast: "EstimandAST"
) -> int:
    node = ast.get_node_data(node_id)

    if isinstance(node, PNode):
        new_id = target_ast.add_node(PNode({new if v == old else v for v in node.variables}))
    elif isinstance(node, CondNode):
        new_id = target_ast.add_node(
            CondNode(
                {new if v == old else v for v in node.num_vars},
                {new if v == old else v for v in node.den_vars},
            )
        )
    elif isinstance(node, MargNode):
        new_id = target_ast.add_node(
            MargNode({new if v == old else v for v in node.marginalize_vars})
        )
    elif isinstance(node, InstNode):
        new_id = target_ast.add_node(
            InstNode({new if v == old else v for v in node.variables})
        )
    else:
        new_id = target_ast.add_node(node)

    for child_id, _ in ast.get_outgoing_edges(node_id):
        new_child_id = _rename_var_in_subtree_recursive(
            ast, child_id, old, new, target_ast
        )
        target_ast.add_edge(new_id, new_child_id)

    return new_id


def _simplify_collect(
    ast: "EstimandAST", node_id: int, in_pow: bool
) -> "tuple[EstimandAST, set]":
    """Recursive worker for ``simplify_ast``.

    Returns ``(new_ast, sum_vars)`` where ``sum_vars`` is the set of
    variables this subtree wants to bubble up as an outer MARG (always empty
    when ``in_pow`` is True).
    """
    node = ast.get_node_data(node_id)

    # ── Leaves ────────────────────────────────────────────────────────────
    if isinstance(node, (PNode, ConstantNode)):
        leaf = EstimandAST()
        leaf.add_node(node)
        return leaf, set()

    # ── MARG ──────────────────────────────────────────────────────────────
    if isinstance(node, MargNode):
        child_id = ast.get_outgoing_edges(node_id)[0][0]
        c_ast, c_sum = _simplify_collect(ast, child_id, in_pow)
        c_node = c_ast.get_node_data(c_ast.get_root())

        if isinstance(c_node, PNode):
            if node.marginalize_vars.issuperset(c_node.variables):
                # Full marginalization → 1
                result = EstimandAST()
                result.add_node(ConstantNode(1))
                return result, set()
            # Partial marginalization: keep MARG(S)[P(V)] intact — do NOT strip to P(V-S)
            new_ast = make_marg(node.marginalize_vars, c_ast)
            return new_ast, set()

        all_sum = node.marginalize_vars | c_sum
        if in_pow:
            new_ast = make_marg(all_sum, c_ast)
            return new_ast, set()
        return c_ast, all_sum

    # ── PROD / DETPROD ────────────────────────────────────────────────────
    if isinstance(node, (ProdNode, DetProdNode)):
        node_cls = type(node)
        results: list = []
        for cid, _ in ast.get_outgoing_edges(node_id):
            c_ast, c_sum = _simplify_collect(ast, cid, in_pow)
            results.append([c_ast, c_sum])

        child_free = [_get_ast_free_vars(r[0], r[0].get_root()) for r in results]
        all_names_in_use: set = set()
        for i in range(len(results)):
            all_names_in_use |= child_free[i] | results[i][1]

        for i, (c_ast, c_sum) in enumerate(results):
            if not c_sum:
                continue
            sibling_free: set = set()
            for j in range(len(results)):
                if j != i:
                    sibling_free |= child_free[j]

            new_c_sum = set(c_sum)
            for v in list(c_sum):
                if v not in sibling_free:
                    continue
                new_v = v + "'"
                while new_v in all_names_in_use:
                    new_v += "'"
                c_ast = _rename_var_in_subtree(c_ast, c_ast.get_root(), v, new_v)
                results[i][0] = c_ast
                new_c_sum.discard(v)
                new_c_sum.add(new_v)
                all_names_in_use.add(new_v)
                child_free[i] = _get_ast_free_vars(results[i][0], results[i][0].get_root())

            results[i][1] = new_c_sum

        all_sum: set = set()
        simplified_children: list = []

        for c_ast, c_sum in results:
            all_sum |= c_sum
            # Do NOT flatten nested ProdNode/DetProdNode — preserve binary structure
            simplified_children.append(c_ast)

        new_ast = _make_binary_prod(node_cls, simplified_children)

        if in_pow and all_sum:
            new_ast = make_marg(all_sum, new_ast)
            return new_ast, set()

        return new_ast, all_sum

    # ── POW ───────────────────────────────────────────────────────────────
    if isinstance(node, PowNode):
        child_id = ast.get_outgoing_edges(node_id)[0][0]
        c_ast, _ = _simplify_collect(ast, child_id, in_pow=True)
        new_ast = make_pow(node.power, c_ast)
        return new_ast, set()

    # ── INST ──────────────────────────────────────────────────────────────
    if isinstance(node, InstNode):
        child_id = ast.get_outgoing_edges(node_id)[0][0]
        c_ast, c_sum = _simplify_collect(ast, child_id, in_pow)
        new_ast = make_inst(node.variables, c_ast)
        return new_ast, c_sum

    # Fallback
    fallback = EstimandAST()
    fallback.add_node(node)
    return fallback, set()


def simplify_ast(ast: "EstimandAST") -> "EstimandAST":
    """Canonicalize an estimand AST.

    Transformations applied:
    - ``MARG(V)[P(V)]`` → ``ConstantNode(1)`` (full marginalization only).
    - ``MARG(S)[P(V)]`` where S ⊊ V is left intact (circuit structure preserved).
    - Integral MARGs inside ``PROD``/``DETPROD`` are bubbled up to a single
      top-level MARG, with integration-variable name clashes resolved.
    - Integral MARGs inside ``POW`` stay in place.
    - Binary structure of ``PROD``/``DETPROD`` nodes is preserved (no flattening).
    """
    new_ast, sum_vars = _simplify_collect(ast, ast.get_root(), in_pow=False)
    if sum_vars:
        new_ast = make_marg(sum_vars, new_ast)
    return new_ast


def present_ast(ast: "EstimandAST") -> "EstimandAST":
    """
    Format AST for visual presentation.
    - Rewrites PROD/DETPROD(A, POW(-1, B)) into COND(A, B) for visualization.
    """

    def _build_cond_ast(src_ast: EstimandAST, src_root: int) -> EstimandAST:
        node = src_ast.get_node_data(src_root)

        if isinstance(node, (ProdNode, DetProdNode)):
            edges = src_ast.get_outgoing_edges(src_root)
            if len(edges) == 2:
                c1_id, _ = edges[0]
                c2_id, _ = edges[1]
                n1 = src_ast.get_node_data(c1_id)
                n2 = src_ast.get_node_data(c2_id)

                num_id, den_id = None, None
                if isinstance(n2, PowNode) and n2.power == -1:
                    num_id = c1_id
                    den_id = src_ast.get_outgoing_edges(c2_id)[0][0]
                elif isinstance(n1, PowNode) and n1.power == -1:
                    num_id = c2_id
                    den_id = src_ast.get_outgoing_edges(c1_id)[0][0]

                if num_id and den_id:
                    num_ast = _build_cond_ast(src_ast, num_id)
                    den_ast = _build_cond_ast(src_ast, den_id)

                    num_vars = get_vars(num_ast)
                    den_vars = get_vars(den_ast)

                    res_ast = EstimandAST()
                    res_root = res_ast.add_node(CondNode(num_vars - den_vars, den_vars))
                    num_root_in_res = _copy_subtree(num_ast, num_ast.get_root(), res_ast)
                    # We keep numerator to show what it is derived from visually
                    res_ast.add_edge(res_root, num_root_in_res)
                    return res_ast

        # Otherwise deep copy and recurse
        res_ast = EstimandAST()
        res_root = res_ast.add_node(node)
        for child_id, edge_data in src_ast.get_outgoing_edges(src_root):
            c_ast = _build_cond_ast(src_ast, child_id)
            c_root_in_res = _copy_subtree(c_ast, c_ast.get_root(), res_ast)
            res_ast.add_edge(res_root, c_root_in_res, edge_data)

        return res_ast

    return _build_cond_ast(ast, ast.get_root())
