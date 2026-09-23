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

    def __str__(self):
        return self.pretty()

    def pretty(self) -> str:
        """Human-readable tree rendering, e.g. for `print(ast)`."""
        lines = []

        def _rec(node_id: int, prefix: str, is_last: bool, is_root: bool):
            node = self.get_node_data(node_id)
            connector = "" if is_root else ("└── " if is_last else "├── ")
            lines.append(f"{prefix}{connector}{node}")
            children = self.get_outgoing_edges(node_id)
            if not is_root:
                prefix += "    " if is_last else "│   "
            for i, (child_id, _) in enumerate(children):
                _rec(child_id, prefix, i == len(children) - 1, False)

        _rec(self.get_root(), "", True, True)
        return "\n".join(lines)

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


def _copy_subtree(source_ast: EstimandAST, source_node_id: int, target_ast: EstimandAST) -> int:
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


def make_cond(num: set, den: set, child_ast: EstimandAST) -> EstimandAST:
    """Create P(A|B) as P(A,B) * P(B)^-1"""
    ast = EstimandAST()
    prod_id = ast.add_node(CondNode(num_vars=num, den_vars=den))
    child_id = _copy_subtree(child_ast, child_ast.get_root(), ast)
    ast.add_edge(prod_id, child_id)
    return ast


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
        return set(node.num_vars) | set(node.den_vars)
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
