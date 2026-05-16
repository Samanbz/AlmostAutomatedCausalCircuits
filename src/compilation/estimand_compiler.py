"""
Estimand Compiler: Compile an EstimandAST into a new SymbolicArithmeticCircuit.

Instead of evaluating causal estimands algebraically at forward time (which breaks
for multi-layer circuits), this module constructs a NEW symbolic circuit that
computes the estimand.  The new circuit goes through the standard pipeline:
fold → fuse → compile (Monarch or Tensorized), and then standard forward()
evaluates the query.

The compilation implements the circuit algebra from Wang et al. (2025):
- PNode       → copy of the base SPN
- MargNode(W) → aggregate (replace leaves for W with constant-1)
- PowNode(p)  → elementwise mapping (negate weights & leaves for p=-1)
- DetProdNode → PROD-SCMP (support-compatible product, h stays h)
- ProdNode    → PROD-CMP (compatible product, h → h², Kronecker/Monarch weights)
"""

from copy import deepcopy

from src.symbolic.arithmetic.circuit import SymbolicArithmeticCircuit
from src.symbolic.arithmetic.nodes import (
    ConstantLeafNode,
    HadamardProductNode,
    InverseLeafNode,
    KroneckerProductNode,
    LeafNode,
    ProductLeafNode,
    SumNode,
    UnaryProductNode,
    UniversalSumNode,
)
from src.symbolic.id_ast import (
    DetProdNode,
    EstimandAST,
    InstNode,
    MargNode,
    PNode,
    PowNode,
    ProdNode,
)
from src.utils.node_allocator import NodeAllocator


def compile_estimand(
    spn: SymbolicArithmeticCircuit,
    ast: EstimandAST,
    ast_root: str,
    var_map: dict[str, int],
) -> SymbolicArithmeticCircuit:
    """Compile an estimand AST into a new symbolic circuit.

    Strips the outermost MargNode (if present) and compiles the inner
    expression into a circuit.  The outermost marginalization is handled
    at query time via NaN-marginalization.

    Args:
        spn: The base symbolic arithmetic circuit (SPN).
        ast: The estimand AST from the identification algorithm.
        ast_root: Root node ID of the AST.
        var_map: Maps variable names (str) to column indices (int).

    Returns:
        A new SymbolicArithmeticCircuit computing the estimand.
    """
    allocator = NodeAllocator()

    # Strip outermost MargNode — those variables are marginalized at query time
    node = ast.get_node_data(ast_root)
    if isinstance(node, MargNode):
        inner_root = ast.get_children(ast_root)[0]
    else:
        inner_root = ast_root

    return _compile_node(spn, ast, inner_root, var_map, allocator)


def _compile_node(
    spn: SymbolicArithmeticCircuit,
    ast: EstimandAST,
    node_id: str,
    var_map: dict[str, int],
    allocator: NodeAllocator,
) -> SymbolicArithmeticCircuit:
    """Recursively compile an AST node into a symbolic circuit."""
    node = ast.get_node_data(node_id)

    if isinstance(node, PNode):
        return _copy_spn(spn, allocator)

    elif isinstance(node, MargNode):
        child_id = ast.get_children(node_id)[0]
        child_circuit = _compile_node(spn, ast, child_id, var_map, allocator)
        marg_cols = {var_map[v] for v in node.marginalize_vars if v in var_map}
        return _aggregate(child_circuit, marg_cols)

    elif isinstance(node, PowNode):
        child_id = ast.get_children(node_id)[0]
        child_circuit = _compile_node(spn, ast, child_id, var_map, allocator)
        return _elementwise_map(child_circuit, node.power)

    elif isinstance(node, DetProdNode):
        children = ast.get_children(node_id)
        left_circuit = _compile_node(spn, ast, children[0], var_map, allocator)
        right_circuit = _compile_node(spn, ast, children[1], var_map, allocator)
        return _product_scmp(left_circuit, right_circuit, allocator)

    elif isinstance(node, ProdNode):
        children = ast.get_children(node_id)
        left_circuit = _compile_node(spn, ast, children[0], var_map, allocator)
        right_circuit = _compile_node(spn, ast, children[1], var_map, allocator)
        return _product_cmp(left_circuit, right_circuit, allocator)

    elif isinstance(node, InstNode):
        child_id = ast.get_children(node_id)[0]
        return _compile_node(spn, ast, child_id, var_map, allocator)

    raise ValueError(f"Unknown AST node type: {type(node)}")


# ---------------------------------------------------------------------------
# Primitive operations
# ---------------------------------------------------------------------------


def _copy_spn(
    spn: SymbolicArithmeticCircuit, allocator: NodeAllocator
) -> SymbolicArithmeticCircuit:
    """Deep-copy an SPN with fresh node IDs."""
    new_circuit = SymbolicArithmeticCircuit()
    old_to_new: dict[int, int] = {}

    for old_id in spn.topological_sort():
        new_id = allocator.next_id()
        old_to_new[old_id] = new_id
        new_circuit._add_node(new_id, deepcopy(spn.get_node_data(old_id)))

    for old_id in spn.topological_sort():
        for child_id, edge_data in spn._adj[old_id].items():
            new_circuit._add_edge(old_to_new[old_id], old_to_new[child_id], deepcopy(edge_data))

    return new_circuit


def _aggregate(
    circuit: SymbolicArithmeticCircuit, marg_cols: set[int]
) -> SymbolicArithmeticCircuit:
    """Replace leaves for marginalized variables with constant-1 nodes.

    This is the AGG operation (Algorithm 1 in Wang et al. 2025).
    The circuit structure is preserved; only leaf functions change.
    """
    for node_id in list(circuit._nodes):
        node = circuit.get_node_data(node_id)
        if isinstance(node, LeafNode) and hasattr(node, "var") and node.var in marg_cols:
            constant = ConstantLeafNode(
                var=node.var,
                unit_count=node.unit_count,
                md_set=node.md_set,
            )
            # Preserve unit_supports from the original leaf
            constant.unit_supports = node.unit_supports
            circuit._nodes[node_id] = constant

    return circuit


def _elementwise_map(circuit: SymbolicArithmeticCircuit, power: int) -> SymbolicArithmeticCircuit:
    """Apply elementwise mapping τ_power to the circuit.

    For power=-1: negate all sum-node weights and wrap leaves in InverseLeafNode.
    Per Theorem 4 (Wang et al. 2025), this distributes over sums when
    the circuit is deterministic (which we enforce via marginal determinism).
    """
    for node_id in list(circuit._nodes):
        node = circuit.get_node_data(node_id)

        if isinstance(node, LeafNode):
            if isinstance(node, ConstantLeafNode):
                # τ(1) = 1 for any power — constant leaves are unaffected
                continue
            inverse = InverseLeafNode(base_leaf=node, power=power)
            circuit._nodes[node_id] = inverse

        elif isinstance(node, SumNode):
            # Negate edge weights (sum-node weights are stored on edges)
            # In the SPN, edge data from sum→product is the weight.
            # However, in this codebase weights are learned during compilation,
            # not stored on edges at the symbolic level.  The POW operation
            # is tracked structurally and applied during folding/compilation.
            pass

    # Mark the circuit as power-mapped so the compiler can handle it
    circuit._power_map = power
    return circuit


# ---------------------------------------------------------------------------
# Circuit products
# ---------------------------------------------------------------------------


def _get_root(circuit: SymbolicArithmeticCircuit) -> int:
    """Get the root node ID of a circuit."""
    roots = circuit.get_roots()
    assert len(roots) == 1, f"Expected 1 root, got {len(roots)}"
    return roots[0]


def _get_children(circuit: SymbolicArithmeticCircuit, node_id: int) -> list[int]:
    """Get ordered children of a node."""
    return list(circuit._adj[node_id].keys())


def _product_scmp(
    circuit_a: SymbolicArithmeticCircuit,
    circuit_b: SymbolicArithmeticCircuit,
    allocator: NodeAllocator,
) -> SymbolicArithmeticCircuit:
    """Support-compatible product (PROD-SCMP, Algorithm 3).

    Pairs sum-node children 1-to-1 (Hadamard).  h stays h.
    Requires the circuits to be support-compatible (guaranteed when
    both are derived from the same base SPN with marginal determinism).
    """
    result = SymbolicArithmeticCircuit()
    root_a = _get_root(circuit_a)
    root_b = _get_root(circuit_b)

    node_map: dict[tuple[int, int], int] = {}
    _product_scmp_recursive(circuit_a, circuit_b, root_a, root_b, result, allocator, node_map)
    return result


def _product_scmp_recursive(
    ca: SymbolicArithmeticCircuit,
    cb: SymbolicArithmeticCircuit,
    id_a: int,
    id_b: int,
    result: SymbolicArithmeticCircuit,
    allocator: NodeAllocator,
    node_map: dict[tuple[int, int], int],
) -> int:
    """Recursive PROD-SCMP implementation."""
    key = (id_a, id_b)
    if key in node_map:
        return node_map[key]

    node_a = ca.get_node_data(id_a)
    node_b = cb.get_node_data(id_b)

    children_a = _get_children(ca, id_a)
    children_b = _get_children(cb, id_b)

    # Leaf × Leaf → ProductLeafNode
    if isinstance(node_a, LeafNode) and isinstance(node_b, LeafNode):
        new_node = ProductLeafNode(leaf_a=node_a, leaf_b=node_b)
        new_id = allocator.next_id()
        result._add_node(new_id, new_node)
        node_map[key] = new_id
        return new_id

    # Product × Product → recurse into children
    if isinstance(node_a, (KroneckerProductNode, HadamardProductNode, UnaryProductNode)):
        assert len(children_a) == len(children_b), (
            f"Product node children count mismatch: {len(children_a)} vs {len(children_b)}"
        )
        child_ids = []
        for ca_child, cb_child in zip(children_a, children_b):
            child_id = _product_scmp_recursive(
                ca, cb, ca_child, cb_child, result, allocator, node_map
            )
            child_ids.append(child_id)

        # Derive unit_count from children (SCMP preserves h)
        child_units = [result.get_node_data(cid).unit_count for cid in child_ids]
        if isinstance(node_a, KroneckerProductNode):
            new_unit_count = child_units[0] * child_units[1]
        elif isinstance(node_a, HadamardProductNode):
            new_unit_count = child_units[0]
        else:
            new_unit_count = child_units[0]

        new_node = type(node_a)(
            support=node_a.support,
            unit_count=new_unit_count,
            md_set=node_a.md_set,
        )
        new_id = allocator.next_id()
        result._add_node(new_id, new_node)
        for child_id in child_ids:
            result._add_edge(new_id, child_id)

        node_map[key] = new_id
        return new_id

    # Sum × Sum → pair children 1-to-1 (Hadamard), weights multiply
    if isinstance(node_a, SumNode) and isinstance(node_b, SumNode):
        assert len(children_a) == len(children_b), (
            f"Sum node children count mismatch: {len(children_a)} vs {len(children_b)}"
        )
        # The product of two sum nodes with support compatibility is a new sum
        # with the same number of children, each being the product of paired children.
        new_node = type(node_a)(
            support=node_a.support,
            unit_count=node_a.unit_count,
            md_set=node_a.md_set,
        )
        new_id = allocator.next_id()
        result._add_node(new_id, new_node)

        for ca_child, cb_child in zip(children_a, children_b):
            child_id = _product_scmp_recursive(
                ca, cb, ca_child, cb_child, result, allocator, node_map
            )
            result._add_edge(new_id, child_id)

        node_map[key] = new_id
        return new_id

    raise ValueError(f"PROD-SCMP: incompatible node types {type(node_a)} × {type(node_b)}")


def _product_cmp(
    circuit_a: SymbolicArithmeticCircuit,
    circuit_b: SymbolicArithmeticCircuit,
    allocator: NodeAllocator,
) -> SymbolicArithmeticCircuit:
    """Compatible product (PROD-CMP, Algorithm 2).

    Cross-products sum-node children (Kronecker).  h → h².
    The resulting sum blocks have Kronecker-product weight matrices,
    which compile to Monarch/Tucker layers.
    """
    result = SymbolicArithmeticCircuit()
    root_a = _get_root(circuit_a)
    root_b = _get_root(circuit_b)

    node_map: dict[tuple[int, int], int] = {}
    _product_cmp_recursive(circuit_a, circuit_b, root_a, root_b, result, allocator, node_map)
    return result


def _product_cmp_recursive(
    ca: SymbolicArithmeticCircuit,
    cb: SymbolicArithmeticCircuit,
    id_a: int,
    id_b: int,
    result: SymbolicArithmeticCircuit,
    allocator: NodeAllocator,
    node_map: dict[tuple[int, int], int],
) -> int:
    """Recursive PROD-CMP implementation."""
    key = (id_a, id_b)
    if key in node_map:
        return node_map[key]

    node_a = ca.get_node_data(id_a)
    node_b = cb.get_node_data(id_b)

    children_a = _get_children(ca, id_a)
    children_b = _get_children(cb, id_b)

    scope_a = node_a.scope if hasattr(node_a, "scope") else set()
    scope_b = node_b.scope if hasattr(node_b, "scope") else set()

    # Disjoint scope → simple product node
    if not scope_a.intersection(scope_b):
        new_node = KroneckerProductNode(
            support=node_a.support.merge(node_b.support)
            if hasattr(node_a.support, "merge")
            else node_a.support,
            unit_count=node_a.unit_count * node_b.unit_count,
            md_set=node_a.md_set,
        )
        new_id = allocator.next_id()
        result._add_node(new_id, new_node)

        # Copy subtrees from both circuits into result
        id_a_copy = _copy_subtree_into(ca, id_a, result, allocator)
        id_b_copy = _copy_subtree_into(cb, id_b, result, allocator)
        result._add_edge(new_id, id_a_copy)
        result._add_edge(new_id, id_b_copy)

        node_map[key] = new_id
        return new_id

    # Leaf × Leaf → ProductLeafNode
    if isinstance(node_a, LeafNode) and isinstance(node_b, LeafNode):
        new_node = ProductLeafNode(leaf_a=node_a, leaf_b=node_b)
        new_id = allocator.next_id()
        result._add_node(new_id, new_node)
        node_map[key] = new_id
        return new_id

    # Product × Product → recurse into children, then compute unit_count from children
    if isinstance(node_a, (KroneckerProductNode, HadamardProductNode, UnaryProductNode)):
        assert len(children_a) == len(children_b)

        child_ids = []
        for ca_child, cb_child in zip(children_a, children_b):
            child_id = _product_cmp_recursive(
                ca, cb, ca_child, cb_child, result, allocator, node_map
            )
            child_ids.append(child_id)

        # Compute unit_count from the children's actual unit_counts
        child_units = [result.get_node_data(cid).unit_count for cid in child_ids]
        if isinstance(node_a, KroneckerProductNode):
            new_unit_count = child_units[0] * child_units[1]
        elif isinstance(node_a, HadamardProductNode):
            new_unit_count = child_units[0]  # Hadamard: same h as children
        else:
            new_unit_count = child_units[0]  # Unary

        new_node = type(node_a)(
            support=node_a.support,
            unit_count=new_unit_count,
            md_set=node_a.md_set,
        )
        new_id = allocator.next_id()
        result._add_node(new_id, new_node)
        for child_id in child_ids:
            result._add_edge(new_id, child_id)

        node_map[key] = new_id
        return new_id

    # Sum × Sum → cross-product at unit level (Kronecker), h → h²
    if isinstance(node_a, SumNode) and isinstance(node_b, SumNode):
        # Recurse into children first to determine their unit counts
        child_ids = []
        for ca_child in children_a:
            for cb_child in children_b:
                child_id = _product_cmp_recursive(
                    ca, cb, ca_child, cb_child, result, allocator, node_map
                )
                child_ids.append(child_id)

        # The product creates a new UNIVERSAL sum (dense) with h_a * h_b units.
        # The weight matrix is the Kronecker product W_a ⊗ W_b → Monarch/Tucker.
        new_node = UniversalSumNode(
            support=node_a.support,
            unit_count=node_a.unit_count * node_b.unit_count,
            md_set=node_a.md_set,
        )
        new_id = allocator.next_id()
        result._add_node(new_id, new_node)
        for child_id in child_ids:
            result._add_edge(new_id, child_id)

        node_map[key] = new_id
        return new_id

    raise ValueError(f"PROD-CMP: incompatible node types {type(node_a)} × {type(node_b)}")


def _copy_subtree_into(
    source: SymbolicArithmeticCircuit,
    node_id: int,
    target: SymbolicArithmeticCircuit,
    allocator: NodeAllocator,
    visited: dict[int, int] | None = None,
) -> int:
    """Copy a subtree from source into target with fresh IDs."""
    if visited is None:
        visited = {}
    if node_id in visited:
        return visited[node_id]

    new_id = allocator.next_id()
    visited[node_id] = new_id
    target._add_node(new_id, deepcopy(source.get_node_data(node_id)))

    for child_id, edge_data in source._adj[node_id].items():
        new_child = _copy_subtree_into(source, child_id, target, allocator, visited)
        target._add_edge(new_id, new_child, deepcopy(edge_data))

    return new_id
