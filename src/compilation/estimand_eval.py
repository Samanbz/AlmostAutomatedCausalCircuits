import torch

from src.compilation.tensorized_circuit import (
    CompiledCircuit,
    TensorizedCPT,
    TensorizedFusedLayer,
    TensorizedGaussianInput,
    TensorizedSparseHadamard,
    TensorizedSparseKronecker,
    TensorizedTucker,
)
from src.symbolic.id_ast import (
    DetProdNode,
    EstimandAST,
    InstNode,
    MargNode,
    PNode,
    PowNode,
    ProdNode,
    get_vars,
)


LOG_ZERO_THRESH = -1e10
OOB_THRESH = -1e10


def _ast_cols(ast: EstimandAST, node_id: str, var_map: dict) -> set[int]:
    """Return the set of column indices for the free variables of an AST subtree."""
    return {var_map[v] for v in get_vars(ast, node_id) if v in var_map}


def _is_scope_active(
    ast: EstimandAST, node_id: str, layer_scope: set | None, var_map: dict | None
) -> bool:
    """Check if an AST subtree's free variables intersect the layer's scope."""
    if layer_scope is None or var_map is None:
        return True
    return bool(_ast_cols(ast, node_id, var_map).intersection(layer_scope))


def get_dense_log_w(layer: TensorizedFusedLayer) -> torch.Tensor:
    """
    Extracts and homogenizes the base log_w into a dense [N, h_out, h_left * h_right] format.
    This makes algebraic Kronecker and Hadamard combinations universally compatible.
    """
    N = layer.num_nodes
    if isinstance(layer, TensorizedTucker):
        return layer.log_w.clone()
    elif isinstance(layer, TensorizedSparseKronecker):
        # Block-diagonal: log_w is [N, h_out, h_in] where h_in = h_child / h_out.
        # Unpack into dense [N, h_out, h_left * h_right] with -inf for off-block entries.
        h_child = layer.h_left * layer.h_right
        dense_w = torch.full((N, layer.h_out, h_child), -1e20, device=layer.log_w.device)
        h_in = layer.h_in
        for k in range(layer.h_out):
            dense_w[:, k, k * h_in : (k + 1) * h_in] = layer.log_w[:, k, :]
        return dense_w
    elif isinstance(layer, TensorizedSparseHadamard):
        # Hadamard does L + R on the same dimension. In dense Kronecker space (h_child * h_child),
        # this corresponds to elements strictly on the diagonal.
        h_child = layer.h_child
        dense_w = torch.full((N, layer.h_out, h_child * h_child), -1e20, device=layer.log_w.device)
        idx = torch.arange(h_child, device=layer.log_w.device)
        diag_idx = idx * h_child + idx

        log_w_flat = layer.log_w.view(N, h_child)
        for c in range(h_child):
            j = c // layer.h_in
            dense_w[:, j, diag_idx[c]] = log_w_flat[:, c]
        return dense_w
    elif isinstance(layer, TensorizedCPT):
        # CPT is a dense Hadamard layer: weight is [N, h_out, h_child] but in Kronecker
        # space (h_child × h_child) each output unit only uses diagonal entries L[k]+R[k].
        h_child = layer.h_child
        dense_w = torch.full(
            (N, layer.h_out, h_child * h_child),
            -1e20,
            device=layer.log_w.device,
            dtype=layer.log_w.dtype,
        )
        diag_idx = torch.arange(h_child, device=layer.log_w.device) * (h_child + 1)
        for k in range(layer.h_out):
            dense_w[:, k, diag_idx] = layer.log_w[:, k, :]
        return dense_w
    return layer.log_w.clone()


def get_algebraic_weights(
    layer: TensorizedFusedLayer,
    ast: EstimandAST,
    node_id: str,
    layer_scope: set | None = None,
    var_map: dict | None = None,
):
    """
    Recursively evaluates the AST to build the algebraic weights for a Fused Layer.
    layer_scope: the set of variable COLUMNS in this layer's scope (for scope-aware POW).
    var_map: maps variable names to column indices.
    Returns: (log_w, h_out, h_L, h_R)
    """
    node = ast.get_node_data(node_id)

    if isinstance(node, PNode):
        dense_w = get_dense_log_w(layer)
        if isinstance(layer, (TensorizedTucker, TensorizedSparseKronecker)):
            h_L, h_R = layer.h_left, layer.h_right
        elif isinstance(layer, (TensorizedSparseHadamard, TensorizedCPT)):
            h_L, h_R = layer.h_child, layer.h_child
        return dense_w, layer.h_out, h_L, h_R

    elif isinstance(node, PowNode):
        child_id = ast.get_children(node_id)[0]
        w, h_out, h_L, h_R = get_algebraic_weights(layer, ast, child_id, layer_scope, var_map)

        # Scope check: POW only modifies weights if the child's free variables
        # intersect this layer's scope.  When they don't, the entire POW subtree
        # evaluates to a scalar at this layer (all its vars are marginalized away),
        # so the elementwise mapping is a no-op on the weights.
        if not _is_scope_active(ast, child_id, layer_scope, var_map):
            return w, h_out, h_L, h_R

        structural_zero = w < LOG_ZERO_THRESH
        result = node.power * w
        result = result.masked_fill(structural_zero, -1e20)
        return result, h_out, h_L, h_R

    elif isinstance(node, (MargNode, InstNode)):
        return get_algebraic_weights(layer, ast, ast.get_children(node_id)[0], layer_scope, var_map)

    elif isinstance(node, DetProdNode):
        children = ast.get_children(node_id)
        active_0 = _is_scope_active(ast, children[0], layer_scope, var_map)
        active_1 = _is_scope_active(ast, children[1], layer_scope, var_map)

        # If a child's scope doesn't intersect this layer, it evaluates to a
        # scalar here (weight contribution = 0 in log domain = multiplicative identity).
        if not active_0 and not active_1:
            return get_algebraic_weights(layer, ast, children[0], layer_scope, var_map)
        if not active_1:
            return get_algebraic_weights(layer, ast, children[0], layer_scope, var_map)
        if not active_0:
            return get_algebraic_weights(layer, ast, children[1], layer_scope, var_map)

        w1, ho1, hl1, hr1 = get_algebraic_weights(layer, ast, children[0], layer_scope, var_map)
        w2, ho2, hl2, hr2 = get_algebraic_weights(layer, ast, children[1], layer_scope, var_map)

        # Element-wise addition, but preserve structural zeros: if either operand
        # is -inf (structural zero in the sparse→dense expansion), the result stays -inf.
        structural_zero = (w1 < LOG_ZERO_THRESH) | (w2 < LOG_ZERO_THRESH)
        w_sum = w1 + w2
        w_sum = w_sum.masked_fill(structural_zero, -1e20)
        return w_sum, ho1, hl1, hr1

    elif isinstance(node, ProdNode):
        children = ast.get_children(node_id)
        # ProdNode always does full Kronecker expansion at both leaf and weight
        # levels to keep them consistent.  Scope-based skipping is only safe for
        # DetProdNode / PowNode (Hadamard / elementwise) where h doesn't change.
        w1, ho1, hl1, hr1 = get_algebraic_weights(layer, ast, children[0], layer_scope, var_map)
        w2, ho2, hl2, hr2 = get_algebraic_weights(layer, ast, children[1], layer_scope, var_map)

        N = w1.shape[0]
        w1 = w1.view(N, ho1, hl1, hr1)
        w2 = w2.view(N, ho2, hl2, hr2)

        w1_exp = w1.view(N, ho1, 1, hl1, 1, hr1, 1)
        w2_exp = w2.view(N, 1, ho2, 1, hl2, 1, hr2)
        w_new = w1_exp + w2_exp

        # Preserve structural zeros from Kronecker expansion
        sz1 = w1_exp < LOG_ZERO_THRESH
        sz2 = w2_exp < LOG_ZERO_THRESH
        w_new = w_new.masked_fill(sz1 | sz2, -1e20)

        ho_new = ho1 * ho2
        hl_new = hl1 * hl2
        hr_new = hr1 * hr2

        return w_new.view(N, ho_new, (hl_new) * (hr_new)), ho_new, hl_new, hr_new


def get_algebraic_leaves(
    layer: TensorizedGaussianInput,
    ast: EstimandAST,
    node_id: str,
    data: torch.Tensor,
    var_map: dict,
    leaf_scope: set | None = None,
) -> torch.Tensor:
    """
    Recursively evaluates the AST to build the algebraic leaves for an Input Layer.
    leaf_scope: set of variable columns present in this input layer (for scope-aware Prod).
    Returns: tensor of shape [N, h_alg, B]
    """
    node = ast.get_node_data(node_id)

    if isinstance(node, PNode):
        return layer.forward(data)

    elif isinstance(node, PowNode):
        child_id = ast.get_children(node_id)[0]
        leaves = get_algebraic_leaves(layer, ast, child_id, data, var_map, leaf_scope)

        # Scope check: same logic as weights — POW is a no-op when the child's
        # free variables don't intersect the leaf scope.
        if leaf_scope is not None and not _is_scope_active(ast, child_id, leaf_scope, var_map):
            return leaves

        oob_mask = leaves < OOB_THRESH
        result = node.power * leaves
        result = result.masked_fill(oob_mask, -1e20)
        return result

    elif isinstance(node, DetProdNode):
        children = ast.get_children(node_id)
        active_0 = leaf_scope is None or _is_scope_active(ast, children[0], leaf_scope, var_map)
        active_1 = leaf_scope is None or _is_scope_active(ast, children[1], leaf_scope, var_map)

        if not active_0 and not active_1:
            return get_algebraic_leaves(layer, ast, children[0], data, var_map, leaf_scope)
        if not active_1:
            return get_algebraic_leaves(layer, ast, children[0], data, var_map, leaf_scope)
        if not active_0:
            return get_algebraic_leaves(layer, ast, children[1], data, var_map, leaf_scope)

        l_leaves = get_algebraic_leaves(layer, ast, children[0], data, var_map, leaf_scope)
        r_leaves = get_algebraic_leaves(layer, ast, children[1], data, var_map, leaf_scope)

        oob = (l_leaves < OOB_THRESH) | (r_leaves < OOB_THRESH)
        result = l_leaves + r_leaves
        result = result.masked_fill(oob, -1e20)
        return result

    elif isinstance(node, ProdNode):
        children = ast.get_children(node_id)
        # ProdNode always does full Kronecker expansion (see matching comment in
        # get_algebraic_weights).
        l_leaves = get_algebraic_leaves(layer, ast, children[0], data, var_map, leaf_scope)
        r_leaves = get_algebraic_leaves(layer, ast, children[1], data, var_map, leaf_scope)

        N, h1, B = l_leaves.shape
        _, h2, _ = r_leaves.shape
        l_exp = l_leaves.view(N, h1, 1, B)
        r_exp = r_leaves.view(N, 1, h2, B)
        result = (l_exp + r_exp).view(N, h1 * h2, B)
        oob = (l_exp < OOB_THRESH) | (r_exp < OOB_THRESH)
        result = result.masked_fill(oob.view(N, h1 * h2, B), -1e20)
        return result

    elif isinstance(node, MargNode):
        child_leaves = get_algebraic_leaves(
            layer, ast, ast.get_children(node_id)[0], data, var_map, leaf_scope
        )

        mask = torch.zeros((layer.num_nodes, 1, 1), dtype=torch.bool, device=data.device)
        marg_indices = [var_map[v] for v in node.marginalize_vars if v in var_map]
        for i, var_idx in enumerate(layer.scopes):
            if var_idx in marg_indices:
                mask[i, 0, 0] = True
        return child_leaves.masked_fill(mask, 0.0)  # log(1) = 0 for integrated vars

    elif isinstance(node, InstNode):
        mod_data = data.clone()
        for var, val in node.variables.items():
            if var in var_map:
                mod_data[:, var_map[var]] = val
        return get_algebraic_leaves(
            layer, ast, ast.get_children(node_id)[0], mod_data, var_map, leaf_scope
        )


@torch.no_grad()
def eval_estimand(
    circuit: CompiledCircuit, ast: EstimandAST, data: torch.Tensor, var_map: dict[str, int]
) -> torch.Tensor:
    """
    Evaluates the causal estimand AST using dynamic block evaluation.
    This avoids re-indexing the static flat buffer while perfectly executing the algebraic circuit.
    """
    data = circuit._pad(data)
    B = data.shape[0]
    ast_root_id = ast.get_root()

    # Dictionary mapping the base circuit's flat offset to the algebraic tensor block [h_alg, B]
    offset_to_alg_tensor = {}

    # 1. Process Input Layers — each input node may need a different algebraic h
    #    depending on which AST ProdNode children are active at that node's scope.
    for il in circuit.input_layers:
        N_nodes = il.num_nodes
        base_scatter = il.scatter_idx.view(N_nodes, -1)[:, 0].tolist()

        # Group nodes by scope to batch nodes that share the same scope.
        scope_groups: dict[frozenset, list[int]] = {}
        for i in range(N_nodes):
            key = frozenset({il.scopes[i]})
            scope_groups.setdefault(key, []).append(i)

        for scope_key, node_indices in scope_groups.items():
            leaf_scope = set(scope_key)
            # Build a sub-layer view for this group of nodes
            idx = torch.tensor(node_indices, dtype=torch.long)
            sub_means = il.means[idx]
            sub_stds = il.stds[idx]
            sub_lows = il.lows[idx]
            sub_highs = il.highs[idx]
            sub_scopes = [il.scopes[i] for i in node_indices]

            sub_layer = TensorizedGaussianInput(
                sub_means, sub_stds, sub_lows, sub_highs, sub_scopes
            )
            # Suppress the jitter that the constructor adds
            sub_layer.means.data.copy_(il.means[idx])
            sub_layer.stds.data.copy_(il.stds[idx])

            sub_leaves = get_algebraic_leaves(
                sub_layer, ast, ast_root_id, data, var_map, leaf_scope
            )

            for j, i in enumerate(node_indices):
                offset_to_alg_tensor[base_scatter[i]] = sub_leaves[j]

    # 2. Process Fused Sum Layers
    node_offset = sum(il.num_nodes for il in circuit.input_layers)
    for layer in circuit.layers:
        # Use the scope of the first node in this fused layer (all nodes in a
        # fused layer share the same scope by construction).
        layer_scope = circuit.node_scopes[node_offset] if circuit.node_scopes else None
        alg_w, _, h_L, h_R = get_algebraic_weights(layer, ast, ast_root_id, layer_scope, var_map)
        N_nodes = layer.num_nodes

        # Extract the base offset pointing to the children
        base_lo = layer.gather_L.view(N_nodes, -1)[:, 0].tolist()
        base_ro = layer.gather_R.view(N_nodes, -1)[:, 0].tolist()

        # Retrieve the dynamically sized algebraic tensors from children
        L_list = [offset_to_alg_tensor[base_lo[i]] for i in range(N_nodes)]
        R_list = [offset_to_alg_tensor[base_ro[i]] for i in range(N_nodes)]

        L_tensor = torch.stack(L_list, dim=0)  # [N, h_L_actual, B]
        R_tensor = torch.stack(R_list, dim=0)  # [N, h_R_actual, B]

        # Dense algebraic contraction
        S = (L_tensor.unsqueeze(2) + R_tensor.unsqueeze(1)).view(N_nodes, h_L * h_R, B)
        m = S.max(dim=1, keepdim=True)[0]
        exp_S = torch.exp(S - m)
        exp_W = torch.exp(alg_w)

        out = torch.bmm(exp_W, exp_S)
        result = torch.log(out.clamp(min=1e-20)) + m  # [N, h_out, B]

        # Save output blocks
        base_scatter = layer.scatter_idx.view(N_nodes, -1)[:, 0].tolist()
        for i in range(N_nodes):
            offset_to_alg_tensor[base_scatter[i]] = result[i]

        node_offset += N_nodes

    # The final root node is the last node registered in the circuit
    final_offset = circuit._node_flat_offsets[-1]
    root_val = offset_to_alg_tensor[final_offset]  # [h_alg, B]
    # Root should have h_out=1; squeeze to [B] for consistency with circuit.forward()
    if root_val.shape[0] == 1:
        return root_val.squeeze(0)
    return root_val
