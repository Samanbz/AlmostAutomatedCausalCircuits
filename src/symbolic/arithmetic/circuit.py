import contextlib
from typing import Any, Dict, List, Optional, Type

import torch
from wandb.util import np

from src.utils import BitSet, NodeAllocator

from ..base import DirectedAcyclicGraph
from ..vtree import VTree
from .nodes import (
    ArithmeticNode,
    ConstantLayer,
    LeafLayer,
    SumLayer,
)


def _prepare_log_for_print(tensor: torch.Tensor, threshold: float = -25.0) -> torch.Tensor:
    """Replace very small log-probabilities with -inf for cleaner printing."""
    t = tensor.detach().clone()
    t[t < threshold] = float("-inf")
    return t


class SymbolicArithmeticCircuit(DirectedAcyclicGraph[int, ArithmeticNode, Any]):
    """
    A DAG representing a symbolic arithmetic circuit.
    Nodes are identified by integers and contain Node objects (SumLayer, etc.).
    Edges can optionally contain data (e.g., weights for sum nodes) and represent descendence and
    not computational flow.
    """

    def __init__(self, node_allocator: NodeAllocator = None, vtree: VTree = None):
        super().__init__(node_allocator=node_allocator)
        self.vtree = vtree
        self.vtree_to_sum: Dict[int, int] = {}
        self.sum_to_vtree: Dict[int, int] = {}

    def to(self, device: torch.device) -> "SymbolicArithmeticCircuit":
        """Moves all nodes and parameters in the circuit to the given device."""
        for node in self._nodes.values():
            if hasattr(node, "to") and callable(node.to):
                node.to(device)
        return self

    def get_nodes_with_scope(self, scope: BitSet, node_type: Type[ArithmeticNode]) -> List[int]:
        """Returns a list of node IDs that have the given scope and are of the specified type."""
        return [
            node_id
            for node_id, node in self._nodes.items()
            if isinstance(node, node_type) and node.scope == scope
        ]  # TODO: improve by using a top-down pruning search

    def add_edge(self, source: int, target: int, data: Any = None) -> None:
        """Adds a directed edge from source to target with optional data."""
        super().add_edge(source, target, data)

    def is_sum_node(self, node_id: int) -> bool:
        return isinstance(self.get_node_data(node_id), SumLayer) or isinstance(
            self.get_node_data(node_id), LeafLayer
        )

    def num_parameters(self, verbose: bool = False) -> int:
        """Total number of non-zero weights referenced by the circuit.

        Sum layers contribute their ``Weights.num_parameters()`` (sparse layers
        count only mask-nonzero entries; ``ProductWeights`` counts the non-zero
        entries of its logical broadcast, derived from the factors'
        zero-patterns); leaf layers contribute their parameter tensors.
        Weight objects and tensors are deduplicated by identity, so query
        circuits that share parameter tensors with the base circuit (or with
        each other) never double-count them.
        """
        from .weights import Weights

        total = 0
        seen_weights: set = set()
        seen_tensors: set = set()

        def count_weights(w) -> int:
            if id(w) in seen_weights:
                return 0
            seen_weights.add(id(w))
            return w.num_parameters()

        def count_tensors(node) -> int:
            """Count a leaf node's own parameters (weights + tensors), recursing
            into wrapped leaves (MixtureLeafLayer.base_dist, ProductLeafLayer.leaf_a/b)."""
            count = 0
            w = getattr(node, "log_weights", None)
            if isinstance(w, Weights):
                count += count_weights(w)
            elif isinstance(w, torch.Tensor) and id(w) not in seen_tensors:
                seen_tensors.add(id(w))
                count += int(w.numel())
            for attr in ("means", "_raw_stddevs", "_log_heights", "_b1", "logits"):
                t = getattr(node, attr, None)
                if isinstance(t, torch.Tensor) and id(t) not in seen_tensors:
                    seen_tensors.add(id(t))
                    count += int(t.numel())
            for child_name in ("base_dist", "leaf_a", "leaf_b"):
                child = getattr(node, child_name, None)
                if isinstance(child, LeafLayer):
                    count += count_tensors(child)
            return count

        for node_id in self.topological_sort():
            node = self.get_node_data(node_id)
            if isinstance(node, LeafLayer):
                count = count_tensors(node)
            else:
                weights = getattr(node, "log_weights", None)
                if isinstance(weights, Weights):
                    count = count_weights(weights)
                elif isinstance(weights, torch.Tensor):
                    if id(weights) not in seen_tensors:
                        seen_tensors.add(id(weights))
                        count = int(weights.numel())
                    else:
                        count = 0
                else:
                    count = 0
            total += count
            if verbose and count:
                print(f"  node {node_id}: {type(node).__name__} scope={node.scope} params={count}")
        return total

    def is_product_node(self, node_id: int) -> bool:
        return False

    def set_edge_data(self, source: int, target: int, data: Any) -> None:
        """Sets the data for an existing edge from source to target."""
        if source not in self._adj or target not in self._adj[source]:
            raise KeyError(f"Edge from '{source}' to '{target}' does not exist.")
        self._adj[source][target] = data
        self._rev_adj[target][source] = data

    def zero_grad(self):
        """Zero all gradients in the circuit nodes."""
        seen_tensors = set()

        def _zero(node):
            if (
                hasattr(node, "log_weights")
                and isinstance(node.log_weights, torch.Tensor)
                and node.log_weights.grad is not None
            ):
                if id(node.log_weights) not in seen_tensors:
                    seen_tensors.add(id(node.log_weights))
                    node.log_weights.grad.zero_()
            if (
                hasattr(node, "means")
                and isinstance(node.means, torch.Tensor)
                and node.means.grad is not None
            ):
                if id(node.means) not in seen_tensors:
                    seen_tensors.add(id(node.means))
                    node.means.grad.zero_()
            if (
                hasattr(node, "_raw_stddevs")
                and isinstance(node._raw_stddevs, torch.Tensor)
                and node._raw_stddevs.grad is not None
            ):
                if id(node._raw_stddevs) not in seen_tensors:
                    seen_tensors.add(id(node._raw_stddevs))
                    node._raw_stddevs.grad.zero_()
            if hasattr(node, "base_dist"):
                _zero(node.base_dist)
            if hasattr(node, "leaf_a"):
                _zero(node.leaf_a)
            if hasattr(node, "leaf_b"):
                _zero(node.leaf_b)

        for node in self._nodes.values():
            _zero(node)

    def leaf_parameters(self):
        """Yields optimizable parameters from leaf nodes.

        New leaf classes can simply store tensors with ``requires_grad=True`` as
        attributes; they will be picked up automatically. Cached outputs and
        structural objects are excluded.
        """
        seen_params = set()
        denylist = {"_last_forward_output", "support", "node_supports"}

        def _yield_params(node):
            for attr_name, attr in node.__dict__.items():
                if attr_name in denylist:
                    continue
                if isinstance(attr, torch.Tensor) and attr.requires_grad:
                    if id(attr) not in seen_params:
                        seen_params.add(id(attr))
                        yield attr
            if hasattr(node, "base_dist"):
                yield from _yield_params(node.base_dist)
            if hasattr(node, "leaf_a"):
                yield from _yield_params(node.leaf_a)
            if hasattr(node, "leaf_b"):
                yield from _yield_params(node.leaf_b)

        for node in self._nodes.values():
            if isinstance(node, LeafLayer):
                yield from _yield_params(node)

    def get_node_config(
        self, show_node_id: bool = False, show_node_supports: bool = False
    ) -> Dict[type, Dict[str, Any]]:
        self._show_node_supports = show_node_supports
        config = super().get_node_config(show_node_id=show_node_id)
        return config

    def _get_base_node_config(self) -> Dict[type, Dict[str, Any]]:
        config = super()._get_base_node_config()
        show_node_supports = getattr(self, "_show_node_supports", False)

        def leaf_label(node: LeafLayer) -> str:
            label = type(node).__name__ if type(node).__name__ == "MixtureLeafLayer" else "Leaf"
            if hasattr(node, "scope"):
                label += f"\n{sorted(node.scope)}"
            if show_node_supports:
                if getattr(node, "node_supports", None) is not None and len(node.node_supports) > 1:
                    supports_str = "\n".join(str(s) for s in node.node_supports)
                    label += f"\n{supports_str}"
                elif hasattr(node, "var_support"):
                    # node: Distribution
                    label += f"\n{node.var_support}"
            return label

        def node_label(node: ArithmeticNode) -> str:
            label = ""
            if isinstance(node, SumLayer):
                label = "+"
                if not getattr(node, "sparse", True):
                    label += " (Dense)"
            elif isinstance(node, LeafLayer):
                label = leaf_label(node)

            if hasattr(node, "num_groups") and node.num_groups > 1:
                label += f"\n(#groups={node.num_groups})"

            if hasattr(node, "num_nodes") and node.num_nodes > 1:
                label += f"\n(#nodes={node.num_nodes})"

            if show_node_supports and getattr(node, "node_supports", None) is not None:
                supports_str = "\n".join(str(s) for s in node.node_supports)
                label += f"\n{supports_str}"

            return label

        config.update(
            {
                SumLayer: {"color": "#ccffcc", "label": node_label, "shape": "box"},
                ConstantLayer: {"color": "#ffcccc", "label": node_label, "shape": "ellipse"},
                LeafLayer: {"color": "#ffcccc", "label": node_label, "shape": "ellipse"},
            }
        )
        return config

    def print_weights(self, log_domain: bool = True):
        """Prints the weights of the circuit for each vtree node's corresponding sum layer."""
        if not self.vtree:
            print("No vtree available to iterate over.")
            return

        for vtree_id in self.vtree.topological_sort(reverse=True):
            if vtree_id not in self.vtree_to_sum:
                continue
            sum_id = self.vtree_to_sum[vtree_id]
            node = self.get_node_data(sum_id)

            if not hasattr(node, "log_weights") or node.log_weights is None:
                continue

            w_obj = node.log_weights
            wt = (
                w_obj.log_weights.detach().cpu()
                if hasattr(w_obj, "log_weights")
                else w_obj.detach().cpu()
            )
            G, H, G_L, H_L, G_R, H_R = wt.shape
            wt_to_print = wt if log_domain else torch.exp(wt)
            if log_domain:
                wt_to_print = _prepare_log_for_print(wt_to_print)
            wt_rounded = torch.round(wt_to_print * 1000) / 1000.0
            wt_rounded = wt_rounded + 0.0
            wt_rounded = wt_rounded.reshape(G, H, G_L * H_L, G_R * H_R)

            print(
                f"DEBUG: VTree Node {vtree_id:2d} -> Sum Node {sum_id:2d} ({type(node).__name__}) (scope: {list(node.scope) if node.scope and not node.scope.is_empty else 'N/A'}) G={G}, H={H}, G_L={G_L}, H_L={H_L}, G_R={G_R}, H_R={H_R}"
            )
            for g in range(G):
                for h in range(H):
                    print(
                        f"DEBUG: VTree Node {vtree_id:2d} -> Sum Node {sum_id:2d} ({type(node).__name__}) (scope: {list(node.scope) if node.scope and not node.scope.is_empty else 'N/A'}) weights for group {g}, node {h}:\n{wt_rounded[g, h]}"
                    )


def eval_circuit(
    ac: SymbolicArithmeticCircuit,
    data: torch.Tensor,
    verbose: bool = False,
    show_leaves_only: bool = False,
    show_weights: bool = False,
    log_domain: bool = True,
    debug_check_finite: bool = False,
    keep_intermediates: bool = True,
    eval_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Evaluates the circuit on the given data.

    Args:
        debug_check_finite: If True, raise a ``ValueError`` when the root output
            is non-finite for any row.  This makes structural disconnects
            explicit instead of silently producing huge NLLs.
        keep_intermediates: If False, free each node's output as soon as all of
            its parents have consumed it.  This reduces peak memory for
            inference; backward still works because autograd holds its own
            references.
        eval_batch_size: Chunk size for the batch dimension. Inputs larger than
            this are evaluated in row chunks and concatenated — every node op is
            row-independent, so results are bit-identical to a single big
            evaluation while per-layer transients stay bounded (this is what
            makes whole-grid plot evaluations memory-safe). Defaults to 8192
            rows on CPU / 32768 on CUDA; pass 0 or None-ish <= 0 to disable.

    When gradients are disabled (the usual inference path), the evaluation runs
    under ``torch.inference_mode()``: no autograd metadata or version counters
    are recorded, which is measurably faster and lower-memory on both CPU and
    GPU. Autograd is still supported: with grad enabled each chunk keeps its
    own graph and the concatenated output backpropagates into all of them.
    """
    if eval_batch_size is None:
        eval_batch_size = 1 << 15 if data.is_cuda else 1 << 13

    if eval_batch_size > 0 and data.shape[0] > eval_batch_size:
        if torch.is_grad_enabled():
            ctx: Any = contextlib.nullcontext()
        else:
            ctx = torch.inference_mode()
        with ctx:
            parts = [
                _eval_circuit_single(
                    ac,
                    data[i : i + eval_batch_size],
                    verbose=verbose,
                    show_leaves_only=show_leaves_only,
                    show_weights=show_weights,
                    log_domain=log_domain,
                    debug_check_finite=debug_check_finite,
                    keep_intermediates=keep_intermediates,
                )
                for i in range(0, data.shape[0], eval_batch_size)
            ]
        return torch.cat(parts, dim=0)

    if not torch.is_grad_enabled():
        with torch.inference_mode():
            return _eval_circuit_single(
                ac,
                data,
                verbose=verbose,
                show_leaves_only=show_leaves_only,
                show_weights=show_weights,
                log_domain=log_domain,
                debug_check_finite=debug_check_finite,
                keep_intermediates=keep_intermediates,
            )

    return _eval_circuit_single(
        ac,
        data,
        verbose=verbose,
        show_leaves_only=show_leaves_only,
        show_weights=show_weights,
        log_domain=log_domain,
        debug_check_finite=debug_check_finite,
        keep_intermediates=keep_intermediates,
    )


def _eval_circuit_single(
    ac: SymbolicArithmeticCircuit,
    data: torch.Tensor,
    verbose: bool = False,
    show_leaves_only: bool = False,
    show_weights: bool = False,
    log_domain: bool = True,
    debug_check_finite: bool = False,
    keep_intermediates: bool = True,
) -> torch.Tensor:
    """Single-chunk circuit evaluation (see :func:`eval_circuit`)."""
    outputs = {}
    roots = ac.get_roots()
    root_set = set(roots)
    remaining_parents = None
    if not keep_intermediates:
        # dict() copy: the eval loop decrements counts, the cache must not.
        remaining_parents = dict(ac.in_degrees())

    for node_id in ac.topological_sort(reverse=True):
        node = ac.get_node_data(node_id)

        # print("Evaluating node with scope", node.scope)

        if show_weights and isinstance(node, SumLayer) and (0 in node.scope and 1 in node.scope):
            w_obj = node.log_weights
            wt = (
                w_obj.log_weights.detach().cpu()
                if hasattr(w_obj, "log_weights")
                else w_obj.detach().cpu()
            )
            G, H, G_L, H_L, G_R, H_R = wt.shape
            wt_to_print = wt if log_domain else torch.exp(wt)
            if log_domain:
                wt_to_print = _prepare_log_for_print(wt_to_print)
            wt_rounded = torch.round(wt_to_print * 1000) / 1000.0
            wt_rounded = wt_rounded + 0.0
            wt_rounded = wt_rounded.reshape(G, H, G_L * H_L, G_R * H_R)
            print(
                f"DEBUG: Node {node_id:2d} ({type(node).__name__}) (scope: {list(node.scope) if node.scope and not node.scope.is_empty else 'N/A'}) G={G}, H={H}, G_L={G_L}, H_L={H_L}, G_R={G_R}, H_R={H_R}"
            )
            for g in range(G):
                for h in range(H):
                    print(
                        f"DEBUG: Node {node_id:2d} ({type(node).__name__}) (scope: {list(node.scope) if node.scope and not node.scope.is_empty else 'N/A'}) weights for group {g}, node {h}:\n{wt_rounded[g, h]}"
                    )
                # elementwise product of all group weights over all nodes (ignoring NaNs)
                prod = np.nansum(wt_rounded[g], axis=0)
                prod_torch = torch.from_numpy(prod)
                print(
                    f"DEBUG: Node {node_id:2d} ({type(node).__name__}) (scope: {list(node.scope) if node.scope and not node.scope.is_empty else 'N/A'}) sum of weights for group {g} over all nodes:\n{prod_torch}"
                )

            # wt_rounded has shape (G, H, G_L * H_L, G_R * H_R)
            wt_np = wt_rounded.detach().cpu().numpy()
            wt_np = np.where(np.isinf(wt_np), np.nan, wt_np).clip(min=-6)

            # 2. Compute standard deviation ignoring NaNs
            per_node_stds_over_group = np.nansum(np.nanstd(wt_np, axis=0, ddof=0), axis=0)
            per_group_stds_over_left_child = np.nanstd(wt_np, axis=(1, 2), ddof=0)

            # 3. Convert back to torch if needed
            per_node_stds_over_group = torch.from_numpy(per_node_stds_over_group)
            per_group_stds_over_left_child = torch.from_numpy(per_group_stds_over_left_child)

            # print(
            #     f"DEBUG: STDs for node {node_id:2d} ({type(node).__name__}) (scope: {list(node.scope) if node.scope and not node.scope.is_empty else 'N/A'}) of nodes (overlapped) over groups:\n{per_node_stds_over_group}\nmean:{per_node_stds_over_group.mean()}"
            # )

            # for g in range(G):
            #     print(
            #         f"DEBUG: STDs for node {node_id:2d} ({type(node).__name__}) (scope: {list(node.scope) if node.scope and not node.scope.is_empty else 'N/A'}) of group {g} over all nodes over left children:\n{per_group_stds_over_left_child[g]}\nmean:{per_group_stds_over_left_child[g].mean()}"
            #     )

        child_ids = ac.get_children(node_id)
        child_outs = [outputs[cid] for cid in child_ids]
        # print(f"Evaluating node with scope {node.scope}")
        out = node.forward(data, child_outs)
        outputs[node_id] = out

        if not keep_intermediates:
            for cid in child_ids:
                remaining_parents[cid] -= 1
                if remaining_parents[cid] == 0 and cid not in root_set:
                    outputs.pop(cid, None)

        if verbose:
            if not show_leaves_only or isinstance(node, LeafLayer):
                if out.numel() > 0:
                    # Convert to linear domain if requested and remove numerical artifacts
                    samples = out[:5] if log_domain else torch.exp(out[:5])
                    if log_domain:
                        samples = _prepare_log_for_print(samples)
                    samples = torch.round(samples * 1000) / 1000.0
                    samples = samples + 0.0
                    samples_str = f"\n{samples}"
                else:
                    samples_str = "N/A"
                print(
                    f"DEBUG: Node {node_id:2d} ({type(node).__name__}) (scope: {list(node.scope) if node.scope and not node.scope.is_empty else 'N/A'}) output: shape={list(out.shape)}, samples={samples_str}"
                )

    roots = ac.get_roots()
    if not roots:
        return torch.empty(data.shape[0], 0)
    # Return the first root's output
    out = outputs[roots[0]]
    if debug_check_finite and not torch.isfinite(out).all():
        bad = (~torch.isfinite(out)).nonzero(as_tuple=False)
        raise ValueError(
            f"eval_circuit produced non-finite root output for {bad.shape[0]} row(s); "
            f"first bad row index: {bad[0].tolist()}"
        )
    return out
