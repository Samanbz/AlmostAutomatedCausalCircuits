from typing import Any, Dict, List, Type

import torch

from src.utils import BitSet, NodeAllocator

from ..base import DirectedAcyclicGraph
from ..vtree import VTree
from .nodes import (
    ArithmeticNode,
    ConstantRegionNode,
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
        """Yields optimizable parameters from leaf nodes."""
        seen_params = set()

        def _yield_params(node):
            if (
                hasattr(node, "means")
                and isinstance(node.means, torch.Tensor)
                and node.means.requires_grad
            ):
                if id(node.means) not in seen_params:
                    seen_params.add(id(node.means))
                    yield node.means
            if (
                hasattr(node, "_raw_stddevs")
                and isinstance(node._raw_stddevs, torch.Tensor)
                and node._raw_stddevs.requires_grad
            ):
                if id(node._raw_stddevs) not in seen_params:
                    seen_params.add(id(node._raw_stddevs))
                    yield node._raw_stddevs
            if (
                hasattr(node, "log_weights")
                and isinstance(node.log_weights, torch.Tensor)
                and node.log_weights.requires_grad
            ):
                if id(node.log_weights) not in seen_params:
                    seen_params.add(id(node.log_weights))
                    yield node.log_weights
            if (
                hasattr(node, "logits")
                and isinstance(node.logits, torch.Tensor)
                and node.logits.requires_grad
            ):
                if id(node.logits) not in seen_params:
                    seen_params.add(id(node.logits))
                    yield node.logits
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

            if hasattr(node, "unit_count") and node.unit_count > 1:
                label += f"\n(units={node.unit_count})"

            if show_node_supports and getattr(node, "node_supports", None) is not None:
                supports_str = "\n".join(str(s) for s in node.node_supports)
                label += f"\n{supports_str}"

            return label

        config.update(
            {
                SumLayer: {"color": "#ccffcc", "label": node_label, "shape": "box"},
                ConstantRegionNode: {"color": "#ffcccc", "label": node_label, "shape": "ellipse"},
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
) -> torch.Tensor:
    """Evaluates the circuit on the given data."""
    outputs = {}
    for node_id in ac.topological_sort(reverse=True):
        node = ac.get_node_data(node_id)

        if show_weights and isinstance(node, SumLayer):
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

        child_ids = ac.get_children(node_id)
        child_outs = [outputs[cid] for cid in child_ids]
        out = node.forward(data, child_outs)
        outputs[node_id] = out

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
    return outputs[roots[0]]
