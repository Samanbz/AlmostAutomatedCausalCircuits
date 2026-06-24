import copy
from typing import TYPE_CHECKING, Any, Dict, List, Type

import torch

from src.utils import BitSet, NodeAllocator

from ..base import DirectedAcyclicGraph
from ..vtree import VTree
from .nodes import (
    ArithmeticNode,
    CartesianLeafNode,
    ConstantLeafNode,
    InverseLeafNode,
    KroneckerProductNode,
    LeafNode,
    ProductLeafNode,
    ProductNode,
    SumNode,
)


if TYPE_CHECKING:
    from .properties import Property


class SymbolicArithmeticCircuit(DirectedAcyclicGraph[int, ArithmeticNode, Any]):
    """
    A DAG representing a symbolic arithmetic circuit.
    Nodes are identified by integers and contain Node objects (SumNode, KroneckerProductNode, etc.).
    Edges can optionally contain data (e.g., weights for sum nodes) and represent descendence and
    not computational flow.
    """

    def __init__(self, node_allocator: NodeAllocator = None, vtree: VTree = None):
        super().__init__(node_allocator=node_allocator)
        self.vtree = vtree
        self.vtree_to_sum: Dict[int, List[int]] = {}
        self.sum_to_vtree: Dict[int, int] = {}

    def get_nodes_with_scope(self, scope: BitSet, node_type: Type[ArithmeticNode]) -> List[int]:
        """Returns a list of node IDs that have the given scope and are of the specified type."""
        return [
            node_id
            for node_id, node in self._nodes.items()
            if isinstance(node, node_type) and node.scope == scope
        ]  # TODO: improve by using a top-down pruning search

    def unfold(self) -> "SymbolicArithmeticCircuit":
        """
        Unfolds the tensorized circuit into an explicit scalar circuit.
        Returns a new SymbolicArithmeticCircuit where all nodes have unit_count=1.
        Edges carry explicit scalar weights if available.
        """

        new_ac = SymbolicArithmeticCircuit()

        # Mapping from original node ID to a list of unfolded node IDs
        unfold_map: Dict[int, List[int]] = {}

        for node_id in self.topological_sort(reverse=True):
            node = self.get_node_data(node_id)
            h = node.unit_count
            unfolded_ids = []

            if isinstance(node, LeafNode):
                for i in range(h):
                    new_node = copy.copy(node)
                    new_node.unit_count = 1
                    if hasattr(node, "unit_supports") and node.unit_supports:
                        new_node.unit_supports = [node.unit_supports[i]]
                    # Do not copy weights as leaf nodes don't have them in the same way sum nodes do
                    new_id = new_ac.add_node(new_node)
                    unfolded_ids.append(new_id)

            elif isinstance(node, KroneckerProductNode):
                children = self.get_children(node_id)
                assert len(children) == 2, "Kronecker product must have exactly two children"
                left_child, right_child = children
                left_unfolded = unfold_map[left_child]
                right_unfolded = unfold_map[right_child]

                for i in range(len(left_unfolded)):
                    for j in range(len(right_unfolded)):
                        new_node = KroneckerProductNode(support=node.support, unit_count=1)
                        new_id = new_ac.add_node(new_node)
                        new_ac.add_edge(new_id, left_unfolded[i])
                        new_ac.add_edge(new_id, right_unfolded[j])
                        unfolded_ids.append(new_id)

            elif isinstance(node, SumNode):
                children = self.get_children(node_id)
                assert len(children) == 1, (
                    "Sum node currently expects exactly one product child layer"
                )
                child_id = children[0]
                child_unfolded = unfold_map[child_id]
                h_in = len(child_unfolded)

                if getattr(node, "sparse", True):
                    prods_per_sum = h_in // h
                    for i in range(h):
                        new_node = SumNode(
                            support=node.support, unit_count=1, md_set=node.md_set, sparse=True
                        )
                        new_id = new_ac.add_node(new_node)
                        unfolded_ids.append(new_id)

                        start_idx = i * prods_per_sum
                        end_idx = start_idx + prods_per_sum
                        for j, child_prod_idx in enumerate(range(start_idx, end_idx)):
                            weight = (
                                node.weights[i, j].item()
                                if hasattr(node, "weights")
                                and node.weights.shape[0] > i
                                and node.weights.shape[1] > j
                                else None
                            )
                            new_ac.add_edge(new_id, child_unfolded[child_prod_idx], data=weight)
                else:
                    for i in range(h):
                        new_node = SumNode(
                            support=node.support, unit_count=1, md_set=node.md_set, sparse=False
                        )
                        new_id = new_ac.add_node(new_node)
                        unfolded_ids.append(new_id)

                        for j in range(h_in):
                            weight = node.weights[i, j].item() if hasattr(node, "weights") else None
                            new_ac.add_edge(new_id, child_unfolded[j], data=weight)

            unfold_map[node_id] = unfolded_ids

        return new_ac

    def check_property(self, property: "Property") -> bool:
        """Checks if the given property holds for the specified node."""
        for node_id in self.topological_sort():
            if not property.check(node_id, self):
                return False
        return True

    def add_edge(self, source: int, target: int, data: Any = None) -> None:
        """Adds a directed edge from source to target with optional data."""
        super().add_edge(source, target, data)

    def is_sum_node(self, node_id: int) -> bool:
        return isinstance(self.get_node_data(node_id), SumNode) or isinstance(
            self.get_node_data(node_id), LeafNode
        )

    def is_product_node(self, node_id: int) -> bool:
        return isinstance(self.get_node_data(node_id), ProductNode)

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
                hasattr(node, "mean")
                and isinstance(node.mean, torch.Tensor)
                and node.mean.grad is not None
            ):
                if id(node.mean) not in seen_tensors:
                    seen_tensors.add(id(node.mean))
                    node.mean.grad.zero_()
            if (
                hasattr(node, "stddev")
                and isinstance(node.stddev, torch.Tensor)
                and node.stddev.grad is not None
            ):
                if id(node.stddev) not in seen_tensors:
                    seen_tensors.add(id(node.stddev))
                    node.stddev.grad.zero_()
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
                hasattr(node, "mean")
                and isinstance(node.mean, torch.Tensor)
                and node.mean.requires_grad
            ):
                if id(node.mean) not in seen_params:
                    seen_params.add(id(node.mean))
                    yield node.mean
            if (
                hasattr(node, "stddev")
                and isinstance(node.stddev, torch.Tensor)
                and node.stddev.requires_grad
            ):
                if id(node.stddev) not in seen_params:
                    seen_params.add(id(node.stddev))
                    yield node.stddev
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
            if isinstance(node, LeafNode):
                yield from _yield_params(node)

    def get_node_config(
        self, show_node_id: bool = False, show_unit_supports: bool = False
    ) -> Dict[type, Dict[str, Any]]:
        self._show_unit_supports = show_unit_supports
        config = super().get_node_config(show_node_id=show_node_id)
        return config

    def to_dot(self, show_weights: bool = True, show_unit_supports: bool = False, **kwargs) -> str:
        """Returns a highly readable string representation of the circuit."""
        lines = []

        def get_composite_leaf_label(node: LeafNode) -> str:
            if isinstance(node, InverseLeafNode):
                return f"Inverse(^ {node.power})\n  " + get_composite_leaf_label(
                    node.base_leaf
                ).replace("\n", "\n  ")
            elif isinstance(node, CartesianLeafNode):
                return f"Cartesian\n  (a): {get_composite_leaf_label(node.leaf_a).replace(chr(10), chr(10) + '  ')}\n  (b): {get_composite_leaf_label(node.leaf_b).replace(chr(10), chr(10) + '  ')}"
            elif isinstance(node, ProductLeafNode):
                return f"ProductLeaf\n  (a): {get_composite_leaf_label(node.leaf_a).replace(chr(10), chr(10) + '  ')}\n  (b): {get_composite_leaf_label(node.leaf_b).replace(chr(10), chr(10) + '  ')}"
            elif type(node).__name__ == "GaussianMixture":
                return (
                    f"GaussianMixture\n  weights: {node.log_weights.shape}, {node.log_weights} \n  base: "
                    + get_composite_leaf_label(node.base_dist).replace("\n", "\n  ")
                )
            elif isinstance(node, ConstantLeafNode):
                return "Const(1.0)"
            else:
                var_str = f"Var {node.var}" if hasattr(node, "var") else ""
                return f"{type(node).__name__} {var_str}".strip()

        for node_id in self.topological_sort():
            node = self.get_node_data(node_id)
            label_str = f"[{node_id}] {type(node).__name__}"

            if isinstance(node, SumNode) or isinstance(node, LeafNode):
                if node_id in getattr(self, "sum_to_vtree", {}):
                    v_id = self.sum_to_vtree[node_id]
                    v_node = self.vtree.get_node_data(v_id)
                    scope_str = "{" + ",".join(str(x) for x in sorted(list(v_node.scope))) + "}"
                    md_set_str = "None"
                    if v_node.md_set is not None:
                        if v_node.md_set.is_universal:
                            md_set_str = "Univ"
                        else:
                            md_set_str = (
                                "{" + ",".join(str(x) for x in sorted(list(v_node.md_set))) + "}"
                            )
                    label_str += f" (scope={scope_str}, md={md_set_str})"

            lines.append(label_str)

            if (
                show_weights
                and isinstance(node, SumNode)
                and getattr(node, "log_weights", None) is not None
            ):
                w = torch.exp(node.log_weights)

                h_l, h_r = None, None
                children = self.get_children(node_id)
                if children and len(children) == 1:
                    child_id = children[0]
                    child_node = self.get_node_data(child_id)
                    if isinstance(child_node, ProductNode):
                        prod_children = self.get_children(child_id)
                        if len(prod_children) == 2:
                            l_child = self.get_node_data(prod_children[0])
                            r_child = self.get_node_data(prod_children[1])
                            if w.shape[1] == l_child.unit_count * r_child.unit_count:
                                h_l = l_child.unit_count
                                h_r = r_child.unit_count

                if h_l is not None and h_r is not None and w.numel() <= 1024:
                    w_3d = w.view(-1, h_l, h_r)
                    lines.append(f"  weights: shape={list(w_3d.shape)}")
                    for i in range(h_l):
                        row_strs = []
                        for u in range(w_3d.shape[0]):
                            row_str = "[" + ", ".join(f"{x:.5f}" for x in w_3d[u, i]) + "]"
                            row_strs.append(row_str)
                        lines.append("    " + " | ".join(row_strs))
                elif w.numel() <= 64:
                    lines.append("  weights:")
                    if w.ndim == 1:
                        w_str = "[" + ", ".join(f"{x:.5f}" for x in w) + "]"
                        lines.append(f"    {w_str}")
                    elif w.ndim == 2:
                        for row in w:
                            w_str = "[" + ", ".join(f"{x:.5f}" for x in row) + "]"
                            lines.append(f"    {w_str}")
                else:
                    lines.append(f"  weights: shape={list(w.shape)}")

            if isinstance(node, LeafNode):
                comp_label = get_composite_leaf_label(node)
                lines.append(f"  {comp_label.replace(chr(10), chr(10) + '  ')}")

            children = self.get_children(node_id)
            if children:
                lines.append(f"  children: {children}")

            lines.append("")

        return "\n".join(lines)

    def _get_base_node_config(self) -> Dict[type, Dict[str, Any]]:
        config = super()._get_base_node_config()
        show_unit_supports = getattr(self, "_show_unit_supports", False)

        def leaf_label(node: LeafNode) -> str:
            label = type(node).__name__ if type(node).__name__ == "GaussianMixture" else "Leaf"
            if hasattr(node, "scope"):
                label += f"\n{sorted(node.scope)}"
            if show_unit_supports:
                if getattr(node, "unit_supports", None) is not None and len(node.unit_supports) > 1:
                    supports_str = "\n".join(str(s) for s in node.unit_supports)
                    label += f"\n{supports_str}"
                elif hasattr(node, "var_support"):
                    # node: Distribution
                    label += f"\n{node.var_support}"
            return label

        def node_label(node: ArithmeticNode) -> str:
            label = ""
            if isinstance(node, SumNode):
                label = "+"
                if not getattr(node, "sparse", True):
                    label += " (Dense)"
            elif isinstance(node, KroneckerProductNode):
                label = "x_K"
            elif isinstance(node, LeafNode):
                label = leaf_label(node)

            if hasattr(node, "unit_count") and node.unit_count > 1:
                label += f"\n(units={node.unit_count})"

            if show_unit_supports and getattr(node, "unit_supports", None) is not None:
                supports_str = "\n".join(str(s) for s in node.unit_supports)
                label += f"\n{supports_str}"

            return label

        config.update(
            {
                SumNode: {"color": "#ccffcc", "label": node_label, "shape": "box"},
                KroneckerProductNode: {"color": "#ffcc99", "label": node_label, "shape": "box"},
                CartesianLeafNode: {"color": "#ffcccc", "label": node_label, "shape": "ellipse"},
                ConstantLeafNode: {"color": "#ffcccc", "label": node_label, "shape": "ellipse"},
            }
        )
        return config


def eval_circuit(
    ac: SymbolicArithmeticCircuit,
    data: torch.Tensor,
    verbose: bool = False,
    show_leaves_only: bool = False,
) -> torch.Tensor:
    """Evaluates the circuit on the given data."""
    outputs = {}
    for node_id in ac.topological_sort(reverse=True):
        node = ac.get_node_data(node_id)
        child_ids = ac.get_children(node_id)
        child_outs = [outputs[cid] for cid in child_ids]
        out = node.forward(data, child_outs)
        outputs[node_id] = out

        if verbose:
            if not show_leaves_only or isinstance(node, LeafNode):
                print(
                    f"DEBUG: Node {node_id} ({type(node).__name__}) (scope: {getattr(getattr(node, 'support', None), 'scope', 'N/A')}) output: shape={out.shape},min={out.min().item():.4f},max={out.max().item():.4f},mean={out.mean().item():.4f},samples={out[:1].squeeze()}"
                )

    roots = ac.get_roots()
    if not roots:
        return torch.empty(data.shape[0], 0)
    # Return the first root's output
    return outputs[roots[0]]
