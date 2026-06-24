from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.logger import logger as g_logger
from src.symbolic import (
    GaussianDistribution,
)
from src.symbolic.arithmetic.circuit import SymbolicArithmeticCircuit
from src.symbolic.arithmetic.nodes import (
    ConstantLeafNode,
    InverseLeafNode,
    KroneckerProductNode,
    LeafNode,
    ProductLeafNode,
    SumNode,
)
from src.symbolic.base import DirectedAcyclicGraph, Node


logger = g_logger.getChild(__name__)


class FoldedLayer(Node):
    def __init__(self, num_nodes: int, h_out: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.h_out = h_out

    @property
    def output_shape(self) -> Tuple[int, int]:
        return (self.num_nodes, self.h_out)


class FoldedInputLayer(FoldedLayer):
    pass


class FoldedProductLayer(FoldedLayer):
    pass


class FoldedKroneckerProductLayer(FoldedProductLayer):
    def __init__(
        self,
        num_nodes: int,
        h_out: int,
        left_indices: List[int],
        right_indices: List[int],
        h_left: int,
        h_right: int,
    ):
        super().__init__(num_nodes, h_out)
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.h_left = h_left
        self.h_right = h_right


class FoldedSumLayer(FoldedLayer):
    """Base class for sum layers. Subclassed per product-child type."""

    def __init__(
        self,
        num_nodes: int,
        h_out: int,
        h_in: int,
        child_indices: List[int],
        h_child: int,
    ):
        super().__init__(num_nodes, h_out)
        self.h_in = h_in
        self.child_indices = child_indices
        self.h_child = h_child

    @property
    def output_shape(self) -> Tuple[int, int]:
        return (self.num_nodes, self.h_out)


class FoldedSparseKroneckerSumLayer(FoldedSumLayer):
    """Sum layer over Kronecker (or Unary) product children.

    Each output unit k can receive any of the h_child product units from each
    child. The weight tensor selects which units contribute and with what
    strength. This is a dense sum — no partition constraint.

    Attributes:
        child_indices: [num_nodes] — product node indices
        weights:       [num_nodes, h_out, prods_per_sum] — normalized over (prods_per_sum)
        h_child:       units per product child (h*h for Kronecker, h for Unary)
    """

    pass


class FoldedTuckerSumLayer(FoldedSumLayer):
    """Universal dense sum layer over Kronecker product children.

    Unlike SparseKronecker, there is no partitioning. Every output unit sums over all
    h_child product units. The weight tensor maps all h_child units to h_out.

    Attributes:
        child_indices: [num_nodes, num_children]
        h_child: units per product child (h*h)
    """

    pass


class FoldedSymbolicCircuit(DirectedAcyclicGraph[int, FoldedLayer, Any]):
    def _get_base_node_config(self) -> Dict[type, Dict[str, Any]]:
        config = super()._get_base_node_config()

        def label_maker(layer_type: str):
            def label_fn(node: FoldedLayer) -> str:
                label_str = f"{layer_type}\nNodes: {node.num_nodes}"
                if hasattr(node, "is_unary") and node.is_unary:
                    label_str += "\n(Unary)"
                label_str += f"\nOutput: {node.output_shape}"
                return label_str

            return label_fn

        config.update(
            {
                FoldedSparseKroneckerSumLayer: {
                    "color": "#99ccff",
                    "label": lambda n: f"SparseKroneckerSum (x{n.num_nodes})\nUnits: {n.h_out}\nProds/sum: {n.h_child // n.h_out}",
                },
                FoldedTuckerSumLayer: {
                    "shape": "box3d",
                    "color": "#ffb3b3",
                    "label": label_maker("TuckerSumLayer"),
                },
                FoldedKroneckerProductLayer: {
                    "color": "#ffcccc",
                    "label": lambda n: f"KroneckerProduct (x{n.num_nodes})\nUnits: {n.h_out}",
                },
                FoldedInputLayer: {
                    "shape": "cylinder",
                    "color": "#b3ffb3",
                    "label": label_maker("InputLayer"),
                },
            }
        )
        return config


class FoldedGaussianInputLayer(FoldedInputLayer):
    def __init__(
        self,
        node_ids: List[int],
        means: torch.Tensor,
        stddevs: torch.Tensor,
        highs: torch.Tensor,
        lows: torch.Tensor,
    ):
        assert (
            len(node_ids) == means.shape[0] == stddevs.shape[0] == highs.shape[0] == lows.shape[0]
        ), "Length of node_ids, means, stddevs, highs, and lows must be the same"
        assert means.dim() == stddevs.dim() == highs.dim() == lows.dim() == 2, (
            "means, stddevs, highs, and lows must be 2D tensors"
        )
        assert means.shape[1] == stddevs.shape[1] == highs.shape[1] == lows.shape[1], (
            "The second dimension (node_h) of means, stddevs, highs, and lows must be the same"
        )
        super().__init__(num_nodes=len(node_ids), h_out=means.shape[1])
        self.node_ids = node_ids
        self.means = means
        self.stddevs = stddevs
        self.highs = highs
        self.lows = lows


class CircuitFolder:
    def __init__(self, ac: SymbolicArithmeticCircuit):
        self.ac = ac
        self.folded_circuit = FoldedSymbolicCircuit()

    def _extract_gaussian_params_extended(
        self, node_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract Gaussian params + leaf_modes for regular and compiled leaves.

        Returns (means, stddevs, lows, highs, leaf_modes) where leaf_modes[i]:
          1.0 = normal Gaussian (log_pdf)
          0.0 = constant (always 0)
          -1.0 = inverse (-log_pdf)
        """
        n = len(node_ids)
        h = self.ac.get_node_data(node_ids[0]).unit_count

        means_np = np.zeros((n, h), dtype=np.float32)
        stddevs_np = np.ones((n, h), dtype=np.float32)
        lows_np = np.full((n, h), float("-inf"), dtype=np.float32)
        highs_np = np.full((n, h), float("inf"), dtype=np.float32)
        modes_np = np.zeros(n, dtype=np.float32)

        for i, nid in enumerate(node_ids):
            raw_node = self.ac.get_node_data(nid)
            base_node, mode = self._resolve_leaf(raw_node)
            modes_np[i] = mode

            if mode == 0.0:
                # Constant leaf: dummy params, forward returns 0
                continue

            # Extract Gaussian params from the base node
            if hasattr(base_node, "mean"):
                mean_val = base_node.mean
                std_val = base_node.stddev

                # If mean_val is a tensor, ensure it matches h
                if torch.is_tensor(mean_val):
                    means_np[i, :] = mean_val.detach().cpu().numpy()
                else:
                    means_np[i, :] = mean_val

                if torch.is_tensor(std_val):
                    stddevs_np[i, :] = std_val.detach().cpu().numpy()
                else:
                    stddevs_np[i, :] = std_val

                us = base_node.unit_supports
                if us:
                    var = base_node.var
                    limit = min(h, len(us))
                    for j in range(limit):
                        iv = us[j].get(var)
                        if iv is not None:
                            lows_np[i, j] = iv.low
                            highs_np[i, j] = iv.high

        return (
            torch.from_numpy(means_np),
            torch.from_numpy(stddevs_np),
            torch.from_numpy(lows_np),
            torch.from_numpy(highs_np),
            torch.from_numpy(modes_np),
        )

    def build(self) -> FoldedSymbolicCircuit:
        node_to_folded: Dict[int, Tuple[int, int]] = {}
        layer_global_offset: Dict[int, int] = {}
        current_offset = 0

        for node_layer in self.ac.layered_topological_sort(reverse=True):
            first = self.ac._nodes[node_layer[0]]
            is_leaf_layer = isinstance(first, LeafNode)
            is_product_layer = isinstance(first, (KroneckerProductNode,))
            is_sum_layer = isinstance(first, SumNode)

            if is_leaf_layer:
                self._fold_leaf_layer(
                    node_layer, node_to_folded, layer_global_offset, current_offset
                )
                for nid in node_layer:
                    fold_id, _ = node_to_folded[nid]
                    if fold_id not in layer_global_offset:
                        layer_global_offset[fold_id] = current_offset
                        current_offset += self.folded_circuit.get_node_data(fold_id).num_nodes

            elif is_product_layer:
                self._fold_product_layer(node_layer, node_to_folded, layer_global_offset)

            elif is_sum_layer:
                self._fold_sum_layer(
                    node_layer, node_to_folded, layer_global_offset, current_offset
                )
                for nid in node_layer:
                    fold_id, _ = node_to_folded[nid]
                    if fold_id not in layer_global_offset:
                        layer_global_offset[fold_id] = current_offset
                        current_offset += self.folded_circuit.get_node_data(fold_id).num_nodes

        self._fix_layer_global_offsets(layer_global_offset, node_to_folded)

        return self.folded_circuit

    def _fix_layer_global_offsets(
        self,
        layer_global_offset: Dict[int, int],
        node_to_folded: Dict[int, Tuple[int, int]],
    ) -> None:
        new_offset: Dict[int, int] = {}
        current = 0
        for fold_id in self.folded_circuit.topological_sort():
            fnode = self.folded_circuit.get_node_data(fold_id)
            if isinstance(fnode, (FoldedInputLayer, FoldedSumLayer)):
                new_offset[fold_id] = current
                current += fnode.num_nodes

        if new_offset == layer_global_offset:
            return

        fold_to_syms: Dict[int, List[Tuple[int, int]]] = {}
        for sym_nid, (fold_id, local_idx) in node_to_folded.items():
            fold_to_syms.setdefault(fold_id, []).append((local_idx, sym_nid))

        adj = self.ac._adj

        for fold_id in self.folded_circuit.topological_sort():
            fnode = self.folded_circuit.get_node_data(fold_id)
            if not isinstance(fnode, (FoldedKroneckerProductLayer,)):
                continue

            sym_entries = sorted(fold_to_syms.get(fold_id, []), key=lambda x: x[0])
            n = len(sym_entries)
            new_left: List[int] = [0] * n
            new_right: List[int] = [0] * n

            for i, (_, sym_nid) in enumerate(sym_entries):
                children = list(adj[sym_nid])
                left_fold_id, left_idx = node_to_folded[children[0]]
                right_fold_id, right_idx = node_to_folded[children[1]]
                new_left[i] = new_offset[left_fold_id] + left_idx
                new_right[i] = new_offset[right_fold_id] + right_idx

            fnode.left_indices = new_left
            fnode.right_indices = new_right

        layer_global_offset.clear()
        layer_global_offset.update(new_offset)

    def _resolve_leaf(self, node: LeafNode) -> tuple[LeafNode, float]:
        if isinstance(node, ProductLeafNode):
            base_a, mode_a = self._resolve_leaf(node.leaf_a)
            base_b, mode_b = self._resolve_leaf(node.leaf_b)
            return base_a, mode_a + mode_b

        if isinstance(node, ConstantLeafNode):
            return node, 0.0
        if isinstance(node, InverseLeafNode):
            base, inner_mode = self._resolve_leaf(node.base_leaf)
            return base, -inner_mode
        return node, 1.0

    def _fold_leaf_layer(
        self,
        node_layer: List[int],
        node_to_folded: Dict[int, Tuple[int, int]],
        layer_global_offset: Dict[int, int],
        current_offset: int,
    ) -> None:
        gaussian_ids = []

        for nid in node_layer:
            node = self.ac.get_node_data(nid)
            if isinstance(node, (ConstantLeafNode, InverseLeafNode, ProductLeafNode)):
                gaussian_ids.append(nid)
            elif isinstance(node, (GaussianDistribution)):
                gaussian_ids.append(nid)

        if gaussian_ids:
            means, stddevs, lows, highs, leaf_modes = self._extract_gaussian_params_extended(
                gaussian_ids
            )
            layer = FoldedGaussianInputLayer(gaussian_ids, means, stddevs, highs, lows)
            layer.leaf_modes = leaf_modes
            layer.scopes = [self.ac._nodes[nid].var for nid in gaussian_ids]
            layer.md_sets = [getattr(self.ac._nodes[nid], "md_set", None) for nid in gaussian_ids]
            layer_id = self.folded_circuit._add_node(layer)
            node_to_folded.update({nid: (layer_id, i) for i, nid in enumerate(gaussian_ids)})

    def _fold_product_layer(
        self,
        node_layer: List[int],
        node_to_folded: Dict[int, Tuple[int, int]],
        layer_global_offset: Dict[int, int],
    ) -> None:
        kronecker_ids = []

        for nid in node_layer:
            node = self.ac.get_node_data(nid)
            if isinstance(node, KroneckerProductNode):
                kronecker_ids.append(nid)

        if kronecker_ids:
            self._fold_kronecker_product_layer(kronecker_ids, node_to_folded, layer_global_offset)

    def _fold_kronecker_product_layer(
        self,
        product_node_ids: List[int],
        node_to_folded: Dict[int, Tuple[int, int]],
        layer_global_offset: Dict[int, int],
    ) -> None:
        """Fold Kronecker product nodes into a single layer."""
        num_nodes = len(product_node_ids)
        h_out = self.ac.get_node_data(product_node_ids[0]).unit_count

        global_left = [0] * num_nodes
        global_right = [0] * num_nodes
        edge_left: Dict[int, List[int]] = {}
        edge_right: Dict[int, List[int]] = {}

        adj = self.ac._adj
        for i, nid in enumerate(product_node_ids):
            children = list(adj[nid])
            left_id, right_id = children[0], children[1]

            left_fold_id, left_idx = node_to_folded[left_id]
            right_fold_id, right_idx = node_to_folded[right_id]

            global_left[i] = layer_global_offset[left_fold_id] + left_idx
            global_right[i] = layer_global_offset[right_fold_id] + right_idx

            edge_left.setdefault(left_fold_id, []).append(left_idx)
            edge_right.setdefault(right_fold_id, []).append(right_idx)

        h_left = self.ac.get_node_data(self.ac.get_children(product_node_ids[0])[0]).unit_count
        h_right = self.ac.get_node_data(self.ac.get_children(product_node_ids[0])[1]).unit_count

        layer = FoldedKroneckerProductLayer(
            num_nodes=num_nodes,
            h_out=h_out,
            left_indices=global_left,
            right_indices=global_right,
            h_left=h_left,
            h_right=h_right,
        )
        layer_id = self.folded_circuit._add_node(layer)

        for child_fold_id, idx_list in edge_left.items():
            self.folded_circuit._add_edge(child_fold_id, layer_id, data=("left", idx_list))
        for child_fold_id, idx_list in edge_right.items():
            self.folded_circuit._add_edge(child_fold_id, layer_id, data=("right", idx_list))

        node_to_folded.update({nid: (layer_id, i) for i, nid in enumerate(product_node_ids)})

    def _fold_sum_layer(
        self,
        node_layer: List[int],
        node_to_folded: Dict[int, Tuple[int, int]],
        layer_global_offset: Dict[int, int],
        current_offset: int,
    ) -> None:
        from collections import defaultdict

        groups = defaultdict(list)

        adj = self.ac._adj
        nodes_dict = self.ac._nodes
        for nid in node_layer:
            sum_node = nodes_dict[nid]
            h_out = sum_node.unit_count
            child_type = self._classify_sum_children(nid)
            arity = len(adj[nid])
            groups[(child_type, arity, h_out)].append(nid)

        for (child_type, _arity, _h_out), sum_node_ids in groups.items():
            if child_type == "kronecker":
                self._create_sparse_kronecker_sum_layer(sum_node_ids, node_to_folded)
            elif child_type == "universal":
                self._create_tucker_sum_layer(sum_node_ids, node_to_folded)
            elif child_type == "cpt":
                self._create_cpt_sum_layer(sum_node_ids, node_to_folded)
            else:
                self._create_sparse_kronecker_sum_layer(sum_node_ids, node_to_folded)

    def _classify_sum_children(self, sum_node_id: int) -> str:
        """Classify a sum node by its children's product type or its own type."""
        sum_node = self.ac.get_node_data(sum_node_id)
        if not getattr(sum_node, "sparse", True):
            return "universal"

        for nid in [sum_node_id]:
            child = self.ac.get_node_data(self.ac.get_children(nid)[0])
            if isinstance(child, KroneckerProductNode):
                return "kronecker"
        return "unary"

    def _collect_child_indices(
        self, sum_node_ids: List[int], node_to_folded: Dict[int, Tuple[int, int]]
    ) -> Tuple[List[int], set]:
        """Collect child folded-layer indices for a group of sum nodes.
        Returns (indices, child_fold_ids).
        """
        all_indices: List[int] = []
        fold_ids: set = set()
        adj = self.ac._adj

        for nid in sum_node_ids:
            cid = next(iter(adj[nid]))
            fid, fidx = node_to_folded[cid]
            all_indices.append(fidx)
            fold_ids.add(fid)

        return all_indices, fold_ids

    def _register_sum_layer(
        self,
        layer: FoldedSumLayer,
        sum_node_ids: List[int],
        node_to_folded: Dict[int, Tuple[int, int]],
        child_fold_ids: set,
    ) -> None:
        """Add the sum layer to the folded circuit with edges and mapping."""
        layer.md_sets = [getattr(self.ac._nodes[nid], "md_set", None) for nid in sum_node_ids]

        first_node = self.ac.get_node_data(sum_node_ids[0])
        if hasattr(first_node, "weights") and first_node.weights is not None:
            import torch

            layer.log_weights = torch.stack(
                [self.ac.get_node_data(nid).weights for nid in sum_node_ids]
            )

        layer_id = self.folded_circuit._add_node(layer)
        for cfid in child_fold_ids:
            self.folded_circuit._add_edge(cfid, layer_id)
        node_to_folded.update({nid: (layer_id, i) for i, nid in enumerate(sum_node_ids)})

    def _create_sparse_kronecker_sum_layer(
        self,
        sum_node_ids: List[int],
        node_to_folded: Dict[int, Tuple[int, int]],
    ) -> None:
        """Create a FoldedSparseKroneckerSumLayer for sums over Kronecker/Unary product children.

        Partition: h_child units are split into h_out groups of (h_child // h_out).
        Group i = product units [i * prods_per_sum, (i+1) * prods_per_sum).
        Each output unit i sums over group i from all children.

        Weight shape: [num_nodes, h_out, num_children, prods_per_sum]
        Normalized over (num_children, prods_per_sum).
        """
        num_nodes = len(sum_node_ids)
        h_out = self.ac.get_node_data(sum_node_ids[0]).unit_count
        h_child = self.ac.get_node_data(self.ac.get_children(sum_node_ids[0])[0]).unit_count
        h_in = h_child // h_out

        indices, fold_ids = self._collect_child_indices(sum_node_ids, node_to_folded)

        layer = FoldedSparseKroneckerSumLayer(
            num_nodes=num_nodes,
            h_out=h_out,
            h_in=h_in,
            child_indices=indices,
            h_child=h_child,
        )
        self._register_sum_layer(layer, sum_node_ids, node_to_folded, fold_ids)

    def _create_tucker_sum_layer(
        self,
        sum_node_ids: List[int],
        node_to_folded: Dict[int, Tuple[int, int]],
    ) -> None:
        """Create a FoldedTuckerSumLayer for universal dense sums.

        No partition: each output unit connects to all h_child product units.
        Normalized over (num_children, h_child).
        """
        num_nodes = len(sum_node_ids)
        h_out = self.ac.get_node_data(sum_node_ids[0]).unit_count
        h_child = self.ac.get_node_data(self.ac.get_children(sum_node_ids[0])[0]).unit_count
        h_in = h_child  # No partition

        indices, fold_ids = self._collect_child_indices(sum_node_ids, node_to_folded)

        layer = FoldedTuckerSumLayer(
            num_nodes=num_nodes,
            h_out=h_out,
            h_in=h_in,
            child_indices=indices,
            h_child=h_child,
        )
        self._register_sum_layer(layer, sum_node_ids, node_to_folded, fold_ids)
