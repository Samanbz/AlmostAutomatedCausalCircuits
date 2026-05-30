from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.logger import logger as g_logger
from src.symbolic import (
    CategoricalDistribution,
    GaussianDistribution,
    TruncatedCategoricalDistribution,
    TruncatedGaussianDistribution,
    TruncatedUniformDistribution,
    UniformDistribution,
)
from src.symbolic.arithmetic.circuit import SymbolicArithmeticCircuit
from src.symbolic.arithmetic.nodes import (
    ConstantLeafNode,
    HadamardProductNode,
    InverseLeafNode,
    KroneckerProductNode,
    LeafNode,
    ProductLeafNode,
    SumNode,
    UniversalSumNode,
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


class FoldedHadamardProductLayer(FoldedProductLayer):
    def __init__(
        self,
        num_nodes: int,
        h_out: int,
        left_indices: List[int],
        right_indices: List[int],
        h_child: int,
    ):
        super().__init__(num_nodes, h_out)
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.h_child = h_child


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


class FoldedSparseHadamardSumLayer(FoldedSumLayer):
    """Sum layer over Hadamard product children.

    Each Hadamard child has h units. Output unit i sums strictly over
    unit i from every child — no cross-index mixing.

    Attributes:
        child_indices: [num_nodes, num_children] — product node indices
        weights:       [num_nodes, h_out, num_children, prods_per_sum] — normalized over (num_children, prods_per_sum)
        h_child:       units per product child (h for Hadamard)
    """

    pass


class FoldedCPTSumLayer(FoldedSumLayer):
    """Dense sum layer over Hadamard product children (CPT / Candecomp-transposed).

    Unlike SparseHadamard, there is no block-diagonal constraint. Every output
    unit sums over all h_child Hadamard product units with its own weight vector.

    Attributes:
        child_indices: [num_nodes, num_children]
        h_child: units per product child (h for Hadamard)
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
                    "shape": "box3d",
                    "color": "#ffb3b3",
                    "label": label_maker("SparseKronSumLayer"),
                },
                FoldedSparseHadamardSumLayer: {
                    "shape": "box3d",
                    "color": "#ffcccc",
                    "label": label_maker("SparseHadSumLayer"),
                },
                FoldedCPTSumLayer: {
                    "shape": "box3d",
                    "color": "#ffd9b3",
                    "label": label_maker("CPTSumLayer"),
                },
                FoldedTuckerSumLayer: {
                    "shape": "box3d",
                    "color": "#ffb3b3",
                    "label": label_maker("TuckerSumLayer"),
                },
                FoldedKroneckerProductLayer: {
                    "shape": "folder",
                    "color": "#b3b3ff",
                    "label": label_maker("KronProductLayer"),
                },
                FoldedHadamardProductLayer: {
                    "shape": "folder",
                    "color": "#b3ffff",
                    "label": label_maker("HadProductLayer"),
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


# TODO: Is this correct?
class FoldedCategoricalInputLayer(FoldedInputLayer):
    def __init__(self, node_ids: List[int], logits: torch.Tensor):
        assert len(node_ids) == logits.shape[0], "Length of node_ids and logits must be the same"
        assert logits.dim() == 2, "logits must be a 2D tensor"
        super().__init__(num_nodes=len(node_ids), h_out=logits.shape[1])
        self.node_ids = node_ids
        self.logits = logits


class FoldedUniformInputLayer(FoldedInputLayer):
    def __init__(self, node_ids: List[int], lows: torch.Tensor, highs: torch.Tensor):
        assert len(node_ids) == lows.shape[0] == highs.shape[0], (
            "Length of node_ids, lows, and highs must be the same"
        )
        assert lows.dim() == highs.dim() == 2, "lows and highs must be 2D tensors"
        assert lows.shape[1] == highs.shape[1], (
            "The second dimension (node_h) of lows and highs must be the same"
        )
        super().__init__(num_nodes=len(node_ids), h_out=lows.shape[1])
        self.node_ids = node_ids
        self.lows = lows
        self.highs = highs


class CircuitFolder:
    def __init__(self, ac: SymbolicArithmeticCircuit):
        self.ac = ac
        self.folded_circuit = FoldedSymbolicCircuit()

    def _extract_gaussian_params(
        self, node_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract (means, stddevs, lows, highs) tensors from Gaussian leaf nodes."""
        nodes = [self.ac.get_node_data(nid) for nid in node_ids]
        h = nodes[0].unit_count
        n = len(nodes)

        # Vectorized mean/stddev extraction — all Gaussian nodes share the same mean/stddev
        # across their h units (support splitting only affects lows/highs).
        means_1d = np.array([node.mean for node in nodes], dtype=np.float32)
        stddevs_1d = np.array([node.stddev for node in nodes], dtype=np.float32)
        means = torch.from_numpy(np.broadcast_to(means_1d[:, None], (n, h)).copy())
        stddevs = torch.from_numpy(np.broadcast_to(stddevs_1d[:, None], (n, h)).copy())

        lows_np = np.full((n, h), float("-inf"), dtype=np.float32)
        highs_np = np.full((n, h), float("inf"), dtype=np.float32)

        for i, node in enumerate(nodes):
            us = node.unit_supports
            if us:
                var = node.var
                limit = min(h, len(us))
                for j in range(limit):
                    iv = us[j].get(var)
                    if iv is not None:
                        lows_np[i, j] = iv.low
                        highs_np[i, j] = iv.high

        return means, stddevs, torch.from_numpy(lows_np), torch.from_numpy(highs_np)

    def _extract_gaussian_params_extended(
        self, node_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract Gaussian params + leaf_modes for regular and compiled leaves.

        Returns (means, stddevs, lows, highs, leaf_modes) where leaf_modes[i]:
          0 = normal Gaussian (log_pdf)
          1 = constant (always 0)
          2 = inverse (-log_pdf)
          3 = indicator (0 if in-support, -inf if OOB)
        """
        n = len(node_ids)
        h = self.ac.get_node_data(node_ids[0]).unit_count

        means_np = np.zeros((n, h), dtype=np.float32)
        stddevs_np = np.ones((n, h), dtype=np.float32)
        lows_np = np.full((n, h), float("-inf"), dtype=np.float32)
        highs_np = np.full((n, h), float("inf"), dtype=np.float32)
        modes_np = np.zeros(n, dtype=np.int64)

        for i, nid in enumerate(node_ids):
            raw_node = self.ac.get_node_data(nid)
            base_node, mode = self._resolve_leaf(raw_node)
            modes_np[i] = mode

            if mode == 1:
                # Constant leaf: dummy params, forward returns 0
                continue

            # Extract Gaussian params from the base node
            if hasattr(base_node, "mean"):
                mean_val = base_node.mean
                std_val = base_node.stddev
                means_np[i, :] = mean_val
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

    def _extract_uniform_params(self, node_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract (lows, highs) tensors from Uniform leaf nodes."""
        nodes = [self.ac.get_node_data(nid) for nid in node_ids]
        h = nodes[0].unit_count
        n = len(nodes)

        lows_np = np.zeros((n, h), dtype=np.float32)
        highs_np = np.ones((n, h), dtype=np.float32)

        for i, node in enumerate(nodes):
            var = node.var
            vs_low = node.var_support.low
            vs_high = node.var_support.high
            us = node.unit_supports
            if us:
                for j in range(h):
                    if j < len(us):
                        iv = us[j].get(var)
                        if iv is not None:
                            lows_np[i, j] = iv.low
                            highs_np[i, j] = iv.high
                        else:
                            lows_np[i, j] = vs_low
                            highs_np[i, j] = vs_high
                    else:
                        lows_np[i, j] = vs_low
                        highs_np[i, j] = vs_high
            else:
                lows_np[i, :] = vs_low
                highs_np[i, :] = vs_high

        return torch.from_numpy(lows_np), torch.from_numpy(highs_np)

    def _extract_categorical_params(self, node_ids: List[int]) -> torch.Tensor:
        """Extract logits tensor from Categorical leaf nodes."""
        nodes = [self.ac.get_node_data(nid) for nid in node_ids]
        h = nodes[0].unit_count
        n = len(nodes)
        logits = torch.zeros(n, h)
        return logits

    def build(self) -> FoldedSymbolicCircuit:
        node_to_folded: Dict[int, Tuple[int, int]] = {}
        # Track global offset for each folded layer (position in V_flat).
        # Assigned inline here and then corrected in _fix_layer_global_offsets.
        layer_global_offset: Dict[int, int] = {}
        current_offset = 0

        for node_layer in self.ac.layered_topological_sort(reverse=True):
            # All nodes within a layer are the same type — check only the first.
            first = self.ac._nodes[node_layer[0]]
            is_leaf_layer = isinstance(first, LeafNode)
            is_product_layer = isinstance(first, (HadamardProductNode, KroneckerProductNode))
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

        # Recompute offsets in folded_circuit.topological_sort() order and patch product layers.
        # layered_topological_sort may assign different orderings within a level than
        # topological_sort (e.g. when h-decay creates multiple fold layers at the same depth).
        # TensorizedCircuit builds _node_flat_offsets in topological_sort order, so the
        # gather indices in product layers must use that same order.
        self._fix_layer_global_offsets(layer_global_offset, node_to_folded)

        return self.folded_circuit

    def _fix_layer_global_offsets(
        self,
        layer_global_offset: Dict[int, int],
        node_to_folded: Dict[int, Tuple[int, int]],
    ) -> None:
        """Recompute layer_global_offset in folded_circuit.topological_sort() order
        and update product-layer gather indices so they match TensorizedCircuit's
        _node_flat_offsets ordering."""

        # Step 1: Compute correct offsets in fold-circuit topological order.
        # Input and sum layers contribute nodes to _node_flat_offsets; product layers don't.
        new_offset: Dict[int, int] = {}
        current = 0
        for fold_id in self.folded_circuit.topological_sort():
            fnode = self.folded_circuit.get_node_data(fold_id)
            if isinstance(fnode, (FoldedInputLayer, FoldedSumLayer)):
                new_offset[fold_id] = current
                current += fnode.num_nodes

        if new_offset == layer_global_offset:
            return  # Orderings already agree — nothing to patch

        # Step 2: Build reverse mapping fold_id → [(local_idx, sym_nid), ...].
        fold_to_syms: Dict[int, List[Tuple[int, int]]] = {}
        for sym_nid, (fold_id, local_idx) in node_to_folded.items():
            fold_to_syms.setdefault(fold_id, []).append((local_idx, sym_nid))

        adj = self.ac._adj

        # Step 3: Recompute left_indices / right_indices for every product fold layer.
        for fold_id in self.folded_circuit.topological_sort():
            fnode = self.folded_circuit.get_node_data(fold_id)
            if not isinstance(fnode, (FoldedHadamardProductLayer, FoldedKroneckerProductLayer)):
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

    def _resolve_leaf(self, node: LeafNode) -> tuple[LeafNode, int]:
        """Resolve a possibly-wrapped leaf into (base_gaussian, mode).

        Modes:
          0 = normal Gaussian (returns log_pdf)
          1 = constant (returns 0 everywhere)
          2 = inverse (returns -log_pdf)
          3 = indicator (returns 0 if in-support, -inf if out-of-bounds)
              Arises from Gaussian × InverseGaussian: log_pdf + (-log_pdf) = 0
              in-support, but -inf for out-of-support bins.
        """
        if isinstance(node, ProductLeafNode):
            base_a, mode_a = self._resolve_leaf(node.leaf_a)
            base_b, mode_b = self._resolve_leaf(node.leaf_b)
            # Constant + anything = anything (adding 0 in log domain is identity)
            if mode_a == 1:
                return base_b, mode_b
            if mode_b == 1:
                return base_a, mode_a
            # Normal + Inverse = indicator (cancels density but preserves OOB structure)
            if mode_a == 0 and mode_b == 2:
                return base_a, 3
            if mode_a == 2 and mode_b == 0:
                return base_a, 3
            # Indicator + normal = normal (indicator selects bin, Gaussian adds density)
            # Indicator + constant = indicator (adding 0 doesn't change indicator)
            # Indicator + inverse = constant (indicator 0 + (-log_pdf) cancels in-support;
            #   OOB entries stay -inf from indicator side)
            if mode_a == 3 and mode_b == 0:
                return base_b, 0
            if mode_a == 0 and mode_b == 3:
                return base_a, 0
            if mode_a == 3 and mode_b == 1:
                return base_a, 3
            if mode_a == 1 and mode_b == 3:
                return base_b, 3
            if mode_a == 3 and mode_b == 3:
                return base_a, 3
            return base_a, mode_a

        if isinstance(node, ConstantLeafNode):
            return node, 1
        if isinstance(node, InverseLeafNode):
            base, inner_mode = self._resolve_leaf(node.base_leaf)
            if inner_mode == 1:
                return base, 1  # inverse of constant is still constant
            if inner_mode == 0:
                return base, 2  # inverse of normal = inverse
            if inner_mode == 2:
                return base, 0  # inverse of inverse = normal
            return base, inner_mode
        # Regular distribution leaf
        return node, 0

    def _fold_leaf_layer(
        self,
        node_layer: List[int],
        node_to_folded: Dict[int, Tuple[int, int]],
        layer_global_offset: Dict[int, int],
        current_offset: int,
    ) -> None:
        gaussian_ids = []
        categorical_ids = []
        uniform_ids = []

        for nid in node_layer:
            node = self.ac.get_node_data(nid)
            # Estimand-compiled leaves (Constant, Inverse, Product) resolve to
            # Gaussian-compatible entries with a mode flag.
            if isinstance(node, (ConstantLeafNode, InverseLeafNode, ProductLeafNode)):
                gaussian_ids.append(nid)
            elif isinstance(node, (GaussianDistribution, TruncatedGaussianDistribution)):
                gaussian_ids.append(nid)
            elif isinstance(node, (CategoricalDistribution, TruncatedCategoricalDistribution)):
                categorical_ids.append(nid)
            elif isinstance(node, (UniformDistribution, TruncatedUniformDistribution)):
                uniform_ids.append(nid)

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

        if categorical_ids:
            logits = self._extract_categorical_params(categorical_ids)
            layer = FoldedCategoricalInputLayer(categorical_ids, logits)
            layer_id = self.allocator.next_id()
            layer_id = self.folded_circuit._add_node(layer)
            node_to_folded.update({nid: (layer_id, i) for i, nid in enumerate(categorical_ids)})

        if uniform_ids:
            lows, highs = self._extract_uniform_params(uniform_ids)
            layer = FoldedUniformInputLayer(uniform_ids, lows, highs)
            layer_id = self.allocator.next_id()
            layer_id = self.folded_circuit._add_node(layer)
            node_to_folded.update({nid: (layer_id, i) for i, nid in enumerate(uniform_ids)})

    def _fold_product_layer(
        self,
        node_layer: List[int],
        node_to_folded: Dict[int, Tuple[int, int]],
        layer_global_offset: Dict[int, int],
    ) -> None:
        hadamard_ids = []
        kronecker_ids = []

        for nid in node_layer:
            node = self.ac.get_node_data(nid)
            if isinstance(node, HadamardProductNode):
                hadamard_ids.append(nid)
            elif isinstance(node, KroneckerProductNode):
                kronecker_ids.append(nid)

        if hadamard_ids:
            self._fold_binary_product(
                hadamard_ids, node_to_folded, layer_global_offset, is_hadamard=True
            )

        if kronecker_ids:
            self._fold_binary_product(
                kronecker_ids, node_to_folded, layer_global_offset, is_hadamard=False
            )

    def _fold_binary_product(
        self,
        node_ids: List[int],
        node_to_folded: Dict[int, Tuple[int, int]],
        layer_global_offset: Dict[int, int],
        is_hadamard: bool,
    ) -> None:
        """Fold Hadamard or Kronecker product nodes into a single layer."""
        n = len(node_ids)
        global_left = [0] * n
        global_right = [0] * n
        edge_left: Dict[int, List[int]] = {}
        edge_right: Dict[int, List[int]] = {}

        adj = self.ac._adj
        for i, nid in enumerate(node_ids):
            children_dict = adj[nid]
            it = iter(children_dict)
            left_id = next(it)
            right_id = next(it)

            left_fold_id, left_idx = node_to_folded[left_id]
            right_fold_id, right_idx = node_to_folded[right_id]

            global_left[i] = layer_global_offset[left_fold_id] + left_idx
            global_right[i] = layer_global_offset[right_fold_id] + right_idx

            edge_left.setdefault(left_fold_id, []).append(left_idx)
            edge_right.setdefault(right_fold_id, []).append(right_idx)

        h_out = self.ac.get_node_data(node_ids[0]).unit_count
        first_node_children = self.ac.get_children(node_ids[0])
        h_left = self.ac.get_node_data(first_node_children[0]).unit_count
        h_right = self.ac.get_node_data(first_node_children[1]).unit_count

        layer_id = self.allocator.next_id()

        if is_hadamard:
            layer = FoldedHadamardProductLayer(
                num_nodes=len(node_ids),
                h_out=h_out,
                left_indices=global_left,
                right_indices=global_right,
                h_child=h_left,
            )
        else:
            layer = FoldedKroneckerProductLayer(
                num_nodes=len(node_ids),
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

        node_to_folded.update({nid: (layer_id, i) for i, nid in enumerate(node_ids)})

    def _fold_sum_layer(
        self,
        node_layer: List[int],
        node_to_folded: Dict[int, Tuple[int, int]],
        layer_global_offset: Dict[int, int],
        current_offset: int,
    ) -> None:
        """Fold sum nodes into typed sum layers (grouped by child-product type and arity)."""
        from collections import defaultdict

        # Group sum nodes by (child_type, arity, h_out).
        # h_out must be included so that nodes created at different h levels
        # (e.g. under h-decay, a mixing node's synthesizing children have h=64
        # while sibling nodes at the same topological depth may have h=32) are
        # placed in separate folded layers with the correct output dimension.
        groups = defaultdict(list)

        adj = self.ac._adj
        nodes_dict = self.ac._nodes
        for nid in node_layer:
            sum_node = nodes_dict[nid]
            h_out = sum_node.unit_count
            if isinstance(sum_node, UniversalSumNode):
                # Check if child is Hadamard (CPT) or Kronecker (Tucker)
                child_is_hadamard = False
                for child_id in adj[nid]:
                    child = nodes_dict[child_id]
                    if isinstance(child, HadamardProductNode):
                        child_is_hadamard = True
                        break
                child_type = "cpt" if child_is_hadamard else "universal"
            else:
                child_type = "unary"
                for child_id in adj[nid]:
                    child = nodes_dict[child_id]
                    if isinstance(child, HadamardProductNode):
                        child_type = "hadamard"
                        break
                    elif isinstance(child, KroneckerProductNode):
                        child_type = "kronecker"
                        break
            arity = len(adj[nid])
            groups[(child_type, arity, h_out)].append(nid)

        for (child_type, _arity, _h_out), sum_node_ids in groups.items():
            if child_type == "kronecker":
                self._create_sparse_kronecker_sum_layer(sum_node_ids, node_to_folded)
            elif child_type == "hadamard":
                self._create_sparse_hadamard_sum_layer(sum_node_ids, node_to_folded)
            elif child_type == "universal":
                self._create_tucker_sum_layer(sum_node_ids, node_to_folded)
            elif child_type == "cpt":
                self._create_cpt_sum_layer(sum_node_ids, node_to_folded)
            else:
                self._create_sparse_kronecker_sum_layer(sum_node_ids, node_to_folded)

    def _classify_sum_children(self, sum_node_id: int) -> str:
        """Classify a sum node by its children's product type or its own type."""
        sum_node = self.ac.get_node_data(sum_node_id)
        if isinstance(sum_node, UniversalSumNode):
            return "universal"

        for child_id in self.ac.get_children(sum_node_id):
            child = self.ac.get_node_data(child_id)
            if isinstance(child, HadamardProductNode):
                return "hadamard"
            elif isinstance(child, KroneckerProductNode):
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
        if hasattr(first_node, "log_weights"):
            import torch

            layer.log_weights = torch.stack(
                [self.ac.get_node_data(nid).log_weights for nid in sum_node_ids]
            )

        layer_id = self.allocator.next_id()
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

    def _create_sparse_hadamard_sum_layer(
        self,
        sum_node_ids: List[int],
        node_to_folded: Dict[int, Tuple[int, int]],
    ) -> None:
        """Create a FoldedSparseHadamardSumLayer for sums over Hadamard product children.

        Partition: h_child units are split into h_out groups of (h_child // h_out).
        Output unit i sums strictly over its group from every child.
        Normalized over (num_children, prods_per_sum).
        """
        num_nodes = len(sum_node_ids)
        h_out = self.ac.get_node_data(sum_node_ids[0]).unit_count
        h_child = self.ac.get_node_data(self.ac.get_children(sum_node_ids[0])[0]).unit_count
        h_in = h_child // h_out

        indices, fold_ids = self._collect_child_indices(sum_node_ids, node_to_folded)

        # Uniform initialization
        layer = FoldedSparseHadamardSumLayer(
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

    def _create_cpt_sum_layer(
        self,
        sum_node_ids: List[int],
        node_to_folded: Dict[int, Tuple[int, int]],
    ) -> None:
        """Create a FoldedCPTSumLayer for dense sums over Hadamard product children.

        Like Tucker but over Hadamard products (h units, not h*h).
        Each output unit connects to all h_child product units.
        """
        num_nodes = len(sum_node_ids)
        h_out = self.ac.get_node_data(sum_node_ids[0]).unit_count
        h_child = self.ac.get_node_data(self.ac.get_children(sum_node_ids[0])[0]).unit_count
        h_in = h_child  # Dense: no partition

        indices, fold_ids = self._collect_child_indices(sum_node_ids, node_to_folded)

        layer = FoldedCPTSumLayer(
            num_nodes=num_nodes,
            h_out=h_out,
            h_in=h_in,
            child_indices=indices,
            h_child=h_child,
        )
        self._register_sum_layer(layer, sum_node_ids, node_to_folded, fold_ids)
