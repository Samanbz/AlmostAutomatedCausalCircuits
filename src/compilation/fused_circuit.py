"""
Fused Circuit: intermediate representation where Product+Sum layer pairs are
fused into single tensor contraction layers (Tucker, CPT, DenseSum).

No forward pass is implemented here — this is a structural description that
the tensorized or monarch compiler consumes.
"""

from typing import Any, Dict, List, Tuple

import torch

from src.compilation.folded_circuit import (
    FoldedGaussianInputLayer,
    FoldedKroneckerProductLayer,
    FoldedSparseKroneckerSumLayer,
    FoldedSymbolicCircuit,
    FoldedTuckerSumLayer,
)
from src.logger import logger as g_logger
from src.symbolic.base import DirectedAcyclicGraph, Node

from .folded_circuit import FoldedLayer


logger = g_logger.getChild("CircuitFuser")


class FusedNode(Node):
    """Base class for all fused-circuit layers."""

    pass


class FusedInputLayer(FusedNode):
    """Leaf input layer — carries the same data as the folded input layer.

    Attributes:
        folded_layer: the original FoldedGaussianInputLayer (or other types)
            containing node_ids, means, stddevs, lows, highs, etc.
        scopes: List[int] — which input variable each node reads from.
    """

    def __init__(self, folded_layer: FoldedLayer, scopes: List[int]):
        self.folded_layer = folded_layer
        self.scopes = scopes
        self.num_nodes = folded_layer.num_nodes
        self.h_out = folded_layer.h_out

    @property
    def output_shape(self) -> Tuple[int, int]:
        return (self.num_nodes, self.h_out)


class SparseKroneckerLayer(FusedNode):
    """Fused Kronecker-Product + SparseKronecker-Sum layer.

    Attributes:
        left_indices:  [N, num_children] — global indices for left operands
        right_indices: [N, num_children] — global indices for right operands
        weights:       [N, h_out, num_children, prods_per_sum] — block-sparse sum weights
        h_out:         output hidden dimension
        h_child:       h_left * h_right (product units per child)
        h_left:        hidden dimension of left child
        h_right:       hidden dimension of right child
        num_nodes:     N
    """

    def __init__(
        self,
        left_indices: torch.Tensor,
        right_indices: torch.Tensor,
        num_nodes: int,
        h_out: int,
        h_in: int,
        h_left: int,
        h_right: int,
    ):
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.num_nodes = num_nodes
        self.h_out = h_out
        self.h_in = h_in
        self.h_left = h_left
        self.h_right = h_right

    @property
    def output_shape(self) -> Tuple[int, int]:
        return (self.num_nodes, self.h_out)


class SparseHadamardLayer(FusedNode):
    """Fused Hadamard-Product + SparseHadamard-Sum layer.

    Computes, for each of N sum nodes and each output unit i:
        out[i] = logsumexp_{ch}( L[ch, i] + R[ch, i] + W[i, ch] )

    Strict index alignment: output unit i only touches unit i from each child.

    Attributes:
        left_indices:  [N, num_children] — global indices for left operands
        right_indices: [N, num_children] — global indices for right operands
        weights:       [N, node_h, num_children, prods_per_sum] — mixing weights
        num_nodes:     N
    """

    def __init__(
        self,
        left_indices: torch.Tensor,
        right_indices: torch.Tensor,
        num_nodes: int,
        h_out: int,
        h_in: int,
        h_child: int,
    ):
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.num_nodes = num_nodes
        self.h_out = h_out
        self.h_in = h_in
        self.h_child = h_child

    @property
    def output_shape(self) -> Tuple[int, int]:
        return (self.num_nodes, self.h_out)


class CPTLayer(FusedNode):
    """Fused Hadamard-Product + Dense-Sum layer (Candecomp-transposed).

    Like SparseHadamard but with dense (non-partitioned) weights.
    Each output unit sums over ALL h_child Hadamard product units.
    """

    def __init__(
        self,
        left_indices: torch.Tensor,
        right_indices: torch.Tensor,
        num_nodes: int,
        h_out: int,
        h_child: int,
    ):
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.num_nodes = num_nodes
        self.h_out = h_out
        self.h_child = h_child

    @property
    def output_shape(self) -> Tuple[int, int]:
        return (self.num_nodes, self.h_out)


class TuckerLayer(FusedNode):
    """Fused Kronecker-Product + Tucker-Sum layer (Universal).

    Computes a dense sum over all Kronecker product units.
    """

    def __init__(
        self,
        left_indices: torch.Tensor,
        right_indices: torch.Tensor,
        num_nodes: int,
        h_out: int,
        h_left: int,
        h_right: int,
    ):
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.num_nodes = num_nodes
        self.h_out = h_out
        self.h_left = h_left
        self.h_right = h_right

    @property
    def output_shape(self) -> Tuple[int, int]:
        return (self.num_nodes, self.h_out)


class FusedCircuit(DirectedAcyclicGraph[int, FusedNode, Any]):
    """DAG of fused layers. No forward pass — consumed by a compiler."""

    def _get_base_node_config(self) -> Dict[type, Dict[str, Any]]:
        config = super()._get_base_node_config()

        def _label(tag: str):
            def fn(node: FusedNode) -> str:
                s = f"{tag}\n{node.output_shape}"
                if hasattr(node, "weights"):
                    s += f"\nW: {list(node.weights.shape)}"
                return s

            return fn

        config.update(
            {
                FusedInputLayer: {
                    "shape": "cylinder",
                    "color": "#b3ffb3",
                    "label": _label("Input"),
                },
                SparseKroneckerLayer: {
                    "shape": "box3d",
                    "color": "#ffb3b3",
                    "label": _label("SparseKron"),
                },
                SparseHadamardLayer: {
                    "shape": "box3d",
                    "color": "#ffcccc",
                    "label": _label("SparseHad"),
                },
                CPTLayer: {"shape": "box3d", "color": "#ffd9b3", "label": _label("CPT")},
                TuckerLayer: {"shape": "box3d", "color": "#b3b3ff", "label": _label("Tucker")},
            }
        )
        return config


class CircuitFuser:
    """Builds a FusedCircuit from a FoldedSymbolicCircuit.

    Walks the folded circuit in topological order and fuses consecutive
    (Product, Sum) layer pairs into single Tucker / CPT / DenseSum layers.
    """

    def __init__(self, folded: FoldedSymbolicCircuit):
        self.folded = folded
        self.fused = FusedCircuit()

    def build(self) -> FusedCircuit:
        # Map folded-layer id → fused-layer id
        folded_to_fused: Dict[int, int] = {}

        for fid in self.folded.topological_sort():
            fnode = self.folded.get_node_data(fid)

            if isinstance(fnode, FoldedGaussianInputLayer):
                self._fuse_input(fid, fnode, folded_to_fused)

            elif isinstance(
                fnode,
                (
                    FoldedSparseKroneckerSumLayer,
                    FoldedSparseHadamardSumLayer,
                    FoldedTuckerSumLayer,
                    FoldedCPTSumLayer,
                ),
            ):
                self._fuse_sum(fid, fnode, folded_to_fused)

            # Product layers and other input types are consumed by their parent sum
            # during _fuse_sum — no standalone fused node needed.

        return self.fused

    # ----- helpers ----------------------------------------------------------

    def _fuse_input(self, fid: int, fnode, folded_to_fused: Dict[int, int]) -> None:
        """Wrap a folded input layer into a FusedInputLayer."""
        scopes = getattr(fnode, "scopes", [])

        fused_layer = FusedInputLayer(fnode, scopes)
        fused_layer.md_sets = getattr(fnode, "md_sets", None)
        fused_id = self.fused._add_node(fused_layer)
        folded_to_fused[fid] = fused_id

    def _fuse_sum(self, fid: int, fnode, folded_to_fused: Dict[int, int]) -> None:
        """Fuse a Product + Sum pair into a SparseKronecker, SparseHadamard, or Tucker layer."""
        product_parents = self.folded.get_parents(fid)
        assert product_parents, f"Sum layer {fid} has no product-layer parents"

        prod_fid = product_parents[0]
        prod_layer = self.folded.get_node_data(prod_fid)

        if isinstance(fnode, FoldedSparseKroneckerSumLayer) and isinstance(
            prod_layer, FoldedKroneckerProductLayer
        ):
            self._fuse_sparse_kronecker(fid, fnode, prod_fid, prod_layer, folded_to_fused)

        elif isinstance(fnode, FoldedTuckerSumLayer) and isinstance(
            prod_layer, FoldedKroneckerProductLayer
        ):
            self._fuse_tucker(fid, fnode, prod_fid, prod_layer, folded_to_fused)

        else:
            raise ValueError(
                f"Unexpected product+sum combination: {type(prod_layer).__name__} + "
                f"{type(fnode).__name__}"
            )

    def _fuse_sparse_kronecker(
        self,
        sum_fid: int,
        sum_layer: FoldedSparseKroneckerSumLayer,
        prod_fid: int,
        prod_layer: FoldedKroneckerProductLayer,
        folded_to_fused: Dict[int, int],
    ) -> None:
        """Fuse KroneckerProduct + SparseKroneckerSum → SparseKroneckerLayer."""

        # Resolve the product layer's left/right sources back to fused layer ids
        left_global = torch.tensor(prod_layer.left_indices, dtype=torch.long)
        right_global = torch.tensor(prod_layer.right_indices, dtype=torch.long)

        # Build per-sum-node left/right index tensors
        # child_indices[n] gives the product-node indices for sum node n
        child_idx = torch.tensor(sum_layer.child_indices, dtype=torch.long)  # [N, num_children]
        left_per_sum = left_global[child_idx]  # [N, num_children]
        right_per_sum = right_global[child_idx]  # [N, num_children]

        if sum_layer.h_child != prod_layer.h_left * prod_layer.h_right:
            logger.error(
                f"Dimension mismatch for SparseKronecker fusion at sum {sum_fid} and product {prod_fid}: "
                f"h_child={sum_layer.h_child} vs h_left * h_right={prod_layer.h_left * prod_layer.h_right}"
            )
            raise ValueError("Cannot fuse layers with incompatible dimensions")

        layer = SparseKroneckerLayer(
            left_indices=left_per_sum,
            right_indices=right_per_sum,
            num_nodes=sum_layer.num_nodes,
            h_out=sum_layer.h_out,
            h_in=sum_layer.h_in,
            h_left=prod_layer.h_left,
            h_right=prod_layer.h_right,
        )
        if hasattr(sum_layer, "log_weights"):
            layer.log_weights = sum_layer.log_weights
        layer.md_sets = getattr(sum_layer, "md_sets", None)
        fused_id = self.fused._add_node(layer)

        # Connect to input fused layers
        self._connect_to_sources(prod_fid, fused_id, folded_to_fused)
        folded_to_fused[sum_fid] = fused_id

    def _fuse_sparse_hadamard(
        self,
        sum_fid: int,
        sum_layer: FoldedSparseHadamardSumLayer,
        prod_fid: int,
        prod_layer: FoldedHadamardProductLayer,
        folded_to_fused: Dict[int, int],
    ) -> None:
        """Fuse HadamardProduct + SparseHadamardSum → SparseHadamardLayer."""

        left_global = torch.tensor(prod_layer.left_indices, dtype=torch.long)
        right_global = torch.tensor(prod_layer.right_indices, dtype=torch.long)

        child_idx = torch.tensor(sum_layer.child_indices, dtype=torch.long)
        left_per_sum = left_global[child_idx]
        right_per_sum = right_global[child_idx]

        layer = SparseHadamardLayer(
            left_indices=left_per_sum,
            right_indices=right_per_sum,
            num_nodes=sum_layer.num_nodes,
            h_out=sum_layer.h_out,
            h_in=sum_layer.h_in,
            h_child=prod_layer.h_child,
        )
        if hasattr(sum_layer, "log_weights"):
            layer.log_weights = sum_layer.log_weights
        layer.md_sets = getattr(sum_layer, "md_sets", None)
        fused_id = self.fused._add_node(layer)

        self._connect_to_sources(prod_fid, fused_id, folded_to_fused)
        folded_to_fused[sum_fid] = fused_id

    def _fuse_tucker(
        self,
        sum_fid: int,
        sum_layer: FoldedTuckerSumLayer,
        prod_fid: int,
        prod_layer: FoldedKroneckerProductLayer,
        folded_to_fused: Dict[int, int],
    ) -> None:
        """Fuse KroneckerProduct + TuckerSum → TuckerLayer."""

        left_global = torch.tensor(prod_layer.left_indices, dtype=torch.long)
        right_global = torch.tensor(prod_layer.right_indices, dtype=torch.long)

        child_idx = torch.tensor(sum_layer.child_indices, dtype=torch.long)
        left_per_sum = left_global[child_idx]
        right_per_sum = right_global[child_idx]

        if sum_layer.h_child != prod_layer.h_left * prod_layer.h_right:
            logger.error(
                f"Dimension mismatch for Tucker fusion at sum {sum_fid} and product {prod_fid}: "
                f"h_child={sum_layer.h_child} vs h_left * h_right={prod_layer.h_left * prod_layer.h_right}"
            )
            raise ValueError("Cannot fuse layers with incompatible dimensions")

        layer = TuckerLayer(
            left_indices=left_per_sum,
            right_indices=right_per_sum,
            num_nodes=sum_layer.num_nodes,
            h_out=sum_layer.h_out,
            h_left=prod_layer.h_left,
            h_right=prod_layer.h_right,
        )
        if hasattr(sum_layer, "log_weights"):
            layer.log_weights = sum_layer.log_weights
        layer.md_sets = getattr(sum_layer, "md_sets", None)
        fused_id = self.fused._add_node(layer)

        self._connect_to_sources(prod_fid, fused_id, folded_to_fused)
        folded_to_fused[sum_fid] = fused_id

    def _fuse_cpt(
        self,
        sum_fid: int,
        sum_layer: FoldedCPTSumLayer,
        prod_fid: int,
        prod_layer: FoldedHadamardProductLayer,
        folded_to_fused: Dict[int, int],
    ) -> None:
        """Fuse HadamardProduct + CPTSum → CPTLayer (dense Hadamard)."""

        left_global = torch.tensor(prod_layer.left_indices, dtype=torch.long)
        right_global = torch.tensor(prod_layer.right_indices, dtype=torch.long)

        child_idx = torch.tensor(sum_layer.child_indices, dtype=torch.long)
        left_per_sum = left_global[child_idx]
        right_per_sum = right_global[child_idx]

        layer = CPTLayer(
            left_indices=left_per_sum,
            right_indices=right_per_sum,
            num_nodes=sum_layer.num_nodes,
            h_out=sum_layer.h_out,
            h_child=prod_layer.h_child,
        )
        if hasattr(sum_layer, "log_weights"):
            layer.log_weights = sum_layer.log_weights
        layer.md_sets = getattr(sum_layer, "md_sets", None)
        fused_id = self.fused._add_node(layer)

        self._connect_to_sources(prod_fid, fused_id, folded_to_fused)
        folded_to_fused[sum_fid] = fused_id

    def _connect_to_sources(
        self, prod_fid: int, fused_id: int, folded_to_fused: Dict[int, int]
    ) -> None:
        """Add edges from the fused source layers to this fused layer."""
        # The product layer's parents in the folded DAG are the input/sum layers
        for source_fid in self.folded.get_parents(prod_fid):
            if source_fid in folded_to_fused:
                src_fused = folded_to_fused[source_fid]
                if src_fused not in self.fused._adj or fused_id not in self.fused._adj.get(
                    src_fused, {}
                ):
                    self.fused._add_edge(src_fused, fused_id)
