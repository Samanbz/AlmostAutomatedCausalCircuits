from typing import Dict, Optional

import torch

from src.construction.region_graph_builder import RegionGraphBuilder
from src.logger import logger as g_logger
from src.symbolic.arithmetic import (
    Distribution,
    HadamardProductNode,
    KroneckerProductNode,
    SumNode,
    SymbolicArithmeticCircuit,
    UniversalSumNode,
)
from src.symbolic.region_graph import (
    LayerType,
    RegionGraph,
    RegionNode,
)
from src.symbolic.vtree import VTree
from src.utils import BitSet, Support
from src.utils.node_allocator import IncrementalNodeAllocator


logger = g_logger.getChild("CircuitBuilder")


def get_support(scope: BitSet, input_dists: Dict[int, Distribution]) -> Support:
    """Utility function to create a full support for a given scope."""
    intervals = {i: input_dists[i].support for i in scope}
    return Support(intervals)


class CircuitBuilder:
    """Build a Marginally-Deterministic arithmetic circuit from a region graph.

    Layer types by md-vtree node kind
    ----------------------------------
    **Mixing** (LEFT_MIXING or RIGHT_MIXING):
        Always HadamardProductNode.  h expands toward the leaves by decay_factor
        (h_child = h_here × decay_factor), so each sum node aggregates
        decay_factor product nodes.  One child may be MD-sparse (single active
        unit per sample) while the other is dense; diagonal pairing still works
        because output[k] = L[k] + R[k] inherits -inf from the sparse side for
        all k except the active bin.

        - Constrained (ψ(m) ⊆ ψ(parent)): SparseSum enforces disjoint support.
        - Unconstrained: UniversalSumNode (dense CPT, maximum expressivity).

    **Synthesizing / non-mixing** (SYNTHESIZING or UNIVERSAL):
        KroneckerProductNode at h_min.  Both children are MD-sparse with
        independently-determined active indices, so the full outer product
        (h_min²) is required to preserve the single valid (k_L, k_R) cell.

        - Constrained: SparseSum — block-diagonal compression h_min² → h_min.
        - Unconstrained / UNIVERSAL: UniversalSumNode (Tucker CPT).
    """

    def __init__(
        self,
        rg: RegionGraph,
        h: int,
        input_dists: Dict[int, Distribution] = None,
        h_max: Optional[int] = None,
        decay_factor: int = 2,
        initialize_weights: bool = False,
    ):
        self.rg = rg
        self.h_min = h
        self.h_max = h_max if h_max is not None else h
        self.h = h  # backward-compat alias
        self.decay_factor = decay_factor
        self.node_allocator = IncrementalNodeAllocator()
        self.input_dists = input_dists
        self.initialize_weights = initialize_weights

    def _child_h(self, h_here: int, is_mixing: bool) -> int:
        """Return the h that children of a Hadamard mixing node must produce."""
        if is_mixing:
            target = h_here * self.decay_factor
            if target <= self.h_max:
                return target
            # If target > h_max, find the largest multiple of h_here <= h_max
            # to maintain clean block-diagonal structure.
            best_h = (self.h_max // h_here) * h_here
            return max(h_here, best_h)
        return self.h_min

    def _build_recursive(
        self,
        ac: SymbolicArithmeticCircuit,
        rg_id: int,
        h_out: Optional[int] = None,
        parent_rg_id: int = None,
    ) -> int:
        rg_node: RegionNode = self.rg._nodes[rg_id]
        h_here = h_out if h_out is not None else self.h_min

        if not self.rg._adj[rg_id]:  # Leaf region
            var_id = rg_node.scope.min
            dist = self.input_dists[var_id]

            if rg_node.is_constrained:
                splits = dist.split(h_here)
                leaf_supports = [s.support for s in splits]
            else:
                leaf_supports = [rg_node.support] * h_here

            cls = type(dist)
            leaf_node = cls.__new__(cls)
            leaf_node.__dict__ = dist.__dict__.copy()
            leaf_node.unit_count = h_here
            leaf_node.unit_supports = leaf_supports
            leaf_node.md_set = rg_node.md_set
            leaf_id = ac._add_node(leaf_node)

            return leaf_id

        is_mixing = rg_node.layer_type in {LayerType.LEFT_MIXING, LayerType.RIGHT_MIXING}
        is_universal = rg_node.layer_type == LayerType.UNIVERSAL

        part_id = next(iter(self.rg._adj[rg_id]))
        p_children_dict = self.rg._adj[part_id]
        n_children = len(p_children_dict)

        assert n_children == 2, (
            f"Expected exactly 2 children for partition node {part_id}, got {n_children}"
        )

        it = iter(p_children_dict)
        l_region_id = next(it)
        r_region_id = next(it)

        if is_mixing:
            h_child = self._child_h(h_here, is_mixing=True)
            node_type = HadamardProductNode
            num_units = h_child
        else:
            # Non-mixing (Synthesizing): full Kronecker outer product at h_min.
            h_child = self.h_min
            node_type = KroneckerProductNode
            num_units = h_child * h_child

        l_child_id = self._build_recursive(ac, l_region_id, h_out=h_child, parent_rg_id=rg_id)
        r_child_id = self._build_recursive(ac, r_region_id, h_out=h_child, parent_rg_id=rg_id)

        prod_node = node_type(support=rg_node.support, unit_count=num_units)
        prod_id = ac._add_node(prod_node)
        ac._add_edge(prod_id, l_child_id)
        ac._add_edge(prod_id, r_child_id)

        # Ensure h_in is a multiple of h_out for regular SumNode
        if is_universal or not rg_node.is_constrained:
            sum_node = UniversalSumNode(
                support=rg_node.support, unit_count=h_here, md_set=rg_node.md_set
            )
            if self.initialize_weights:
                sum_node.weights = torch.rand(h_here, num_units)
                sum_node.weights = sum_node.weights / sum_node.weights.sum(dim=-1, keepdim=True)
        else:
            sum_node = SumNode(support=rg_node.support, unit_count=h_here, md_set=rg_node.md_set)
            if self.initialize_weights:
                h_in = num_units // h_here
                sum_node.weights = torch.rand(h_here, h_in)
                sum_node.weights = sum_node.weights / sum_node.weights.sum(dim=-1, keepdim=True)

        sum_id = ac._add_node(sum_node)
        ac._add_edge(sum_id, prod_id)

        return sum_id

    def build(self) -> SymbolicArithmeticCircuit:
        """Construct a blockified SymbolicArithmeticCircuit from the region graph."""
        circuit = SymbolicArithmeticCircuit()

        rg_root = self.rg.get_roots()
        assert len(rg_root) == 1, f"Expected exactly one root region, got {len(rg_root)}"

        root_id = self._build_recursive(circuit, rg_root[0], h_out=1)
        return circuit


def create_md_circuit(
    input_dists: Dict[int, Distribution],
    md_var_decomp: VTree,
    h: int,
    h_max: Optional[int] = None,
    decay_factor: int = 2,
    initialize_weights: bool = False,
) -> SymbolicArithmeticCircuit:
    """Creates an MD-Circuit end-to-end."""
    rg_builder = RegionGraphBuilder(input_dists, md_var_decomp)
    rg = rg_builder.build()

    c_builder = CircuitBuilder(
        rg=rg,
        h=h,
        input_dists=input_dists,
        h_max=h_max,
        decay_factor=decay_factor,
        initialize_weights=initialize_weights,
    )
    return c_builder.build()
