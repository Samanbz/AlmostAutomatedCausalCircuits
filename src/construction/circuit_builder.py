import random
from typing import Dict, Optional

import numpy as np
import torch

from src.construction.region_graph_builder import RegionGraphBuilder
from src.logger import logger as g_logger
from src.symbolic.arithmetic import (
    Distribution,
    GaussianMixture,
    KroneckerProductNode,
    SumNode,
    SymbolicArithmeticCircuit,
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
logger.setLevel("DEBUG")


def get_support(scope: BitSet, input_dists: Dict[int, Distribution]) -> Support:
    """Utility function to create a full support for a given scope."""
    intervals = {i: input_dists[i].support for i in scope}
    return Support(intervals)


class CircuitBuilder:
    """Build a Marginally-Deterministic arithmetic circuit from a region graph.

    This builder maintains a constant hidden dimension 'h' for all internal Sum
    and Leaf nodes, except for the root node which always has unit_count=1.
    """

    def __init__(
        self,
        rg: RegionGraph,
        leaf_h: int,
        sum_h: int,
        input_dists: Dict[int, Distribution] = None,
        initialize_weights: bool = False,
    ):
        self.rg = rg
        self.leaf_h = leaf_h
        self.sum_h = sum_h
        self.node_allocator = IncrementalNodeAllocator()
        self.input_dists = input_dists
        self.initialize_weights = initialize_weights

    def _build_recursive(
        self,
        ac: SymbolicArithmeticCircuit,
        rg_id: int,
        h_out: Optional[int] = None,
    ) -> int:
        rg_node: RegionNode = self.rg._nodes[rg_id]

        h_here = h_out if h_out is not None else self.sum_h
        vtree_id = self.rg.region_to_vtree[rg_id]

        if not self.rg._adj[rg_id]:  # Leaf region
            var_id = rg_node.scope.min
            dist = self.input_dists[var_id]

            if not rg_node.is_constrained and rg_node.md_set.is_universal:
                leaf_supports = [rg_node.support] * self.leaf_h
            else:
                leaf_supports = [
                    Support({var_id: iv})
                    for iv in dist.split_support(self.leaf_h, strategy="perturbed_quantile")
                ]

            leaf_type = type(dist)
            leaf_node = leaf_type.__new__(leaf_type)
            leaf_node.__dict__ = dist.__dict__.copy()
            leaf_node.unit_count = self.leaf_h
            leaf_node.unit_supports = leaf_supports
            leaf_node.md_set = rg_node.md_set

            # If the leaf node has parameters (e.g. Gaussian), initialize them as independent vectors
            if hasattr(leaf_node, "mean") and hasattr(leaf_node, "stddev"):
                # We start with the global standard deviation for all units
                base_std = float(torch.as_tensor(dist.stddev).detach().mean())
                std_vec = torch.full(
                    (1, self.leaf_h), base_std, dtype=torch.float32
                ).requires_grad_(True)

                # For means, if constrained, we try to place the mean within the interval to avoid dead units
                means = []
                for i in range(self.leaf_h):
                    if rg_node.is_constrained:
                        iv = leaf_supports[i].intervals[var_id]
                        if iv.low > float("-inf") and iv.high < float("inf"):
                            m = (iv.low + iv.high) / 2.0
                        elif iv.low > float("-inf"):
                            m = iv.low + base_std
                        elif iv.high < float("inf"):
                            m = iv.high - base_std
                        else:
                            m = float(torch.as_tensor(dist.mean).detach().mean())
                    else:
                        base_m = float(torch.as_tensor(dist.mean).detach().mean())
                        m = base_m + float(np.random.uniform(-base_std, base_std))
                    means.append(m)

                mean_vec = torch.tensor([means], dtype=torch.float32).requires_grad_(True)

                leaf_node.mean = mean_vec
                leaf_node.stddev = std_vec

            if not rg_node.is_constrained:
                diagonal_weight = 0.95
                off_diagonal_weight = (
                    (1.0 - diagonal_weight) / max(1, self.leaf_h - 1) if self.leaf_h > 1 else 0.0
                )
                probs = (
                    torch.eye(self.leaf_h) * diagonal_weight
                    + (1 - torch.eye(self.leaf_h)) * off_diagonal_weight
                )
                log_weights = torch.log(probs).requires_grad_(True)
                mixture_leaf_node = GaussianMixture(base_dist=leaf_node, log_weights=log_weights)
                leaf_node = mixture_leaf_node

            leaf_id = ac._add_node(leaf_node)
            ac.sum_to_vtree[leaf_id] = vtree_id
            ac.vtree_to_sum.setdefault(vtree_id, []).append(leaf_id)
            return leaf_id

        is_mixing = rg_node.layer_type in {LayerType.LEFT_MIXING, LayerType.RIGHT_MIXING}
        is_universal = rg_node.layer_type == LayerType.UNIVERSAL

        part_id = next(iter(self.rg._adj[rg_id]))
        p_children_dict = self.rg._adj[part_id]

        it = iter(p_children_dict)
        l_region_id = next(it)
        r_region_id = next(it)

        node_type = KroneckerProductNode
        l_child_id = self._build_recursive(ac, l_region_id)
        r_child_id = self._build_recursive(ac, r_region_id)

        l_child = ac.get_node_data(l_child_id)
        r_child = ac.get_node_data(r_child_id)
        num_units = l_child.unit_count * r_child.unit_count

        prod_node = node_type(support=rg_node.support, unit_count=num_units)
        prod_id = ac._add_node(prod_node)
        ac._add_edge(prod_id, l_child_id)
        ac._add_edge(prod_id, r_child_id)

        sum_node = SumNode(
            support=rg_node.support,
            unit_count=h_here,
            md_set=rg_node.md_set,
            sparse=rg_node.is_constrained or is_mixing,
        )
        logger.info(
            f"Building SumNode {sum_node} with scope {rg_node.scope}, layer_type={rg_node.layer_type}, is_constrained={rg_node.is_constrained}"
        )
        w = CircuitBuilder._generate_weights(
            h_here,
            l_child.unit_count,
            r_child.unit_count,
            rg_node.layer_type,
            rg_node.is_constrained,
        )

        w = w.reshape(h_here, num_units)
        # Handle cases where a row might be completely zero due to h_here > num_combs
        row_sums = w.sum(dim=-1, keepdim=True)
        row_sums[row_sums == 0] = 1.0  # Prevent division by zero
        w = w / row_sums

        sum_node.log_weights = torch.log(w + 1e-20).detach().requires_grad_(True)

        sum_id = ac._add_node(sum_node)
        ac._add_edge(sum_id, prod_id)

        ac.sum_to_vtree[sum_id] = vtree_id
        ac.vtree_to_sum.setdefault(vtree_id, []).append(sum_id)

        return sum_id

    def build(self) -> SymbolicArithmeticCircuit:
        """Construct a blockified SymbolicArithmeticCircuit from the region graph."""
        circuit = SymbolicArithmeticCircuit(vtree=self.rg.vtree)

        rg_root = self.rg.get_roots()
        assert len(rg_root) == 1, f"Expected exactly one root region, got {len(rg_root)}"

        self._build_recursive(circuit, rg_root[0], h_out=1)
        return circuit

    @staticmethod
    def _generate_weights(
        h_sum: int, h_l: int, h_r: int, layer_type: LayerType, is_constrained: bool
    ) -> torch.Tensor:
        """Generates sparse weight assignments ensuring determinism constraints for mixing layers."""

        logger.info(
            f"_generate_weights: layer_type={layer_type}, is_constrained={is_constrained}, h_sum={h_sum}, h_l={h_l}, h_r={h_r}"
        )

        w = torch.zeros((h_sum, h_l, h_r))

        if layer_type == LayerType.UNIVERSAL:
            if is_constrained:
                raise ValueError("UNIVERSAL layer cannot be constrained")
            else:
                logger.info("UNIVERSAL: unconstrained case")
                w = torch.rand((h_sum, h_l, h_r))

        elif layer_type == LayerType.LEFT_MIXING:
            if is_constrained:
                logger.info("LEFT_MIXING: constrained case")
                j_list = list(range(h_l))
                random.shuffle(j_list)
                k_pool = []
                while len(k_pool) < h_l:
                    batch = list(range(h_r))
                    random.shuffle(batch)
                    k_pool.extend(batch)
                for idx, j in enumerate(j_list):
                    i = idx % h_sum
                    k = k_pool[idx]
                    w[i, j, k] = torch.rand(1).item()
            else:
                logger.info("LEFT_MIXING: unconstrained case")
                for i in range(h_sum):
                    for j in range(h_l):
                        k = random.randint(0, h_r - 1)
                        w[i, j, k] = torch.rand(1).item()

        elif layer_type == LayerType.RIGHT_MIXING:
            if is_constrained:
                logger.info("RIGHT_MIXING: constrained case")
                k_list = list(range(h_r))
                random.shuffle(k_list)
                j_pool = []
                while len(j_pool) < h_r:
                    batch = list(range(h_l))
                    random.shuffle(batch)
                    j_pool.extend(batch)
                for idx, k in enumerate(k_list):
                    i = idx % h_sum
                    j = j_pool[idx]
                    w[i, j, k] = torch.rand(1).item()
            else:
                logger.info("RIGHT_MIXING: unconstrained case")
                for i in range(h_sum):
                    for k in range(h_r):
                        j = random.randint(0, h_l - 1)
                        w[i, j, k] = torch.rand(1).item()

        elif layer_type == LayerType.SYNTHESIZING:
            if is_constrained:
                logger.info("SYNTHESIZING: constrained case")
                pairs = [(j, k) for j in range(h_l) for k in range(h_r)]
                random.shuffle(pairs)
                for idx, (j, k) in enumerate(pairs):
                    i = idx % h_sum
                    w[i, j, k] = torch.rand(1).item()
            else:
                logger.info("SYNTHESIZING: unconstrained case")
                w = torch.rand((h_sum, h_l, h_r))
        else:
            logger.info(f"Default case: unknown layer_type {layer_type}")
            w = torch.rand((h_sum, h_l, h_r))

        return w


def create_md_circuit(
    input_dists: Dict[int, Distribution],
    md_var_decomp: VTree,
    leaf_h: int,
    sum_h: int,
    initialize_weights: bool = False,
) -> SymbolicArithmeticCircuit:
    """Creates an MD-Circuit end-to-end."""
    rg_builder = RegionGraphBuilder(input_dists, md_var_decomp)
    rg = rg_builder.build()

    c_builder = CircuitBuilder(
        rg=rg,
        leaf_h=leaf_h,
        sum_h=sum_h,
        input_dists=input_dists,
        initialize_weights=initialize_weights,
    )
    return c_builder.build()
