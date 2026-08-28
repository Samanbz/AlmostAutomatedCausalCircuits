import math
import random
from enum import Enum
from typing import Dict

import torch

from src.logger import logger as g_logger
from src.symbolic.arithmetic.circuit import SymbolicArithmeticCircuit
from src.symbolic.arithmetic.nodes.leaf_layer import (
    CategoricalDistribution,
    Distribution,
    GaussianDistribution,
    MixtureLeafLayer,
)
from src.symbolic.arithmetic.nodes.sum_layer import SumLayer
from src.symbolic.arithmetic.weights import DenseWeights, SparseWeights
from src.symbolic.vtree import VTree
from src.utils import BitSet, Support
from src.utils.node_allocator import IncrementalNodeAllocator


logger = g_logger.getChild("CircuitBuilder")
logger.setLevel("ERROR")


def get_support(scope: BitSet, input_dists: Dict[int, Distribution]) -> Support:
    """Utility function to create a full support for a given scope."""
    intervals = {i: input_dists[i].support for i in scope}
    return Support(intervals)


class LayerType(Enum):
    UNIVERSAL = "UNIVERSAL"
    LEFT_MIXING = "LEFT_MIXING"
    RIGHT_MIXING = "RIGHT_MIXING"
    SYNTHESIZING = "SYNTHESIZING"


class CircuitBuilder:
    """Constructs a SymbolicArithmeticCircuit directly from a VTree."""

    def __init__(
        self,
        vtree: VTree,
        num_nodes: int,
        input_dists: Dict[int, Distribution] = None,
        initialize_weights: bool = False,
        fairness_temperature: float = 1.0,
    ):
        self.vtree = vtree
        self.num_nodes = num_nodes
        self.node_allocator = IncrementalNodeAllocator()
        self.input_dists = input_dists
        self.initialize_weights = initialize_weights
        self.fairness_temperature = fairness_temperature
        self.built_nodes = {}

    def _get_layer_type(self, vid: int) -> LayerType:
        vnode = self.vtree.get_node_data(vid)
        if vnode.md_set.is_universal:
            return LayerType.UNIVERSAL

        children = self.vtree.get_children(vid)
        assert len(children) == 2, "VTree must be binary"
        l_vnode = self.vtree.get_node_data(children[0])
        r_vnode = self.vtree.get_node_data(children[1])

        if l_vnode.md_set == vnode.md_set:
            return LayerType.LEFT_MIXING
        elif r_vnode.md_set == vnode.md_set:
            return LayerType.RIGHT_MIXING
        else:
            return LayerType.SYNTHESIZING

    def _get_child_dimensions(
        self,
        layer_type: LayerType,
        l_vid: int,
        r_vid: int,
        n: int,
        l_is_constrained: bool,
        r_is_constrained: bool,
    ):
        if layer_type == LayerType.UNIVERSAL:
            return n, 1, n, 1
        elif layer_type == LayerType.LEFT_MIXING:
            return 1, n, n, 1
        elif layer_type == LayerType.RIGHT_MIXING:
            return n, 1, 1, n
        elif layer_type == LayerType.SYNTHESIZING:
            a = 1
            for i in range(math.floor(math.sqrt(n)), 0, -1):
                if n % i == 0:
                    a = i
                    break
            b = n // a

            if a == b:
                return a, a, a, a

            l_child_type = (
                self._get_layer_type(l_vid)
                if not self.vtree.is_leaf(l_vid)
                else LayerType.UNIVERSAL
            )
            r_child_type = (
                self._get_layer_type(r_vid)
                if not self.vtree.is_leaf(r_vid)
                else LayerType.UNIVERSAL
            )

            l_is_synth_like = l_child_type == LayerType.SYNTHESIZING or (
                self.vtree.is_leaf(l_vid) and l_is_constrained
            )
            r_is_synth_like = r_child_type == LayerType.SYNTHESIZING or (
                self.vtree.is_leaf(r_vid) and r_is_constrained
            )

            if l_is_synth_like and not r_is_synth_like:
                favor_left = True
            elif r_is_synth_like and not l_is_synth_like:
                favor_left = False
            else:
                favor_left = True

            if favor_left:
                h_l, G_l = b, a
                G_r, h_r = b, a
            else:
                h_r, G_r = b, a
                h_l, G_l = a, b

            return G_l, h_l, G_r, h_r

    def _build_recursive(
        self,
        ac: SymbolicArithmeticCircuit,
        vid: int,
        G: int = 1,
        H: int = 1,
    ) -> int:
        if vid in self.built_nodes:
            return self.built_nodes[vid]

        vnode = self.vtree.get_node_data(vid)
        parent_vid = self.vtree.get_parent(vid)

        if parent_vid is None:  # root
            is_constrained = False
        else:
            parent_vnode = self.vtree.get_node_data(parent_vid)
            is_constrained = not vnode.md_set.is_universal and vnode.md_set.is_subset(
                parent_vnode.md_set
            )

        if self.vtree.is_leaf(vid):
            var_id = vnode.scope.min
            dist = self.input_dists[var_id]

            if not is_constrained and vnode.md_set.is_universal:
                leaf_supports = [dist.support] * H
            else:
                leaf_supports = [Support({var_id: iv}) for iv in dist.split_support(H)]

            if isinstance(dist, GaussianDistribution):
                from src.symbolic.arithmetic.nodes.leaf_layer import GaussianLeafLayer

                leaf_layer = GaussianLeafLayer(
                    dist,
                    num_nodes=H,
                    num_groups=G,
                    node_supports=leaf_supports if is_constrained else None,
                )
            elif isinstance(dist, CategoricalDistribution):
                from src.symbolic.arithmetic.nodes.leaf_layer import CategoricalLeafLayer

                leaf_layer = CategoricalLeafLayer(dist, num_nodes=H, num_groups=G)
            else:
                raise NotImplementedError(f"Unsupported distribution type: {type(dist)}")

            leaf_layer.node_supports = leaf_supports
            leaf_layer.md_set = vnode.md_set

            if not is_constrained:
                raw_weights = (
                    torch.randn(
                        (
                            G,
                            H,
                            G,
                            H,
                            1,
                            1,
                        )
                    )
                    * 0.1
                )
                for g in range(G):
                    raw_weights[g, :, g, :, :, :] += 0.5
                weights = torch.exp(raw_weights)
                weights = weights / weights.sum(dim=(2, 3, 4, 5), keepdim=True)
                log_weights = torch.log(weights + 1e-20).detach().requires_grad_(True)
                logger.debug(
                    f"Generated log weights for mixture leaf node {vid}:\n{log_weights.view(G, H, G, H)}"
                )
                mixture_leaf_layer = MixtureLeafLayer(
                    base_dist=leaf_layer, log_weights=DenseWeights(log_weights)
                )
                leaf_layer = mixture_leaf_layer

            leaf_id = ac._add_node(leaf_layer)
            ac.sum_to_vtree[leaf_id] = vid
            ac.vtree_to_sum[vid] = leaf_id
            self.built_nodes[vid] = leaf_id
            return leaf_id

        # Internal node logic
        layer_type = self._get_layer_type(vid)

        children = self.vtree.get_children(vid)
        assert len(children) == 2, "VTree must be binary"

        l_vnode = self.vtree.get_node_data(children[0])
        r_vnode = self.vtree.get_node_data(children[1])
        l_is_constrained = not l_vnode.md_set.is_universal and l_vnode.md_set.is_subset(
            vnode.md_set
        )
        r_is_constrained = not r_vnode.md_set.is_universal and r_vnode.md_set.is_subset(
            vnode.md_set
        )

        G_L, H_L, G_R, H_R = self._get_child_dimensions(
            layer_type, children[0], children[1], self.num_nodes, l_is_constrained, r_is_constrained
        )
        logger.debug(
            f"\nBuilding node {vid} with layer type {layer_type}, G={G}, H={H}, G_L={G_L}, H_L={H_L}, G_R={G_R}, H_R={H_R}"
        )

        l_id = self._build_recursive(ac, children[0], G=G_L, H=H_L)
        r_id = self._build_recursive(ac, children[1], G=G_R, H=H_R)

        l_node = ac.get_node_data(l_id)
        r_node = ac.get_node_data(r_id)

        if l_node.support is not None and r_node.support is not None:
            support = l_node.support.union(r_node.support)
        elif l_node.support is not None:
            support = l_node.support
        else:
            support = r_node.support

        sum_layer = SumLayer(
            support=support,
            num_nodes=H,
            num_groups=G,
            md_set=vnode.md_set,
        )

        logger.debug(f"Generating weights for node {vid} with scope {vnode.scope}")
        w = CircuitBuilder._generate_weights(
            G, H, G_L, H_L, G_R, H_R, layer_type, is_constrained, self.fairness_temperature
        )

        log_w = torch.log(w + 1e-20).detach().requires_grad_(True)
        sparse_flag = layer_type != LayerType.UNIVERSAL

        if sparse_flag:
            mask = (w > 0).float()
            sum_layer.log_weights = SparseWeights(log_w, mask=mask)
        else:
            sum_layer.log_weights = DenseWeights(log_w)

        sum_id = ac._add_node(sum_layer)
        ac._add_edge(sum_id, l_id)
        ac._add_edge(sum_id, r_id)

        ac.sum_to_vtree[sum_id] = vid
        ac.vtree_to_sum[vid] = sum_id
        self.built_nodes[vid] = sum_id
        return sum_id

    def build(self) -> SymbolicArithmeticCircuit:
        """Construct a SymbolicArithmeticCircuit directly from the VTree."""
        circuit = SymbolicArithmeticCircuit(vtree=self.vtree)
        root_vid = self.vtree.get_roots()[0]
        self._build_recursive(circuit, root_vid, G=1, H=1)
        return circuit

    @staticmethod
    def _sample_balanced_assignment(num_items: int, num_bins: int, temperature: float) -> list[int]:
        """Sample an assignment of items to bins biased toward balanced loads.

        The score for each bin is ``-temperature * (count / target)`` plus a small
        Gaussian jitter. ``temperature = 0`` recovers uniform random assignment;
        larger values make perfectly balanced assignments the most likely outcome
        while still allowing unbalanced ones.
        """
        if num_bins == 0:
            raise ValueError("num_bins must be positive")
        if num_items == 0:
            return []

        counts = [0] * num_bins
        target = num_items / num_bins
        assignments: list[int] = []

        for _ in range(num_items):
            scores = torch.tensor(
                [
                    -temperature * (counts[b] / target if target > 0 else 0.0)
                    + random.gauss(0.0, 1.0)
                    for b in range(num_bins)
                ],
                dtype=torch.float32,
            )
            probs = torch.softmax(scores, dim=0)
            sampled = int(torch.multinomial(probs, 1).item())
            assignments.append(sampled)
            counts[sampled] += 1

        return assignments

    @staticmethod
    def _generate_weights(
        G: int,
        H: int,
        G_L: int,
        H_L: int,
        G_R: int,
        H_R: int,
        layer_type: LayerType,
        is_constrained: bool,
        fairness_temperature: float = 2.0,
        weight_softmax_temperature: float = 0.5,
    ) -> torch.Tensor:
        """Generates global weight assignments ensuring determinism constraints."""

        w = torch.zeros((G, H, G_L, H_L, G_R, H_R))

        if layer_type == LayerType.UNIVERSAL:
            w = torch.exp(torch.randn((G, H, G_L, H_L, G_R, H_R)) * weight_softmax_temperature)

        elif layer_type == LayerType.LEFT_MIXING and not is_constrained:
            g_l_assignments = CircuitBuilder._sample_balanced_assignment(
                H_L, G_L, fairness_temperature
            )  # H_L -> G_L mapping
            logger.debug(list(enumerate(g_l_assignments)))
            for g in range(G):
                for n in range(H):
                    # dense over all left child nodes for given sum node and group
                    for i in range(H_L):
                        g_l = g_l_assignments[i]
                        # dense over the right nodes for given left node and group
                        w[g, n, g_l, i, :, :] = torch.exp(
                            torch.randn(G_R, H_R) * weight_softmax_temperature
                        )

        elif layer_type == LayerType.LEFT_MIXING and is_constrained:
            g_l_assignments = CircuitBuilder._sample_balanced_assignment(
                H_L, G_L, fairness_temperature
            )  # H_L -> G_L mapping
            h_l_assignments = CircuitBuilder._sample_balanced_assignment(
                H_L, H, fairness_temperature
            )  # H_L -> H mapping
            logger.debug(list(enumerate(g_l_assignments)))
            logger.debug(list(enumerate(h_l_assignments)))
            for g in range(G):
                for i, n in enumerate(h_l_assignments):
                    # i = left child node index, n = assigned sum node index
                    g_l = g_l_assignments[i]
                    # dense over the right nodes for given left node and group
                    w[g, n, g_l, i, :, :] = torch.exp(
                        torch.randn(G_R, H_R) * weight_softmax_temperature
                    )

        elif layer_type == LayerType.RIGHT_MIXING and not is_constrained:
            g_r_assignments = CircuitBuilder._sample_balanced_assignment(
                H_R, G_R, fairness_temperature
            )  # H_R -> G_R mapping
            logger.debug(list(enumerate(g_r_assignments)))
            for g in range(G):
                for n in range(H):
                    # dense over all right child nodes for given sum node and group
                    for i in range(H_R):
                        g_r = g_r_assignments[i]
                        # dense over the left nodes for given right node and group
                        w[g, n, :, :, g_r, i] = torch.exp(
                            torch.randn(G_L, H_L) * weight_softmax_temperature
                        )

        elif layer_type == LayerType.RIGHT_MIXING and is_constrained:
            g_r_assignments = CircuitBuilder._sample_balanced_assignment(
                H_R, G_R, fairness_temperature
            )  # H_R -> G_R mapping
            h_r_assignments = CircuitBuilder._sample_balanced_assignment(
                H_R, H, fairness_temperature
            )  # H_R -> H mapping
            logger.debug(list(enumerate(g_r_assignments)))
            logger.debug(list(enumerate(h_r_assignments)))
            for g in range(G):
                for i, n in enumerate(h_r_assignments):
                    # i = right child node index, n = assigned sum node index
                    g_l = i
                    g_r = g_r_assignments[i]
                    # dense over the left nodes for given right node and group
                    w[g, n, :, :, g_r, i] = torch.exp(
                        torch.randn(G_L, H_L) * weight_softmax_temperature
                    )

        elif layer_type == LayerType.SYNTHESIZING and not is_constrained:
            # Here we have the relationship H_L=G_R H_R=G_L
            for g in range(G):
                for n in range(H):
                    # dense over all combinations of left and right child nodes for given sum node and group
                    for i in range(H_L):
                        for k in range(H_R):
                            g_l = k  # right node index determines left node group
                            g_r = i  # left node index determines right node group)
                            w[g, n, g_l, i, g_r, k] = torch.exp(
                                torch.randn(1) * weight_softmax_temperature
                            )

        elif layer_type == LayerType.SYNTHESIZING and is_constrained:
            # assign each combination of left and right child nodes to a unique sum node
            flat_assignments = CircuitBuilder._sample_balanced_assignment(
                H_L * H_R, H, fairness_temperature
            )
            assignments = [flat_assignments[i * H_R : (i + 1) * H_R] for i in range(H_L)]
            logger.debug(f"Assignments for SYNTHESIZING constrained: {assignments}")
            for g in range(G):
                for i, row in enumerate(assignments):
                    for k, n in enumerate(row):
                        g_l = k  # right node index determines left node group
                        g_r = i  # left node index determines right node group
                        w[g, n, g_l, i, g_r, k] = torch.exp(
                            torch.randn(1) * weight_softmax_temperature
                        )

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        w_sum = w.sum(dim=(2, 3, 4, 5), keepdim=True)

        safe_w_sum = torch.where(w_sum == 0, torch.ones_like(w_sum), w_sum)
        w = w / safe_w_sum

        for g in range(G):
            for n in range(H):
                logger.debug(
                    f"LayerType={layer_type}, is_constrained={is_constrained}, g={g}, n={n}, w[g,n,:,:,:, :]=\n{w[g, n, :, :, :, :].view(G_L * H_L, G_R * H_R)}"
                )

        return w


def create_md_circuit(
    input_dists: Dict[int, Distribution],
    md_var_decomp: VTree,
    num_nodes: int,
    initialize_weights: bool = False,
    fairness_temperature: float = 1.0,
) -> SymbolicArithmeticCircuit:
    """Creates an MD-Circuit end-to-end."""

    c_builder = CircuitBuilder(
        vtree=md_var_decomp,
        num_nodes=num_nodes,
        input_dists=input_dists,
        initialize_weights=initialize_weights,
        fairness_temperature=fairness_temperature,
    )
    return c_builder.build()
