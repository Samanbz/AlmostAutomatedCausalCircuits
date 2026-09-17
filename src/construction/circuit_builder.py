import math
import random
from enum import Enum
from typing import Dict, Optional

import torch

from src.logger import logger as g_logger
from src.symbolic.arithmetic.circuit import SymbolicArithmeticCircuit
from src.symbolic.arithmetic.nodes.leaf_layer import (
    CategoricalDistribution,
    Distribution,
    GaussianDistribution,
    GaussianLeafLayer,
    LinearSplineDistribution,
    LinearSplineLeafLayer,
    LogLinearSplineDistribution,
    LogLinearSplineLeafLayer,
    MixtureLeafLayer,
    QuadraticSplineDistribution,
    QuadraticSplineLeafLayer,
    RationalQuadraticSplineDistribution,
    RationalQuadraticSplineLeafLayer,
)
from src.symbolic.arithmetic.nodes.sum_layer import SumLayer
from src.symbolic.arithmetic.weights import DenseWeights, SparseWeights
from src.symbolic.vtree import VNode, VTree
from src.utils import BitSet, Support
from src.utils.node_allocator import IncrementalNodeAllocator


logger = g_logger.getChild("CircuitBuilder")
logger.setLevel("ERROR")


def find_closest_factor(n: int, max_val: int = None) -> int:
    """Find the largest factor of n that is less than or equal to sqrt(n)."""
    max_val = math.floor(max_val)
    if max_val > n:
        raise ValueError("max_val must be less than or equal to n")
    for i in range(max_val, 0, -1):
        if n % i == 0:
            return i
    return 1


def get_support(scope: BitSet, input_dists: Dict[int, Distribution]) -> Support:
    """Utility function to create a full support for a given scope."""
    intervals = {i: input_dists[i].support for i in scope}
    return Support(intervals)


class LayerType(Enum):
    UNIVERSAL = "UNIVERSAL"
    LEFT_MIXING = "LEFT_MIXING"
    PS_LEFT_MIXING = "PS_LEFT_MIXING"
    RIGHT_MIXING = "RIGHT_MIXING"
    PS_RIGHT_MIXING = "PS_RIGHT_MIXING"
    SYNTHESIZING = "SYNTHESIZING"


class CircuitBuilder:
    """Constructs a SymbolicArithmeticCircuit directly from a VTree."""

    def __init__(
        self,
        vtree: VTree,
        num_nodes: int,
        input_dists: Dict[int, Distribution] = None,
        initialize_weights: bool = False,
        weight_softmax_temperature: float = 0.5,
        max_leaf_num_nodes: Optional[int] = None,
        max_leaf_num_groups: Optional[int] = None,
        max_sum_num_groups: Optional[int] = None,
        leaf_mixture_num_nodes: Optional[int] = None,
        leaf_mixture_num_groups: Optional[int] = None,
        fairness_temperature: float = 1.0,
    ):
        self.vtree = vtree
        self.num_nodes = num_nodes
        self.input_dists = input_dists
        self.node_allocator = IncrementalNodeAllocator()

        self.weight_softmax_temperature = weight_softmax_temperature
        self.initialize_weights = initialize_weights
        self.fairness_temperature = fairness_temperature

        self.max_leaf_num_nodes = max_leaf_num_nodes
        self.max_leaf_num_groups = max_leaf_num_groups
        self.max_sum_num_groups = max_sum_num_groups
        self.leaf_mixture_num_nodes = leaf_mixture_num_nodes
        self.leaf_mixture_num_groups = leaf_mixture_num_groups

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
            return (
                LayerType.LEFT_MIXING if r_vnode.md_set.is_universal else LayerType.PS_LEFT_MIXING
            )
        elif r_vnode.md_set == vnode.md_set:
            return (
                LayerType.RIGHT_MIXING if l_vnode.md_set.is_universal else LayerType.PS_RIGHT_MIXING
            )
        else:
            return LayerType.SYNTHESIZING

    def _handle_leaf(
        self,
        vnode: VNode,
        G: int,
        H: int,
        is_constrained: bool,
    ):
        var_id = vnode.scope.min
        dist = self.input_dists[var_id]

        G_b = (
            G
            if (is_constrained or self.leaf_mixture_num_groups is None)
            else self.leaf_mixture_num_groups
        )
        H_b = (
            H
            if (is_constrained or self.leaf_mixture_num_nodes is None)
            else self.leaf_mixture_num_nodes
        )

        if not is_constrained and vnode.md_set.is_universal:
            leaf_supports = [dist.support] * H_b
        else:
            leaf_supports = [Support({var_id: iv}) for iv in dist.split_support(H_b)]

        if isinstance(dist, GaussianDistribution) or len(leaf_supports) == 1:
            base_mean = float(getattr(dist, "base_mean", 0.0))
            base_stddev = float(getattr(dist, "base_stddev", 1.0))
            if base_stddev <= 0:
                base_stddev = 1.0
            leaf_layer = GaussianLeafLayer(
                GaussianDistribution(var=var_id, base_mean=base_mean, base_stddev=base_stddev),
                num_nodes=H_b,
                num_groups=G_b,
                node_supports=leaf_supports if is_constrained else None,
            )
        elif isinstance(dist, CategoricalDistribution):
            from src.symbolic.arithmetic.nodes.leaf_layer import CategoricalLeafLayer

            leaf_layer = CategoricalLeafLayer(dist, num_nodes=H, num_groups=G)
        elif is_constrained and isinstance(dist, LogLinearSplineDistribution):
            leaf_layer = LogLinearSplineLeafLayer(
                dist, num_nodes=H_b, num_groups=G_b, node_supports=leaf_supports
            )
        elif is_constrained and isinstance(dist, LinearSplineDistribution):
            leaf_layer = LinearSplineLeafLayer(
                dist, num_nodes=H_b, num_groups=G_b, node_supports=leaf_supports
            )
        elif is_constrained and isinstance(dist, QuadraticSplineDistribution):
            leaf_layer = QuadraticSplineLeafLayer(
                dist, num_nodes=H_b, num_groups=G_b, node_supports=leaf_supports
            )
        elif is_constrained and isinstance(dist, RationalQuadraticSplineDistribution):
            leaf_layer = RationalQuadraticSplineLeafLayer(
                dist, num_nodes=H_b, num_groups=G_b, node_supports=leaf_supports
            )
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
                        G_b,
                        H_b,
                        1,
                        1,
                    )
                )
                * 0.1
            )
            for g in range(G_b):
                raw_weights[g % G, :, g, :, :, :] += 0.5
            weights = torch.exp(raw_weights)
            weights = weights / weights.sum(dim=(2, 3, 4, 5), keepdim=True)
            log_weights = torch.log(weights + 1e-20).detach().requires_grad_(True)
            logger.debug(
                f"Generated log weights for mixture leaf node with scope {vnode.scope}:\n{log_weights.view(G, H, G_b, H_b)}"
            )
            mixture_leaf_layer = MixtureLeafLayer(
                base_dist=leaf_layer, log_weights=DenseWeights(log_weights)
            )
            leaf_layer = mixture_leaf_layer

        return leaf_layer

    def _sample_balanced_assignment(
        self, num_items: int, num_bins: int, temperature: float
    ) -> list[list[int]]:
        """Sample an assignment of items to bins biased toward balanced loads.

        The score for each bin is ``-temperature * (count / target)`` plus a small
        Gaussian jitter. ``temperature = 0`` recovers uniform random assignment;
        larger values make perfectly balanced assignments the most likely outcome
        while still allowing unbalanced ones.
        """
        if num_bins == 0:
            raise ValueError("num_bins must be positive")
        if num_items == 0:
            return [[] for _ in range(num_bins)]

        counts = [0] * num_bins
        target = num_items / num_bins
        assignments: list[list[int]] = [[] for _ in range(num_bins)]

        for item in range(num_items):
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
            assignments[sampled].append(item)
            counts[sampled] += 1

        return assignments

    def _get_child_dimensions(
        self,
        layer_type: LayerType,
        l_vid: int,
        r_vid: int,
        n: int,
        l_is_constrained: bool,
        r_is_constrained: bool,
    ):
        G_l, H_l, G_r, H_r = 1, n, 1, n

        if self.max_sum_num_groups is not None:
            G_l = min(G_l, self.max_sum_num_groups)
            G_r = min(G_r, self.max_sum_num_groups)
            H_l = n // G_l
            H_r = n // G_r
            return G_l, H_l, G_r, H_r

        if layer_type == LayerType.UNIVERSAL:
            # doesn't really matter since the weights are dense over everything anyways
            pass
        elif layer_type in [LayerType.LEFT_MIXING, LayerType.PS_LEFT_MIXING]:
            a = find_closest_factor(n, max_val=math.sqrt(n))
            b = n // a
            G_l, H_l, G_r, H_r = a, b, b, a
        elif layer_type in [LayerType.RIGHT_MIXING, LayerType.PS_RIGHT_MIXING]:
            a = find_closest_factor(n, max_val=math.sqrt(n))
            b = n // a
            G_l, H_l, G_r, H_r = b, a, a, b
        elif layer_type == LayerType.SYNTHESIZING:
            a = find_closest_factor(n, max_val=math.sqrt(n))
            b = n // a

            if a == b:
                G_l, H_l, G_r, H_r = a, a, a, a

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
                favor_right = True
            elif r_is_synth_like and not l_is_synth_like:
                favor_right = False
            else:
                favor_right = True

            if favor_right:
                h_l, G_l = b, a
                G_r, h_r = b, a
            else:
                h_r, G_r = b, a
                h_l, G_l = a, b

            return G_l, h_l, G_r, h_r

        # Handle leaf cases
        if (
            self.vtree.is_leaf(l_vid)
            and l_is_constrained
            and (not self.vtree.is_leaf(r_vid) or not r_is_constrained)
            and self.max_leaf_num_nodes is not None
        ):
            # if only one child is a constrained leaf and max_leaf_num_nodes is set, then adapt the sibling sum layer's num_groups to match
            G_l = (
                n // self.max_leaf_num_nodes
                if self.max_leaf_num_groups is None
                else self.max_leaf_num_groups
            )
            H_l = self.max_leaf_num_nodes
            G_r = self.max_leaf_num_nodes
            H_r = n // self.max_leaf_num_nodes
        elif (
            self.vtree.is_leaf(r_vid)
            and r_is_constrained
            and (not self.vtree.is_leaf(l_vid) or not r_is_constrained)
            and self.max_leaf_num_nodes is not None
        ):
            G_r = (
                n // self.max_leaf_num_nodes
                if self.max_leaf_num_groups is None
                else self.max_leaf_num_groups
            )
            H_r = self.max_leaf_num_nodes
            G_l = self.max_leaf_num_nodes
            H_l = n // self.max_leaf_num_nodes
        elif (
            self.vtree.is_leaf(l_vid)
            and l_is_constrained
            and self.vtree.is_leaf(r_vid)
            and r_is_constrained
            and self.max_leaf_num_nodes is not None
        ):
            # if both children are constrained leaves and max_leaf_num_nodes is set, then adapt both sum layers' num_groups to match
            G_l = (
                n // self.max_leaf_num_nodes
                if self.max_leaf_num_groups is None
                else self.max_leaf_num_groups
            )
            H_l = self.max_leaf_num_nodes
            G_r = (
                n // self.max_leaf_num_nodes
                if self.max_leaf_num_groups is None
                else self.max_leaf_num_groups
            )
            H_r = self.max_leaf_num_nodes

        if G_l == 1 or G_r == 1 and layer_type == LayerType.SYNTHESIZING:
            logger.warning(
                f"Synthesizing SumLayer and vtree children ({l_vid}, {r_vid}) has G_l={G_l}, G_r={G_r}. This may lead to degenerate behavior."
            )
        return G_l, H_l, G_r, H_r

    def _generate_weights(
        self,
        G: int,
        H: int,
        G_L: int,
        H_L: int,
        G_R: int,
        H_R: int,
        layer_type: LayerType,
        is_constrained: bool,
        fairness_temperature: float = 2.0,
    ) -> torch.Tensor:
        """Generates global weight assignments ensuring determinism constraints."""

        w = torch.zeros((G, H, G_L, H_L, G_R, H_R))

        if layer_type == LayerType.UNIVERSAL:
            w = torch.exp(torch.randn((G, H, G_L, H_L, G_R, H_R)) * self.weight_softmax_temperature)

        elif layer_type in [LayerType.LEFT_MIXING, LayerType.PS_LEFT_MIXING]:
            assert is_constrained, "LEFT_MIXING should only occur for constrained nodes"
            h_l_assignments = self._sample_balanced_assignment(H_L, H, fairness_temperature)
            g_l_assignments = self._sample_balanced_assignment(H_L, G_L, fairness_temperature)
            g_r_assignments = self._sample_balanced_assignment(H_R * H_L, G_R, fairness_temperature)
            g_rs = [
                [h_r * H_L + h_l in g_r_assignments[g_t] for g_t in range(G_R)].index(True)
                for h_r in range(H_R)
                for h_l in range(H_L)
            ]
            g_rs = torch.tensor(g_rs).reshape(H_R, H_L)
            for g in range(G):
                for h in range(H):
                    h_ls = h_l_assignments[h]
                    g_ls = [
                        [h_l in g_l_assignments[g_t] for g_t in range(G_L)].index(True)
                        for h_l in h_ls
                    ]
                    if layer_type == LayerType.PS_LEFT_MIXING:
                        for idx, h_l in enumerate(h_ls):
                            for h_r in range(H_R):
                                g_r = g_rs[h_r, h_l]
                                w[g, h, g_ls[idx], h_l, g_r, h_r] = torch.exp(
                                    torch.randn(1) * self.weight_softmax_temperature
                                )
                    else:
                        w[g, h, g_ls, h_ls, :, :] = torch.exp(
                            torch.randn((G_R, H_R)) * self.weight_softmax_temperature
                        )

        elif layer_type in [LayerType.RIGHT_MIXING, LayerType.PS_RIGHT_MIXING]:
            assert is_constrained, "RIGHT_MIXING should only occur for constrained nodes"
            h_r_assignments = self._sample_balanced_assignment(H_R, H, fairness_temperature)
            g_r_assignments = self._sample_balanced_assignment(H_R, G_R, fairness_temperature)
            g_l_assignments = self._sample_balanced_assignment(H_L * H_R, G_L, fairness_temperature)
            g_ls = [
                [h_l * H_R + h_r in g_l_assignments[g_t] for g_t in range(G_L)].index(True)
                for h_l in range(H_L)
                for h_r in range(H_R)
            ]
            g_ls = torch.tensor(g_ls).reshape(H_L, H_R)
            for g in range(G):
                for h in range(H):
                    h_rs = h_r_assignments[h]
                    g_rs = [
                        [h_r in g_r_assignments[g_t] for g_t in range(G_R)].index(True)
                        for h_r in h_rs
                    ]
                    print(f"g={g}, h={h},\nh_rs={h_rs},\ng_rs={g_rs}")
                    if layer_type == LayerType.PS_RIGHT_MIXING:
                        for idx, h_r in enumerate(h_rs):
                            for h_l in range(H_L):
                                g_l = g_ls[h_l, h_r]
                                w[g, h, g_l, h_l, g_rs[idx], h_r] = torch.exp(
                                    torch.randn(1) * self.weight_softmax_temperature
                                )
                    else:
                        w[g, h, :, :, g_rs, h_rs] = torch.exp(
                            torch.randn((G_L, H_L)) * self.weight_softmax_temperature
                        )

        elif layer_type == LayerType.SYNTHESIZING:
            assert is_constrained, "SYNTHESIZING should only occur for constrained nodes"
            # assign each combination of left and right child nodes to a unique sum node
            flat_assignments = self._sample_balanced_assignment(H_L * H_R, H, fairness_temperature)
            assignments = [
                [(i // H_R, i % H_R) for i in row] for row in flat_assignments
            ]  # convert to (left, right) pairs
            for g in range(G):
                for h in range(H):
                    for h_l, h_r in assignments[h]:
                        # If H_R < G_L or H_L < G_R, then some groups will be unused,
                        # but this only occurs when both leaves are constrained and max_leaf_num_nodes is set (i.e. synthesizing).
                        # If we know this will happen, we can additionally set max_leaf_num_groups to avoid the wasted leaf groups.
                        g_l = h_r % G_L
                        g_r = h_l % G_R
                        w[g, h, g_l, h_l, g_r, h_r] = torch.exp(
                            torch.randn(1) * self.weight_softmax_temperature
                        )
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        w_sum = w.sum(dim=(2, 3, 4, 5), keepdim=True)

        safe_w_sum = torch.where(w_sum == 0, torch.ones_like(w_sum), w_sum)
        w = w / safe_w_sum

        for g in range(G):
            for n in range(H):
                w_to_print = (w > 0).int()
                logger.debug(
                    f"LayerType={layer_type}, is_constrained={is_constrained}, g={g}, n={n},\nG_L={G_L}, H_L={H_L}, G_R={G_R}, H_R={H_R},\nw[g,n,:,:,:, :]=\n{w_to_print[g, n, :, :, :, :].view(G_L * H_L, G_R * H_R)}"
                )

        return w

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
        is_constrained = not vnode.md_set.is_universal

        if self.vtree.is_leaf(vid):
            leaf_layer = self._handle_leaf(
                vnode,
                G if self.max_leaf_num_groups is None else self.max_leaf_num_groups,
                H if self.max_leaf_num_nodes is None else self.max_leaf_num_nodes,
                is_constrained,
            )

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
        l_is_constrained = not l_vnode.md_set.is_universal
        r_is_constrained = not r_vnode.md_set.is_universal

        G_L, H_L, G_R, H_R = self._get_child_dimensions(
            layer_type,
            children[0],
            children[1],
            self.num_nodes,
            l_is_constrained,
            r_is_constrained,
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

        print(f"Generating weights for node {vid} with scope {vnode.scope}")
        w = self._generate_weights(
            G,
            H,
            G_L,
            H_L,
            G_R,
            H_R,
            layer_type,
            is_constrained,
            self.fairness_temperature,
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


def create_md_circuit(
    input_dists: Dict[int, Distribution],
    md_var_decomp: VTree,
    num_nodes: int,
    initialize_weights: bool = False,
    fairness_temperature: float = 1.0,
    weight_softmax_temperature: float = 0.5,
    max_leaf_num_nodes: Optional[int] = None,
    max_leaf_num_groups: Optional[int] = None,
    max_sum_num_groups: Optional[int] = None,
    leaf_mixture_num_nodes: Optional[int] = None,
    leaf_mixture_num_groups: Optional[int] = None,
) -> SymbolicArithmeticCircuit:
    """Creates an MD-Circuit end-to-end."""

    c_builder = CircuitBuilder(
        vtree=md_var_decomp,
        num_nodes=num_nodes,
        input_dists=input_dists,
        initialize_weights=initialize_weights,
        fairness_temperature=fairness_temperature,
        weight_softmax_temperature=weight_softmax_temperature,
        max_leaf_num_nodes=max_leaf_num_nodes,
        max_leaf_num_groups=max_leaf_num_groups,
        max_sum_num_groups=max_sum_num_groups,
        leaf_mixture_num_nodes=leaf_mixture_num_nodes,
        leaf_mixture_num_groups=leaf_mixture_num_groups,
    )
    return c_builder.build()
